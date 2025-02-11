#!/usr/bin/env python

# Acados adds
from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import casadi as ca

# Common lib
import numpy as np
import scipy.linalg
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import math

class GemCarModel(object):
    def __init__(self,):

        model = AcadosModel()
        constraint = ca.types.SimpleNamespace()
        length = 2.565
        k = 1

        # control inputs
        v = ca.SX.sym('v')
        fai = ca.SX.sym('fai')
        controls = ca.vertcat(v, fai) # fai: front angle

        # model states
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(x, y, theta)
        

        # dynamic
        rhs = [v*ca.cos(theta), v*ca.sin(theta), v*ca.tan(fai)/(k*length)]

        # function
        f = ca.Function('f', [states, controls], [ca.vcat(rhs)], ['state', 'control_input'], ['rhs'])

        # acados model
        x_dot = ca.SX.sym('x_dot', len(rhs))
        f_impl = x_dot - f(states, controls)

        # other settings
        model.f_expl_expr = f(states, controls)
        model.f_impl_expr = f_impl
        model.x = states
        model.xdot = x_dot
        model.u = controls
        model.p = []
        model.name = 'GemCarModel'

        # constraints
        constraint.v_max = 5.0
        constraint.v_min = 0
        constraint.fai_max = np.pi / 5.2   # np.pi/5.2
        constraint.fai_min = -np.pi/ 5.2 # -np.pi
        constraint.expr = ca.vcat([v, fai])

        self.model = model
        self.constraint = constraint

class GemCarOptimizer(object):

    def __init__(self, m_model, m_constraint, t_horizon, dt, obstacles, target):

        model = m_model

        self.T = t_horizon
        self.dt = dt
        self.N = int(t_horizon / dt)

        # Car Info
        self.car_width = 1.5
        self.car_length = 2.565
        self.Epi = 3000

        self.target_x = target[0]
        self.target_y = target[1]
        self.target_theta = target[2]

        # Obstacle set here
        self.circle_obstacles_1 = {'x': 0, 'y': 20, 'r': 1.0}
        self.circle_obstacles_2 = {'x': 1, 'y': 25, 'r': 1.0}
        self.circle_obstacles_3 = {'x': -1, 'y': 30, 'r': 1.0}

        self.plot_figures = True

        # Ensure current working directory is current folder
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        self.acados_models_dir = './acados_models'
        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, acados_source_path)

        # Acados degrees (necessary)
        nx = model.x.size()[0]
        self.nx = nx
        nu = model.u.size()[0]
        self.nu = nu
        ny = nx + nu
        n_params = len(model.p) # number of extra parameters (0 here)

        # create OCP
        ocp = AcadosOcp()
        ocp.acados_include_path = acados_source_path + '/include'
        ocp.acados_lib_path = acados_source_path + '/lib'
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.T

        # initialize parameters
        ocp.dims.np = n_params
        ocp.parameter_values = np.zeros(n_params)

        # Cost settings ***
        Q = np.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.01]])
        R = np.array([[0.05, 0.0], [0.0, 0.05]])

        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = np.diag([1.0, 5.0, 0.5])
        ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
        ocp.model.cost_y_expr_e = model.x

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)

        # set constraints
        ocp.constraints.lbu = np.array([m_constraint.v_min, m_constraint.fai_min])
        ocp.constraints.ubu = np.array([m_constraint.v_max, m_constraint.fai_max])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.lbx = np.array([-10, -100, -np.pi, 0])
        ocp.constraints.ubx = np.array([10, 100, np.pi, 100])
        ocp.constraints.idxbx = np.array([0, 1, 2, 3])

        x_ref = np.zeros(nx)
        u_ref = np.zeros(nu)

        # obstacles
        x = ocp.model.x[0]
        y = ocp.model.x[1]

        obs_num = len(obstacles)
        obs = obstacles

        con_h_expr = []

        for i in range(obs_num):
            obs_x, obs_y = obs[i, 0], obs[i, 1]
            obs_radius = obs[i, 2]

            # nonlinear cons
            distance = ((x - obs_x) ** 2 + (y - obs_y) ** 2) - ((obs_radius) ** 2)

            # add to the list
            con_h_expr.append(distance)

        if con_h_expr:
            ocp.model.con_h_expr = ca.vertcat(*con_h_expr)
            ocp.constraints.lh = np.zeros((len(con_h_expr),))
            ocp.constraints.uh = 1000 * np.ones((len(con_h_expr),))

            # slack variable configuration
            nsh = len(con_h_expr)
            ocp.constraints.lsh = np.zeros(nsh)
            ocp.constraints.ush = np.zeros(nsh)
            ocp.constraints.idxsh = np.array(range(nsh))

            ns = len(con_h_expr)
            ocp.cost.zl = 1000 * np.ones((ns,))
            ocp.cost.Zl = 100 * np.ones((ns,))
            ocp.cost.zu = 0 * np.ones((ns,))
            ocp.cost.Zu = 0 * np.ones((ns,))


        # initial state
        ocp.constraints.x0 = x_ref
        ocp.cost.yref = np.concatenate((x_ref, u_ref))
        ocp.cost.yref_e = x_ref

        # solver options
        # ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'EXACT'
        ocp.solver_options.integrator_type = 'ERK' # 'IRK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP_RTI' #'SQP_RTI'

        # compile acados ocp
        json_file = os.path.join('./'+model.name+'_acados_ocp.json')
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)
        self.integrator = AcadosSimSolver(ocp, json_file=json_file)



    def solve(self, x_real, y_real, theta_real, vel_real):

        # current state
        x0 = np.array([x_real, y_real, theta_real])

        # terminal state
        xs = np.array([self.target_x, self.target_y, self.target_theta])

        # setting
        simX = np.zeros((self.N+1, self.nx))
        simU = np.zeros((self.N, self.nu))
        x_current = x0
        simX[0, :] = x0.reshape(1, -1)  

        # reference
        vel_ref = np.zeros(int(self.N))

        delta_x = self.target_x - x_real
        delta_y = self.target_y - y_real
        theta_between = np.abs(math.atan2(self.target_y - y_real, self.target_x - x_real))

        for i in range(self.N):
            xs_between = np.array([
                self.target_x, # delta_x / self.N * i + x_real,
                self.target_y, # delta_y / self.N * i + y_real,
                self.target_theta, # theta_between,
                0,
                0
            ])
            self.solver.set(i, 'yref', xs_between)
        self.solver.set(self.N, 'yref', xs)

        # Start Solving

        self.solver.set(0, 'lbx', x_current)
        self.solver.set(0, 'ubx', x_current)     
        status = self.solver.solve()

        if status != 0 :
            raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))
        
        simX[0, :] = self.solver.get(0, 'x')

        for i in range(self.N):
            # solve ocp
            simU[i, :] = self.solver.get(i, 'u')
            simX[i+1, :] = self.solver.get(i+1, 'x')

        # next state
        next_x = simX[1, 0]
        next_y = simX[1, 1]
        next_theta = simX[1, 2]
        aim_vel = simU[0, 0]
        aim_omega = simU[0, 1]

        return next_x, next_y, next_theta, aim_vel, aim_omega


    # plot function
    def plot_results(self, start_x, start_y, theta_log, x_log, y_log, x_real_log, y_real_log, v_log, fai_log):
        
        plt.figure()
        a = np.arange(0, (len(v_log)), 1)*self.dt
        plt.plot(a, v_log, 'r-', label='current_velocity')
        plt.xlabel('time')
        plt.ylabel('value')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.plot(a, y_log, 'r-', label='y trans')
        plt.xlabel('time')
        plt.ylabel('y-value')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.plot(a, fai_log, 'r-', label='desired fai')
        plt.xlabel('time')
        plt.ylabel('fai-value')
        plt.legend()
        plt.grid(True)
        plt.show()


        # Plot for angles
        t = np.arange(0, (len(theta_log)), 1)*self.dt
        plt.plot(t, theta_log, 'r-', label='desired theta')
        plt.xlabel('time')
        plt.ylabel('theta')
        plt.legend()
        plt.grid(True)
        plt.show()

        ## Plot for circle obstacles and x-y positions
        
        plt.plot(x_log, y_log, 'r-', label='desired path')
        plt.plot(x_real_log, y_real_log, color='b', linestyle='--', label='real path')
        plt.plot(self.target_x,self.target_y,'bo')
        plt.plot(start_x, start_y, 'go')
        plt.xlabel('pos_x')
        plt.ylabel('pos_y')

        target_circle1 = plt.Circle((self.circle_obstacles_1['x'], self.circle_obstacles_1['y']), self.circle_obstacles_1['r'], color='whitesmoke', fill=True)
        target_circle2 = plt.Circle((self.circle_obstacles_2['x'], self.circle_obstacles_2['y']), self.circle_obstacles_2['r'], color='whitesmoke', fill=True)
        target_circle3 = plt.Circle((self.circle_obstacles_3['x'], self.circle_obstacles_3['y']), self.circle_obstacles_3['r'], color='whitesmoke', fill=True)
        target_circle4 = plt.Circle((self.circle_obstacles_1['x'], self.circle_obstacles_1['y']), self.circle_obstacles_1['r'], color='k', fill=False)
        target_circle5 = plt.Circle((self.circle_obstacles_2['x'], self.circle_obstacles_2['y']), self.circle_obstacles_2['r'], color='k', fill=False)
        target_circle6 = plt.Circle((self.circle_obstacles_3['x'], self.circle_obstacles_3['y']), self.circle_obstacles_3['r'], color='k', fill=False)
        
        plt.gcf().gca().add_artist(target_circle1)
        plt.gcf().gca().add_artist(target_circle2)
        plt.gcf().gca().add_artist(target_circle3)
        plt.gcf().gca().add_artist(target_circle4)
        plt.gcf().gca().add_artist(target_circle5)
        plt.gcf().gca().add_artist(target_circle6)
        plt.axis('equal')
        plt.legend()
        plt.show()


    def main(self, x_init, y_init, theta_init, velocity_init):
                    
        x_0, y_0, theta, vel= x_init, y_init, theta_init, velocity_init
        x_real, y_real, theta_real, vel_real = x_init, y_init, theta_init, velocity_init

        x_log, y_log = [], []
        theta_log = []

        x_real_log, y_real_log = [], []
        v_log, fai_log = [], []

        with tqdm(total=100, desc='cpu%', position=1) as cpubar, tqdm(total=100, desc='ram%', position=0) as rambar:
            for i in tqdm(range(self.Epi)):

                try:
                    x_0, y_0, theta, vel, fai_0 = self.solve(x_real, y_real, theta_real, vel_real)
                    

                    x_real, y_real, theta_real, vel_real = x_0, y_0, theta, vel
                    
                    x_log.append(x_0)
                    y_log.append(y_0)
                    theta_log.append(theta)

                    x_real_log.append(x_real)
                    y_real_log.append(y_real)
                    v_log.append(vel)
                    fai_log.append(fai_0)

                    if (x_0 - self.target_x) ** 2 + (y_0 - self.target_y) ** 2 < 1:
                        # break
                        print("reach the target", theta)
                        if self.plot_figures == True:
                            self.plot_results(x_init, y_init, theta_log, x_log, y_log, x_real_log, y_real_log, v_log, fai_log)
                        return [1, theta_log], x_log, y_log

                except RuntimeError:
                    print("Infesible", theta)
                    if self.plot_figures == True:
                        self.plot_results(x_init, y_init, theta_log, x_log, y_log, x_real_log, y_real_log, v_log, fai_log)
                    return [0, theta_log], x_log, y_log

            print("not reach the target", theta)
            if self.plot_figures == True:
                self.plot_results(x_init, y_init, theta_log, x_log, y_log, x_real_log, y_real_log, v_log, fai_log)
            return [0, theta_log], x_log, y_log

if __name__ == '__main__':

    obstacles = np.array([
    [0.0, 20, 1],        #x, y, r 20 25 30
    [-1.0, 25, 1],
    [1.0, 30, 1]
    ])

    start_x, start_y, theta, vel = -0.25, -0.3, np.pi/2, 0.0007
    terminal = np.array([0.0, 40.0, np.pi/2])


    car_model = GemCarModel()
    opt = GemCarOptimizer(m_model=car_model.model, 
                               m_constraint=car_model.constraint, t_horizon=1.0, dt=0.05, obstacles = obstacles, target=terminal)
    opt.main(start_x, start_y, theta, vel)