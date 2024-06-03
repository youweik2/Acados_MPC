import numpy as np
from casadi import *
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosMultiphaseOcp, AcadosModel, AcadosSimSolver
import matplotlib.pyplot as plt
from copy import deepcopy
import json
import scipy.linalg
import pickle
from time import sleep
from tqdm import tqdm


class base_nlcons_MPC(object):
    def __init__(self, T, dt, target_x, target_y):
        
        # Time constant
        self.dt = dt # time frequency 20Hz
        self.T = T # time horizon
        self.N = int(self.T/self.dt) # number of control intervals

        self.Epi = 500 # number of episodes
        self.plot_figures = True
        self.step_plotting = False # plot or not

        # Target Input
        self.target_x = target_x
        self.target_y = target_y
        
        # Basic Info
        self.initial_pos_sin_obs = 1  # initial position of sin obstacles
        self.upper_limit = 1.5
        self.lower_limit = -2.0

        # Dimension
        self.u_dim = 2
        self.x_dim = 3

        # Car Info
        self.car_width = 1.5 # 55.5in = 1.4107m
        self.car_length = 2.7 # 103in = 2.6162m

        # obstacles in simulation
        self.circle_obstacles_1 = {'x': 0, 'y': 20, 'r': 1.0}
        self.circle_obstacles_2 = {'x': 1, 'y': 25, 'r': 1.0}
        self.circle_obstacles_3 = {'x': -1, 'y': 30, 'r': 1.0}

    
    def base_mpc_model(self) -> AcadosModel:
        
        model_name = 'base_mpc_model'

        # Vector
        v = SX.sym('v')
        omega = SX.sym('omega')

        self.u = vertcat(v, omega)

        x = SX.sym('x')
        y = SX.sym('y')
        theta = SX.sym('theta')

        self.x = vertcat(x, y, theta)


        rhs = [v*cos(theta), v*sin(theta), omega]
        self.x_dot = SX.sym('x_dot', len(rhs))

        self.f = Function('f', [self.x, self.u], [vcat(rhs)], ['state', 'control_input'], ['rhs'])
        
        # dynamics
        self.f_impl = self.x_dot - self.f(self.x, self.u)
        
        model = AcadosModel()

        model.f_expl_expr = self.f(self.x, self.u)
        model.f_impl_expr = self.f_impl
        model.x_dot = self.x_dot
        model.x = self.x
        model.u = self.u
        model.name = model_name


        return model


    def solver_mpc(self, m_model, x_init, y_init, theta_init, obstacles):
        model = m_model
        ocp = AcadosOcp()

        # num of state & extra para
        nx = self.x_dim
        nu = self.u_dim
        ny = nu + nx
        ny_e = nx

        # set ocp
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.T

        # set optimizer
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.print_level = 0

        # cost function type: LINEAR or NONLINEAR?
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'

        # constraint
        v_limit = 1.5
        omega_limit = 3.0
        ocp.constraints.lbu = np.array([-v_limit, -omega_limit])
        ocp.constraints.ubu = np.array([v_limit, omega_limit])
        ocp.constraints.idxbu = np.array([0, 1])

        # init position
        x0 = np.zeros((nx))
        x0[0] = x_init
        x0[1] = y_init
        x0[2] = theta_init

        ocp.constraints.x0 = x0

        # cost function
        Q = np.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0001]])
        R = np.array([[0.5, 0.0], [0.0, 0.05]])
        Q_e = 2 * np.diag([2, 5, 0])

        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Q_e

        ocp.model.cost_y_expr = vertcat(model.x, model.u)
        ocp.model.cost_y_expr_e = model.x

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)

        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((nx,))

        x = ocp.model.x[0]  # x position
        y = ocp.model.x[1]  # y position

        obs_num, obs_dim = obstacles.shape
        obs = obstacles

        '''
        con_h_expr = []  # list to collect constraints
        
        for i in range(obs_num):
            obs_x, obs_y = obs[i, 0], obs[i, 1]
            obs_radius = obs[i, 2]

            # nonlinear cons
            distance = ((x - obs_x)**2 + (y - obs_y)**2) - ((obs_radius)**2)

            # add to the list
            con_h_expr.append(distance)

        if con_h_expr:
            ocp.model.con_h_expr = vertcat(*con_h_expr)
            ocp.constraints.lh = np.zeros((len(con_h_expr),))
            ocp.constraints.uh = 1000 * np.ones((len(con_h_expr),))
        
        '''

        # save model
        solver_json = 'acados_ocp_' + model.name + '.json'
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json)
        acados_integrator = AcadosSimSolver(ocp, json_file=solver_json)

        return acados_ocp_solver, acados_integrator


 
    def solve(self, x_init, y_init, theta_init, obss): 

        # init solver

        nx = self.x_dim

        mpc_model = self.base_mpc_model()

        ocp_solver, acados_integrator = self.solver_mpc(mpc_model, x_init, y_init, theta_init, obss)
        
        
        # define update info
        self.sim_x = np.zeros((self.N+1, self.x_dim))
        self.sim_u = np.zeros((self.N, self.u_dim))
        self.sim_t = np.zeros((self.N))

        # initial position

        x0 = np.zeros((nx,))
        x0[0] = x_init
        x0[1] = y_init
        x0[2] = theta_init

        x_current = x0
        self.sim_x[0, :] = x0 

        # final position
        xe = np.zeros((nx,))
        xe[0] = self.target_x
        xe[1] = self.target_y
        xe[2] = np.pi/2
        
        ocp_solver.set(self.N, 'yref', xe)

        # middle position
        u_control = np.zeros(2)
        x_middle = np.concatenate((xe, u_control))
        
        for i in range(self.N-1):
            ocp_solver.set(i, 'yref', x_middle)
        
        # do some initial iterations to start with a good initial guess
        # num_iter_initial = 5
        # for _ in range(num_iter_initial):
        #     ocp_solver.solve_for_x0(x0_bar = x0)


        for i in range(self.N):

            ocp_solver.set(0, 'lbx', x_current)
            ocp_solver.set(0, 'ubx', x_current)

            status = ocp_solver.solve()

            if status != 0 :
                raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))

            self.sim_u[i, :] = ocp_solver.get(0, 'u')
            # SIMULATION

            acados_integrator.set('x', x_current)
            acados_integrator.set('u', self.sim_u[i, :])

			# CALCULATE
            status_s = acados_integrator.solve()
            if status_s != 0:
                raise Exception('acados integrator returned status {}. Exiting.'.format(status))

            #print(x_current)
            x_current = acados_integrator.get('x')
            self.sim_x[i+1, :] = x_current

        # next state
        next_x = self.sim_x[1, 0]
        next_y = self.sim_x[1, 1]
        next_theta = self.sim_x[1, 2]
        velocity = self.sim_u[1, 0]
        omega = self.sim_u[1, 1]

        return next_x, next_y, next_theta, self.sim_u, self.sim_x, velocity, omega
    

    def main(self, x_init, y_init, theta_init, obstacles):
        
        start_x, start_y = x_init, y_init                
        x_0, y_0, theta = start_x, start_y, theta_init
        x_real, y_real, theta_real = start_x, start_y, theta_init
        theta_0 = theta_init            # Save the initial theta
        U_real = np.array([0.0, 0.0])

        x_log, y_log = [x_0], [y_0]
        theta_log = [theta]
        U_log = []

        x_real_log, y_real_log = [x_real], [y_real]
        theta_real_log = [theta_real]
        U_real_log = []

        with tqdm(total=100, desc='cpu%', position=1) as cpubar, tqdm(total=100, desc='ram%', position=0) as rambar:
            for i in tqdm(range(self.Epi)):

                try:
                    x_0, y_0, theta, U, X, vel, dtht = self.solve(x_real, y_real, theta_real, obstacles)

                    print(x_0, y_0, theta, vel, dtht)
                    #print('x',self.sim_x)
                    #print('u',self.sim_u)
                    x_real, y_real, theta_real = x_0, y_0, theta
                    desire_ctrl = U.T[0]
                    U_real = desire_ctrl
                    
                    x_log.append(x_0)
                    y_log.append(y_0)
                    theta_log.append(theta)
                    U_log.append(desire_ctrl)

                    x_real_log.append(x_real)
                    y_real_log.append(y_real)
                    theta_real_log.append(theta_real)
                    U_real_log.append(U_real)

                    if (x_0 - self.target_x) ** 2 + (y_0 - self.target_y) ** 2 < 0.01:
                        # break
                        print("reach the target", theta_0)
                        if self.plot_figures == True:
                            self.plot_results(start_x, start_y, theta_log, U_log, x_log, y_log, x_real_log, y_real_log, U_real_log, theta_real_log)
                        return [1, theta_log], x_log, y_log

                except RuntimeError:
                    print("Infesible", theta_0)
                    if self.plot_figures == True:
                        self.plot_results(start_x, start_y, theta_log, U_log, x_log, y_log, x_real_log, y_real_log, U_real_log, theta_real_log)
                    return [0, theta_log], x_log, y_log

            print("not reach the target", theta_0)
            if self.plot_figures == True:
                self.plot_results(start_x, start_y, theta_log, U_log, x_log, y_log, x_real_log, y_real_log, U_real_log, theta_real_log)
            return [0, theta_log], x_log, y_log
            
        # Plot for control signals


    def plot_results(self, start_x, start_y, theta_log, U_log, x_log, y_log, x_real_log, y_real_log, U_real_log, theta_real_log):

        tt = np.arange(0, (len(U_log)), 1)*self.dt
        t = np.arange(0, (len(theta_log)), 1)*self.dt
        plt.plot(tt, U_log, 'r-', label='desired U')
        plt.plot(tt, U_real_log, 'b-', label='U_real', linestyle='--')
        plt.xlabel('time')
        plt.ylabel('U')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot for angles
        
        plt.plot(t, theta_log, 'r-', label='desired theta')
        # plt.plot(t, theta_real_log, 'b-', label='theta_real')
        plt.xlabel('time')
        plt.ylabel('theta')
        plt.legend()
        plt.grid(True)
        plt.show()
        '''
        ## Plot for sin obstacles and x-y positions env1
        if self.env_numb == 1:
            plt.plot(x_log, y_log, 'r-', label='desired path')
            plt.plot(x_real_log, y_real_log, 'b-', label='real path', linestyle='--')
            plt.plot(self.target_x,self.target_y,'bo')
            plt.plot(start_x, start_y, 'go')
            plt.xlabel('pos_x')
            plt.ylabel('pos_y')
            x = np.arange(start_x-1,4,0.01)
            y = np.sin(0.5 * pi * x) + self.initial_pos_sin_obs
            plt.plot(x, y, 'g-', label='upper limit')
            plt.plot(x, y-2*self.initial_pos_sin_obs, 'b-', label='lower limit')
            plt.show()
        '''
        ## Plot for circle obstacles and x-y positions env2
        ## if self.env_numb == 2:
        plt.plot(x_log, y_log, 'r-', label='desired path')
        plt.plot(x_real_log, y_real_log, 'b-', label='real path', linestyle='--')
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
        # plt.axis([-5.0, 1.5, -2.4, 2.4])
        plt.axis('equal')
        # x = np.arange(start_x-1,4,0.01)
        # plt.plot(x, len(x)*[self.upper_limit], 'g-', label='upper limit')
        # plt.plot(x, len(x)*[self.lower_limit], 'b-', label='lower limit')
        plt.legend()
        plt.show()

        with open('single_traj_mpc_50hz.pkl', 'wb') as f:
            pickle.dump([x_log, y_log], f)


if __name__ == "__main__":

    target_x, target_y = 1.0, 0.1               # ENV 2 target point
    start_x, start_y = 0.0, 0.0                # ENV 2 start point


    mpc = base_nlcons_MPC(T = 2.0, dt = 0.05, target_x=target_x, target_y=target_y)
    
    theta = 0.5 * np.pi
    
    obstacles = np.array([
    [0, 20, 1],        #x, y, r 20 25 30
    [-1, 25, 1],
    [1.0, 30, 1]
    ])
    
    mpc.main(start_x, start_y, theta, obstacles)





