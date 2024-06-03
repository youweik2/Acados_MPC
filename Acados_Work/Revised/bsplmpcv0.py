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
from B_spline import Bspline, Bspline_basis


class bspline_MPC(object):
    def __init__(self, T, dt, target_x, target_y):
        
        # Time constant
        self.dt = dt # time frequency 20Hz
        self.T = T # time horizon
        self.N = int(self.T/self.dt) # number of control intervals

        self.Epi = 500 # number of episodes
        self.plot_figures = False
        self.step_plotting = False # plot or not

        # Target Input
        self.target_x = target_x
        self.target_y = target_y
        
        # Basic Info
        self.initial_pos_sin_obs = 1  # initial position of sin obstacles
        self.upper_limit = 1.5
        self.lower_limit = -2.0
        self.poly_degree = 3
        self.num_ctrl_points = 4

        # Dimension
        self.u_dim = 8
        self.x_dim = 3

        # Car Info
        self.car_width = 1.5 # 55.5in = 1.4107m
        self.car_length = 2.7 # 103in = 2.6162m

        # obstacles in simulation
        self.circle_obstacles_1 = {'x': 0, 'y': 20, 'r': 1.0}
        self.circle_obstacles_2 = {'x': 1, 'y': 25, 'r': 1.0}
        self.circle_obstacles_3 = {'x': -1, 'y': 30, 'r': 1.0}

    
    def bspline_mpc_model(self) -> AcadosModel:
        
        model_name = 'bspline_mpc_model'
        
        ## Vector ##

        # Controls

        self.u = SX.sym("u",self.u_dim)

        # State
        x = SX.sym('x')
        y = SX.sym('y')
        theta = SX.sym('theta')

        self.x = vertcat(x, y, theta)
        
        # time vectors for Bspline
        self.tau_0 = SX.sym('tau_0') # time
        self.tau_i = SX.sym('tau_i')   # time interval i
        self.tau_i1 = SX.sym('tau_i1')   # time interval i+1


        # xdot
        self.dx = SX.sym('dx')
        self.dy = SX.sym('dy')
        self.omega = SX.sym('omega')

        self.x_dot = vertcat(self.dx, self.dy, self.omega)

        # k list
        self.k32 = (3*(self.tau_0 - self.tau_i))/(self.tau_i - self.tau_i1) + (3*(self.tau_0 - self.tau_i)**2)/(self.tau_i - self.tau_i1)**2 + (self.tau_0 - self.tau_i)**3/(self.tau_i - self.tau_i1)**3 + 1
        self.k11 = np.cos(self.x[2])*self.k32
        self.k21 = np.sin(self.x[2])*self.k32
        self.k34 = - (3*(self.tau_0 - self.tau_i))/(self.tau_i - self.tau_i1) - (6*(self.tau_0 - self.tau_i)**2)/(self.tau_i - self.tau_i1)**2 - (3*(self.tau_0 - self.tau_i)**3)/(self.tau_i - self.tau_i1)**3
        self.k13 = np.cos(self.x[2])*self.k34
        self.k23 = np.sin(self.x[2])*self.k34
        self.k36 = (3*(self.tau_0 - self.tau_i)**2)/(self.tau_i - self.tau_i1)**2 + (3*(self.tau_0 - self.tau_i)**3)/(self.tau_i - self.tau_i1)**3
        self.k15 = np.cos(self.x[2])*self.k36
        self.k25 = np.sin(self.x[2])*self.k36
        self.k38 = -(self.tau_0 - self.tau_i)**3/(self.tau_i - self.tau_i1)**3
        self.k17 = np.cos(self.x[2])*self.k38
        self.k27 = np.sin(self.x[2])*self.k38

        dx = self.k11*self.u[0] + self.k13*self.u[2] + self.k15*self.u[4] + self.k17*self.u[6]
        dy = self.k21*self.u[0] + self.k23*self.u[2] + self.k25*self.u[4] + self.k27*self.u[6]
        omega = self.k32*self.u[1] + self.k34*self.u[3] + self.k36*self.u[5] + self.k38*self.u[7]

        rhs = [dx, dy, omega]

        self.x_dot = SX.sym('x_dot', self.x_dim)

        self.f = Function('f', [self.x, self.u, self.tau_0, self.tau_i, self.tau_i1], [vcat(rhs)])

        # dynamics
        self.f_expl = self.f(self.x, self.u, self.tau_0, self.tau_i, self.tau_i1)
        
        self.f_impl = self.x_dot - self.f_expl

        model = AcadosModel()

        model.f_impl_expr = self.f_impl
        model.f_expl_expr = self.f_expl
        model.x_dot = self.x_dot
        model.x = self.x
        model.u = self.u
        # model.p = [] #extra parameters
        model.name = model_name

        return model


    def get_coef(self, tau_i, tau_i1):

        gm11 = tau_i1/7 - tau_i/7
        gm21 = tau_i1/14 - tau_i/14
        gm31 = tau_i1/35 - tau_i/35
        gm41 = tau_i1/140 - tau_i/140
        gm12 = gm21
        gm22 = (3*tau_i1)/35 - (3*tau_i)/35
        gm32 = (9*tau_i1)/140 - (9*tau_i)/140
        gm42 = tau_i1/35 - tau_i/35
        gm13 = gm31
        gm23 = gm32
        gm33 = (3*tau_i1)/35 - (3*tau_i)/35
        gm43 = tau_i1/14 - tau_i/14
        gm14 = gm41
        gm24 = gm42
        gm34 = gm43
        gm44 = tau_i1/7 - tau_i/7
        
        return [gm11, gm21, gm31, gm41, gm12, gm22, gm32, gm42, gm13, gm23, gm33, gm43, gm14, gm24, gm34, gm44]

    def cost_function_ctrlpoints(self, cp, tau_i, tau_i1):
        gm = self.get_coef(tau_i, tau_i1)
        cost = 0
        for i in range(4):
            for j in range(4):
                cost +=  gm[i*4+j] * cp[j] @ cp[i].T 
        return cost
    

    def solver_mpc(self, m_model, x_init, y_init, theta_init, obstacles):

        model = m_model
        ocp = AcadosOcp()

        # num of state & extra para
        nx = self.x_dim
        nu = self.u_dim
        ny = nu + nx
        ny_cost = 16 + nx
        ny_e = nx

        # set ocp
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.T

        # simplify the expression of x & u
        u = ocp.model.u
        x = ocp.model.x[0]  # x position
        y = ocp.model.x[1]  # y position
        theta = ocp.model.x[2]

        # set optimizer
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.print_level = 0

        # cost function type: LINEAR or NONLINEAR? Here use external type
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        # cost function        
        ctrl_point_1 = np.array([u[0], u[1]])
        ctrl_point_2 = np.array([u[2], u[3]])
        ctrl_point_3 = np.array([u[4], u[5]])
        ctrl_point_4 = np.array([u[6], u[7]])
        cp = [ctrl_point_1, ctrl_point_2, ctrl_point_3, ctrl_point_4]

        Q = 2 * np.diag([10, 10, 1])
        Q_e = 2 * np.diag([10, 10, 0])

        cost_ctrl = self.cost_function_ctrlpoints(cp, 0, 1)

        ocp.model.cost_expr_ext_cost = model.x.T @ Q @ model.x + cost_ctrl
        ocp.model.cost_expr_ext_cost_e = model.x.T @ Q_e @ model.x

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)

        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((nx,))

        # constraint
        v_limit = 1.5
        omega_limit = 3.0
        constraint_k = omega_limit/v_limit

        ocp.constraints.lbu = np.array([-v_limit, -omega_limit])
        ocp.constraints.ubu = np.array([v_limit, omega_limit])
        ocp.constraints.idxbu = np.array([0, 1])

        # define the constraints in u
        ctrl_constraint_leftupper = lambda ctrl_point: constraint_k*ctrl_point + omega_limit
        ctrl_constraint_rightlower = lambda ctrl_point: constraint_k*ctrl_point - omega_limit
        ctrl_constraint_leftlower = lambda ctrl_point: -constraint_k*ctrl_point - omega_limit
        ctrl_constraint_rightupper = lambda ctrl_point: -constraint_k*ctrl_point + omega_limit

        con_h_expr = []  # list to collect constraints

        for i in range(4):
            con_lu = ctrl_constraint_leftupper(u[i*2]) - ctrl_constraint_leftupper(u[i*2+1])
            con_ru = ctrl_constraint_rightupper(u[i*2]) - ctrl_constraint_rightupper(u[i*2+1])
            con_ll = ctrl_constraint_leftlower(u[i*2+1]) - ctrl_constraint_leftlower(u[i*2])
            con_rl = ctrl_constraint_rightlower(u[i*2+1]) - ctrl_constraint_rightlower(u[i*2])

            con_h_expr.append(con_lu)
            con_h_expr.append(con_ru)
            con_h_expr.append(con_ll)
            con_h_expr.append(con_rl)

        # add the obstacles constraints

        # assume the point is the center of the car

        obs_num, obs_dim = obstacles.shape
        obs = obstacles

        xlu = self.car_length*cos(theta)/2 - self.car_width*sin(theta)/2 + x
        xll = -self.car_length*cos(theta)/2 - self.car_width*sin(theta)/2 + x
        xru = self.car_length*cos(theta)/2 + self.car_width*sin(theta)/2 + x
        xrl = -self.car_length*cos(theta)/2 + self.car_width*sin(theta)/2 + x

        ylu = self.car_length*sin(theta)/2 + self.car_width*cos(theta)/2 + y
        yll = -self.car_length*sin(theta)/2 + self.car_width*cos(theta)/2 + y
        yru = self.car_length*cos(theta)/2 - self.car_width*cos(theta)/2 + y
        yrl = -self.car_length*cos(theta)/2 - self.car_width*cos(theta)/2 + y
        
        for i in range(obs_num):

            obs_x, obs_y = obs[i, 0], obs[i, 1]
            obs_radius = obs[i, 2]

            # nonlinear cons
            dis_lu = ((xlu - obs_x)**2 + (ylu - obs_y)**2) - ((obs_radius)**2)
            dis_ll = ((xll - obs_x)**2 + (yll - obs_y)**2) - ((obs_radius)**2)
            dis_ru = ((xru - obs_x)**2 + (yru - obs_y)**2) - ((obs_radius)**2)
            dis_rl = ((xrl - obs_x)**2 + (yrl - obs_y)**2) - ((obs_radius)**2)

            # add to the list
            con_h_expr.append(dis_lu)
            con_h_expr.append(dis_ll)
            con_h_expr.append(dis_ru)
            con_h_expr.append(dis_rl)

        ocp.model.con_h_expr = vertcat(*con_h_expr)
        ocp.constraints.lh = np.zeros((len(con_h_expr),))
        ocp.constraints.uh = 1000 * np.ones((len(con_h_expr),))
        

        # save model
        solver_json = 'acados_ocp_' + model.name + '.json'
        acados_ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json)
        acados_integrator = AcadosSimSolver(ocp, json_file=solver_json)

        return acados_ocp_solver, acados_integrator

    def solve(self, x_init, y_init, theta_init, obss): 

        # init solver

        nx = self.x_dim

        mpc_model = self.bspline_mpc_model()

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
        x_middle = np.concatenate((xe, np.zeros(8)))

        for i in range(self.N-1):

            ocp_solver.set(i, 'yref', x_middle)

        # solve for next state
        for i in range(self.N):

            ocp_solver.set(0, 'lbx', x_current)
            ocp_solver.set(0, 'ubx', x_current)

            # solve ocp and get next control input

            self.sim_u[i,:] = ocp_solver.solve_for_x0(x0_bar = self.sim_x[i, :])

            self.sim_t[i] = ocp_solver.get_stats('time_tot')

            self.sim_x[i+1, :] = acados_integrator.simulate(x=self.sim_x[i, :], u=self.sim_u[i,:])
        
        # next state
        next_x = self.sim_x[1, 0]
        next_y = self.sim_x[1, 1]
        next_theta = self.sim_x[1, 2]
        
        return next_x, next_y, next_theta, self.sim_u, self.sim_x
    
    def dynamic_model_bspline(self, x, y, theta, ctrls):
        len_ctrls = len(ctrls)
        xx = []
        yy = []
        ttheta = []
        for i in range(len_ctrls):
            v = ctrls[i][0]
            w = ctrls[i][1]
            x = x + v * np.cos(theta) * self.dt/len_ctrls
            y = y + v * np.sin(theta) * self.dt/len_ctrls
            theta = theta + w * self.dt/len_ctrls
            xx.append(x)
            yy.append(y)
            ttheta.append(theta)

        x_next, y_next, theta_next = x, y, theta
        return x_next, y_next, theta_next, xx, yy, ttheta
    
    
    def main(self, x_init, y_init, theta_init, obss):

        x_simulation = []
        y_simulation = []
        #default obstacle
       
        obstacles = obss

        obs_num, obs_dim = obstacles.shape  # get the num of obstacles

        start_x, start_y = x_init, y_init                   
        x_0, y_0, theta = start_x, start_y, theta_init
        x_real, y_real, theta_real = start_x, start_y, theta_init

        theta_0 = theta_init            # Save the initial theta
        U_real = np.array([0.0, 0.0])

        x_0, y_0, theta, U, X = self.solve(x_real, y_real, theta_real, obss)
        print(x_0, y_0, theta)
        '''
        x_log, y_log = [x_0], [y_0]
        theta_log = [theta]
        U_log = []

        x_real_log, y_real_log = [x_real], [y_real]
        theta_real_log = [theta_real]
        U_real_log = []

        with tqdm(total=100, desc='cpu%', position=1) as cpubar, tqdm(total=100, desc='ram%', position=0) as rambar:
            for i in tqdm(range(self.Epi)):

                try:
                    x_0, y_0, theta, U, X = self.solve(x_real, y_real, theta_real, obss)
                    desire_ctrl = U.T[0]
                    x_real, y_real, theta_real = x_0, y_0, theta

                    x_log.append(x_0)
                    y_log.append(y_0)
                    theta_log.append(theta)
                    U_log.append(desire_ctrl)

                    x_real_log.append(x_real)
                    y_real_log.append(y_real)
                    theta_real_log.append(theta_real)
                    U_real_log.append(U_real)

                    if (x_0 - self.target_x) ** 2 + (y_0 - self.target_y) ** 2 < 0.01:
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
        '''
if __name__ == "__main__":

    target_x, target_y = 0.0, 40.0               # ENV 2 target point
    start_x, start_y = 0.0, 0.0                # ENV 2 start point


    mpc = bspline_MPC(T = 1.0, dt = 0.02, target_x=target_x, target_y=target_y)
    
    theta = 0.5 * np.pi

    obstacles =np.array([
    [-5, 20, 6]
    ])
    '''
    obstacles = np.array([
    [0, 20, 1],        #x, y, r
    [1, 25, 1],
    [-1.0, 30, 1]
    ])
    '''
    mpc.main(start_x, start_y, theta, obstacles)





