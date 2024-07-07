#!/usr/bin/env python

import os
import sys
import shutil
import errno
import timeit

from GemCarBspl import GemCarBsplModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import logging
import subprocess

import casadi as ca
import numpy as np
import scipy.linalg
from tqdm import tqdm
from time import sleep

import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
import math
from B_spline import Bspline, Bspline_basis
import matplotlib.pyplot as plt



def safe_mkdir_recursive(directory, overwrite=False):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(directory):
                pass
            else:
                raise
    else:
        if overwrite:
            try:
                shutil.rmtree(directory)
            except:
                print('Error while removing directory {}'.format(directory))

def coefficients(tau_i, tau_i1):

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

def cost_function_ctrlpoints(cp, tau_i, tau_i1):

    gm = coefficients(tau_i, tau_i1)
    cost = 0
    for i in range(4):
        for j in range(4):
            cost +=  gm[i*4+j] * cp[j] @ cp[i].T

    return cost

def find_correct_index(array, value):

    indicator = 0
    index = 0
    while indicator == 0:
        indicator = 1 if array[index] <= value < array[index+1] else 0
        index += 1
    return index - 1



class GemCarOptimizer(object):

    def __init__(self, m_model, t_horizon, dt, obstacles):

        model = m_model

        self.T = t_horizon
        self.dt = dt
        self.N = int(t_horizon/dt)

        # Car Info
        self.car_width = 1.5 # 55.5in = 1.4107m
        self.car_length = 2.7 # 103in = 2.6162m
        self.Epi = 3000

        self.car_collision = 0 # consider car info or not **** Important param

        self.target_x = 0.0
        self.target_y = 40.0
        self.target_theta = np.pi/2

        # some info of K knots
        self.poly_degree = 3
        self.num_ctrl_points = 4

        # basic info
        self.circle_obstacles_1 = {'x': 0, 'y': 20, 'r': 1.0}
        self.circle_obstacles_2 = {'x': 1, 'y': 25, 'r': 1.0}
        self.circle_obstacles_3 = {'x': -1, 'y': 30, 'r': 1.0}

        self.plot_figures = False
        self.step_plotting = False
        self.env_numb = 2 

        self.gap = 2.5   # gap between upper and lower limit
        self.initial_pos_sin_obs = self.gap/2   # initial position of sin obstacles

        self.upper_limit = 1.5 
        self.lower_limit = -2.0 

        # Ensure current working directory is current folder
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        self.acados_models_dir = './acados_models'
        safe_mkdir_recursive(os.path.join(os.getcwd(), self.acados_models_dir))
        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, acados_source_path)

        nx = model.x.size()[0]
        self.nx = nx
        nu = model.u.size()[0]
        self.nu = nu
        ny = nx + nu
        ntau = 3

        # create OCP
        ocp = AcadosOcp()
        ocp.acados_include_path = acados_source_path + '/include'
        ocp.acados_lib_path = acados_source_path + '/lib'
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.T

        # initialize parameters
        ocp.dims.np = ntau
        ocp.parameter_values = np.zeros(ntau)

        # cost type

        # calculate cost function

        Q = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.01]]) # X states cost
        #R = coefficients(0, 1)

        U = ocp.model.u
        ctrl_point_1 = np.array([U[0], U[1]])
        ctrl_point_2 = np.array([U[2], U[3]])
        ctrl_point_3 = np.array([U[4], U[5]])
        ctrl_point_4 = np.array([U[6], U[7]])
        cp = [ctrl_point_1, ctrl_point_2, ctrl_point_3, ctrl_point_4]

        x_aim = np.array([self.target_x, self.target_y, self.target_theta])
        x_gap = x_aim - ocp.model.x

        ocp.model.cost_expr_ext_cost = x_gap.T @ Q @ x_gap + cost_function_ctrlpoints(cp, 0, 1)
        ocp.model.cost_expr_ext_cost_e = x_gap.T @ Q @ x_gap
        
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        '''
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        '''

        # set constraints
        
        v_limit = 1.5
        omega_limit = 3.0
        constraint_k = omega_limit/v_limit

        ocp.constraints.constr_type = 'BGH'

        # define the constraints in u
        ctrl_constraint_leftupper = lambda ctrl_point: constraint_k*ctrl_point + omega_limit
        ctrl_constraint_rightlower = lambda ctrl_point: constraint_k*ctrl_point - omega_limit
        ctrl_constraint_leftlower = lambda ctrl_point: -constraint_k*ctrl_point - omega_limit
        ctrl_constraint_rightupper = lambda ctrl_point: -constraint_k*ctrl_point + omega_limit

        # obstacles
        x = ocp.model.x[0]  # x position
        y = ocp.model.x[1]  # y position

        obs_num, obs_dim = obstacles.shape
        obs = obstacles

        con_h_expr = []  # list to collect constraints

        
        for i in range(4):
            con_lu = ctrl_constraint_leftupper(U[i*2]) - ctrl_constraint_leftupper(U[i*2+1])
            con_ru = ctrl_constraint_rightupper(U[i*2]) - ctrl_constraint_rightupper(U[i*2+1])
            con_ll = ctrl_constraint_leftlower(U[i*2+1]) - ctrl_constraint_leftlower(U[i*2])
            con_rl = ctrl_constraint_rightlower(U[i*2+1]) - ctrl_constraint_rightlower(U[i*2])

            con_h_expr.append(con_lu)
            con_h_expr.append(con_ru)
            con_h_expr.append(con_ll)
            con_h_expr.append(con_rl)


        for i in range(obs_num):
            obs_x, obs_y = obs[i, 0], obs[i, 1]
            obs_radius = obs[i, 2]

            # nonlinear cons
            distance = ((x - obs_x)**2 + (y - obs_y)**2) - ((obs_radius)**2)

            # add to the list
            con_h_expr.append(distance)


        if con_h_expr:
            ocp.model.con_h_expr = ca.vertcat(*con_h_expr)
            ocp.constraints.lh = np.zeros((len(con_h_expr),))
            ocp.constraints.uh = 10 * np.ones((len(con_h_expr),))

            #slack variable configuration:

            nsh = len(con_h_expr)
            ocp.constraints.lsh = np.zeros(nsh)             # Lower bounds on slacks corresponding to soft lower bounds for nonlinear constraints
            ocp.constraints.ush = np.zeros(nsh)             # Lower bounds on slacks corresponding to soft upper bounds for nonlinear constraints
            ocp.constraints.idxsh = np.array(range(nsh))    # Jsh


            ns = len(con_h_expr)
            ocp.cost.zl = 10 * np.ones((ns,)) # gradient wrt lower slack at intermediate shooting nodes (1 to N-1)
            ocp.cost.Zl = 1 * np.ones((ns,))    # diagonal of Hessian wrt lower slack at intermediate shooting nodes (1 to N-1)
            ocp.cost.zu = 0 * np.ones((ns,))    
            ocp.cost.Zu = 1 * np.ones((ns,))  

        
        # initial state **
        x_0 = np.array([0, 0, np.pi/2])
        ocp.constraints.x0 = x_0

        # solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES' # 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'EXACT' # 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_tol_eq = 1e-4
        ocp.solver_options.nlp_solver_tol_ineq = 1e-4
        ocp.solver_options.nlp_solver_type = 'SQP_RTI' #'SQP'

        # compile acados ocp
        json_file = os.path.join('./'+model.name+'_acados_ocp.json')
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)
        self.integrator = AcadosSimSolver(ocp, json_file=json_file)


    def solve(self, x_real, y_real, theta_real):

        x0 = np.zeros(3)
        x0[0] = x_real
        x0[1] = y_real
        x0[2] = theta_real

        xs = np.zeros(3)
        xs[0] = self.target_x
        xs[1] = self.target_y
        xs[2] = self.target_theta

        simX = np.zeros((self.N+1, self.nx))
        simU = np.zeros((self.N, self.nu))
        x_current = x0
        simX[0, :] = x0.reshape(1, -1)  

        # Start Solving
        self.solver.set(0, 'lbx', x_current)
        self.solver.set(0, 'ubx', x_current)

        # set tau 
        time_interval = np.arange(0, self.N) *self.dt
        t = np.array([0]*self.poly_degree + list(range(self.num_ctrl_points-self.poly_degree+1)) + [self.num_ctrl_points-self.poly_degree]*self.poly_degree,dtype='int')
        
        for k in range(self.N): # loop over control intervals
            index_ = find_correct_index(t, time_interval[k])
            timei = t[index_]
            timei1 = t[index_+1]
            tau_in = np.array([index_, timei, timei1])
            self.solver.set(k, 'p', tau_in)

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
        next_U = simU[1,:]

        return next_x, next_y, next_theta, simX, next_U


    # plot function for case 2 --unchanged
    def plot_results(self, start_x, start_y, theta_log, U_log, x_log, y_log, x_real_log, y_real_log, U_real_log, theta_real_log):
        
        plt.figure()
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

        ## Plot for circle obstacles and x-y positions
        
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

    # ---- post-processing        ------

    def main(self, x_init, y_init, theta_init):
        ### One time testing
        start_x, start_y = x_init, y_init                   # ENV2 start point
        # start_x, start_y = -3.0, 1.0                # ENV1 start point
        x_0, y_0, theta = start_x, start_y, theta_init
        x_real, y_real, theta_real = start_x, start_y, theta
        U_real = np.array([0.0, 0.0])
        theta_0 = theta_init            # Save the initial theta


        # x_0, y_0, theta = -7, 1, np.pi*-0.3
        # x_real, y_real, theta_real = -7, 1, np.pi*-0.3

        x_log, y_log = [x_0], [y_0]
        theta_log = [theta]
        U_log = []

        x_real_log, y_real_log = [x_real], [y_real]
        theta_real_log = [theta_real]

        curve_degree = 3
        control_pt_num = 4
        time_knots_num = control_pt_num + curve_degree + 1

        U_last = np.array([0, 0])
        
        with tqdm(total=100, desc='cpu%', position=1) as cpubar, tqdm(total=100, desc='ram%', position=0) as rambar:

            for i in tqdm(range(self.Epi)):
                # rambar.n=psutil.virtual_memory().percent
                # cpubar.n=psutil.cpu_percent()
                # rambar.refresh()
                # cpubar.refresh()
                # sleep(0.5)
                try:
                    x_0, y_0, theta, X, U = self.solve(x_real, y_real, theta_real)
                    ctrl_point_1 = [U[0], U[1]]
                    ctrl_point_2 = [U[2], U[3]]
                    ctrl_point_3 = [U[4], U[5]]
                    ctrl_point_4 = [U[6], U[7]]
                    ctrl_points = np.array([ctrl_point_1, ctrl_point_2, ctrl_point_3, ctrl_point_4])
                    
                    t = np.array([0]*self.poly_degree + list(range(self.num_ctrl_points-self.poly_degree+1)) + [self.num_ctrl_points-self.poly_degree]*self.poly_degree,dtype='int')
                    traj_prime = Bspline_basis()
                    bspline_curve_prime = traj_prime.bspline_basis(ctrl_points, t, curve_degree)
                    desire_ctrl = bspline_curve_prime[0:5]
                    
                    

                    x_real, y_real, theta_real, x_real_more, y_real_more, theta_real_more = self.dynamic_model_bspline(x_real, y_real, theta_real, desire_ctrl)
                    # print("desire_ctrl", desire_ctrl)
                    # print("ctrl_points" ,ctrl_points)
                    # print("real_pos", x_real, y_real, theta_real)
                    # print("desire_pos", x_0, y_0, theta)
                    # print("states_more", x_real_more, y_real_more, theta_real_more)
                    # print("desire_states_more", X[0,:], X[1,:], X[2,:])
                    # print("bspline_curve_prime", len(bspline_curve_prime))


                    x_real_log.append(x_real)
                    y_real_log.append(y_real)
                    U_log.append(desire_ctrl)

                    x_log.append(x_0)
                    y_log.append(y_0)
                    theta_log.append(theta)
                    if self.step_plotting == True:
                        plt.plot(X[0,:], X[1,:], 'r-')
                        plt.plot(x_0, y_0, 'bo')
                        plt.plot(X[0,0], X[1,0], 'go')
                        x = np.arange(-7,4,0.01)
                        y = np.sin(0.5 * pi * x) + self.initial_pos_sin_obs
                        plt.plot(x, y, 'g-', label='upper limit')
                        plt.plot(x, y-self.gap, 'b-', label='lower limit')
                        plt.show()

                        ctrl_point_1 = [U[0], U[4]]
                        ctrl_point_2 = [U[1], U[5]]
                        ctrl_point_3 = [U[2], U[6]]
                        ctrl_point_4 = [U[3], U[7]]
                        ctrl_points = np.array([ctrl_point_1, ctrl_point_2, ctrl_point_3, ctrl_point_4])
                        print("ctrl_points" ,ctrl_points)
                        # t1 = np.array([0]*curve_degree + list(range(len(ctrl_points)-curve_degree+1)) + [len(ctrl_points)-curve_degree]*curve_degree,dtype='int')
                        # t1 = t1 * dt *N
                        # print(t1)

                        ### Plot for B-spline basis
                        # t2 = np.array(list(range(len(ctrl_points)+curve_degree+1)))*dt/N
                        t = np.array([0]*self.poly_degree + list(range(self.num_ctrl_points-self.poly_degree+1)) + [self.num_ctrl_points-self.poly_degree]*self.poly_degree,dtype='int')
                        t2 = np.array(list(range(len(ctrl_points)+curve_degree+1)))*self.dt*self.N/(len(ctrl_points)+curve_degree)
                        print(t2)
                        plt.plot(ctrl_points[:,0],ctrl_points[:,1], 'o-', label='Control Points')
                        traj_prime = Bspline_basis()
                        bspline_curve_prime = traj_prime.bspline_basis(ctrl_points, t, curve_degree)
                        plt.plot(bspline_curve_prime[:,0], bspline_curve_prime[:,1], label='B-spline Curve')
                        plt.gca().set_aspect('equal', adjustable='box')
                        len_bspline_curve_prime = len(bspline_curve_prime)
                        half_len = int(len_bspline_curve_prime/2)
                        plt.arrow(bspline_curve_prime[half_len,0], bspline_curve_prime[half_len,1], bspline_curve_prime[half_len+1,0]-bspline_curve_prime[half_len,0], bspline_curve_prime[half_len+1,1]-bspline_curve_prime[half_len,1], head_width=0.1, head_length=0.3, fc='k', ec='k')
                        plt.legend(loc='upper right')
                        plt.show()
                    if (x_0 - self.target_x) ** 2 + (y_0 - self.target_y) ** 2 < 0.01:
                        print("reach the target", theta_0)
                        if self.plot_figures == True:
                            self.plot_results(x_log, y_log, x_real_log, y_real_log, theta_log, theta_real_log, start_x, start_y)
                        return [1, theta_log], x_log, y_log
                except RuntimeError:
                    print("Infesible", theta_0)
                    if self.plot_figures == True:
                        self.plot_results(x_log, y_log, x_real_log, y_real_log, theta_log, theta_real_log, start_x, start_y)
                    return [0, theta_log], x_log, y_log
            print("not reach the target", theta_0)
            if self.plot_figures == True:
                self.plot_results(x_log, y_log, x_real_log, y_real_log, theta_log, theta_real_log, start_x, start_y)
            return [0, theta_log], x_log, y_log
        
    def plot_results(self, x_log, y_log, x_real_log, y_real_log, theta_log, theta_real_log, start_x, start_y):
        tt = np.arange(0, (len(x_log))*self.dt, self.dt)
        t = np.arange(0, len(x_log), 1)
        plt.plot(t, theta_log, 'r-')
        plt.show()
        # print(self.optimizer_time)

        # print("x_log", x_log)
        # print("y_log", y_log)

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
            plt.plot(x, y-self.gap, 'b-', label='lower limit')
            plt.show()

        ## Plot for circle obstacles and x-y positions env2
        if self.env_numb == 2:
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
            plt.axis([-5.0, 1.5, -2.4, 2.4])
            # plt.axis('equal')
            x = np.arange(start_x-1,4,0.01)
            plt.plot(x, len(x)*[self.upper_limit], 'g-', label='upper limit')
            plt.plot(x, len(x)*[self.lower_limit], 'b-', label='lower limit')
            plt.legend()
            plt.show()

        if self.env_numb == 0:
            plt.plot(x_log, y_log, 'r-', label='desired path')
            plt.plot(x_real_log, y_real_log, 'b-', label='real path', linestyle='--')
            plt.plot(self.target_x,self.target_y,'bo')
            plt.plot(start_x, start_y, 'go')
            plt.xlabel('pos_x')
            plt.ylabel('pos_y')
            plt.show()

    def mutli_init_theta(self):
        self.plot_figures = False
        THETA = np.arange(-np.pi, np.pi, 0.1)
        LOG_theta = []
        LOG_traj = []
        ii = 0
        start_x, start_y = -4, 0
        for theta in THETA:
            print("epsidoe", ii)
            Data_vel, Data_tarj_x, Data_tarj_y = self.main(start_x, start_y, theta)
            LOG_theta.append(Data_vel)
            LOG_traj.append([Data_tarj_x, Data_tarj_y])
            ii += 1
        
        with open('LOG_initial_theta_env25.pkl', 'wb') as f:
            pickle.dump(LOG_theta, f)

        with open('LOG_traj_env_25.pkl', 'wb') as f:
            pickle.dump(LOG_traj, f)
    
if __name__ == '__main__':

    obstacles = np.array([
    [0.0, 20, 1],        #x, y, r 20 25 30
    [-1.0, 25, 0.1],
    [1.0, 30, 0.1]
    ])

    start_x, start_y, theta = -0.0, 0.0, np.pi/2

    car_model = GemCarBsplModel()
    opt = GemCarOptimizer(m_model=car_model.model, 
                              t_horizon=1, dt=0.05, obstacles = obstacles)
    opt.main(start_x, start_y, theta)
    opt.mutli_init_theta