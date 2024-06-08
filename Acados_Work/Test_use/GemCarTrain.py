#!/usr/bin/env python

import os
import sys
import shutil
import errno
import timeit

from GemCar import GemCarModel
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

from draw import Draw_MPC_point_stabilization_v1

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


class GemCarOptimizer(object):

    def __init__(self, m_model, m_constraint, t_horizon, dt, obstacles):

        model = m_model

        self.T = t_horizon
        self.dt = dt
        self.N = int(t_horizon/dt)

        # Car Info
        self.car_width = 1.5 # 55.5in = 1.4107m
        self.car_length = 2.7 # 103in = 2.6162m
        self.Epi = 3000

        self.target_x = 0.0
        self.target_y = 50.0
        self.target_theta = np.pi/2

        self.plot_figures = True

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
        n_params = len(model.p)

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

        # cost type
        Q = np.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, .1]])
        R = np.array([[0.5, 0.0], [0.0, 0.05]])
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Q
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)

        # set constraints
        ocp.constraints.lbu = np.array([m_constraint.v_min, m_constraint.omega_min])
        ocp.constraints.ubu = np.array([m_constraint.v_max, m_constraint.omega_max])
        ocp.constraints.idxbu = np.array([0, 1])

        x_ref = np.zeros(nx)
        u_ref = np.zeros(nu)

        # obstacles

        x = ocp.model.x[0]  # x position
        y = ocp.model.x[1]  # y position

        obs_num, obs_dim = obstacles.shape
        obs = obstacles

        con_h_expr = []  # list to collect constraints
        
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
            ocp.constraints.uh = 1000 * np.ones((len(con_h_expr),))


            #slack variable configuration:

            nsh = len(con_h_expr)
            ocp.constraints.lsh = np.zeros(nsh)             # Lower bounds on slacks corresponding to soft lower bounds for nonlinear constraints
            ocp.constraints.ush = np.zeros(nsh)             # Lower bounds on slacks corresponding to soft upper bounds for nonlinear constraints
            ocp.constraints.idxsh = np.array(range(nsh))    # Jsh


            ns = len(con_h_expr)
            ocp.cost.zl = 1000 * np.ones((ns,)) # gradient wrt lower slack at intermediate shooting nodes (1 to N-1)
            ocp.cost.Zl = 1 * np.ones((ns,))    # diagonal of Hessian wrt lower slack at intermediate shooting nodes (1 to N-1)
            ocp.cost.zu = 0 * np.ones((ns,))    
            ocp.cost.Zu = 1 * np.ones((ns,))  
                

        # initial state
        ocp.constraints.x0 = x_ref
        ocp.cost.yref = np.concatenate((x_ref, u_ref))
        ocp.cost.yref_e = x_ref

        # solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        # explicit Runge-Kutta integrator
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

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
        xs_between = np.concatenate((xs, np.zeros(2)))
        time_record = np.zeros(self.N)

        # closed loop
        self.solver.set(self.N, 'yref', xs)
        for i in range(self.N):
            self.solver.set(i, 'yref', xs_between)

        for i in range(self.N):
            # solve ocp
            start = timeit.default_timer()
            ##  set inertial (stage 0)
            self.solver.set(0, 'lbx', x_current)
            self.solver.set(0, 'ubx', x_current)
            status = self.solver.solve()

            if status != 0 :
                raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))

            simU[i, :] = self.solver.get(0, 'u')
            time_record[i] =  timeit.default_timer() - start
            # simulate system
            self.integrator.set('x', x_current)
            self.integrator.set('u', simU[i, :])

            status_s = self.integrator.solve()
            if status_s != 0:
                raise Exception('acados integrator returned status {}. Exiting.'.format(status))

            # update
            x_current = self.integrator.get('x')
            simX[i+1, :] = x_current

        #Draw_MPC_point_stabilization_v1(rob_diam=0.3, init_state=x0, target_state=xs, robot_states=simX, )

        # next state
        next_x = simX[1, 0]
        next_y = simX[1, 1]
        next_theta = simX[1, 2]

        return next_x, next_y, next_theta, simX, simU


    # plot function for case 2 --unchanged
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


    def main(self, x_init, y_init, theta_init):
        
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

                    x_0, y_0, theta, X, U = self.solve(x_real, y_real, theta_real)

                    print(x_0, y_0, theta)
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

    
if __name__ == '__main__':

    obstacles = np.array([
    [0.0, 20, 1],        #x, y, r 20 25 30
    [-1.0, 25, 1],
    [1.0, 30, 1]
    ])

    start_x, start_y, theta = 0.0, 0.0, np.pi/2.8

    car_model = GemCarModel()
    opt = GemCarOptimizer(m_model=car_model.model, 
                               m_constraint=car_model.constraint, t_horizon=2, dt=0.02, obstacles = obstacles)
    opt.main(start_x, start_y, theta)