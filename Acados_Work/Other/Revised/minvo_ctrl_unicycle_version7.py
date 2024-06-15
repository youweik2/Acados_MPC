import numpy as np
from casadi import *
import math
from B_spline import Bspline, Bspline_basis
import matplotlib.pyplot as plt
from unicycle_pd import UnicyclePDController
import pickle

from time import sleep
import psutil
from tqdm import tqdm

class mpc_bspline_ctrl:
    def __init__(self, target_x, target_y):
        self.N = 20 # number of control intervals
        self.Epi = 500 # number of episodes
        
        self.target_x = target_x
        self.target_y = target_y

        self.gap = 2.5   # gap between upper and lower limit
        self.initial_pos_sin_obs = self.gap/2   # initial position of sin obstacles

        self.upper_limit = 1.5 
        self.lower_limit = -2.0 

        self.tau = SX.sym("tau")    # time
        self.u = SX.sym("u", 8)    # control
        self.x = SX.sym("x", 3)  # state
        self.tau_i = SX.sym("tau_i")   # time interval i
        self.tau_i1 = SX.sym("tau_i1")   # time interval i+1

        # self.k32 = (3*(self.tau - self.tau_i))/(self.tau_i - self.tau_i1) - (3*(self.tau - self.tau_i)**2)/(self.tau_i - self.tau_i1)**2 - (self.tau - self.tau_i)**3/(self.tau_i - self.tau_i1)**3 - 1
        # self.k11 = np.cos(self.x[2])*self.k32
        # self.k21 = np.sin(self.x[2])*self.k32
        # self.k34 = (6*(self.tau - self.tau_i))/(self.tau_i - self.tau_i1) + (3*(self.tau - self.tau_i)**2)/(self.tau_i - self.tau_i1)**2 + 3
        # self.k13 = np.cos(self.x[2])*self.k34
        # self.k23 = np.sin(self.x[2])*self.k34
        # self.k36 = - (3*(self.tau - self.tau_i))/(self.tau_i - self.tau_i1) - 3 
        # self.k15 = np.cos(self.x[2])*self.k36
        # self.k25 = np.sin(self.x[2])*self.k36
        # self.k38 = 1
        # self.k17 = np.cos(self.x[2])*self.k38
        # self.k27 = np.sin(self.x[2])*self.k38

        self.k32 = (3*(self.tau - self.tau_i))/(self.tau_i - self.tau_i1) + (3*(self.tau - self.tau_i)**2)/(self.tau_i - self.tau_i1)**2 + (self.tau - self.tau_i)**3/(self.tau_i - self.tau_i1)**3 + 1
        self.k11 = np.cos(self.x[2])*self.k32
        self.k21 = np.sin(self.x[2])*self.k32
        self.k34 = - (3*(self.tau - self.tau_i))/(self.tau_i - self.tau_i1) - (6*(self.tau - self.tau_i)**2)/(self.tau_i - self.tau_i1)**2 - (3*(self.tau - self.tau_i)**3)/(self.tau_i - self.tau_i1)**3
        self.k13 = np.cos(self.x[2])*self.k34
        self.k23 = np.sin(self.x[2])*self.k34
        self.k36 = (3*(self.tau - self.tau_i)**2)/(self.tau_i - self.tau_i1)**2 + (3*(self.tau - self.tau_i)**3)/(self.tau_i - self.tau_i1)**3
        self.k15 = np.cos(self.x[2])*self.k36
        self.k25 = np.sin(self.x[2])*self.k36
        self.k38 = -(self.tau - self.tau_i)**3/(self.tau_i - self.tau_i1)**3
        self.k17 = np.cos(self.x[2])*self.k38
        self.k27 = np.sin(self.x[2])*self.k38

        self.Kp = 0.5
        self.Kd = 0.1
        self.dt1 = 0.05
        self.dt2 = 0.0025
        
        # ---- dynamic constraints --------
        # xdot = self.k11*self.u[0] + self.k13*self.u[1] + self.k15*self.u[2] + self.k17*self.u[3]
        # ydot = self.k21*self.u[0] + self.k23*self.u[1] + self.k25*self.u[2] + self.k27*self.u[3]
        # thetadot = self.k32*self.u[4] + self.k34*self.u[5] + self.k36*self.u[6] + self.k38*self.u[7]

        xdot = self.k11*self.u[0] + self.k13*self.u[2] + self.k15*self.u[4] + self.k17*self.u[6]
        ydot = self.k21*self.u[0] + self.k23*self.u[2] + self.k25*self.u[4] + self.k27*self.u[6]
        thetadot = self.k32*self.u[1] + self.k34*self.u[3] + self.k36*self.u[5] + self.k38*self.u[7]

        self.x_dot = vertcat(xdot, ydot, thetadot)

        self.f = Function('f', [self.x, self.u, self.tau, self.tau_i, self.tau_i1],[self.x_dot])
        self.dt = 0.05 # length of a control interval
        self.poly_degree = 3
        self.num_ctrl_points = 4

        self.step_plotting = False
        self.use_low_level_ctrl = False

        self.circle_obstacles_1 = {'x': 0.5, 'y': 0.5, 'r': 0.5}
        self.circle_obstacles_2 = {'x': -0.5, 'y': -0.5, 'r': 0.6}
        self.circle_obstacles_3 = {'x': -1.0, 'y': 0.8, 'r': 0.5}

        self.env_numb = 2           # 1: sin wave obstacles, 2: circle obstacles
        self.plot_figures = True

        self.optimizer_time = []
        # def distance_circle_obs(self, x, y, circle_obstacles):
        #     return (x - circle_obstacles['x']) ** 2 + (y - circle_obstacles['y']) ** 2 - circle_obstacles['r'] ** 2
    def distance_circle_obs(self, x, y, circle_obstacles):
        return (x - circle_obstacles['x']) ** 2 + (y - circle_obstacles['y']) ** 2 - circle_obstacles['r'] ** 2


    def find_floor(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        if array[idx] > value:
            idx = idx - 1
        return idx

    def find_ceil(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        if array[idx] < value:
            idx = idx + 1
        return idx

    def find_correct_index(self, array, value):
        indicator = 0
        index = 0
        while indicator == 0:
            indicator = 1 if array[index] <= value < array[index+1] else 0
            index += 1
        return index - 1
    
    def coefficients(self, tau_i, tau_i1):
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
        gm = self.coefficients(tau_i, tau_i1)
        cost = 0
        for i in range(4):
            for j in range(4):
                cost +=  gm[i*4+j] * cp[j] @ cp[i].T 
        return cost



    def solver_mpc(self, x_init, y_init, theta_init):

        opti = Opti() # Optimization problem
        time_interval = np.arange(0, self.N) *self.dt #+ current_time # time interval
        # ---- decision variables ---------
        X = opti.variable(3, self.N+1) # state trajectory
        pos_x = X[0,:]
        pos_y = X[1,:]
        theta = X[2,:]

        U = opti.variable(8, 1)   # control points (8*1)
        ctrl_point_1 = np.array([U[0], U[1]])
        ctrl_point_2 = np.array([U[2], U[3]])
        ctrl_point_3 = np.array([U[4], U[5]])
        ctrl_point_4 = np.array([U[6], U[7]])
        cp = [ctrl_point_1, ctrl_point_2, ctrl_point_3, ctrl_point_4]

        # Clamped uniform time knots
        # time_knots = np.array([0]*poly_degree + list(range(num_ctrl_points-poly_degree+1)) + [num_ctrl_points-poly_degree]*poly_degree,dtype='int')

        # Uniform B spline time knots
        t = np.array([0]*self.poly_degree + list(range(self.num_ctrl_points-self.poly_degree+1)) + [self.num_ctrl_points-self.poly_degree]*self.poly_degree,dtype='int')
        # Objective term
        State_xy = X[0:2, :] - [self.target_x, self.target_y]
        Last_term = X[:,-1]
        LL = sumsqr(Last_term[:2] - [self.target_x, self.target_y]) + sumsqr(Last_term[2])
        L = 10*sumsqr(State_xy) + 10 * LL # sum of QP terms
        L += 1 * self.cost_function_ctrlpoints(cp, 0, 1)

        # ---- objective          ---------
        opti.minimize(L) # race in minimal time 

        for k in range(self.N): # loop over control intervals
            # Runge-Kutta 4 integration
            index_ = self.find_correct_index(t, time_interval[k])
            timei = t[index_]
            timei1 = t[index_+1]

            # k11, k12, k13 = self.f(X[:,k],         U[:], time_interval[k], timei, timei1)
            # k21, k22, k23 = self.f(X[:,k]+self.dt/2*k11, U[:], time_interval[k], timei, timei1)
            # k31, k32, k33 = self.f(X[:,k]+self.dt/2*k21, U[:], time_interval[k], timei, timei1)
            # k41, k42, k43 = self.f(X[:,k]+self.dt*k31,   U[:], time_interval[k], timei, timei1)
            # x_next = X[0,k] + self.dt/6*(k11+2*k21+2*k31+k41)
            # y_next = X[1,k] + self.dt/6*(k12+2*k22+2*k32+k42)
            # theta_next = X[2,k] + self.dt/6*(k13+2*k23+2*k33+k43)
            # opti.subject_to(X[0,k+1]==x_next)
            # opti.subject_to(X[1,k+1]==y_next)
            # opti.subject_to(X[2,k+1]==theta_next)   # close the gaps

            k1 = self.f(X[:,k],         U[:], time_interval[k], timei, timei1)
            k2 = self.f(X[:,k]+self.dt/2*k1, U[:], time_interval[k], timei, timei1)
            k3 = self.f(X[:,k]+self.dt/2*k2, U[:], time_interval[k], timei, timei1)
            k4 = self.f(X[:,k]+self.dt*k3,   U[:], time_interval[k], timei, timei1)
            x_next = X[0,k] + self.dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
            y_next = X[1,k] + self.dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
            theta_next = X[2,k] + self.dt/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
            opti.subject_to(X[0,k+1]==x_next) # close the gaps
            opti.subject_to(X[1,k+1]==y_next) # close the gaps
            opti.subject_to(X[2,k+1]==theta_next) # close the gaps

            # L += self.cost_function_ctrlpoints(time_interval[k], cp, timei, timei1)


       # ---- path constraints 1 -----------
        if self.env_numb == 1:
            limit_upper = lambda pos_x: sin(0.5*pi*pos_x) + self.initial_pos_sin_obs
            limit_lower = lambda pos_x: sin(0.5*pi*pos_x) + self.initial_pos_sin_obs - self.gap
            opti.subject_to(limit_lower(pos_x)<pos_y)
            opti.subject_to(limit_upper(pos_x)>pos_y)   # state constraints 
            
        # ---- path constraints 2 -----------
        if self.env_numb == 2:
            opti.subject_to(pos_y<=self.upper_limit)
            opti.subject_to(pos_y>=self.lower_limit)
            opti.subject_to(self.distance_circle_obs(pos_x, pos_y, self.circle_obstacles_1) >= 0.01)
            opti.subject_to(self.distance_circle_obs(pos_x, pos_y, self.circle_obstacles_2) >= 0.01)
            opti.subject_to(self.distance_circle_obs(pos_x, pos_y, self.circle_obstacles_3) >= 0.01)

        # ---- input constraints --------
        v_limit = 1.0
        omega_limit = 3.0
        constraint_k = omega_limit/v_limit

        ctrl_constraint_leftupper = lambda ctrl_point: constraint_k*ctrl_point + omega_limit
        ctrl_constraint_rightlower = lambda ctrl_point: constraint_k*ctrl_point - omega_limit
        ctrl_constraint_leftlower = lambda ctrl_point: -constraint_k*ctrl_point - omega_limit
        ctrl_constraint_rightupper = lambda ctrl_point: -constraint_k*ctrl_point + omega_limit
        opti.subject_to(ctrl_constraint_rightlower(ctrl_point_1[0])<=ctrl_point_1[1])
        opti.subject_to(ctrl_constraint_leftupper(ctrl_point_1[0])>=ctrl_point_1[1])
        opti.subject_to(ctrl_constraint_leftlower(ctrl_point_1[0])<=ctrl_point_1[1])
        opti.subject_to(ctrl_constraint_rightupper(ctrl_point_1[0])>=ctrl_point_1[1])

        opti.subject_to(ctrl_constraint_rightlower(ctrl_point_2[0])<=ctrl_point_2[1])
        opti.subject_to(ctrl_constraint_leftupper(ctrl_point_2[0])>=ctrl_point_2[1])
        opti.subject_to(ctrl_constraint_leftlower(ctrl_point_2[0])<=ctrl_point_2[1])
        opti.subject_to(ctrl_constraint_rightupper(ctrl_point_2[0])>=ctrl_point_2[1])

        opti.subject_to(ctrl_constraint_rightlower(ctrl_point_3[0])<=ctrl_point_3[1])
        opti.subject_to(ctrl_constraint_leftupper(ctrl_point_3[0])>=ctrl_point_3[1])
        opti.subject_to(ctrl_constraint_leftlower(ctrl_point_3[0])<=ctrl_point_3[1])
        opti.subject_to(ctrl_constraint_rightupper(ctrl_point_3[0])>=ctrl_point_3[1])

        opti.subject_to(ctrl_constraint_rightlower(ctrl_point_4[0])<=ctrl_point_4[1])
        opti.subject_to(ctrl_constraint_leftupper(ctrl_point_4[0])>=ctrl_point_4[1])
        opti.subject_to(ctrl_constraint_leftlower(ctrl_point_4[0])<=ctrl_point_4[1])
        opti.subject_to(ctrl_constraint_rightupper(ctrl_point_4[0])>=ctrl_point_4[1])

        # opti.subject_to(opti.bounded(-v_limit, U[0], v_limit))
        # opti.subject_to(opti.bounded(-v_limit, U[2], v_limit))
        # opti.subject_to(opti.bounded(-v_limit, U[4], v_limit))
        # opti.subject_to(opti.bounded(-v_limit, U[6], v_limit))
        # opti.subject_to(opti.bounded(-omega_limit, U[1], omega_limit))
        # opti.subject_to(opti.bounded(-omega_limit, U[3], omega_limit))
        # opti.subject_to(opti.bounded(-omega_limit, U[5], omega_limit))
        # opti.subject_to(opti.bounded(-omega_limit, U[7], omega_limit))

        # ---- boundary conditions --------
        opti.subject_to(pos_x[0]==x_init)
        opti.subject_to(pos_y[0]==y_init)   
        opti.subject_to(theta[0]==theta_init)


        # ---- solve NLP              ------
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        
        opti.solver("ipopt", opts) # set numerical backend
        # opti.solver("ipopt") # set numerical backend
        sol = opti.solve()   # actual solve

        # opti.debug.value(U)
        # casadi_time = sol.stats()['t_wall_total']
        # print("time", casadi_time)
        # self.optimizer_time.append(casadi_time)

        return sol.value(pos_x[1]), sol.value(pos_y[1]), sol.value(theta[1]), sol.value(U), sol.value(X)
    

    def low_level_ctrl(self, ctrl_ref, theta, x, y, ctrls):
        pd_controller = UnicyclePDController(self.Kp, self.Kd, self.dt1)

        initial_x = x
        initial_y = y
        initial_theta = theta
        initial_ctrls = ctrls

        Log_x, Log_y = [initial_x], [initial_y]
        Log_ctrls_v, Log_ctrls_w = [], []
        Log_desire_ctrls_v, Log_desire_ctrls_w = [], []


        for i in range(len(ctrl_ref)):
            time_steps, positions, ctrls, desire_ctrl = pd_controller.simulate_unicycle(ctrl_ref[i], initial_ctrls, theta, x, y)
            initial_x = positions[-1][0]
            initial_y = positions[-1][1]
            initial_theta = positions[-1][2]
            initial_ctrls = ctrls[-1]
            Log_x.extend(np.array(positions).T[0])
            Log_y.extend(np.array(positions).T[1])
            Log_ctrls_v.extend(np.array(ctrls)[:,0])
            Log_ctrls_w.extend(np.array(ctrls)[:,1])
            Log_desire_ctrls_v.extend(np.array(desire_ctrl)[:,0])
            Log_desire_ctrls_w.extend(np.array(desire_ctrl)[:,1])

        if self.step_plotting == True:

            time_plotting = np.arange(0, len(ctrl_ref)*self.dt1, self.dt2)
            plt.figure(figsize=(8, 6))
            plt.plot(time_plotting, Log_ctrls_v, label='Control Signals_v')
            plt.plot(time_plotting, Log_ctrls_w, label='Control Signals_w')
            plt.plot(time_plotting, Log_desire_ctrls_v, label='Desired Control Signals_v', linestyle='--')
            plt.plot(time_plotting, Log_desire_ctrls_w, label='Desired Control Signals_w', linestyle='--')
            plt.xlabel('Time Steps')
            plt.ylabel('Values')
            plt.legend()
            plt.title('Control Signals')
            plt.grid(True)
            plt.show()
        
        return initial_x, initial_y, initial_theta, initial_ctrls
    

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
                    x_0, y_0, theta, U, X = self.solver_mpc(x_real, y_real, theta_real)
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


if __name__ == "__main__":
    # try:
    target_x, target_y = 0.5, -0.5
    mpc_bspline = mpc_bspline_ctrl(target_x=target_x, target_y=target_y)
    # start_x, start_y = -4, 0


    # theta = -0.4
    # mpc_bspline.main(start_x, start_y, theta)

    mpc_bspline.mutli_init_theta()

