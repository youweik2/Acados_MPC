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

class mpc_ctrl:
    def __init__(self, target_x, target_y):
        self.dt = 0.05 # time frequency 20Hz
        self.N = 20 # number of control intervals
        # self.dt = 0.1 # time frequency 10Hz
        # self.N = 10 # number of control intervals
        # self.dt = 0.02 # time frequency 20Hz
        # self.N = 50 # number of control intervals
        self.Epi = 3000 # number of episodes

        self.target_x = target_x
        self.target_y = target_y
        

        # self.gap = 1.   # gap between upper and lower limit
        self.initial_pos_sin_obs = 1  # initial position of sin obstacles
        self.upper_limit = 1.5
        self.lower_limit = -2.0

        
        self.u = SX.sym("u", 2)    # control
        self.x = SX.sym("x", 3)  # state

        self.x_next = SX.sym("x_next", 3)  # state

        self.low_level_ = False

        xdot = np.cos(self.x[2])*self.u[0]
        ydot = np.sin(self.x[2])*self.u[0]
        thetadot = self.u[1]

        self.x_dot =  vertcat(xdot, ydot, thetadot)

        self.Kp = 0.5
        self.Kd = 0.1
        self.dt1 = 0.05
        self.dt2 = 0.0025

        self.step_plotting = False

        # self.f = Function('f', [self.x, self.u],[xdot, ydot, thetadot])
        self.f = Function('f', [self.x, self.u],[self.x_dot])
        
        self.v_limit = 5.0
        self.omega_limit = 3.0
        self.constraint_k = self.omega_limit/self.v_limit

        self.circle_obstacles_1 = {'x': 0, 'y': 20, 'r': 1.0}
        self.circle_obstacles_2 = {'x': 1, 'y': 25, 'r': 1.0}
        self.circle_obstacles_3 = {'x': -1, 'y': 30, 'r': 1.0}

        self.env_numb = 2          # 1: sin wave obstacles, 2: circle obstacles
        self.plot_figures = True

        self.casadi_time = []

    def distance_circle_obs(self, x, y, circle_obstacles):
        return (x - circle_obstacles['x']) ** 2 + (y - circle_obstacles['y']) ** 2 - circle_obstacles['r'] ** 2

    def solver_mpc(self, x_init, y_init, theta_init):
        # ---- decision variables ---------
        opti = Opti() # Optimization problem
        X = opti.variable(3, self.N+1) # state trajectory
        pos_x = X[0,:]
        pos_y = X[1,:]
        theta = X[2,:]

        U = opti.variable(2, self.N+1)   # control points (2*1)

        State_xy = X[0:2, :] - [self.target_x, self.target_y]
        V = U[0, :]
        
        Last_term = X[:,-1]
        LL = sumsqr(Last_term[:2] - [self.target_x, self.target_y]) # + sumsqr(Last_term[2])

        L = 10*sumsqr(State_xy) + sumsqr(U) + 10*LL # sum of QP terms


        # ---- objective          ---------
        opti.minimize(L) # race in minimal time 

        for k in range(self.N): # loop over control intervals
            # Runge-Kutta 4 integration
            # k11, k12, k13 = self.f(X[:,k],         U[:,k])
            # k21, k22, k23 = self.f(X[:,k]+self.dt/2*k11, U[:,k])
            # k31, k32, k33 = self.f(X[:,k]+self.dt/2*k21, U[:,k])
            # k41, k42, k43 = self.f(X[:,k]+self.dt*k31,   U[:,k])
            # x_next = X[0,k] + self.dt/6*(k11+2*k21+2*k31+k41)
            # y_next = X[1,k] + self.dt/6*(k12+2*k22+2*k32+k42)
            # theta_next = X[2,k] + self.dt/6*(k13+2*k23+2*k33+k43)
            # opti.subject_to(X[0,k+1]==x_next)
            # opti.subject_to(X[1,k+1]==y_next)
            # opti.subject_to(X[2,k+1]==theta_next)   # close the gaps
            k1 = self.f(X[:,k],         U[:,k])
            k2 = self.f(X[:,k]+self.dt/2*k1, U[:,k])
            k3 = self.f(X[:,k]+self.dt/2*k2, U[:,k])
            k4 = self.f(X[:,k]+self.dt*k3,   U[:,k])
            x_next = X[0,k] + self.dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
            y_next = X[1,k] + self.dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
            theta_next = X[2,k] + self.dt/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
            opti.subject_to(X[0,k+1]==x_next)
            opti.subject_to(X[1,k+1]==y_next)
            opti.subject_to(X[2,k+1]==theta_next)   # close the gaps


        # for k in range(self.N): # loop over control intervals
        #     # Runge-Kutta 4 integration
        #     k11, k12, k13 = self.f(X[:,k], U[:,k])
        #     x_next = X[0,k] + self.dt*k11
        #     y_next = X[1,k] + self.dt*k12
        #     theta_next = X[2,k] + self.dt*k13
        #     opti.subject_to(X[0,k+1]==x_next)
        #     opti.subject_to(X[1,k+1]==y_next)
        #     opti.subject_to(X[2,k+1]==theta_next)   # close the gaps
    
        # ---- path constraints 1 -----------
        if self.env_numb == 1:
            limit_upper = lambda pos_x: sin(0.5*pi*pos_x) + self.initial_pos_sin_obs
            limit_lower = lambda pos_x: sin(0.5*pi*pos_x) - self.initial_pos_sin_obs
            # opti.subject_to(limit_lower(pos_x)<pos_y)
            # opti.subject_to(limit_upper(pos_x)>pos_y)   # state constraints 
            
        # ---- path constraints 2 -----------
        if self.env_numb == 2:
            # opti.subject_to(pos_y<=self.upper_limit)
            # opti.subject_to(pos_y>=self.lower_limit)
            opti.subject_to(self.distance_circle_obs(pos_x, pos_y, self.circle_obstacles_1) > 1.0)
            # opti.subject_to(self.distance_circle_obs(pos_x, pos_y, self.circle_obstacles_2) > 0.1)
            # opti.subject_to(self.distance_circle_obs(pos_x, pos_y, self.circle_obstacles_3) > 0.1)


        # ---- control constraints ----------
        v_limit = 1.5
        omega_limit = 3.0
        constraint_k = omega_limit/v_limit

        # ctrl_constraint_leftupper = lambda v: constraint_k*v + omega_limit          # omega <= constraint_k*v + omega_limit
        # ctrl_constraint_rightlower = lambda v: constraint_k*v - omega_limit         # omega >= constraint_k*v - omega_limit
        # ctrl_constraint_leftlower = lambda v: -constraint_k*v - omega_limit         # omega >= -constraint_k*v - omega_limit
        # ctrl_constraint_rightupper = lambda v: -constraint_k*v + omega_limit        # omega <= -constraint_k*v + omega_limit
        # opti.subject_to(ctrl_constraint_rightlower(U[0,:])<=U[1,:])
        # opti.subject_to(ctrl_constraint_leftupper(U[0,:])>=U[1,:])
        # opti.subject_to(ctrl_constraint_leftlower(U[0,:])<=U[1,:])
        # opti.subject_to(ctrl_constraint_rightupper(U[0,:])>=U[1,:])

        opti.subject_to(opti.bounded(-v_limit, U[0, :], v_limit))
        opti.subject_to(opti.bounded(-omega_limit, U[1, :], omega_limit))


        
        # ---- boundary conditions --------
        opti.subject_to(pos_x[0]==x_init)
        opti.subject_to(pos_y[0]==y_init)   
        opti.subject_to(theta[0]==theta_init)


        # ---- solve NLP              ------
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        
        opti.solver("ipopt", opts) # set numerical backend
        # opti.solver("ipopt") # set numerical backend
        

        sol = opti.solve()   # actual solve
        # self.casadi_time.append(sol.stats()['t_wall_total'])


        return sol.value(pos_x[1]), sol.value(pos_y[1]), sol.value(theta[1]), sol.value(U), sol.value(X)
    
    def low_level_ctrl(self, desire_ctrl, theta, x, y, ctrls):
        pd_controller = UnicyclePDController(self.Kp, self.Kd, self.dt1)

        initial_x = x
        initial_y = y
        initial_theta = theta
        initial_ctrls = ctrls

        Log_x, Log_y = [initial_x], [initial_y]
        Log_ctrls_v, Log_ctrls_w = [], []
        Log_desire_ctrls_v, Log_desire_ctrls_w = [], []



        time_steps, positions, ctrls, desire_ctrl = pd_controller.simulate_unicycle(desire_ctrl, initial_ctrls, theta, x, y)
        initial_x = positions[-1][0]
        initial_y = positions[-1][1]
        initial_theta = math.atan2(positions[-1][1], positions[-1][0])
        initial_ctrls = ctrls[-1]
        Log_x.extend(np.array(positions).T[0])
        Log_y.extend(np.array(positions).T[1])
        Log_ctrls_v.extend(np.array(ctrls)[0])
        Log_ctrls_w.extend(np.array(ctrls)[1])
        Log_desire_ctrls_v.extend(np.array(desire_ctrl)[:,0])
        Log_desire_ctrls_w.extend(np.array(desire_ctrl)[:,1])

        if self.step_plotting == True:
            # Plotting the results
                # Plotting the results
            plt.figure(figsize=(8, 6))
            print(len(Log_x), len(Log_y))   
            plt.plot(Log_x, Log_y, label='Unicycle Path')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Unicycle Path Controlled by Time-Changing Velocities with PD Controller')
            plt.legend()
            plt.grid(True)
            plt.show()

            time_plotting = np.arange(0, len(desire_ctrl)*self.dt1, self.dt2)
            plt.figure(figsize=(8, 6))
            plt.plot(time_plotting, Log_ctrls_v, label='Control Signals_v')
            plt.plot(time_plotting, Log_ctrls_w, label='Control Signals_w')
            plt.plot(time_plotting, Log_desire_ctrls_v, label='Desired Control Signals_v', linestyle='--')
            plt.plot(time_plotting, Log_desire_ctrls_w, label='Desired Control Signals_w', linestyle='--')
            print(ctrls)
            plt.xlabel('Time Steps')
            plt.ylabel('Values')
            plt.legend()
            plt.title('Control Signals')
            plt.grid(True)
            plt.show()
        
        return initial_x, initial_y, initial_theta, ctrls[-1]
    
    def dynamic_model(self, x, y, theta, v, w):
        x_next = x + self.dt * v * np.cos(theta)
        y_next = y + self.dt * v * np.sin(theta)
        theta_next = theta + self.dt * w
        return x_next, y_next, theta_next

    def main(self, x_init, y_init, theta_init):
        
        start_x, start_y = x_init, y_init                   # ENV2 start point
        # start_x, start_y = -3.0, 1.0                # ENV1 start point
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
                # rambar.n=psutil.virtual_memory().percent
                # cpubar.n=psutil.cpu_percent()
                # rambar.refresh()
                # cpubar.refresh()
                # sleep(0.5)
                try:
                    x_0, y_0, theta, U, X = self.solver_mpc(x_real, y_real, theta_real)
                    desire_ctrl = U.T[0]
                    # print(U, desire_ctrl)
                    # print("desire_ctrl", desire_ctrl)
                    # print("desire_state", x_0, y_0, theta)
                    # print("real_state", x_real, y_real, theta_real)
                    # x_real, y_real, theta_real = self.dynamic_model(x_real, y_real, theta_real, desire_ctrl[0], desire_ctrl[1])
                    U_real = desire_ctrl
                    x_real, y_real, theta_real = x_0, y_0, theta

                    # print("desire_ctrl", desire_ctrl)
                    # print("desire_state", x_0, y_0, theta)
                    # print("real_state", x_real, y_real, theta_real)
                    
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
        # print(len(U_log), U_log)
        # print(len(theta_log))
        # print(len(tt))
        # print(len(t))
        # print(x_log)
        # print(y_log)
        # print(self.casadi_time)
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
            # plt.axis([-5.0, 1.5, -2.4, 2.4])
            plt.axis('equal')
            # x = np.arange(start_x-1,4,0.01)
            # plt.plot(x, len(x)*[self.upper_limit], 'g-', label='upper limit')
            # plt.plot(x, len(x)*[self.lower_limit], 'b-', label='lower limit')
            plt.legend()
            plt.show()

        with open('single_traj_mpc_50hz.pkl', 'wb') as f:
            pickle.dump([x_log, y_log], f)

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
        
        # with open('LOG_initial_theta_env9.pkl', 'wb') as f:         # ENV 2 with square control constraints
        #     pickle.dump(LOG_theta, f)

        # with open('LOG_traj_env_9.pkl', 'wb') as f:
        #     pickle.dump(LOG_traj, f)

        with open('LOG_initial_theta_env26_mpc_sq.pkl', 'wb') as f:         # ENV 2 with longze control constraints
            pickle.dump(LOG_theta, f)

        with open('LOG_traj_env_26_mpc_sq.pkl', 'wb') as f:
            pickle.dump(LOG_traj, f)



if __name__ == "__main__":
    # target_x, target_y = 0.5, -0.5                # ENV 2 target point
    # start_x, start_y = -4.0, 0.0                # ENV 2 start point
    # start_x, start_y = 1, -0.8
    target_x, target_y = 0.0, 40.0              # ENV 1 target point

    start_x, start_y = -0.0, 10.0   

    # theta = 1.4
    mpc = mpc_ctrl(target_x=target_x, target_y=target_y)
    
    theta = 0.5 * np.pi
    mpc.main(start_x, start_y, theta)

    # mpc.mutli_init_theta()
