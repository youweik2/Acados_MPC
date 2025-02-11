#!/usr/bin/env python

# Ros adds

# Acados adds
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
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
        wpg = 20 # lidar width (per grid)
        hpg = 22 # lidar height 

        # control inputs
        a = ca.SX.sym('accel')
        fai = ca.SX.sym('fai')
        controls = ca.vertcat(a, fai) # fai: front angle
        k = 1

        # model states
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        v = ca.SX.sym('v')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(x, y, theta, v)

        # dynamic
        rhs = [v*ca.cos(theta), v*ca.sin(theta), v*ca.tan(fai)/(length * k), a] # v*ca.tan(fai)/length -> v*fai/length as |fai| < 30 degree

        # function
        f = ca.Function('f', [states, controls], [ca.vcat(rhs)], ['state', 'control_input'], ['rhs'])

        # acados model
        x_dot = ca.SX.sym('x_dot', len(rhs))
        f_impl = x_dot - f(states, controls)

        # grid map setting
        grid = ca.SX.sym('grid', wpg*hpg)
        cx = ca.SX.sym('cx')
        cy = ca.SX.sym('cy')

        # other settings
        model.f_expl_expr = f(states, controls)
        model.f_impl_expr = f_impl
        model.x = states
        model.xdot = x_dot
        model.u = controls
        model.p = ca.vcat([grid, cx, cy])
        model.name = 'GemCarModel'

        # constraints
        constraint.a_max = 1.5
        constraint.a_min = -2.0
        constraint.theta_max = np.pi/5.2
        constraint.theta_min = -np.pi/5.2
        constraint.expr = ca.vcat([a, fai])

        self.model = model
        self.constraint = constraint


class GemCarOptimizer(object):
    def __init__(self, m_model, m_constraint, t_horizon, dt, grid, target):
        model = m_model

        # Grid dimensions
        self.wpg = 20
        self.hpg = 22
        self.rtg = grid  # Real-time grid

        # Basic info
        self.T = t_horizon
        self.dt = dt
        self.N = int(t_horizon / dt)

        # Car info
        self.car_width = 1.5
        self.car_length = 2.565
        self.Epi = 3000

        self.target_x = target[0]
        self.target_y = target[1]
        self.target_theta = target[2]
        self.target_velocity = target[3]

        self.plot_figures = True

        # Ensure working directory is current folder
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, acados_source_path)

        # Acados degrees
        nx = model.x.size()[0]
        self.nx = nx
        nu = model.u.size()[0]
        self.nu = nu
        ny = nx + nu
        n_params = self.wpg * self.hpg + 2  # Total parameters

        # Create OCP
        ocp = AcadosOcp()
        ocp.acados_include_path = acados_source_path + '/include'
        ocp.acados_lib_path = acados_source_path + '/lib'
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.T

        # Initialize parameters
        ocp.dims.np = n_params
        ocp.parameter_values = np.zeros((n_params,))

        # Slice parameters
        parms = ocp.model.p
        assert parms.size1() == n_params, f"Expected parameter size {n_params}, got {parms.size1()}"
        grid_1d = parms[0: self.wpg * self.hpg]
        self.grid = grid_1d.reshape((20, 22)).T
        self.cx = parms[self.wpg * self.hpg]
        self.cy = parms[self.wpg * self.hpg + 1]

        # Cost settings ***
        Q = np.diag([1.0, 1.0, 0.0, 0.0])
        R = np.array([[0.0, 0.0], [0.0, 0.0]])

        # Q = np.diag([5.0, 5.0, 15.0, 3.0])  # States
        # R = np.array([[0.0, 0.0],[0.0, 0.0]])  # Controls


        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'

        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = np.diag([5.0, 5.0, 0.5, 2.0])
        # ocp.cost.W_e = np.diag([5.0, 5.0, 0.5, 2.0])
        ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
        ocp.model.cost_y_expr_e = model.x

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)

        # Constraints
        ocp.constraints.lbu = np.array([m_constraint.a_min, m_constraint.theta_min])
        ocp.constraints.ubu = np.array([m_constraint.a_max, m_constraint.theta_max])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.lbx = np.array([-10, -100, 0, 0])
        ocp.constraints.ubx = np.array([10, 100, np.pi, 6])
        ocp.constraints.idxbx = np.array([0, 1, 2, 3])

        x_ref = np.zeros(nx)
        u_ref = np.zeros(nu)

        # Soft constraints
        x_ = ocp.model.x[0]
        y_ = ocp.model.x[1]
        con_h_expr = []

        # safe_condition = [(0, 0)]

        safe_condition = [(0, 1.5), (1.5, 0), (-1.5, 0), (0, 0),
                          (1 * np.sin(np.pi / 4), 1 * np.cos(np.pi / 4)),
                          (1 * np.sin(np.pi * 3 / 4), 1 * np.cos(np.pi * 3 / 4))]
        for xp, yp in safe_condition:
            grid_value = 40 - self.get_gvalue(x_ + xp, y_ + yp, -self.cx + 5.5, -self.cy - 1.2825, self.grid)
            con_h_expr.append(grid_value)

        if con_h_expr:
            ocp.model.con_h_expr = ca.vertcat(*con_h_expr)
            ocp.constraints.lh = np.zeros((len(con_h_expr),))
            ocp.constraints.uh = 100 * np.ones((len(con_h_expr),))  # small means hard

            nsh = len(con_h_expr)
            ocp.constraints.lsh = np.zeros(nsh)
            ocp.constraints.ush = np.zeros(nsh)
            ocp.constraints.idxsh = np.array(range(nsh))

            ns = len(con_h_expr)
            ocp.cost.zl = 1000 * np.ones((ns,))
            ocp.cost.Zl = 10 * np.ones((ns,))
            ocp.cost.zu = 1000 * np.ones((ns,))
            ocp.cost.Zu = 10 * np.ones((ns,))

        # Initial state
        ocp.constraints.x0 = x_ref
        ocp.cost.yref = np.concatenate((x_ref, u_ref))
        ocp.cost.yref_e = x_ref

        # Solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        # Compile Acados OCP
        json_file = os.path.join('./' + model.name + '_acados_ocp.json')
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)
        self.integrator = AcadosSimSolver(ocp, json_file=json_file)

    def get_gvalue(self, cur_x, cur_y, cx, cy, grid):
        h = 0.5

        # Reshape the flattened grid into a 50x50 symbolic matrix
        gridmap = ca.reshape(grid, self.hpg, self.wpg)

        # Compute symbolic grid indices
        grid_x = ca.fmax(ca.floor((cur_x + cx) / h), 0)
        grid_y = ca.fmax(ca.floor((cur_y + cy) / h), 0)

        # Handle boundary cases symbolically
        grid_x = ca.if_else(grid_x < 0, 0, ca.if_else(grid_x >= self.wpg, self.hpg - 1, grid_x))
        grid_x1 = ca.if_else(grid_x + 1 >= self.wpg, self.wpg - 1, grid_x + 1)
        grid_y = ca.if_else(grid_y < 0, 0, ca.if_else(grid_y >= self.hpg, self.wpg - 1, grid_y))
        grid_y1 = ca.if_else(grid_y + 1 >= self.wpg, self.hpg - 1, grid_y + 1)
        
        def symbolic_lookup(matrix, row_idx, col_idx):
            result = 0
            for i in range(self.hpg):
                for j in range(self.wpg):
                    result += ca.if_else(
                        ca.logic_and(row_idx == i, col_idx == j),
                        ca.if_else(matrix[i, j] == -1, 50, matrix[i, j]),
                        0
                    )
            return result
        
        # Access grid values using the updated symbolic_lookup
        gxy = symbolic_lookup(gridmap, grid_x, grid_y)
        gxpy = symbolic_lookup(gridmap, grid_x1, grid_y)
        gxyp = symbolic_lookup(gridmap, grid_x, grid_y1)
        gxpyp = symbolic_lookup(gridmap, grid_x1, grid_y1)

        # Compute weights
        I_x = ca.floor((cur_x + cx) / h)
        I_y = ca.floor((cur_y + cy) / h)
        R_x = (cur_x + cx) / h - I_x
        R_y = (cur_y + cy) / h - I_y

        # Symbolic matrix and vector operations
        m_x = ca.vertcat(1 - R_x, R_x)
        m_g = ca.horzcat(ca.vertcat(gxy, gxpy), ca.vertcat(gxyp, gxpyp))
        m_y = ca.vertcat(1 - R_y, R_y)

        # Compute the value
        g_value = ca.mtimes([m_x.T, m_g, m_y])
        return g_value

    def solve(self, x_real, y_real, theta_real, velocity_real, a_real, fai_real):

        # current state
        x0 = np.array([x_real, y_real, theta_real, velocity_real])

        # terminal state
        xs = np.array([self.target_x, self.target_y, self.target_theta, self.target_velocity])

        # setting
        simX = np.zeros((self.N+1, self.nx))
        simU = np.zeros((self.N, self.nu))
        x_current = x0
        simX[0, :] = x0.reshape(1, -1)  

        # reference
        vel_ref = np.zeros(int(self.N))

        for i in range(int(self.N/4 * 3)):
            vel_ref[i] = velocity_real + a_real * 4 * (i + 1) / (3 * self.N)
        vel_top = vel_ref[int(self.N/4 * 3) - 1]

        for i in range(int(self.N/4 * 3), self.N):
            vel_ref[i] = vel_ref[i-1] -  vel_top * 4 / (3 * self.N + 3)

        delta_x = self.target_x - x_real
        delta_y = self.target_y - y_real
        theta_between = np.abs(math.atan2(self.target_y - y_real, self.target_x - x_real))

        for i in range(self.N):
            if velocity_real > 0.1:
                xs_between = np.array([
                    delta_x / self.N * i + x_real,
                    delta_y / self.N * i + y_real,
                    theta_between,
                    vel_ref[i],
                    a_real,
                    fai_real
                ])
            else:
                xs_between = np.array([
                    (self.target_x - x_real) / self.N * i + x_real,
                    (self.target_y - y_real) / self.N * i + y_real,
                    np.abs(math.atan2(self.target_y - y_real, self.target_x - x_real)),
                    vel_ref[i],
                    0.3,
                    fai_real
                ])
            self.solver.set(i, 'yref', xs_between)
            self.solver.set(i, "p", np.concatenate((self.rtg[0: self.wpg * self.hpg], [x_real, y_real])))            
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
        next_vel = simX[1, 3]
        aim_a = simU[0, 0]
        aim_fai = simU[0, 1]

        return next_x, next_y, next_theta, next_vel, aim_a, aim_fai


    # plot function
    def plot_results(self, start_x, start_y, theta_log, a_log, x_log, y_log, x_real_log, y_real_log, o_log, v_log, grid):
        
        plt.figure()
        a = np.arange(0, (len(a_log)), 1)*self.dt
        plt.plot(a, a_log, 'r-', label='desired a')
        plt.xlabel('time')
        plt.ylabel('value')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.plot(a, v_log, 'r-', label='current_velocity')
        plt.xlabel('time')
        plt.ylabel('value')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.plot(a, o_log, 'r-', label='desired omega')
        plt.xlabel('time')
        plt.ylabel('value')
        plt.legend()
        plt.grid(True)
        plt.show()        

        plt.plot(a, y_log, 'r-', label='y trans')
        plt.xlabel('time')
        plt.ylabel('value')
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

        # Define the square size in meters
        square_size = 0.5

        # Create a color map where 0 is white, 50 is gray, and 100 is black
        color_map = {0: "white", -1: "gray", 100: "black"}
        cx, cy  = -5, 1.2825

        for i in range(len(grid)):
            for j in range(len(grid[i])):
                color = color_map[grid[i][j]]
                # Draw each square with the appropriate color
                plt.gca().add_patch(plt.Rectangle((cx + j * square_size, cy + i * square_size), square_size, square_size, color=color))

        plt.axis('equal')
        plt.legend()
        plt.show()


    def main(self, x_init, y_init, theta_init, velocity_init, a_init, fai_init, grid):
                    
        x_0, y_0, theta, vel= x_init, y_init, theta_init, velocity_init
        x_real, y_real, theta_real, vel_real, a_real, fai_real = x_init, y_init, theta_init, velocity_init, a_init, fai_init

        x_log, y_log = [], []
        theta_log = []
        a_log = []

        x_real_log, y_real_log = [], []
        o_log, v_log = [], []

        with tqdm(total=100, desc='cpu%', position=1) as cpubar, tqdm(total=100, desc='ram%', position=0) as rambar:
            for i in tqdm(range(self.Epi)):

                try:
                    x_0, y_0, theta, vel, a_0, o_0 = self.solve(x_real, y_real, theta_real, vel_real, a_real, fai_real)
                    

                    x_real, y_real, theta_real, vel_real, a_real = x_0, y_0, theta, vel, a_0
                    fai_real = 0.5 * fai_real + 0.5 * o_0
                    
                    x_log.append(x_0)
                    y_log.append(y_0)
                    theta_log.append(theta)
                    a_log.append(a_0)

                    x_real_log.append(x_real)
                    y_real_log.append(y_real)
                    o_log.append(fai_real)
                    v_log.append(vel)

                    if (x_0 - self.target_x) ** 2 + (y_0 - self.target_y) ** 2 < 1:
                        # break
                        print("reach the target", theta)
                        if self.plot_figures == True:
                            self.plot_results(x_init, y_init, theta_log, a_log, x_log, y_log, x_real_log, y_real_log, o_log, v_log, grid)
                        return [1, theta_log], x_log, y_log

                except RuntimeError:
                    print("Infesible", theta)
                    if self.plot_figures == True:
                        self.plot_results(x_init, y_init, theta_log, a_log, x_log, y_log, x_real_log, y_real_log, o_log, v_log, grid)
                    return [0, theta_log], x_log, y_log

            print("not reach the target", theta)
            if self.plot_figures == True:
                self.plot_results(x_init, y_init, theta_log, a_log, x_log, y_log, x_real_log, y_real_log, o_log, v_log, grid)
            return [0, theta_log], x_log, y_log

    
if __name__ == '__main__':

    car_model = GemCarModel()
    
    grid_1d = [0] * 440

    grid_1d = [0, 100, 100, 100, 100, 100, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1,\
               -1, 0, 100, -1, 100, 100, 100, 100, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, \
               0, 100, 0, 0, 0, 100, 100, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 100, 0, -1, -1, 0, 0, 0, 0, \
                0, 0, 100, -1, 100, 100, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, 0, 100, \
                100, 100, 0, 0, 0, 0, 0, 0, 100, 0, 100, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 100, 100, 100, 0, 0,\
                 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
                -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, \
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
                0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, \
                0, 0, 0, 0, 0, 0, 0, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
                0, 0, -1, 0, 0, 0, 0, 0, 0, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, \
                100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, \
                100, 0, -1, 0, 100, 100, 100, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 100, 100, 100, \
                -1, -1, 0, 100, 100, 0, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, \
                -1, -1, -1, 100, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, \
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    
    grid_full = grid_1d
    grid_1d = np.array(grid_1d)

    grid_full.append(0)
    grid_full.append(0)
    grid_full = np.array(grid_full)

    grid = (grid_1d.reshape(20, 22)).T

    start_x, start_y, theta, vel, a0, fai0 = 1, 1.2825, np.pi/2, 0.0007, 0, 0

    terminal = np.array([0.0, 20.0, np.pi/2, 0.0])

    opt = GemCarOptimizer(m_model=car_model.model, 
                               m_constraint=car_model.constraint, t_horizon=1.0, dt=0.05, grid = grid_full, target=terminal)
    opt.main(start_x, start_y, theta, vel, a0, fai0, grid)