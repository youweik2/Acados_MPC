import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import casadi as ca
import pickle
import math

try:
    from GemCar import GemCarModel
except ModuleNotFoundError:
    from .GemCar import GemCarModel


class GemCarOptimizer(object):
    def __init__(self, m_model, m_constraint, t_horizon, dt):
        model = m_model

        self.T = t_horizon
        self.dt = dt
        self.N = int(t_horizon / dt)

        # Basic settings
        self.Epi = 3000
        self.plot_figures = True

        # Grid map definition
        self.matrix = np.zeros((50, 50))  # Replace with your custom grid values
        self.matrix[20:30, 20:30] = 100  # Example obstacle in the grid

        self.row = len(self.matrix)
        self.column = len(self.matrix[0])

        self.gridmap = ca.SX(self.row, self.column)
        for i in range(self.row):
            for j in range(self.column):
                self.gridmap[i, j] = self.matrix[i][j]

        # Target settings
        self.target_x = 0.0
        self.target_y = 50.0
        self.target_theta = np.pi / 2
        self.target_velocity = 5.0

        # Ensure working directory is set correctly
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, acados_source_path)

        # Acados OCP
        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.T

        # Cost settings: external cost
        x_ = ocp.model.x
        u_ = ocp.model.u
        obstacle_penalty = 2

        stage_cost = (
            (x_[0] - self.target_x) ** 2
            + 2 * (x_[1] - self.target_y) ** 2
            + 20 * (x_[3] - self.target_velocity) ** 2
            + 10 * u_[0] ** 2
            + 1000 * u_[1] ** 2
            + obstacle_penalty * self.get_gvalue(x_[0], x_[1], 10, -5)
        )

        terminal_cost = (
            (x_[0] - self.target_x) ** 2 + 2 * (x_[1] - self.target_y) ** 2
        )

        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = stage_cost
        ocp.model.cost_expr_ext_cost_e = terminal_cost

        # Constraints
        ocp.constraints.lbu = np.array([m_constraint.a_min, m_constraint.theta_min])
        ocp.constraints.ubu = np.array([m_constraint.a_max, m_constraint.theta_max])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.lbx = np.array([-10, -100, 0, 0])
        ocp.constraints.ubx = np.array([10, 100, np.pi, 6])
        ocp.constraints.idxbx = np.array([0, 1, 2, 3])

        # Initial state constraint
        x_ref = np.zeros(model.x.size()[0])
        ocp.constraints.x0 = x_ref

        # Solver options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"

        # Compile OCP
        json_file = os.path.join("./" + model.name + "_acados_ocp.json")
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)
        self.integrator = AcadosSimSolver(ocp, json_file=json_file)

    def get_gvalue(self, cur_x, cur_y, cx, cy):

        h = 0.4

        # Compute symbolic grid indices
        grid_x = ca.floor((cur_x + cx) / h)
        grid_y = ca.floor((cur_y + cy) / h)

        # Handle boundary cases symbolically
        grid_x = ca.if_else(grid_x < 0, 0, ca.if_else(grid_x >= self.column, self.column - 1, grid_x))
        grid_x1 = ca.if_else(grid_x + 1 >= self.column, self.column - 1, grid_x + 1)
        grid_y = ca.if_else(grid_y < 0, 0, ca.if_else(grid_y >= self.row, self.row - 1, grid_y))
        grid_y1 = ca.if_else(grid_y + 1 >= self.row, self.row - 1, grid_y + 1)

        def access_matrix(matrix, grid_x, grid_y):
            value = 0
            for i in range(self.row):
                for j in range(self.column):
                    # Use ca.logic_and() instead of &
                    value += ca.if_else(ca.logic_and(grid_x == i, grid_y == j), matrix[i, j], 0)
            return value

        # Interpolate matrix values
        gxy = access_matrix(self.gridmap, grid_y, grid_x)
        gxyp = access_matrix(self.gridmap, grid_y, grid_x1)
        gxpy = access_matrix(self.gridmap, grid_y1, grid_x)
        gxpyp = access_matrix(self.gridmap, grid_y1, grid_x1)

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

    def solve(self, x_real, y_real, theta_real, velocity_real):
        x0 = np.zeros(4)
        x0[0] = x_real
        x0[1] = y_real
        x0[2] = theta_real
        x0[3] = velocity_real

        xs = np.zeros(4)
        xs[0] = self.target_x
        xs[1] = self.target_y
        xs[2] = self.target_theta
        xs[3] = self.target_velocity

        simX = np.zeros((self.N + 1, 4))
        simU = np.zeros((self.N, 2))
        x_current = x0
        simX[0, :] = x0.reshape(1, -1)

        for i in range(self.N):
            xs_between = np.concatenate((xs, np.array([0.0, 0.0])))
            self.solver.set(i, "yref", xs_between)
        self.solver.set(self.N, "yref", xs)

        self.solver.set(0, "lbx", x_current)
        self.solver.set(0, "ubx", x_current)
        status = self.solver.solve()

        if status != 0:
            raise Exception(f"Acados OCP solver returned status {status}.")

        simX[0, :] = self.solver.get(0, "x")

        for i in range(self.N):
            simU[i, :] = self.solver.get(i, "u")
            simX[i + 1, :] = self.solver.get(i + 1, "x")

        next_x, next_y, next_theta, next_vel = simX[1, :]
        aim_a, aim_fai = simU[0, :]

        return next_x, next_y, next_theta, next_vel, aim_a, aim_fai

    def main(self, x_init, y_init, theta_init, velocity_init):
        x_0, y_0, theta, vel = x_init, y_init, theta_init, velocity_init
        x_real, y_real, theta_real, vel_real = x_0, y_0, theta, vel

        x_log, y_log, theta_log, a_log = [x_0], [y_0], [theta], []

        for i in tqdm(range(self.Epi), desc="Solving MPC"):
            try:
                x_0, y_0, theta, vel, a_0, o_0 = self.solve(x_real, y_real, theta_real, vel_real)
                x_log.append(x_0)
                y_log.append(y_0)
                theta_log.append(theta)
                a_log.append(a_0)

                if (x_0 - self.target_x) ** 2 + (y_0 - self.target_y) ** 2 < 1:
                    print("Target reached!")
                    break
            except Exception as e:
                print(f"Error: {e}")
                break


if __name__ == "__main__":
    start_x, start_y, theta, vel = 0.0, 0.0, np.pi / 2, 0.0
    car_model = GemCarModel()
    optimizer = GemCarOptimizer(m_model=car_model.model, m_constraint=car_model.constraint, t_horizon=1.0, dt=0.05)
    optimizer.main(start_x, start_y, theta, vel)


