from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from casadi import SX, vertcat, Function, dot, cross, norm_2, inv, diag, sqrt
import numpy as np
from os import environ, system, removedirs, path
import shutil

def sx_quat_multiply(q, p):

    s1 = q[0]
    v1 = q[1:4]
    s2 = p[0]
    v2 = p[1:4]
    s = s1 * s2 - dot(v1, v2)
    v = s1 * v2 + s2 * v1 + cross(v1, v2)
    return vertcat(s, v)


def sx_quat_inverse(q):

    return SX([1, -1, -1, -1]) * q / (norm_2(q)**2)


def quat_derivative(q, w):

    return sx_quat_multiply(q, vertcat(SX(0), w)) / 2


def rotate_vec3(v, q):

    p = vertcat(SX(0), v)
    p_rotated = sx_quat_multiply(sx_quat_multiply(q, p), sx_quat_inverse(q))
    return p_rotated[1:4]


def quat_err(q, p):

    # assume q = q_delta * p, q_delta = [w, q_err] = q * p^(-1)
    q_d = sx_quat_multiply(q, sx_quat_inverse(p))
    return q_d[1:4]


def e(x, y_ref):

    p_ref = y_ref[0:3]
    q_ref = y_ref[3:7]
    v_ref = y_ref[7:10]
    w_ref = y_ref[10:13]

    p_err = x[0:3] - p_ref
    q_err = quat_err(x[3:7], q_ref)
    v_err = x[7:10] - v_ref
    w_err = x[10:13] - w_ref
    f = x[13:16]
    t = x[16:19]
    
    return vertcat(p_err, q_err, v_err, w_err, f, t)


def h(x):

    # fields in the state variable
    v = x[7:10]
    w = x[10:13]
    f = x[13:16]
    t = x[16:19]
    
    # allocation: y = pinv(B) * wrench
    # y = [sin(a1)f1, cos(a1)f1, ... sin(a6)f6, cos(a6)f6]
    # wrench = [f, t]
    # pinv(B) see matlab script allocation.m, with torque = thrust * 0.06
    B_inv = SX([
        [-1/3,            0     ,         0  ,          -25/149,          0     ,       425/894],
        [ 0  ,            0     ,        -1/6,         -425/447,          0     ,       -25/298],
        [ 1/3,            0     ,         0  ,          -25/149,          0     ,       425/894],
        [ 0  ,            0     ,        -1/6,          425/447,          0     ,        25/298],
        [ 1/6,          390/1351,         0  ,           25/298,        695/4783,       425/894],
        [ 0  ,            0     ,        -1/6,          425/894,        760/923 ,       -25/298],
        [-1/6,         -390/1351,         0  ,           25/298,        695/4783,       425/894],
        [ 0  ,            0     ,        -1/6,         -425/894,       -760/923 ,        25/298],
        [-1/6,          390/1351,         0  ,           25/298,       -695/4783,       425/894],
        [ 0  ,            0     ,        -1/6,         -425/894,        760/923 ,        25/298],
        [ 1/6,         -390/1351,         0  ,           25/298,       -695/4783,       425/894],
        [ 0  ,            0     ,        -1/6,          425/894,       -760/923 ,       -25/298]
    ])
    y = B_inv @ vertcat(f, t)
    f_unit_actuator = vertcat(
        sqrt(y[0]**2 + y[1]**2),
        sqrt(y[2]**2 + y[3]**2),
        sqrt(y[4]**2 + y[5]**2),
        sqrt(y[6]**2 + y[7]**2),
        sqrt(y[8]**2 + y[9]**2),
        sqrt(y[10]**2 + y[11]**2)
    )

    # h = [v, w, f_unit_actuator]
    return vertcat(v, w, f_unit_actuator)
    

def get_omnihex_ode_model():

    # nomial value of parameters
    m = 4.0
    g = SX([0, 0, 9.8])
    com = SX([0, 0, 0.03]) # center of mass position in body FRD frame
    J_xx = 0.08401764152
    J_xy = -0.00000445135
    J_xz = 0.00014163105
    J_yy = 0.08169689992
    J_yz = -0.00000936372
    J_zz = 0.14273598018
    J = SX([
        [J_xx, J_xy, J_xz],
        [J_xy, J_yy, J_yz],
        [J_xz, J_yz, J_zz]
    ])
    J_inv = inv(J)

    # states and controls
    p = SX.sym('p', 3) # position in world NED frame (m)
    q = SX.sym('q', 4) # attitude expressed in quaternion
    v = SX.sym('v', 3) # velocity in body frame (m/s)
    w = SX.sym('w', 3) # angular velocity in body FRD frame (rad/s)
    f = SX.sym('f', 3) # actuator force in body frame (N)
    t = SX.sym('t', 3) # actuator torque in body frame (Nm)

    p_dot = SX.sym('p_dot', 3)
    q_dot = SX.sym('q_dot', 4)
    v_dot = SX.sym('v_dot', 3)
    w_dot = SX.sym('w_dot', 3)
    f_dot_x = SX.sym('f_dot_x', 3)
    t_dot_x = SX.sym('t_dot_x', 3)
    f_dot_u = SX.sym('f_dot_u', 3)
    t_dot_u = SX.sym('t_dot_u', 3)

    x = vertcat(p, q, v, w, f, t)
    u = vertcat(f_dot_u, t_dot_u)

    # dynamics
    x_dot = vertcat(p_dot, q_dot, v_dot, w_dot, f_dot_x, t_dot_x)
    f_expl = vertcat(
        rotate_vec3(v, q),
        quat_derivative(q, w),
        f / m + rotate_vec3(g, sx_quat_inverse(q)) - cross(w, v),
        J_inv @ (t + cross(com, f) - cross(w, J @ w)),
        f_dot_u,
        t_dot_u
    )
    
    f_impl = x_dot - f_expl

    # acados model
    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.x = x
    model.xdot = x_dot
    model.u = u
    model.name = 'omnihex_ode'
    return model


def formulate_omnihex_ocp():

    # prediction horizon and dt
    N = 20
    Tf = 1.0

    # cost weighting matrices
    Q_p = SX([1, 1, 1])
    Q_q = SX([1, 1, 1])
    Q_v = SX([1, 1, 1])
    Q_w = SX([1, 1, 1])
    Q_f = SX([1, 1, 1])
    Q_t = SX([1, 1, 1])
    Q = diag(vertcat(Q_p, Q_q, Q_v, Q_w, Q_f, Q_t))

    R_f_dot = SX([1, 1, 1]) # small R -> aggressive moves, large R -> large error
    R_t_dot = SX([1, 1, 1])
    R = diag(vertcat(R_f_dot, R_t_dot))

    Q_p_e = SX([1, 1, 1])
    Q_q_e = SX([1, 1, 1])
    Q_v_e = SX([1, 1, 1])
    Q_w_e = SX([1, 1, 1])
    Q_f_e = SX([1, 1, 1])
    Q_t_e = SX([1, 1, 1])
    Q_e = diag(vertcat(Q_p_e, Q_q_e, Q_v_e, Q_w_e, Q_f_e, Q_t_e))

    # constraints, u = [f_dot, t_dot], h = [v, w, f_unit_actuator]
    u_lb = np.array([-1, -1, -1, -1, -1, -1])
    u_ub = np.array([1, 1, 1, 1, 1, 1])
    h_lb = np.array([-10, -10, -10, -60, -60, -60, 0, 0, 0, 0, 0, 0])
    h_ub = np.array([10, 10, 10, 60, 60, 60, 10, 10, 10, 10, 10, 10])

    # reference trajectory, with y_ref being the interface
    # set to all 'zero' for now, can be set later when using the solver
    # y_ref = [p_ref, q_ref, v_ref, w_ref]
    y_ref = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # optimal control problem
    # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcp
    ocp = AcadosOcp()

    # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcpDims
    ocp.dims.N = N # prediction horizon, other fields in dim are set automatically

    # https://docs.acados.org/python_interface/index.html#acados_template.acados_model.AcadosModel
    ocp.model = get_omnihex_ode_model() # dynamics
    x = ocp.model.x
    u = ocp.model.u
    ocp.model.con_h_expr = h(x)
    ocp.model.con_h_expr_e = h(x)
    ocp.model.cost_expr_ext_cost = e(x, y_ref).T @ Q @ e(x, y_ref) + u.T @ R @ u
    ocp.model.cost_expr_ext_cost_e = e(x, y_ref).T @ Q_e @ e(x, y_ref)

    # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcpCost
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcpConstraints
    ocp.constraints.constr_type = 'BGH'
    ocp.constraints.lbu = u_lb
    ocp.constraints.ubu = u_ub
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5])
    ocp.constraints.lh = h_lb
    ocp.constraints.uh = h_ub
    ocp.constraints.lh_e = h_lb
    ocp.constraints.uh_e = h_ub

    # https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp.AcadosOcpOptions
    ocp.solver_options.tf = Tf
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'

    ocp.acados_include_path = environ['ACADOS_SOURCE_DIR'] + '/include'
    ocp.acados_lib_path = environ['ACADOS_SOURCE_DIR'] + '/lib'
    ocp.code_export_directory = '../acados_export'

    return ocp


def generate_ref(solver):

    # TODO: How to set reference trajectory?
    # TODO: y_ref and yref?
    # TODO: email, discourse, wechat?

    print('generate simple reference for testing only')

    
if __name__ == '__main__':

    ocp = formulate_omnihex_ocp()

    if path.exists(ocp.code_export_directory):
        shutil.rmtree(ocp.code_export_directory)

    json_path = ocp.code_export_directory + '/acados_ocp_' + ocp.model.name + '.json'
    ocpSolver = AcadosOcpSolver(ocp, json_file = json_path)
    simSolver = AcadosSimSolver(ocp, json_file = json_path)

    generate_ref(ocpSolver)