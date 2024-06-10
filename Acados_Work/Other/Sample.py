from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
import numpy as np
import scipy.linalg
from casadi import SX, vertcat
import matplotlib.pyplot as plt

def export_ode_model():
    model_name = 'minimalsystem'

    # set up states & controls
    x1      = SX.sym('x1')
    x2      = SX.sym('x2')

    F1 = SX.sym('F1')
    F2 = SX.sym('F2')
    u = vertcat(F1,F2)

    # xdot
    x1_dot      = SX.sym('x1_dot')
    x2_dot      = SX.sym('x2_dot')

    x = vertcat(x1, x1_dot, x2, x2_dot)
    xdot = SX.sym('xdot',x.size()[0],1)

    # dynamics
    f_expl = vertcat(x1_dot,
                     F1,
                     x2_dot,
                     F2
                     )

    f_impl = xdot - f_expl

    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    return model

def main():
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_ode_model()
    ocp.model = model

    Tf = 1.5
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx
    N_horizon = 10
    Fmax = 2
    setpoint = np.array([4,4])     
    x0 = np.array([0, 0.0, 0.0, 0.0])

    # set dimensions
    ocp.dims.N = N_horizon

    # set cost module
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'

    Q_mat = np.diag([1e3, 1e0, 1e3, 1e0])
    R_mat = np.diag([1e-2, 1e-2])

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx,:nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.yref  = np.array([setpoint[0],0,setpoint[1],0,0,0]) #np.zeros((ny, ))
    ocp.cost.yref_e = np.array([setpoint[0],0,setpoint[1],0]) #np.zeros((ny_e, ))

    # linear state constraints
    ocp.constraints.constr_type = 'BGH'
    # ocp.constraints.constr_type = 'BGP'
    ocp.constraints.lbu = np.array([-Fmax,-Fmax])
    ocp.constraints.ubu = np.array([+Fmax,+Fmax])
    ocp.constraints.x0 = x0
    ocp.constraints.idxbu = np.array([0, 1])

    # non-linear (BGH) state constraint: circle
    ocp.model.con_h_expr = (model.x[0]-2)**2 + (model.x[2]-2)**2  # x1, x2
    ocp.constraints.lh = np.array([1**2])       # radius
    ocp.constraints.uh = np.array([10e3])       
   
    
    # slack variable configuration:
    nsh = 1
    ocp.constraints.lsh = np.zeros(nsh)             # Lower bounds on slacks corresponding to soft lower bounds for nonlinear constraints
    ocp.constraints.ush = np.zeros(nsh)             # Lower bounds on slacks corresponding to soft upper bounds for nonlinear constraints
    ocp.constraints.idxsh = np.array(range(nsh))    # Jsh
    ns = 1
    ocp.cost.zl = 10e5 * np.ones((ns,)) # gradient wrt lower slack at intermediate shooting nodes (1 to N-1)
    ocp.cost.Zl = 1 * np.ones((ns,))    # diagonal of Hessian wrt lower slack at intermediate shooting nodes (1 to N-1)
    ocp.cost.zu = 0 * np.ones((ns,))    
    ocp.cost.Zu = 1 * np.ones((ns,))  
    

    # default solver params
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.qp_solver_cond_N = N_horizon

    # set prediction horizon
    ocp.solver_options.tf = Tf

    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    Nsim = 200
    simX = np.ndarray((Nsim+1, nx))
    simU = np.ndarray((Nsim, nu))

    simX[0,:] = x0
    xy_predictions = np.zeros((N_horizon,2))    

    # initialize figure
    fig0 = plt.figure()
    ax1 = fig0.add_subplot(1,1,1)
    circle = plt.Circle((2, 2), radius=1, color='blue',alpha=0.1,label='constraint')
    ax1.plot(setpoint[0],setpoint[1], 'go',label='target')
    ax1.add_patch(circle); ax1.set_xlabel('x1'); ax1.set_ylabel('x2'); ax1.grid()
    ax1.legend()

    # closed loop
    runOnce = True
    for i in range(Nsim):
        
        # solve ocp and get next control input
        try:
            simU[i,:] = acados_ocp_solver.solve_for_x0(x0_bar = simX[i, :])
        except:
            ax1.plot(simX[i,0],simX[i,2],'ro',label='infeasible x0')
            ax1.legend()
            break
        
        # extract solution state info
        for j in range(N_horizon):
            xy_predictions[j,0] = acados_ocp_solver.get(j, "x")[0] #x1
            xy_predictions[j,1] = acados_ocp_solver.get(j, "x")[2] #x2
        
        # update figure with predictions
        ax1.plot(simX[i,0],simX[i,2],'ko')
        ax1.plot(xy_predictions[:,0],xy_predictions[:,1],'k-',alpha=0.1)
        
        plt.show()  # put a breakpoint here to F5 and plot through the for-loop.

        # simulate system
        simX[i+1, :] = acados_integrator.simulate(x=simX[i, :], u=simU[i,:])

    # plot results    
    plt.show(block=True)

if __name__ == '__main__':
    main()