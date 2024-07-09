#!/usr/bin/env python

import numpy as np
import casadi as ca
from acados_template import AcadosModel

class GemCarBsplModel(object):
    def __init__(self,):

        model = AcadosModel() #  ca.types.SimpleNamespace()
        constraint = ca.types.SimpleNamespace()

        # control inputs
        controls = ca.SX.sym('u', 8)

        # tau values
        tau_0 = ca.SX.sym('tau_0')
        tau_i = ca.SX.sym('tau_i')
        tau_i1 = ca.SX.sym('tau_i1')
        tau = ca.vertcat(tau_0, tau_i, tau_i1)

        # state inputs
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(x, y, theta)

        # constant paras -> get rhs

        k32 = (3*(tau_0 - tau_i))/(tau_i - tau_i1) + (3*(tau_0 - tau_i)**2)/(tau_i - tau_i1)**2 + (tau_0 - tau_i)**3/(tau_i - tau_i1)**3 + 1
        k11 = np.cos(theta)*k32
        k21 = np.sin(theta)*k32
        k34 = - (3*(tau_0 - tau_i))/(tau_i - tau_i1) - (6*(tau_0 - tau_i)**2)/(tau_i - tau_i1)**2 - (3*(tau_0 - tau_i)**3)/(tau_i - tau_i1)**3
        k13 = np.cos(theta)*k34
        k23 = np.sin(theta)*k34
        k36 = (3*(tau_0 - tau_i)**2)/(tau_i - tau_i1)**2 + (3*(tau_0 - tau_i)**3)/(tau_i - tau_i1)**3
        k15 = np.cos(theta)*k36
        k25 = np.sin(theta)*k36
        k38 = -(tau_0 - tau_i)**3/(tau_i - tau_i1)**3
        k17 = np.cos(theta)*k38
        k27 = np.sin(theta)*k38

        dx = k11*controls[0] + k13*controls[2] + k15*controls[4] + k17*controls[6]
        dy = k21*controls[0] + k23*controls[2] + k25*controls[4] + k27*controls[6]
        omega = k32*controls[1] + k34*controls[3] + k36*controls[5] + k38*controls[7]

        rhs = [dx, dy, omega]

        # function
        f = ca.Function('f', [states, controls, tau], [ca.vcat(rhs)], ['state', 'control_input', 'tau'], ['rhs'])

        # acados model
        x_dot = ca.SX.sym('x_dot', len(rhs))
        f_impl = x_dot - f(states, controls, tau)

        model.f_expl_expr = f(states, controls, tau)
        model.f_impl_expr = f_impl
        model.x = states
        model.xdot = x_dot
        model.u = controls
        model.p = tau
        model.name = 'GemCarModel_Bspline'

        self.model = model