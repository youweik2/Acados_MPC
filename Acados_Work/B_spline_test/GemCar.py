#!/usr/bin/env python

import numpy as np
import casadi as ca
from acados_template import AcadosModel

class GemCarModel(object):
    def __init__(self,):

        model = AcadosModel() #  ca.types.SimpleNamespace()
        constraint = ca.types.SimpleNamespace()
        # control inputs
        v = ca.SX.sym('v')
        omega = ca.SX.sym('omega')
        controls = ca.vertcat(v, omega)
        # n_controls = controls.size()[0]
        # model states
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(x, y, theta)

        rhs = [v*ca.cos(theta), v*ca.sin(theta), omega]

        # function
        f = ca.Function('f', [states, controls], [ca.vcat(rhs)], ['state', 'control_input'], ['rhs'])
        # f_expl = ca.vcat(rhs)
        # acados model
        x_dot = ca.SX.sym('x_dot', len(rhs))
        f_impl = x_dot - f(states, controls)

        model.f_expl_expr = f(states, controls)
        model.f_impl_expr = f_impl
        model.x = states
        model.xdot = x_dot
        model.u = controls
        model.p = []
        model.name = 'GemCarModel_Bspline'

        # constraint
        constraint.v_max = 1.5
        constraint.v_min = -1.5
        constraint.omega_max = 3
        constraint.omega_min = -3
        constraint.expr = ca.vcat([v, omega])

        self.model = model
        self.constraint = constraint