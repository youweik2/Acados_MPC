#!/usr/bin/env python

import numpy as np
import casadi as ca
from acados_template import AcadosModel

class GemCarModel(object):
    def __init__(self):

        model = AcadosModel() #  ca.types.SimpleNamespace()
        constraint = ca.types.SimpleNamespace()

        # control inputs
        a = ca.SX.sym('acceleration')
        theta = ca.SX.sym('theta')
        controls = ca.vertcat(a, theta)

        # n_controls = controls.size()[0]
        # model states
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        v = ca.SX.sym('v')
        states = ca.vertcat(x, y, v)

        rhs = [v*ca.cos(theta), v*ca.sin(theta), a]

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
        model.name = 'GemCarModel'

        # constraint
        constraint.a_max = 0.8
        constraint.a_min = -0.8
        constraint.expr = ca.vcat([a, theta])

        self.model = model
        self.constraint = constraint