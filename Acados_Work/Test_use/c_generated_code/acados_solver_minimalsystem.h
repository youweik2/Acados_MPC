/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SOLVER_minimalsystem_H_
#define ACADOS_SOLVER_minimalsystem_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define MINIMALSYSTEM_NX     4
#define MINIMALSYSTEM_NZ     0
#define MINIMALSYSTEM_NU     2
#define MINIMALSYSTEM_NP     0
#define MINIMALSYSTEM_NBX    0
#define MINIMALSYSTEM_NBX0   4
#define MINIMALSYSTEM_NBU    2
#define MINIMALSYSTEM_NSBX   0
#define MINIMALSYSTEM_NSBU   0
#define MINIMALSYSTEM_NSH    1
#define MINIMALSYSTEM_NSH0   0
#define MINIMALSYSTEM_NSG    0
#define MINIMALSYSTEM_NSPHI  0
#define MINIMALSYSTEM_NSHN   0
#define MINIMALSYSTEM_NSGN   0
#define MINIMALSYSTEM_NSPHIN 0
#define MINIMALSYSTEM_NSPHI0 0
#define MINIMALSYSTEM_NSBXN  0
#define MINIMALSYSTEM_NS     1
#define MINIMALSYSTEM_NS0    0
#define MINIMALSYSTEM_NSN    0
#define MINIMALSYSTEM_NG     0
#define MINIMALSYSTEM_NBXN   0
#define MINIMALSYSTEM_NGN    0
#define MINIMALSYSTEM_NY0    6
#define MINIMALSYSTEM_NY     6
#define MINIMALSYSTEM_NYN    4
#define MINIMALSYSTEM_N      10
#define MINIMALSYSTEM_NH     1
#define MINIMALSYSTEM_NHN    0
#define MINIMALSYSTEM_NH0    0
#define MINIMALSYSTEM_NPHI0  0
#define MINIMALSYSTEM_NPHI   0
#define MINIMALSYSTEM_NPHIN  0
#define MINIMALSYSTEM_NR     0

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct minimalsystem_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */
    // dynamics

    external_function_param_casadi *forw_vde_casadi;
    external_function_param_casadi *expl_ode_fun;




    // cost






    // constraints
    external_function_param_casadi *nl_constr_h_fun_jac;
    external_function_param_casadi *nl_constr_h_fun;







} minimalsystem_solver_capsule;

ACADOS_SYMBOL_EXPORT minimalsystem_solver_capsule * minimalsystem_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int minimalsystem_acados_free_capsule(minimalsystem_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int minimalsystem_acados_create(minimalsystem_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int minimalsystem_acados_reset(minimalsystem_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of minimalsystem_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int minimalsystem_acados_create_with_discretization(minimalsystem_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int minimalsystem_acados_update_time_steps(minimalsystem_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int minimalsystem_acados_update_qp_solver_cond_N(minimalsystem_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int minimalsystem_acados_update_params(minimalsystem_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int minimalsystem_acados_update_params_sparse(minimalsystem_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);

ACADOS_SYMBOL_EXPORT int minimalsystem_acados_solve(minimalsystem_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void minimalsystem_acados_batch_solve(minimalsystem_solver_capsule ** capsules, int N_batch);
ACADOS_SYMBOL_EXPORT int minimalsystem_acados_free(minimalsystem_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void minimalsystem_acados_print_stats(minimalsystem_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int minimalsystem_acados_custom_update(minimalsystem_solver_capsule* capsule, double* data, int data_len);


ACADOS_SYMBOL_EXPORT ocp_nlp_in *minimalsystem_acados_get_nlp_in(minimalsystem_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *minimalsystem_acados_get_nlp_out(minimalsystem_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *minimalsystem_acados_get_sens_out(minimalsystem_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *minimalsystem_acados_get_nlp_solver(minimalsystem_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *minimalsystem_acados_get_nlp_config(minimalsystem_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *minimalsystem_acados_get_nlp_opts(minimalsystem_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *minimalsystem_acados_get_nlp_dims(minimalsystem_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *minimalsystem_acados_get_nlp_plan(minimalsystem_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_minimalsystem_H_
