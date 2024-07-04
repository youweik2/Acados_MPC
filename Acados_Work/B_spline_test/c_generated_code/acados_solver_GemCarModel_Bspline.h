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

#ifndef ACADOS_SOLVER_GemCarModel_Bspline_H_
#define ACADOS_SOLVER_GemCarModel_Bspline_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define GEMCARMODEL_BSPLINE_NX     3
#define GEMCARMODEL_BSPLINE_NZ     0
#define GEMCARMODEL_BSPLINE_NU     8
#define GEMCARMODEL_BSPLINE_NP     3
#define GEMCARMODEL_BSPLINE_NBX    0
#define GEMCARMODEL_BSPLINE_NBX0   3
#define GEMCARMODEL_BSPLINE_NBU    0
#define GEMCARMODEL_BSPLINE_NSBX   0
#define GEMCARMODEL_BSPLINE_NSBU   0
#define GEMCARMODEL_BSPLINE_NSH    0
#define GEMCARMODEL_BSPLINE_NSH0   0
#define GEMCARMODEL_BSPLINE_NSG    0
#define GEMCARMODEL_BSPLINE_NSPHI  0
#define GEMCARMODEL_BSPLINE_NSHN   0
#define GEMCARMODEL_BSPLINE_NSGN   0
#define GEMCARMODEL_BSPLINE_NSPHIN 0
#define GEMCARMODEL_BSPLINE_NSPHI0 0
#define GEMCARMODEL_BSPLINE_NSBXN  0
#define GEMCARMODEL_BSPLINE_NS     0
#define GEMCARMODEL_BSPLINE_NS0    0
#define GEMCARMODEL_BSPLINE_NSN    0
#define GEMCARMODEL_BSPLINE_NG     0
#define GEMCARMODEL_BSPLINE_NBXN   0
#define GEMCARMODEL_BSPLINE_NGN    0
#define GEMCARMODEL_BSPLINE_NY0    0
#define GEMCARMODEL_BSPLINE_NY     0
#define GEMCARMODEL_BSPLINE_NYN    0
#define GEMCARMODEL_BSPLINE_N      20
#define GEMCARMODEL_BSPLINE_NH     0
#define GEMCARMODEL_BSPLINE_NHN    0
#define GEMCARMODEL_BSPLINE_NH0    0
#define GEMCARMODEL_BSPLINE_NPHI0  0
#define GEMCARMODEL_BSPLINE_NPHI   0
#define GEMCARMODEL_BSPLINE_NPHIN  0
#define GEMCARMODEL_BSPLINE_NR     0

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct GemCarModel_Bspline_solver_capsule
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

    external_function_param_casadi *impl_dae_fun;
    external_function_param_casadi *impl_dae_fun_jac_x_xdot_z;
    external_function_param_casadi *impl_dae_jac_x_xdot_u_z;

    external_function_param_casadi *impl_dae_hess;



    // cost

    external_function_param_casadi *ext_cost_fun;
    external_function_param_casadi *ext_cost_fun_jac;
    external_function_param_casadi *ext_cost_fun_jac_hess;





    external_function_param_casadi ext_cost_0_fun;
    external_function_param_casadi ext_cost_0_fun_jac;
    external_function_param_casadi ext_cost_0_fun_jac_hess;




    external_function_param_casadi ext_cost_e_fun;
    external_function_param_casadi ext_cost_e_fun_jac;
    external_function_param_casadi ext_cost_e_fun_jac_hess;



    // constraints







} GemCarModel_Bspline_solver_capsule;

ACADOS_SYMBOL_EXPORT GemCarModel_Bspline_solver_capsule * GemCarModel_Bspline_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int GemCarModel_Bspline_acados_free_capsule(GemCarModel_Bspline_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int GemCarModel_Bspline_acados_create(GemCarModel_Bspline_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int GemCarModel_Bspline_acados_reset(GemCarModel_Bspline_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of GemCarModel_Bspline_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int GemCarModel_Bspline_acados_create_with_discretization(GemCarModel_Bspline_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int GemCarModel_Bspline_acados_update_time_steps(GemCarModel_Bspline_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int GemCarModel_Bspline_acados_update_qp_solver_cond_N(GemCarModel_Bspline_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int GemCarModel_Bspline_acados_update_params(GemCarModel_Bspline_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int GemCarModel_Bspline_acados_update_params_sparse(GemCarModel_Bspline_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);

ACADOS_SYMBOL_EXPORT int GemCarModel_Bspline_acados_solve(GemCarModel_Bspline_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void GemCarModel_Bspline_acados_batch_solve(GemCarModel_Bspline_solver_capsule ** capsules, int N_batch);
ACADOS_SYMBOL_EXPORT int GemCarModel_Bspline_acados_free(GemCarModel_Bspline_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void GemCarModel_Bspline_acados_print_stats(GemCarModel_Bspline_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int GemCarModel_Bspline_acados_custom_update(GemCarModel_Bspline_solver_capsule* capsule, double* data, int data_len);


ACADOS_SYMBOL_EXPORT ocp_nlp_in *GemCarModel_Bspline_acados_get_nlp_in(GemCarModel_Bspline_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *GemCarModel_Bspline_acados_get_nlp_out(GemCarModel_Bspline_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *GemCarModel_Bspline_acados_get_sens_out(GemCarModel_Bspline_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *GemCarModel_Bspline_acados_get_nlp_solver(GemCarModel_Bspline_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *GemCarModel_Bspline_acados_get_nlp_config(GemCarModel_Bspline_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *GemCarModel_Bspline_acados_get_nlp_opts(GemCarModel_Bspline_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *GemCarModel_Bspline_acados_get_nlp_dims(GemCarModel_Bspline_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *GemCarModel_Bspline_acados_get_nlp_plan(GemCarModel_Bspline_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_GemCarModel_Bspline_H_
