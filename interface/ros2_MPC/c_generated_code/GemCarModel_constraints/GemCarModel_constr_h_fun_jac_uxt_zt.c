/* This file was automatically generated by CasADi 3.6.6.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) GemCarModel_constr_h_fun_jac_uxt_zt_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[12] = {5, 3, 0, 2, 4, 6, 2, 3, 2, 3, 2, 3};
static const casadi_int casadi_s4[3] = {3, 0, 0};

/* GemCarModel_constr_h_fun_jac_uxt_zt:(i0[3],i1[2],i2[],i3[])->(o0[3],o1[5x3,6nz],o2[3x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4, a5, a6, a7, a8;
  a0=arg[0]? arg[0][0] : 0;
  a1=casadi_sq(a0);
  a2=arg[0]? arg[0][1] : 0;
  a3=20.;
  a3=(a2-a3);
  a4=casadi_sq(a3);
  a1=(a1+a4);
  a4=1.;
  a1=(a1-a4);
  if (res[0]!=0) res[0][0]=a1;
  a1=-1.;
  a1=(a0-a1);
  a5=casadi_sq(a1);
  a6=25.;
  a6=(a2-a6);
  a7=casadi_sq(a6);
  a5=(a5+a7);
  a5=(a5-a4);
  if (res[0]!=0) res[0][1]=a5;
  a5=(a0-a4);
  a7=casadi_sq(a5);
  a8=30.;
  a2=(a2-a8);
  a8=casadi_sq(a2);
  a7=(a7+a8);
  a7=(a7-a4);
  if (res[0]!=0) res[0][2]=a7;
  a0=(a0+a0);
  if (res[1]!=0) res[1][0]=a0;
  a3=(a3+a3);
  if (res[1]!=0) res[1][1]=a3;
  a1=(a1+a1);
  if (res[1]!=0) res[1][2]=a1;
  a6=(a6+a6);
  if (res[1]!=0) res[1][3]=a6;
  a5=(a5+a5);
  if (res[1]!=0) res[1][4]=a5;
  a2=(a2+a2);
  if (res[1]!=0) res[1][5]=a2;
  return 0;
}

CASADI_SYMBOL_EXPORT int GemCarModel_constr_h_fun_jac_uxt_zt(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int GemCarModel_constr_h_fun_jac_uxt_zt_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int GemCarModel_constr_h_fun_jac_uxt_zt_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void GemCarModel_constr_h_fun_jac_uxt_zt_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int GemCarModel_constr_h_fun_jac_uxt_zt_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void GemCarModel_constr_h_fun_jac_uxt_zt_release(int mem) {
}

CASADI_SYMBOL_EXPORT void GemCarModel_constr_h_fun_jac_uxt_zt_incref(void) {
}

CASADI_SYMBOL_EXPORT void GemCarModel_constr_h_fun_jac_uxt_zt_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int GemCarModel_constr_h_fun_jac_uxt_zt_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int GemCarModel_constr_h_fun_jac_uxt_zt_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real GemCarModel_constr_h_fun_jac_uxt_zt_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* GemCarModel_constr_h_fun_jac_uxt_zt_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* GemCarModel_constr_h_fun_jac_uxt_zt_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* GemCarModel_constr_h_fun_jac_uxt_zt_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* GemCarModel_constr_h_fun_jac_uxt_zt_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s3;
    case 2: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int GemCarModel_constr_h_fun_jac_uxt_zt_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int GemCarModel_constr_h_fun_jac_uxt_zt_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 3*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif