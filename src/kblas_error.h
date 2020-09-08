/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/kblas_error.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef __KBLAS_ERR_CHECK_H__
#define __KBLAS_ERR_CHECK_H__

#include "kblas_defs.h"
// ----------------------------------------
// C++ function is overloaded for different error types,
// which depends on error types being enums to be differentiable.
//inline
int _kblas_error( cudaError_t err, const char* func, const char* file, int line );
int _kblas_error( cublasStatus_t err, const char* func, const char* file, int line );
int _kblas_error( int err, const char* func, const char* file, int line );

#define check_error_ret( err_, ret_ ) \
{if(!_kblas_error( (err_), __func__, __FILE__, __LINE__ )) return ret_;}

#define check_error( err_ ) \
{if(!_kblas_error( (err_), __func__, __FILE__, __LINE__ )) ;}

#define check_error_forward( err_ ) \
{ int ret_val = (err_); if(!_kblas_error( ret_val, __func__, __FILE__, __LINE__ )) return ret_val;}

#define check_ret_error check_error_forward

#ifdef DBG_MSG
#define ECHO_I(_val) printf("%s(%d) ", #_val, (_val));fflush( stdout );
#define ECHO_f(_val) printf("%s(%e) ", #_val, (_val));fflush( stdout );
#define ECHO_p(_val) printf("%s(%p) ", #_val, (_val));fflush( stdout );
#define ECHO_LN printf("line %d\n", __LINE__);fflush( stdout );
#else
#define ECHO_I(_val)
#define ECHO_f(_val)
#define ECHO_p(_val)
#define ECHO_LN
#endif

#endif //__KBLAS_ERR_CHECK_H__
