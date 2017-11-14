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
 * @version 2.0.0
 * @author Ali Charara
 * @date 2017-11-13
 **/

#ifndef __KBLAS_ERR_CHECK_H__
#define __KBLAS_ERR_CHECK_H__

// ----------------------------------------
#define KBLAS_Success 1
#define KBLAS_UnknownError 0
#define KBLAS_NotSupported -1
#define KBLAS_NotImplemented -2
#define KBLAS_cuBLAS_Error -3
#define KBLAS_WrongConfig -4
#define KBLAS_CUDA_Error -5
#define KBLAS_InsufficientWorkspace -6
#define KBLAS_Error_Allocation -7
#define KBLAS_Error_Deallocation -8
#define KBLAS_Error_NotInitialized -9
#define KBLAS_Error_WrongInput -10
#define KBLAS_MAGMA_Error -11
#define KBLAS_SVD_NoConvergence -12
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

#endif //__KBLAS_ERR_CHECK_H__
