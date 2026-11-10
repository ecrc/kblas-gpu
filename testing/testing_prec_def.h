/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/testing_prec_def.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ali Charara
 * @date 2020-12-10
 **/

#ifndef __TESTING_PREC_DEF__
#define __TESTING_PREC_DEF__

#include "kblas_prec_def.h"

#if defined   PREC_s
  #define cublasXtrsm_batched cublasStrsmBatched
  #define LAPACK_GEMM cblas_sgemm
  #define LAPACK_GEMM_BATCH sgemm_batch
  #ifdef USE_MKL
  #define LAPACK_GESVD sgesvd
  #define LAPACK_LACPY slacpy
  #define LAPACK_OMATCPY mkl_somatcopy
  #else
    #define LAPACK_GESVD LAPACK_sgesvd
    #define LAPACK_LACPY LAPACK_slacpy
    #define LAPACK_TRANS LAPACKE_sge_trans
  #endif
  #define LAPACK_SCAL cblas_sscal

  #define Xrand_matrix srand_matrix
  #define Xget_max_error_matrix sget_max_error_matrix
  #define Xmatrix_make_hpd smatrix_make_hpd

#elif defined PREC_d
  #define cublasXtrsm_batched cublasDtrsmBatched
  #define LAPACK_GEMM cblas_dgemm
  #define LAPACK_GEMM_BATCH dgemm_batch
  #ifdef USE_MKL
  #define LAPACK_GESVD dgesvd
  #define LAPACK_LACPY dlacpy
  #define LAPACK_OMATCPY mkl_domatcopy
  #else
    #define LAPACK_GESVD LAPACK_dgesvd
    #define LAPACK_LACPY LAPACK_dlacpy
    #define LAPACK_TRANS LAPACKE_dge_trans
  #endif
    #define LAPACK_SCAL cblas_dscal

  #define Xrand_matrix drand_matrix
  #define Xget_max_error_matrix dget_max_error_matrix
  #define Xget_max_error_matrix_uplo dget_max_error_matrix_uplo
  #define Xmatrix_make_hpd dmatrix_make_hpd

#elif defined PREC_c
  #define cublasXtrsm_batched cublasCtrsmBatched
  #define LAPACK_GEMM cblas_cgemm
  #define LAPACK_GEMM_BATCH cgemm_batch
  #ifdef USE_MKL
  #define LAPACK_GESVD cgesvd
  #define LAPACK_LACPY clacpy
  #define LAPACK_OMATCPY mkl_comatcopy
  #else
    #define LAPACK_GESVD LAPACK_cgesvd
    #define LAPACK_LACPY LAPACK_clacpy
  #endif
  #ifdef USE_MKL
  #define LAPACK_SCAL cscal
  #else
    #define LAPACK_SCAL cblas_cscal
  #endif

  #define Xrand_matrix crand_matrix
  #define Xget_max_error_matrix cget_max_error_matrix
  #define Xmatrix_make_hpd cmatrix_make_hpd

#elif defined PREC_z
  #define cublasXtrsm_batched cublasZtrsmBatched
  #define LAPACK_GEMM cblas_zgemm
  #define LAPACK_GEMM_BATCH zgemm_batch
  #ifdef USE_MKL
  #define LAPACK_GESVD zgesvd
  #define LAPACK_LACPY zlacpy
  #define LAPACK_OMATCPY mkl_zomatcopy
  #else
    #define LAPACK_GESVD LAPACK_zgesvd
    #define LAPACK_LACPY LAPACK_zlacpy
  #endif
  #ifdef USE_MKL
  #define LAPACK_SCAL zscal
  #else
    #define LAPACK_SCAL cblas_zscal
  #endif

  #define Xrand_matrix zrand_matrix
  #define Xget_max_error_matrix zget_max_error_matrix
  #define Xmatrix_make_hpd zmatrix_make_hpd

#endif

#endif //__KBLAS_PREC_DEF__
