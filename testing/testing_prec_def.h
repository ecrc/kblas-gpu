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
 * @version 2.0.0
 * @author Ali Charara
 * @date 2017-11-13
 **/

#ifndef __TESTING_PREC_DEF__
#define __TESTING_PREC_DEF__

#include "kblas_prec_def.h"

#if defined   PREC_s
  #define cublasXtrsm_batched cublasStrsmBatched
  #define LAPACK_GEMM sgemm
  #define LAPACK_GEMM_BATCH sgemm_batch
  #define LAPACK_SYRK ssyrk
  #define LAPACK_AXPY saxpy
  #define LAPACK_LANSY slansy
  #define LAPACK_LANGE slange
  #define LAPACK_TRSM strsm
  #define LAPACK_TRMM strmm
  #define LAPACK_POTRF spotrf
  #define LAPACK_LAUUM slauum
  #define LAPACK_TRTRI strtri
  #define LAPACK_POTRI spotri

  #define Xrand_matrix srand_matrix
  #define Xget_max_error_matrix sget_max_error_matrix
  #define Xmatrix_make_hpd smatrix_make_hpd

#elif defined PREC_d
  #define cublasXtrsm_batched cublasDtrsmBatched
  #define LAPACK_GEMM dgemm
  #define LAPACK_GEMM_BATCH dgemm_batch
  #define LAPACK_SYRK dsyrk
  #define LAPACK_AXPY daxpy
  #define LAPACK_LANSY dlansy
  #define LAPACK_LANGE dlange
  #define LAPACK_TRSM dtrsm
  #define LAPACK_TRMM dtrmm
  #define LAPACK_POTRF dpotrf
  #define LAPACK_LAUUM dlauum
  #define LAPACK_TRTRI dtrtri
  #define LAPACK_POTRI dpotri

  #define Xrand_matrix drand_matrix
  #define Xget_max_error_matrix dget_max_error_matrix
  #define Xmatrix_make_hpd dmatrix_make_hpd

#elif defined PREC_c
  #define cublasXtrsm_batched cublasCtrsmBatched
  #define LAPACK_GEMM cgemm
  #define LAPACK_GEMM_BATCH cgemm_batch
  #define LAPACK_SYRK csyrk
  #define LAPACK_AXPY caxpy
  #define LAPACK_LANSY clansy
  #define LAPACK_LANGE clange
  #define LAPACK_TRSM ctrsm
  #define LAPACK_TRMM ctrmm
  #define LAPACK_POTRF cpotrf
  #define LAPACK_LAUUM clauum
  #define LAPACK_TRTRI ctrtri
  #define LAPACK_POTRI cpotri

  #define Xrand_matrix crand_matrix
  #define Xget_max_error_matrix cget_max_error_matrix
  #define Xmatrix_make_hpd cmatrix_make_hpd

#elif defined PREC_z
  #define cublasXtrsm_batched cublasZtrsmBatched
  #define LAPACK_GEMM zgemm
  #define LAPACK_GEMM_BATCH zgemm_batch
  #define LAPACK_SYRK zsyrk
  #define LAPACK_AXPY zaxpy
  #define LAPACK_LANSY zlansy
  #define LAPACK_LANGE zlange
  #define LAPACK_TRSM ztrsm
  #define LAPACK_TRMM ztrmm
  #define LAPACK_POTRF zpotrf
  #define LAPACK_LAUUM zlauum
  #define LAPACK_TRTRI ztrtri
  #define LAPACK_POTRI zpotri

  #define Xrand_matrix zrand_matrix
  #define Xget_max_error_matrix zget_max_error_matrix
  #define Xmatrix_make_hpd zmatrix_make_hpd

#endif

#endif //__KBLAS_PREC_DEF__
