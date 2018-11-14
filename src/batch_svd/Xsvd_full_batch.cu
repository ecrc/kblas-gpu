/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_svd/Xsvd_full_batch.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include "kblas.h"
#include "operators.h"

// #define DBG_MSG

#include "kblas_struct.h"
#include "kblas_prec_def.h"
#include "kblas_gpu_util.ch"

#include "workspace_queries.ch"

#include "Xblas_core.ch"
#include "Xhelper_funcs.ch"
#include "batch_block_copy.h"
#include "Xsvd_full_batch_core.cuh"

//==============================================================================================
// workspace needed: size of A * batchCount + ??
// on input:  A: host pointer to device buffer, contains the input matrix
//            S,U,V: host pointers to device buffers, preallocated
//            rank: if > 0, the rank required
// on output: S: contains singular values (up to rank if rank > 0)
//            U: contains right singular vectors
//            V: contains left singular vectors scaled by S
//            A: not modified
int kblas_svd_full_batch( kblasHandle_t handle,
                          int m, int n, int &rank,
                          TYPE* A, int lda, int stride_a,
                          TYPE* S, int stride_s,
                          TYPE* U, int ldu, int stride_u,
                          TYPE* V, int ldv, int stride_v,
                          SVD_method variant,
                          kblasRandState_t rand_state,
                          int batchCount)
{
  // not all precision are supported yet
  #if (defined PREC_c) || (defined PREC_z)
    return KBLAS_NotSupported;
  #else
    // return Xsvd_full_batch_core<TYPE>(
    return Xsvd_full_batch_core<TYPE, TYPE*, true>(
                                handle,
                                m, n, rank,
                                A, lda, stride_a,
                                S, stride_s,
                                U, ldu, stride_u,
                                V, ldv, stride_v,
                                variant,
                                rand_state,
                                batchCount);
  #endif
}

extern "C"
int kblasXsvd_full_batch_strided( kblasHandle_t handle,
                                  int m, int n, int rank,
                                  TYPE* A, int lda, int stride_a,
                                  TYPE* S, int stride_s,
                                  TYPE* U, int ldu, int stride_u,
                                  TYPE* V, int ldv, int stride_v,
                                  SVD_method variant,
                                  kblasRandState_t rand_state,
                                  int batchCount)
{
  // not all precision are supported yet
  #if (defined PREC_c) || (defined PREC_z)
    return KBLAS_NotSupported;
  #else
    return Xsvd_full_batch_core<TYPE, TYPE*, true>(
                                handle,
                                m, n, rank,
                                A, lda, stride_a,
                                S, stride_s,
                                U, ldu, stride_u,
                                V, ldv, stride_v,
                                variant,
                                rand_state,
                                batchCount);
  #endif
}

//==============================================================================================
int kblas_svd_full_batch( kblasHandle_t handle,
                          int m, int n, int &rank,
                          TYPE** A, int lda,
                          TYPE** S,
                          TYPE** U, int ldu,
                          TYPE** V, int ldv,
                          SVD_method variant,
                          kblasRandState_t rand_state,
                          int batchCount)
{
  // not all precision are supported yet
  #if (defined PREC_c) || (defined PREC_z)
    return KBLAS_NotSupported;
  #else
    return Xsvd_full_batch_core<TYPE, TYPE**, false>(
                                handle,
                                m, n, rank,
                                A, lda, 0,
                                S, 0,
                                U, ldu, 0,
                                V, ldv, 0,
                                variant,
                                rand_state,
                                batchCount);
  #endif
}

//==============================================================================================
int kblas_svd_full_batch( kblasHandle_t handle,
                          int m, int n,
                          int* m_array, int* n_array,       //m_array, n_array: device buffers
                          TYPE** A, int lda, int* lda_array,//A, lda_array: device buffers
                          TYPE** S,                         //S: device buffer
                          TYPE** U, int ldu, int* ldu_array,//U, ldu_array: device buffers
                          TYPE** V, int ldv, int* ldv_array,//V, ldv_array: device buffers
                          SVD_method variant,
                          kblasRandState_t rand_state,
                          int batchCount,
                          int max_rank, double tolerance,
                          int* ranks_array)                 //ranks_array: device buffer
{
  // not all precision are supported yet
  #if (defined PREC_c) || (defined PREC_z)
    return KBLAS_NotSupported;
  #else
    return Xsvd_full_batch_nonuniform_core<TYPE>(
                                          handle,
                                          m, n,
                                          m_array, n_array,
                                          A, lda, lda_array,
                                          S,
                                          U, ldu, ldu_array,
                                          V, ldv, ldv_array,
                                          variant,
                                          rand_state,
                                          batchCount,
                                          max_rank, tolerance,
                                          ranks_array);
  #endif
}

//----------------------------------------------------------------------------------------------
extern "C"
void kblasXsvd_full_batch_wsquery(kblasHandle_t handle,
                                  int m, int n, int rank,
                                  int batchCount, SVD_method variant)
{
  // not all precision are supported yet
  #if (defined PREC_s) || (defined PREC_d)
    svd_full_batch_wsquery_core<TYPE, false, true>( m, n, rank, batchCount, variant, &(handle->work_space.requested_ws_state) );
  #endif
}
//----------------------------------------------------------------------------------------------
extern "C"
void kblasXsvd_full_batch_nonUniform_wsquery( kblasHandle_t handle,
                                              int m, int n, int rank,
                                              int batchCount, SVD_method variant)
{
  // not all precisions are supported yet
  #if (defined PREC_s) || (defined PREC_d)
    svd_full_batch_wsquery_core<TYPE, false, false>( m, n, rank, batchCount, variant, &(handle->work_space.requested_ws_state) );
  #endif
}
