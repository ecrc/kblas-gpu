/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xtrsm_batch.cu

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
#include <typeinfo>

#include "kblas.h"
#include "kblas_struct.h"
#include "kblas_operators.h"
#include "kblas_defs.h"
#include "kblas_common.h"
#include "workspace_queries.ch"

//==============================================================================================
#include "Xblas_core.ch"
#include "Xhelper_funcs.ch"
#include "Xtrsm_batch_drivers.cuh"

//==============================================================================================
//Non-Strided form

int Xtrsm_batch(kblasHandle_t handle,
                char side, char uplo, char trans, char diag,
                int m, int n,
                TYPE alpha,
                TYPE** A, int A_row_off, int A_col_off, int lda, long strideA,
                TYPE** B, int B_row_off, int B_col_off, int ldb, long strideB,
                int batchCount)
{
  (void)strideA;
  (void)strideB;

  KBlasWorkspaceState ws_needed;
  trsm_batch_wsquery_core<false>( batchCount,
                                  side, m, n,
                                  (kblasWorkspaceState_t)&ws_needed);

  bool suffWorkspace = (ws_needed.d_ptrs_bytes <= handle->work_space.allocated_ws_state.d_ptrs_bytes);

  if(!suffWorkspace){
    return KBLAS_InsufficientWorkspace;
  }

  return Xtrsm_batch_core<TYPE, TYPE**, false>(
                          handle,
                          side, uplo, trans, diag,
                          m, n,
                          alpha,
                          (TYPE**)A, A_row_off, A_col_off, lda, (long)0,
                          (TYPE**)B, B_row_off, B_col_off, ldb, (long)0,
                          batchCount);
}

// workspace needed: device pointers
// A, B: host pointer to array of device pointers to device buffers

int kblas_trsm_batch(kblasHandle_t handle,
                     char side, char uplo, char trans, char diag,
                     const int m, const int n,
                     const TYPE alpha,
                     const TYPE** A, int lda,
                           TYPE** B, int ldb,
                     int batchCount)
{
  return Xtrsm_batch( handle,
                      side, uplo, trans, diag,
                      m, n,
                      alpha,
                      (TYPE**)A, 0, 0, lda, 0,
                      (TYPE**)B, 0, 0, ldb, 0,
                      batchCount);
}

extern "C"
// workspace needed: device pointers
// A, B: host pointer to array of device pointers to device buffers
int kblasXtrsm_batch(kblasHandle_t handle,
                     char side, char uplo, char trans, char diag,
                     const int m, const int n,
                     const TYPE alpha,
                     const TYPE** A, int lda,
                           TYPE** B, int ldb,
                    int batchCount)
{
  return Xtrsm_batch( handle,
                      side, uplo, trans, diag,
                      m, n,
                      alpha,
                      (TYPE**)A, 0, 0, lda, 0,
                      (TYPE**)B, 0, 0, ldb, 0,
                      batchCount);
}

int Xtrsm_batch(kblasHandle_t handle,
                char side, char uplo, char trans, char diag,
                int *m, int *n,
                int max_m, int max_n,
                TYPE alpha,
                TYPE** A, int A_row_off, int A_col_off, int* lda, long strideA,
                TYPE** B, int B_row_off, int B_col_off, int* ldb, long strideB,
                int batchCount)
{
  (void)strideA;
  (void)strideB;
  (void)A_col_off;
  (void)A_row_off;
  (void)B_col_off;
  (void)B_row_off;

  return Xtrsm_batch_nonuniform_core<TYPE>(
                                    handle,
                                    side, uplo, trans, diag,
                                    m, n,
                                    alpha,
                                    A, lda,
                                    B, ldb,
                                    max_m, max_n,
                                    batchCount);
}

int Xtrsm_batch(kblasHandle_t handle,
                char side, char uplo, char trans, char diag,
                int *m, int *n,
                TYPE alpha,
                TYPE** A, int A_row_off, int A_col_off, int* lda, long strideA,
                TYPE** B, int B_row_off, int B_col_off, int* ldb, long strideB,
                int batchCount)
{
  (void)strideA;
  (void)strideB;
  (void)A_col_off;
  (void)A_row_off;
  (void)B_col_off;
  (void)B_row_off;

  return Xtrsm_batch_nonuniform_core<TYPE>(
                                    handle,
                                    side, uplo, trans, diag,
                                    m, n,
                                    alpha,
                                    A, lda,
                                    B, ldb,
                                    0, 0,
                                    batchCount);
}

int kblas_trsm_batch( kblasHandle_t handle,
                      char side, char uplo, char trans, char diag,
                      int *m, int *n,
                      int max_m, int max_n,
                      TYPE alpha,
                      TYPE** A, int* lda,
                      TYPE** B, int* ldb,
                      int batchCount)
{
  return Xtrsm_batch( handle,
                      side, uplo, trans, diag,
                      m, n,
                      max_m, max_n,
                      alpha,
                      A, 0, 0,lda, 0,
                      B, 0, 0, ldb, 0,
                      batchCount);
}
//==============================================================================================
//Strided form


int Xtrsm_batch(kblasHandle_t handle,
                char side, char uplo, char trans, char diag,
                int m, int n,
                TYPE alpha,
                TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                TYPE* B, int B_row_off, int B_col_off, int ldb, long strideB,
                int batchCount)
{
  KBlasWorkspaceState ws_needed;
  trsm_batch_wsquery_core<true>(batchCount,
                                side, m, n,
                                (kblasWorkspaceState_t)&ws_needed);

  bool suffWorkspace = (ws_needed.d_ptrs_bytes <= handle->work_space.allocated_ws_state.d_ptrs_bytes);

  if(!suffWorkspace){
    return KBLAS_InsufficientWorkspace;
  }

  return Xtrsm_batch_core<TYPE, TYPE*, true>(
                          handle,
                          side, uplo, trans, diag,
                          m, n,
                          alpha,
                          (TYPE*)A, A_row_off, A_col_off, lda, strideA,
                          (TYPE*)B, B_row_off, B_col_off, ldb, strideB,
                          batchCount);
}


// workspace needed: device pointers
// A, B: host pointer to array of device pointers to device buffers

int kblas_trsm_batch(kblasHandle_t handle,
                     char side, char uplo, char trans, char diag,
                     const int m, const int n,
                     const TYPE alpha,
                     const TYPE* A, int lda, long strideA,
                           TYPE* B, int ldb, long strideB,
                    int batchCount)
{
  return Xtrsm_batch( handle,
                      side, uplo, trans, diag,
                      m, n,
                      alpha,
                      (TYPE*)A, 0, 0, lda, strideA,
                      (TYPE*)B, 0, 0, ldb, strideB,
                      batchCount);
}

extern "C"
// workspace needed: device pointers
// A, B: host pointer to device buffers
int kblasXtrsm_batch_strided(kblasHandle_t handle,
                             char side, char uplo, char trans, char diag,
                             const int m, const int n,
                             const TYPE alpha,
                             const TYPE* A, int lda, long strideA,
                                   TYPE* B, int ldb, long strideB,
                             int batchCount)
{
  return Xtrsm_batch( handle,
                      side, uplo, trans, diag,
                      m, n,
                      alpha,
                      (TYPE*)A, 0, 0, lda, strideA,
                      (TYPE*)B, 0, 0, ldb, strideB,
                      batchCount);
}

