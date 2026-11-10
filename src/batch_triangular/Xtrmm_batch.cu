/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xtrmm_batch.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ali Charara
 * @date 2020-12-10
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
#include "kblas_gpu_util.ch"
#include "workspace_queries.ch"

//==============================================================================================
#include "Xblas_core.ch"
#include "Xhelper_funcs.ch"
#include "Xtrmm_batch_drivers.cuh"

//==============================================================================================
//Non-Strided form


int Xtrmm_batch(kblasHandle_t handle,
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
  trmm_batch_wsquery_core<false>( batchCount,
                                  side, m, n,
                                  (kblasWorkspaceState_t)&ws_needed);

  if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) )
    return KBLAS_InsufficientWorkspace;

  if( side == KBLAS_Right || uplo == KBLAS_Upper || diag == KBLAS_Unit ){
    #ifdef USE_MAGMA
    if(handle->use_magma){

      // TYPE **dA_array, **dB_array;
      // dA_array = (TYPE**)handle->work_space.d_ptrs;
      // dB_array = dA_array + batchCount;

      // Xset_pointer_2(dA_array, A, lda, strideA,
      //                dB_array, B, ldb, strideB,
      //                batchCount, handle->stream);

#if (MAGMA_VERSION_MAJOR > 2 || (MAGMA_VERSION_MAJOR == 2 && MAGMA_VERSION_MINOR > 3))
      magmablas_Xtrmm_batched_core(
            (magma_side_t)(side == KBLAS_Left ? MagmaLeft : MagmaRight),
            (magma_uplo_t)(uplo == KBLAS_Lower ? MagmaLower : MagmaUpper),
            (magma_trans_t)(MagmaNoTrans + (trans == KBLAS_Trans)),
            (magma_diag_t)(MagmaNonUnit + (diag == KBLAS_Unit)),
            m, n,
            alpha,
            (TYPE**)A, A_row_off, A_col_off, lda,
            (TYPE**)B, B_row_off, B_col_off, ldb,
            batchCount, handle->magma_queue );
#else
      magmablas_Xtrmm_batched_core(
            (magma_side_t)(side == KBLAS_Left ? MagmaLeft : MagmaRight),
            (magma_uplo_t)(uplo == KBLAS_Lower ? MagmaLower : MagmaUpper),
            (magma_trans_t)(MagmaNoTrans + (trans == KBLAS_Trans)),
            (magma_diag_t)(MagmaNonUnit + (diag == KBLAS_Unit)),
            m, n,
            alpha,
            (TYPE**)A, lda,
            (TYPE**)B, ldb,
            A_row_off, A_col_off, B_row_off, B_col_off,
            batchCount, handle->magma_queue );
#endif
      return KBLAS_Success;
    }else
    #endif
    {
      printf("(Right | Upper | Unit) TRMM_BATCH is not implemented yet, enable magma for alternative support. Aborting...\n");
      return KBLAS_NotImplemented;
    }
  }

  return Xtrmm_batch_core<TYPE, TYPE**>(
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
int kblas_trmm_batch(kblasHandle_t handle,
                     char side, char uplo, char trans, char diag,
                     const int m, const int n,
                     const TYPE alpha,
                     const TYPE** A, int lda,
                           TYPE** B, int ldb,
                    int batchCount)
{
  return Xtrmm_batch( handle,
                      side, uplo, trans, diag,
                      m, n,
                      alpha,
                      (TYPE**)A, 0, 0, lda, 0,
                      (TYPE**)B, 0, 0, ldb, 0,
                      batchCount);
}


// workspace needed: device pointers
// A, B: host pointer to array of device pointers to device buffers
extern "C"
int kblasXtrmm_batch(kblasHandle_t handle,
                     char side, char uplo, char trans, char diag,
                     const int m, const int n,
                     const TYPE alpha,
                     const TYPE** A, int lda,
                           TYPE** B, int ldb,
                    int batchCount)
{
  return Xtrmm_batch( handle,
                      side, uplo, trans, diag,
                      m, n,
                      alpha,
                      (TYPE**)A, 0, 0, lda, 0,
                      (TYPE**)B, 0, 0, ldb, 0,
                      batchCount);
}


//==============================================================================================
//Strided form

int Xtrmm_batch(kblasHandle_t handle,
                char side, char uplo, char trans, char diag,
                int m, int n,
                TYPE alpha,
                TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                TYPE* B, int B_row_off, int B_col_off, int ldb, long strideB,
                int batchCount)
{

  KBlasWorkspaceState ws_needed;
  trmm_batch_wsquery_core<true>(batchCount,
                                side, m, n,
                                (kblasWorkspaceState_t)&ws_needed);

  if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) )
    return KBLAS_InsufficientWorkspace;

  if( side == KBLAS_Right || uplo == KBLAS_Upper || diag == KBLAS_Unit ){
    #ifdef USE_MAGMA
    if(handle->use_magma){

      TYPE **dA_array, **dB_array;
      dA_array = (TYPE**)handle->work_space.d_ptrs;
      dB_array = dA_array + batchCount;

      Xset_pointer_2(dA_array, A, lda, strideA,
                     dB_array, B, ldb, strideB,
                     batchCount, handle->stream);

#if (MAGMA_VERSION_MAJOR > 2 || (MAGMA_VERSION_MAJOR == 2 && MAGMA_VERSION_MINOR > 3))
      magmablas_Xtrmm_batched_core(
            (magma_side_t)(side == KBLAS_Left ? MagmaLeft : MagmaRight),
            (magma_uplo_t)(uplo == KBLAS_Lower ? MagmaLower : MagmaUpper),
            (magma_trans_t)(MagmaNoTrans + (trans == KBLAS_Trans)),
            (magma_diag_t)(MagmaNonUnit + (diag == KBLAS_Unit)),
            m, n,
            alpha,
            dA_array, A_row_off, A_col_off, lda,
            dB_array, B_row_off, B_col_off, ldb,
            batchCount, handle->magma_queue );
#else
      magmablas_Xtrmm_batched_core(
            (magma_side_t)(side == KBLAS_Left ? MagmaLeft : MagmaRight),
            (magma_uplo_t)(uplo == KBLAS_Lower ? MagmaLower : MagmaUpper),
            (magma_trans_t)(MagmaNoTrans + (trans == KBLAS_Trans)),
            (magma_diag_t)(MagmaNonUnit + (diag == KBLAS_Unit)),
            m, n,
            alpha,
            dA_array, lda,
            dB_array, ldb,
            A_row_off, A_col_off, B_row_off, B_col_off,
            batchCount, handle->magma_queue );
#endif
      return KBLAS_Success;
    }else
    #endif
    {
      printf("(Right | Upper | Unit) TRMM_BATCH is not implemented yet, enable magma for alternative support. Aborting...\n");
      return KBLAS_NotImplemented;
    }
  }

  return Xtrmm_batch_core<TYPE, TYPE*>(
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
int kblas_trmm_batch(kblasHandle_t handle,
                     char side, char uplo, char trans, char diag,
                     const int m, const int n,
                     const TYPE alpha,
                     const TYPE* A, int lda, long strideA,
                           TYPE* B, int ldb, long strideB,
                    int batchCount)
{
  return Xtrmm_batch( handle,
                      side, uplo, trans, diag,
                      m, n,
                      alpha,
                      (TYPE*)A, 0, 0, lda, strideA,
                      (TYPE*)B, 0, 0, ldb, strideB,
                      batchCount);
}


// workspace needed: device pointers
// A, B: host pointer to device buffers
extern "C"
int kblasXtrmm_batch_strided(kblasHandle_t handle,
                             char side, char uplo, char trans, char diag,
                             const int m, const int n,
                             const TYPE alpha,
                             const TYPE* A, int lda, long strideA,
                                   TYPE* B, int ldb, long strideB,
                             int batchCount)
{
  return Xtrmm_batch( handle,
                      side, uplo, trans, diag,
                      m, n,
                      alpha,
                      (TYPE*)A, 0, 0, lda, strideA,
                      (TYPE*)B, 0, 0, ldb, strideB,
                      batchCount);
}

