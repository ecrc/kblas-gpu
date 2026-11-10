/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xpoti_batch.cu

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
#include "workspace_queries.ch"

//==============================================================================================
#include "Xblas_core.ch"
#include "Xhelper_funcs.ch"
#include "Xpoti_batch_drivers.cuh"

//==============================================================================================
//Non-Strided form

// workspace needed: device pointers
// A: host pointer to device buffer
int Xpoti_batch_offset(kblasHandle_t handle,
                        char uplo,
                        const int n,
                        TYPE** A, int A_row_off, int A_col_off, int lda,
                        int batchCount,
                        int *info_array)
{
  KBlasWorkspaceState ws_needed;
  poti_batch_wsquery_core<false>( n, batchCount, (kblasWorkspaceState_t)&ws_needed);

  if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) ){
    return KBLAS_InsufficientWorkspace;
  }

  return Xpoti_batch_core<TYPE, TYPE**, false>(
                          handle,
                          uplo, n,
                          (TYPE**)A, A_row_off, A_col_off, lda, (long)0,
                          batchCount,
                          info_array);
}

// workspace needed: device pointers
// A: host pointer to device buffer
int kblas_poti_batch(kblasHandle_t handle,
                      char uplo,
                      const int n,
                      TYPE** A, int lda,
                      int batchCount,
                      int *info_array)
{
  return Xpoti_batch_offset( handle,
                              uplo, n,
                              A, 0, 0, lda,
                              batchCount,
                              info_array);
}


// workspace needed: device pointers
// A: host pointer to device buffer
extern "C"
int kblasXpoti_batch(kblasHandle_t handle,
                      char uplo,
                      const int n,
                      TYPE** A, int lda,
                      int batchCount,
                      int *info_array)
{
  return Xpoti_batch_offset( handle,
                              uplo, n,
                              A, 0, 0, lda,
                              batchCount,
                              info_array);
}


//==============================================================================================
//Strided form
// template<>

// workspace needed: device pointers
// A: host pointer to device buffer
int Xpoti_batch_offset(kblasHandle_t handle,
                        char uplo,
                        const int n,
                        TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                        int batchCount,
                        int *info_array)
{
  KBlasWorkspaceState ws_needed;
  poti_batch_wsquery_core<true>( n, batchCount, (kblasWorkspaceState_t)&ws_needed);

  if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) ){
    return KBLAS_InsufficientWorkspace;
  }

  return Xpoti_batch_core<TYPE, TYPE*, true>(
                          handle,
                          uplo, n,
                          (TYPE*)A, A_row_off, A_col_off, lda, strideA,
                          batchCount,
                          info_array);
}

// workspace needed: device pointers
// A: host pointer to device buffer
int kblas_poti_batch(kblasHandle_t handle,
                      char uplo,
                      const int n,
                      TYPE* A, int lda, long strideA,
                      int batchCount,
                      int *info_array)
{
  return Xpoti_batch_offset( handle,
                              uplo, n,
                              A, 0, 0, lda, strideA,
                              batchCount,
                              info_array);
}

// workspace needed: device pointers
// A: host pointer to device buffer
extern "C"
int kblasXpoti_batch_strided(kblasHandle_t handle,
                              char uplo,
                              const int n,
                              TYPE* A, int lda, long strideA,
                              int batchCount,
                              int *info_array)
{
  return Xpoti_batch_offset( handle,
                              uplo, n,
                              A, 0, 0, lda, strideA,
                              batchCount,
                              info_array);
}
