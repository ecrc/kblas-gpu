/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/Xblas_core.cu

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
#include <set>
#include <cublas_v2.h>
#include "kblas.h"
#include "kblas_operators.h"

#include "Xblas_core.ch"
#include "Xhelper_funcs.ch"
#include "workspace_queries.ch"

//####################################################################################################
// wrapper functions around cuBLAS
//####################################################################################################

//==============================================================================================
// A, B, C: host pointers to device buffers
int kblasXgemm( kblasHandle_t handle,
                char transa, char transb,
                int m, int n, int k,
                const TYPE alpha, const TYPE *A, int lda,
                                  const TYPE *B, int ldb,
                const TYPE beta,        TYPE *C, int ldc)
{
  cublasStatus_t status = cublasXgemm( handle->cublas_handle,
                                       transa == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                                       transb == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                                       m, n, k,
                                       &alpha, A, lda,
                                               B, ldb,
                                       &beta,  C, ldc);
  check_error_ret(status, KBLAS_cuBLAS_Error);
  return KBLAS_Success;
}

//==============================================================================================
// A, B: host pointers to device buffers
int kblasXsyrk( kblasHandle_t handle,
                char uplo, char trans,
                int m, int n,
                const TYPE alpha, const TYPE* A, int lda,
                const TYPE beta,        TYPE* B, int ldb)
{
  cublasStatus_t status = cublasXsyrk( handle->cublas_handle,
                                       uplo == KBLAS_Lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
                                       trans == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                                       m, n,
                                       &alpha, A, lda,
                                       &beta,  B, ldb);
  check_error_ret(status, KBLAS_cuBLAS_Error);
  return KBLAS_Success;
}

//=========================================================================
// A, B, C: host pointers to device buffers
int kblasXsymm( kblasHandle_t handle,
                char side, char uplo,
                int m, int n,
                const TYPE alpha, const TYPE *A, int lda,
                                  const TYPE *B, int ldb,
                const TYPE beta,        TYPE *C, int ldc)
{
  cublasStatus_t status = cublasXsymm( handle->cublas_handle,
                                      side == KBLAS_Left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
                                      uplo == KBLAS_Lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
                                      m, n,
                                      &alpha, (const TYPE*) A, lda,
                                              (const TYPE*) B, ldb,
                                      &beta,             C, ldc);
  check_error_ret(status, KBLAS_cuBLAS_Error);
  return KBLAS_Success;
}

//=========================================================================
int kblasXtrsm( kblasHandle_t handle,
                char side, char uplo, char trans, char diag,
                int m, int n,
                const TYPE alpha,
                const TYPE* A, int lda,
                      TYPE* B, int ldb)
{
  //TODO if cuda version >= 8, call cublas instead
  //TODO verify this is better than cublas if cuda >= 8
  cublasStatus_t status = kblasXtrsm( handle->cublas_handle,
                                      side == KBLAS_Left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
                                      uplo == KBLAS_Lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
                                      trans == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                                      diag == KBLAS_NonUnit ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT,
                                      m, n,
                                      &alpha, A, lda,
                                              B, ldb);
  check_error_ret(status, KBLAS_cuBLAS_Error);
  return KBLAS_Success;
}

//=========================================================================
int kblasXtrmm( kblasHandle_t handle,
                char side, char uplo, char trans, char diag,
                int m, int n,
                const TYPE alpha,
                const TYPE* A, int lda,
                      TYPE* B, int ldb)
{
  //TODO if cuda version >= 8, call cublas instead
  //TODO verify this is better than cublas if cuda >= 8
  cublasStatus_t status = kblasXtrmm( handle->cublas_handle,
                                      side == KBLAS_Left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
                                      uplo == KBLAS_Lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
                                      trans == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                                      diag == KBLAS_NonUnit ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT,
                                      m, n,
                                      &alpha, A, lda,
                                              B, ldb);
  check_error_ret(status, KBLAS_cuBLAS_Error);
  return KBLAS_Success;
}

//=========================================================================
int kblasXscal( kblasHandle_t handle,
                int n,
                const TYPE alpha,
                TYPE *x, int incx)
{
  check_error_ret( cublasXscal( handle->cublas_handle,
                                n,
                                &alpha,
                                x,
                                incx), KBLAS_cuBLAS_Error);
  return KBLAS_Success;
}
//=========================================================================
int kblasXgeam( kblasHandle_t handle,
                char transa, char transb,
                int m, int n,
                const TYPE alpha, const TYPE *A, int lda,
                const TYPE beta,  const TYPE *B, int ldb,
                                        TYPE *C, int ldc)
{
  check_error_ret( cublasXgeam( handle->cublas_handle,
                                transa == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                                transb == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                                m, n,
                                &alpha, A, lda,
                                &beta,  B, ldb,
                                        C, ldc), KBLAS_cuBLAS_Error);
  return KBLAS_Success;
}
//=========================================================================

#ifdef USE_MAGMA

//workspace needed: none
// A_array, B_array, C_array: host pointer to array of device pointers to device buffers
int Xsymm_batch(kblasHandle_t handle,
                char side, char uplo,
                int m, int n,
                TYPE alpha, TYPE **dA_array, int ldda, long strideA,
                            TYPE **dB_array, int lddb, long strideB,
                TYPE beta,  TYPE **dC_array, int lddc, long strideC,
                int batchCount)
{
  (void)strideA;
  (void)strideB;
  (void)strideC;
  #if (defined PREC_c) || (defined PREC_z)
    return KBLAS_NotSupported;
  #else
  magma_Xsymm_batched((magma_side_t)(side == KBLAS_Left ? MagmaLeft : MagmaRight),
                      (magma_uplo_t)(uplo == KBLAS_Lower ? MagmaLower : MagmaUpper),
                      m, n,
                      alpha, (TYPE**)dA_array, ldda,
                             (TYPE**)dB_array, lddb,
                      beta,          dC_array, lddc,
                      batchCount, handle->magma_queue );
  return KBLAS_Success;
  #endif
}

//-------------------------------------------------------------------------
// workspace needed: device pointers
// d_A, d_B, d_C: host pointers to device buffers
// TODO better implementation is needed
int Xsymm_batch(kblasHandle_t handle,
                char side, char uplo,
                int m, int n,
                TYPE alpha, TYPE *d_A, int ldda, long strideA,
                            TYPE *d_B, int lddb, long strideB,
                TYPE beta,  TYPE *d_C, int lddc, long strideC,
                int batchCount)
{
  #if (defined PREC_c) || (defined PREC_z)
    return KBLAS_NotSupported;
  #else

  KBlasWorkspaceState ws_needed;
  symm_batch_wsquery_core<true>( batchCount, (kblasWorkspaceState_t)&ws_needed);

  if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) )
    return KBLAS_InsufficientWorkspace;

  TYPE **dA_array, **dB_array, **dC_array;
  dA_array = (TYPE**)handle->work_space.d_ptrs;
  dB_array = dA_array + batchCount;
  dC_array = dB_array + batchCount;

  Xset_pointer_3(dA_array, d_A, ldda, strideA,
                 dB_array, d_B, lddb, strideB,
                 dC_array, d_C, lddc, strideC,
                 batchCount, handle->stream);

  magma_Xsymm_batched((magma_side_t)(side == KBLAS_Left ? MagmaLeft : MagmaRight),
                      (magma_uplo_t)(uplo == KBLAS_Lower ? MagmaLower : MagmaUpper),
                      m, n,
                      alpha, dA_array, ldda,
                             dB_array, lddb,
                      beta,  dC_array, lddc,
                      batchCount, handle->magma_queue );
  return KBLAS_Success;
  #endif
}

int Xsymm_batch(kblasHandle_t handle,
                char side, char uplo,
                int* m, int* n,
                int max_m, int max_n,
                TYPE alpha, TYPE **dA_array, int* ldda, long strideA,
                            TYPE **dB_array, int* lddb, long strideB,
                TYPE beta,  TYPE **dC_array, int* lddc, long strideC,
                int batchCount)
{
  (void)strideA;
  (void)strideB;
  (void)strideC;
  #if (defined PREC_c) || (defined PREC_z)
    return KBLAS_NotSupported;
  #else
  if(handle->use_magma){
    KBlasWorkspaceState ws_needed;
    symm_batch_nonuniform_wsquery_core((kblasWorkspaceState_t)&ws_needed);

    if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) ){
      return KBLAS_InsufficientWorkspace;
    }

    int h_max_mn[2];
    kblasWorkspace_t ws_current = &(handle->work_space);
    int* d_max_mn = (int*)(ws_current->d_data);

    //take care of batch size limitation with magma
    int batch_increment = 65535;
    int batch_start = 0;
    if(max_m > 0 || max_n > 0){
      h_max_mn[0] = max_m;
      h_max_mn[1] = max_n;
    }

    while(batch_start != batchCount)
    {
      int batch_size = kmin(batch_increment, batchCount - batch_start);

      if((batchCount > batch_increment) || (max_m <= 0 && max_n <= 0)){
        // compute the max. dimensions
        kblas_imax_size_2(handle, m, n, *d_max_mn, *(d_max_mn+1), batch_size);
        check_error_ret( cublasGetVectorAsync( 2, sizeof(int), d_max_mn, 1, h_max_mn, 1, handle->stream ), KBLAS_cuBLAS_Error);
        check_error_ret( cudaStreamSynchronize(handle->stream), KBLAS_CUDA_Error );
      }
      magmablas_Xsymm_vbatched_max_nocheck(
                      (magma_side_t)(side == KBLAS_Left ? MagmaLeft : MagmaRight),
                      (magma_uplo_t)(uplo == KBLAS_Lower ? MagmaLower : MagmaUpper),
                      m, n,
                      alpha, dA_array, ldda,
                             dB_array, lddb,
                      beta,  dC_array, lddc,
                      batch_size,
                      h_max_mn[0], h_max_mn[1], handle->magma_queue);

      dA_array += batch_size;
      dB_array += batch_size;
      dC_array += batch_size;
      m += batch_size;
      n += batch_size;
      ldda += batch_size;
      lddb += batch_size;
      lddc += batch_size;

      batch_start += batch_size;
      check_error_ret( cudaGetLastError(), KBLAS_MAGMA_Error);
    }
  return KBLAS_Success;
  }else{
    printf("Configuration error at %s in file %s at line %d, MAGMA required but not enabled!\n", __func__, __FILE__, __LINE__ );
    return KBLAS_WrongConfig;
  }
  #endif
}

//=========================================================================
#else//USE_MAGMA

int Xsymm_batch(kblasHandle_t handle,
                char side, char uplo,
                int m, int n,
                TYPE alpha, TYPE **dA_array, int ldda, long strideA,
                            TYPE **dB_array, int lddb, long strideB,
                TYPE beta,  TYPE **dC_array, int lddc, long strideC,
                int batchCount)
{
  //TODO need to provide this
  return KBLAS_NotSupported;
}

//-------------------------------------------------------------------------
int Xsymm_batch(kblasHandle_t handle,
                char side, char uplo,
                int m, int n,
                TYPE alpha, TYPE *d_A, int ldda, long strideA,
                            TYPE *d_B, int lddb, long strideB,
                TYPE beta,  TYPE *d_C, int lddc, long strideC,
                int batchCount)
{
  //TODO need to provide this
  return KBLAS_NotSupported;
}

#endif//USE_MAGMA
