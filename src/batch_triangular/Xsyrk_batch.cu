/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xsyrk_batch.cu

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

#include "kblas.h"
#include "kblas_struct.h"
#include "kblas_operators.h"
#include "kblas_defs.h"
#include "kblas_common.h"
#include "workspace_queries.ch"

//==============================================================================================
#include "Xblas_core.ch"
#include "Xhelper_funcs.ch"
#include "Xsyrk_batch_drivers.cuh"

//==============================================================================================
//Non-Strided form

// workspace needed: device pointers
// A, B: host pointer to array of device pointers to device buffers
int Xsyrk_batch(kblasHandle_t handle,
                char uplo, char trans,
                int m, int n,
                TYPE alpha, TYPE** A, int A_row_off, int A_col_off, int lda, long strideA,
                TYPE beta,  TYPE** B, int B_row_off, int B_col_off, int ldb, long strideB,
                int batchCount)
{
  (void)strideA;
  (void)strideB;

  return Xsyrk_batch_core<TYPE, TYPE**>(
                          handle,
                          uplo, trans,
                          m, n,
                          alpha, (TYPE**)A, A_row_off, A_col_off, lda,
                          beta,  (TYPE**)B, B_row_off, B_col_off, ldb,
                          batchCount);
}

// workspace needed: device pointers
// A, B: host pointer to array of device pointers to device buffers
int kblas_syrk_batch(kblasHandle_t handle,
                    char uplo, char trans,
                    const int m, const int n,
                    const TYPE alpha, const TYPE** A, int lda,
                    const TYPE beta,        TYPE** B, int ldb,
                    int batchCount)
{
  return Xsyrk_batch_core<TYPE, TYPE**>(
                          handle,
                          uplo, trans,
                          m, n,
                          alpha, (TYPE**)A, 0, 0, lda,
                          beta,  (TYPE**)B, 0, 0, ldb,
                          batchCount);
}

// workspace needed: device pointers
// A, B: host pointer to array of device pointers to device buffers
extern "C"
int kblasXsyrk_batch(kblasHandle_t handle,
                    char uplo, char trans,
                    const int m, const int n,
                    const TYPE alpha, const TYPE** A, int lda,
                    const TYPE beta,        TYPE** B, int ldb,
                    int batchCount)
{
  return Xsyrk_batch_core<TYPE, TYPE**>(
                          handle,
                          uplo, trans,
                          m, n,
                          alpha, (TYPE**)A, 0, 0, lda,
                          beta,  (TYPE**)B, 0, 0, ldb,
                          batchCount);
}

int Xsyrk_batch(kblasHandle_t handle,
                char uplo, char trans,
                int* m, int* n,
                int max_m, int max_n,
                TYPE alpha, TYPE** A, int* lda,
                TYPE beta,  TYPE** B, int* ldb,
                int batchCount)
{
  return Xsyrk_batch_nonuniform_core<TYPE>(
                                    handle,
                                    uplo, trans,
                                    m, n,
                                    alpha, A, lda,
                                    beta,  B, ldb,
                                    max_m, max_n,
                                    batchCount);
}

int kblas_syrk_batch( kblasHandle_t handle,
                      char uplo, char trans,
                      int* m, int* n,
                      int max_m, int max_n,
                      TYPE alpha, TYPE** A, int* lda,
                      TYPE beta,  TYPE** B, int* ldb,
                      int batchCount)
{
  return Xsyrk_batch( handle,
                      uplo, trans,
                      m, n,
                      max_m, max_n,
                      alpha, A, lda,
                      beta,  B, ldb,
                      batchCount);
}

//==============================================================================================
//Strided form

int Xsyrk_batch(kblasHandle_t handle,
                char uplo, char trans,
                int m, int n,
                TYPE alpha, TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                TYPE beta,  TYPE* B, int B_row_off, int B_col_off, int ldb, long strideB,
                int batchCount)
{
  return Xsyrk_batch_strided_core<TYPE, TYPE*>(
                                  handle,
                                  uplo, trans,
                                  m, n,
                                  alpha, (TYPE*)(A) + A_row_off + A_col_off * lda, lda, strideA,
                                  beta,  (TYPE*)(B) + B_row_off + B_col_off * ldb, ldb, strideB,
                                  batchCount);
}

// workspace needed: device pointers
// A, B: host pointer to device buffers
int kblas_syrk_batch( kblasHandle_t handle,
                      char uplo, char trans,
                      const int m, const int n,
                      const TYPE alpha, const TYPE* A, int lda, long strideA,
                      const TYPE beta,        TYPE* B, int ldb, long strideB,
                      int batchCount)
{
  return Xsyrk_batch_strided_core<TYPE, TYPE*>(
                                  handle,
                                  uplo, trans,
                                  m, n,
                                  alpha, (TYPE*)A, lda, strideA,
                                  beta,  (TYPE*)B, ldb, strideB,
                                  batchCount);
}


// workspace needed: device pointers
// A, B: host pointer to device buffers
extern "C"
int kblasXsyrk_batch_strided(kblasHandle_t handle,
                            char uplo, char trans,
                            const int m, const int n,
                            const TYPE alpha, const TYPE* A, int lda, long strideA,
                            const TYPE beta,        TYPE* B, int ldb, long strideB,
                            int batchCount)
{
  return Xsyrk_batch_strided_core<TYPE, TYPE*>(
                                  handle,
                                  uplo, trans,
                                  m, n,
                                  alpha, (TYPE*)A, lda, strideA,
                                  beta,  (TYPE*)B, ldb, strideB,
                                  batchCount);
}

