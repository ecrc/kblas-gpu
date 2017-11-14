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
 * @version 2.0.0
 * @author Ali Charara
 * @date 2017-11-13
 **/

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"

#include "kblas.h"
#include "kblas_struct.h"
#include "operators.h"
#include "defs.h"
#include "kblas_common.h"
#include "batch_common.ch"

//==============================================================================================
#include "Xblas_core.ch"
#include "Xhelper_funcs.ch"
#include "Xsyrk_batch_drivers.cuh"

//==============================================================================================
//Non-Strided form

// workspace needed: device pointers
// A, B: host pointer to array of device pointers to device buffers
int Xsyrk_batch_offset( kblasHandle_t handle,
                        char uplo, char trans,
                        const int m, const int n,
                        const TYPE alpha, const TYPE** A, int A_row_off, int A_col_off, int lda,
                        const TYPE beta,        TYPE** B, int B_row_off, int B_col_off, int ldb,
                        int batchCount){
  return Xsyrk_batch_core(handle,
                          uplo, trans,
                          m, n,
                          alpha, A, A_row_off, A_col_off, lda,
                          beta,  B, B_row_off, B_col_off, ldb,
                          batchCount);
}

// workspace needed: device pointers
// A, B: host pointer to array of device pointers to device buffers
int kblas_syrk_batch(kblasHandle_t handle,
                    char uplo, char trans,
                    const int m, const int n,
                    const TYPE alpha, const TYPE** A, int lda,
                    const TYPE beta,        TYPE** B, int ldb,
                    int batchCount){
  return Xsyrk_batch_core(handle,
                          uplo, trans,
                          m, n,
                          alpha, A, 0, 0, lda,
                          beta,  B, 0, 0, ldb,
                          batchCount);
}

extern "C" {

// workspace needed: device pointers
// A, B: host pointer to array of device pointers to device buffers
int kblasXsyrk_batch(kblasHandle_t handle,
                    char uplo, char trans,
                    const int m, const int n,
                    const TYPE alpha, const TYPE** A, int lda,
                    const TYPE beta,        TYPE** B, int ldb,
                    int batchCount){
  return Xsyrk_batch_core(handle,
                          uplo, trans,
                          m, n,
                          alpha, A, 0, 0, lda,
                          beta,  B, 0, 0, ldb,
                          batchCount);
}
}//extern "C"

//==============================================================================================
//Strided form

// workspace needed: device pointers
// A, B: host pointer to device buffers
int kblas_syrk_batch( kblasHandle_t handle,
                      char uplo, char trans,
                      const int m, const int n,
                      const TYPE alpha, const TYPE* A, int lda, long strideA,
                      const TYPE beta,        TYPE* B, int ldb, long strideB,
                      int batchCount){
  return Xsyrk_batch_strided_core(handle,
                                  uplo, trans,
                                  m, n,
                                  alpha, A, lda, strideA,
                                  beta,  B, ldb, strideB,
                                  batchCount);
}

extern "C" {

// workspace needed: device pointers
// A, B: host pointer to device buffers
int kblasXsyrk_batch_strided(kblasHandle_t handle,
                            char uplo, char trans,
                            const int m, const int n,
                            const TYPE alpha, const TYPE* A, int lda, long strideA,
                            const TYPE beta,        TYPE* B, int ldb, long strideB,
                            int batchCount){
  return Xsyrk_batch_strided_core(handle,
                                  uplo, trans,
                                  m, n,
                                  alpha, A, lda, strideA,
                                  beta,  B, ldb, strideB,
                                  batchCount);
}

}//extern C
