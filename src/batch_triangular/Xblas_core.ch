/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xblas_core.ch

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 2.0.0
 * @author Ali Charara
 * @date 2017-11-13
 **/

#ifndef __XBLAS_CORE__
#define __XBLAS_CORE__


#include "kblas_struct.h"
#include "kblas_prec_def.h"

//==============================================================================================
/// Query workspace needed for GEMM with offset
void kblas_gemm_batch_wsquery(kblasHandle_t handle,
                              int batchCount,
                              int A_row_off, int A_col_off,
                              int B_row_off, int B_col_off,
                              int C_row_off, int C_col_off);

/**
 * @brief Uniform-size batch non-strided GEMM with offset wrapper routine
 */
int kblas_gemm_batch( kblasHandle_t handle,
                      char transA, char transB,
                      const int m, const int n, const int k,
                      const TYPE alpha,
                      const TYPE** A, int A_row_off, int A_col_off, int lda,
                      const TYPE** B, int B_row_off, int B_col_off, int ldb,
                      const TYPE beta,
                            TYPE** C, int C_row_off, int C_col_off, int ldc,
                      int batchCount);

//==============================================================================================
int Xsyrk_batch_offset( kblasHandle_t handle,
                        char uplo, char trans,
                        const int m, const int n,
                        const TYPE alpha, const TYPE** A, int A_row_off, int A_col_off, int lda,
                        const TYPE beta,        TYPE** B, int B_row_off, int B_col_off, int ldb,
                        int batchCount);

//==============================================================================================
int Xtrsm_batch_offset( kblasHandle_t handle,
                        char side, char uplo, char trans, char diag,
                        const int m, const int n,
                        const TYPE alpha,
                        const TYPE** A, int A_row_off, int A_col_off, int lda,
                              TYPE** B, int B_row_off, int B_col_off, int ldb,
                        int batchCount);

int Xtrsm_batch_offset( kblasHandle_t handle,
                        char side, char uplo, char trans, char diag,
                        const int m, const int n,
                        const TYPE alpha,
                        const TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                              TYPE* B, int B_row_off, int B_col_off, int ldb, long strideB,
                        int batchCount);

//==============================================================================================
int Xtrmm_batch_offset( kblasHandle_t handle,
                        char side, char uplo, char trans, char diag,
                        const int m, const int n,
                        const TYPE alpha,
                        const TYPE** A, int A_row_off, int A_col_off, int lda,
                              TYPE** B, int B_row_off, int B_col_off, int ldb,
                        int batchCount);

int Xtrmm_batch_offset( kblasHandle_t handle,
                        char side, char uplo, char trans, char diag,
                        const int m, const int n,
                        const TYPE alpha,
                        const TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                              TYPE* B, int B_row_off, int B_col_off, int ldb, long strideB,
                        int batchCount);

//==============================================================================================
int Xpotrf_batch_offset(kblasHandle_t handle,
                        char uplo,
                        const int n,
                        TYPE** A, int A_row_off, int A_col_off, int lda,
                        int batchCount,
                        int *info_array);

int Xpotrf_batch_offset(kblasHandle_t handle,
                        char uplo,
                        const int n,
                        TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                        int batchCount,
                        int *info_array);

//==============================================================================================
int Xpotrs_batch_offset(kblasHandle_t handle,
                        char side, char uplo,
                        const int m, const int n,
                        const TYPE** A, int A_row_off, int A_col_off, int lda,
                              TYPE** B, int B_row_off, int B_col_off, int ldb,
                        int batchCount);

int Xpotrs_batch_offset(kblasHandle_t handle,
                        char side, char uplo,
                        const int m, const int n,
                        const TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                              TYPE* B, int B_row_off, int B_col_off, int ldb, long strideB,
                        int batchCount);

//==============================================================================================
int Xtrtri_batch_offset(kblasHandle_t handle,
                        char uplo, char diag,
                        const int n,
                        TYPE** A, int A_row_off, int A_col_off, int lda,
                        int batchCount,
                        int *info_array);

int Xtrtri_batch_offset(kblasHandle_t handle,
                        char uplo, char diag,
                        const int n,
                        TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                        int batchCount,
                        int *info_array);

//==============================================================================================
int Xlauum_batch_offset(kblasHandle_t handle,
                        char uplo,
                        const int n,
                        TYPE** A, int A_row_off, int A_col_off, int lda,
                        int batchCount,
                        int *info_array);

int Xlauum_batch_offset(kblasHandle_t handle,
                        char uplo,
                        const int n,
                        TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                        int batchCount,
                        int *info_array);

//==============================================================================================
int Xpotri_batch_offset(kblasHandle_t handle,
                        char uplo,
                        const int n,
                        TYPE** A, int A_row_off, int A_col_off, int lda,
                        int batchCount,
                        int *info_array);

int Xpotri_batch_offset(kblasHandle_t handle,
                        char uplo,
                        const int n,
                        TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                        int batchCount,
                        int *info_array);


#endif// __XBLAS_CORE__
