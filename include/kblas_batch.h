/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file include/kblas_batch.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 2.0.0
 * @author Ali Charara
 * @date 2017-11-13
 **/

#ifndef _KBLAS_BATCH_H_
#define _KBLAS_BATCH_H_

//############################################################################
//BATCH GEMM routines
//wrappers around cuBLAS / MAGMA batch GEMM routines
//############################################################################

/// Workspace query for batch strided GEMM.
void kblas_gemm_batch_strided_wsquery(kblasHandle_t handle, int batchCount);

/** @addtogroup CPP_API
*  @{
*/
#ifdef __cplusplus
    /**
     * @name Uniform-size batched GEMM wrapper functions around cuBLAS / MAGMA batch GEMM routines
     */
    //@{
    //------------------------------------------------------------------------------
    /**
     * @brief Non-Strided uniform-size single precision batched GEMM
     */
    int kblas_gemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int m, const int n, const int k,
                          const float alpha,
                          const float** A, int lda,
                          const float** B, int ldb,
                          const float beta,
                                float** C, int ldc,
                          int batchCount);

    /**
     * @brief Non-Strided uniform-size double precision batched GEMM
     */
    int kblas_gemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int m, const int n, const int k,
                          const double alpha,
                          const double** A, int lda,
                          const double** B, int ldb,
                          const double beta,
                                double** C, int ldc,
                          int batchCount);

    /**
     * @brief Non-Strided uniform-size single-complex precision batched GEMM
     */
    int kblas_gemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int m, const int n, const int k,
                          const cuFloatComplex alpha,
                          const cuFloatComplex** A, int lda,
                          const cuFloatComplex** B, int ldb,
                          const cuFloatComplex beta,
                                cuFloatComplex** C, int ldc,
                          int batchCount);

    /**
     * @brief Non-Strided uniform-size double-complex precision batched GEMM
     */
    int kblas_gemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int m, const int n, const int k,
                          const cuDoubleComplex alpha,
                          const cuDoubleComplex** A, int lda,
                          const cuDoubleComplex** B, int ldb,
                          const cuDoubleComplex beta,
                                cuDoubleComplex** C, int ldc,
                          int batchCount);

    //------------------------------------------------------------------------------
    // Strided

    /**
     * @brief Strided uniform-size single precision batched GEMM
     */
    int kblas_gemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int m, const int n, const int k,
                          const float alpha,
                          const float* A, int lda, long strideA,
                          const float* B, int ldb, long strideB,
                          const float beta,
                                float* C, int ldc, long strideC,
                          int batchCount);

    /**
     * @brief Strided uniform-size double precision batched GEMM
     */
    int kblas_gemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int m, const int n, const int k,
                          const double alpha,
                          const double* A, int lda, long strideA,
                          const double* B, int ldb, long strideB,
                          const double beta,
                                double* C, int ldc, long strideC,
                          int batchCount);

    /**
     * @brief Strided uniform-size single-complex precision batched GEMM
     */
    int kblas_gemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int m, const int n, const int k,
                          const cuFloatComplex alpha,
                          const cuFloatComplex* A, int lda, long strideA,
                          const cuFloatComplex* B, int ldb, long strideB,
                          const cuFloatComplex beta,
                                cuFloatComplex* C, int ldc, long strideC,
                          int batchCount);

    /**
     * @brief Strided uniform-size double-complex precision batched GEMM
     */
    int kblas_gemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int m, const int n, const int k,
                          const cuDoubleComplex alpha,
                          const cuDoubleComplex* A, int lda, long strideA,
                          const cuDoubleComplex* B, int ldb, long strideB,
                          const cuDoubleComplex beta,
                                cuDoubleComplex* C, int ldc, long strideC,
                          int batchCount);
    //@}
#endif
/** @} */

#ifdef __cplusplus
extern "C" {
#endif

/** @addtogroup C_API
*  @{
*/

    /// Non-Strided single precision batched GEMM
    int kblasSgemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int m, const int n, const int k,
                          const float alpha,
                          const float** A, int lda,
                          const float** B, int ldb,
                          const float beta,
                                float** C, int ldc,
                          int batchCount);

    /// Non-Strided single precision batched GEMM
    int kblasDgemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int m, const int n, const int k,
                          const double alpha,
                          const double** A, int lda,
                          const double** B, int ldb,
                          const double beta,
                                double** C, int ldc,
                          int batchCount);

    /// Non-Strided single-complex precision batched GEMM
    int kblasCgemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int m, const int n, const int k,
                          const cuFloatComplex alpha,
                          const cuFloatComplex** A, int lda,
                          const cuFloatComplex** B, int ldb,
                          const cuFloatComplex beta,
                                cuFloatComplex** C, int ldc,
                          int batchCount);

    /// Non-Strided double-complex precision batched GEMM
    int kblasZgemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int m, const int n, const int k,
                          const cuDoubleComplex alpha,
                          const cuDoubleComplex** A, int lda,
                          const cuDoubleComplex** B, int ldb,
                          const cuDoubleComplex beta,
                                cuDoubleComplex** C, int ldc,
                          int batchCount);

    //------------------------------------------------------------------------------
    /// Strided single precision batched GEMM
    int kblasSgemm_batch_strided( kblasHandle_t handle,
                                  char transA, char transB,
                                  const int m, const int n, const int k,
                                  const float alpha,
                                  const float* A, int lda, long strideA,
                                  const float* B, int ldb, long strideB,
                                  const float beta,
                                        float* C, int ldc, long strideC,
                                  int batchCount);

    /// Strided single precision batched GEMM
    int kblasDgemm_batch_strided( kblasHandle_t handle,
                                  char transA, char transB,
                                  const int m, const int n, const int k,
                                  const double alpha,
                                  const double* A, int lda, long strideA,
                                  const double* B, int ldb, long strideB,
                                  const double beta,
                                        double* C, int ldc, long strideC,
                                  int batchCount);

    /// Strided single-complex precision batched GEMM
    int kblasCgemm_batch_strided( kblasHandle_t handle,
                                  char transA, char transB,
                                  const int m, const int n, const int k,
                                  const cuFloatComplex alpha,
                                  const cuFloatComplex* A, int lda, long strideA,
                                  const cuFloatComplex* B, int ldb, long strideB,
                                  const cuFloatComplex beta,
                                        cuFloatComplex* C, int ldc, long strideC,
                                  int batchCount);

    /// Strided double-complex precision batched GEMM
    int kblasZgemm_batch_strided( kblasHandle_t handle,
                                  char transA, char transB,
                                  const int m, const int n, const int k,
                                  const cuDoubleComplex alpha,
                                  const cuDoubleComplex* A, int lda, long strideA,
                                  const cuDoubleComplex* B, int ldb, long strideB,
                                  const cuDoubleComplex beta,
                                        cuDoubleComplex* C, int ldc, long strideC,
                                  int batchCount);
/** @} */
#ifdef __cplusplus
}
#endif


//############################################################################
// KBLAS BATCH routines
//############################################################################

//============================================================================
// batch SYRK

void kblas_syrk_batch_wsquery(kblasHandle_t handle, const int m, int batchCount);

#ifdef __cplusplus

    //------------------------------------------------------------------------------
    // Non-Strided
    int kblas_syrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          const int m, const int n,
                          const float alpha, const float** A, int lda,
                          const float beta,        float** B, int ldb,
                          int batchCount);

    int kblas_syrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          const int m, const int n,
                          const double alpha, const double** A, int lda,
                          const double beta,        double** B, int ldb,
                          int batchCount);

    int kblas_syrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          const int m, const int n,
                          const cuFloatComplex alpha, const cuFloatComplex** A, int lda,
                          const cuFloatComplex beta,        cuFloatComplex** B, int ldb,
                          int batchCount);

    int kblas_syrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          const int m, const int n,
                          const cuDoubleComplex alpha, const cuDoubleComplex** A, int lda,
                          const cuDoubleComplex beta,        cuDoubleComplex** B, int ldb,
                          int batchCount);

    //------------------------------------------------------------------------------
    // Strided
    int kblas_syrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          const int m, const int n,
                          const float alpha, const float* A, int lda, long strideA,
                          const float beta,        float* B, int ldb, long strideB,
                          int batchCount);

    int kblas_syrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          const int m, const int n,
                          const double alpha, const double* A, int lda, long strideA,
                          const double beta,        double* B, int ldb, long strideB,
                          int batchCount);

    int kblas_syrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          const int m, const int n,
                          const cuFloatComplex alpha, const cuFloatComplex* A, int lda, long strideA,
                          const cuFloatComplex beta,        cuFloatComplex* B, int ldb, long strideB,
                          int batchCount);

    int kblas_syrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          const int m, const int n,
                          const cuDoubleComplex alpha, const cuDoubleComplex* A, int lda, long strideA,
                          const cuDoubleComplex beta,        cuDoubleComplex* B, int ldb, long strideB,
                          int batchCount);
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** @addtogroup C_API
*  @{
*/
    //------------------------------------------------------------------------------
    // Non-Strided

    /// Non-Strided uniform-size single precision batch SYRK
    int kblasSsyrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          const int m, const int n,
                          const float alpha, const float** A, int lda,
                          const float beta,        float** B, int ldb,
                          int batchCount);

    /// Non-Strided uniform-size double precision batch SYRK
    int kblasDsyrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          const int m, const int n,
                          const double alpha, const double** A, int lda,
                          const double beta,        double** B, int ldb,
                          int batchCount);

    /// Non-Strided uniform-size single-complex precision batch SYRK
    int kblasCsyrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          const int m, const int n,
                          const cuFloatComplex alpha, const cuFloatComplex** A, int lda,
                          const cuFloatComplex beta,        cuFloatComplex** B, int ldb,
                          int batchCount);

    /// Non-Strided uniform-size double-complex precision batch SYRK
    int kblasZsyrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          const int m, const int n,
                          const cuDoubleComplex alpha, const cuDoubleComplex** A, int lda,
                          const cuDoubleComplex beta,        cuDoubleComplex** B, int ldb,
                          int batchCount);

    //------------------------------------------------------------------------------
    // Strided

    int kblasSsyrk_batch_strided( kblasHandle_t handle,
                                  char uplo, char trans,
                                  const int m, const int n,
                                  const float alpha, const float* A, int lda, long strideA,
                                  const float beta,        float* B, int ldb, long strideB,
                                  int batchCount);

    int kblasDsyrk_batch_strided( kblasHandle_t handle,
                                  char uplo, char trans,
                                  const int m, const int n,
                                  const double alpha, const double* A, int lda, long strideA,
                                  const double beta,        double* B, int ldb, long strideB,
                                  int batchCount);

    int kblasCsyrk_batch_strided( kblasHandle_t handle,
                                  char uplo, char trans,
                                  const int m, const int n,
                                  const cuFloatComplex alpha, const cuFloatComplex* A, int lda, long strideA,
                                  const cuFloatComplex beta,        cuFloatComplex* B, int ldb, long strideB,
                                  int batchCount);

    int kblasZsyrk_batch_strided( kblasHandle_t handle,
                                  char uplo, char trans,
                                  const int m, const int n,
                                  const cuDoubleComplex alpha, const cuDoubleComplex* A, int lda, long strideA,
                                  const cuDoubleComplex beta,        cuDoubleComplex* B, int ldb, long strideB,
                                  int batchCount);
/** @} */
#ifdef __cplusplus
}
#endif

//============================================================================
// batch TRSM

void kblas_trsm_batch_wsquery(kblasHandle_t handle, int batchCount, char side, int m, int n);
void kblas_trsm_batch_strided_wsquery(kblasHandle_t handle, int batchCount, char side, int m, int n);

#ifdef __cplusplus
    //------------------------------------------------------------------------------
    // Non-Strided

    int kblas_trsm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const float alpha,
                         const float** A, int lda,
                               float** B, int ldb,
                        int batchCount);

    int kblas_trsm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const double alpha,
                         const double** A, int lda,
                               double** B, int ldb,
                        int batchCount);

    int kblas_trsm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const cuFloatComplex alpha,
                         const cuFloatComplex** A, int lda,
                               cuFloatComplex** B, int ldb,
                        int batchCount);

    int kblas_trsm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const cuDoubleComplex alpha,
                         const cuDoubleComplex** A, int lda,
                               cuDoubleComplex** B, int ldb,
                        int batchCount);

    //------------------------------------------------------------------------------
    // Strided
    int kblas_trsm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const float alpha,
                         const float* A, int lda, long strideA,
                               float* B, int ldb, long strideB,
                         int batchCount);

    int kblas_trsm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const double alpha,
                         const double* A, int lda, long strideA,
                               double* B, int ldb, long strideB,
                         int batchCount);

    int kblas_trsm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const cuFloatComplex alpha,
                         const cuFloatComplex* A, int lda, long strideA,
                               cuFloatComplex* B, int ldb, long strideB,
                         int batchCount);

    int kblas_trsm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const cuDoubleComplex alpha,
                         const cuDoubleComplex* A, int lda, long strideA,
                               cuDoubleComplex* B, int ldb, long strideB,
                         int batchCount);
#endif

#ifdef __cplusplus
extern "C" {
#endif
/** @addtogroup C_API
*  @{
*/
    //------------------------------------------------------------------------------
    // Non-Strided

    int kblasStrsm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const float alpha,
                         const float** A, int lda,
                               float** B, int ldb,
                        int batchCount);

    int kblasDtrsm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const double alpha,
                         const double** A, int lda,
                               double** B, int ldb,
                        int batchCount);

    int kblasCtrsm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const cuFloatComplex alpha,
                         const cuFloatComplex** A, int lda,
                               cuFloatComplex** B, int ldb,
                        int batchCount);

    int kblasZtrsm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const cuDoubleComplex alpha,
                         const cuDoubleComplex** A, int lda,
                               cuDoubleComplex** B, int ldb,
                        int batchCount);

    //------------------------------------------------------------------------------
    // Strided

    int kblasStrsm_batch_strided(kblasHandle_t handle,
                                 char side, char uplo, char trans, char diag,
                                 const int m, const int n,
                                 const float alpha,
                                 const float* A, int lda, long strideA,
                                       float* B, int ldb, long strideB,
                                 int batchCount);

    int kblasDtrsm_batch_strided(kblasHandle_t handle,
                                 char side, char uplo, char trans, char diag,
                                 const int m, const int n,
                                 const double alpha,
                                 const double* A, int lda, long strideA,
                                       double* B, int ldb, long strideB,
                                 int batchCount);

    int kblasCtrsm_batch_strided(kblasHandle_t handle,
                                 char side, char uplo, char trans, char diag,
                                 const int m, const int n,
                                 const cuFloatComplex alpha,
                                 const cuFloatComplex* A, int lda, long strideA,
                                       cuFloatComplex* B, int ldb, long strideB,
                                 int batchCount);

    int kblasZtrsm_batch_strided(kblasHandle_t handle,
                                 char side, char uplo, char trans, char diag,
                                 const int m, const int n,
                                 const cuDoubleComplex alpha,
                                 const cuDoubleComplex* A, int lda, long strideA,
                                       cuDoubleComplex* B, int ldb, long strideB,
                                 int batchCount);
/** @} */
#ifdef __cplusplus
}
#endif

//============================================================================
// batch TRMM

void kblas_trmm_batch_wsquery(kblasHandle_t handle, int batchCount, char side, int m, int n);
void kblas_trmm_batch_strided_wsquery(kblasHandle_t handle, int batchCount, char side, int m, int n);

#ifdef __cplusplus
    //------------------------------------------------------------------------------
    // Non-Strided

    int kblas_trmm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const float alpha,
                         const float** A, int lda,
                               float** B, int ldb,
                        int batchCount);

    int kblas_trmm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const double alpha,
                         const double** A, int lda,
                               double** B, int ldb,
                        int batchCount);

    int kblas_trmm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const cuFloatComplex alpha,
                         const cuFloatComplex** A, int lda,
                               cuFloatComplex** B, int ldb,
                        int batchCount);

    int kblas_trmm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const cuDoubleComplex alpha,
                         const cuDoubleComplex** A, int lda,
                               cuDoubleComplex** B, int ldb,
                        int batchCount);

    //------------------------------------------------------------------------------
    // Strided
    int kblas_trmm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const float alpha,
                         const float* A, int lda, long strideA,
                               float* B, int ldb, long strideB,
                         int batchCount);

    int kblas_trmm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const double alpha,
                         const double* A, int lda, long strideA,
                               double* B, int ldb, long strideB,
                         int batchCount);

    int kblas_trmm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const cuFloatComplex alpha,
                         const cuFloatComplex* A, int lda, long strideA,
                               cuFloatComplex* B, int ldb, long strideB,
                         int batchCount);

    int kblas_trmm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const cuDoubleComplex alpha,
                         const cuDoubleComplex* A, int lda, long strideA,
                               cuDoubleComplex* B, int ldb, long strideB,
                         int batchCount);
#endif

#ifdef __cplusplus
extern "C" {
#endif
    //------------------------------------------------------------------------------
    // Non-Strided

    int kblasStrmm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const float alpha,
                         const float** A, int lda,
                               float** B, int ldb,
                        int batchCount);

    int kblasDtrmm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const double alpha,
                         const double** A, int lda,
                               double** B, int ldb,
                        int batchCount);

    int kblasCtrmm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const cuFloatComplex alpha,
                         const cuFloatComplex** A, int lda,
                               cuFloatComplex** B, int ldb,
                        int batchCount);

    int kblasZtrmm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const cuDoubleComplex alpha,
                         const cuDoubleComplex** A, int lda,
                               cuDoubleComplex** B, int ldb,
                        int batchCount);

    //------------------------------------------------------------------------------
    // Strided

    int kblasStrmm_batch_strided(kblasHandle_t handle,
                                 char side, char uplo, char trans, char diag,
                                 const int m, const int n,
                                 const float alpha,
                                 const float* A, int lda, long strideA,
                                       float* B, int ldb, long strideB,
                                 int batchCount);

    int kblasDtrmm_batch_strided(kblasHandle_t handle,
                                 char side, char uplo, char trans, char diag,
                                 const int m, const int n,
                                 const double alpha,
                                 const double* A, int lda, long strideA,
                                       double* B, int ldb, long strideB,
                                 int batchCount);

    int kblasCtrmm_batch_strided(kblasHandle_t handle,
                                 char side, char uplo, char trans, char diag,
                                 const int m, const int n,
                                 const cuFloatComplex alpha,
                                 const cuFloatComplex* A, int lda, long strideA,
                                       cuFloatComplex* B, int ldb, long strideB,
                                 int batchCount);

    int kblasZtrmm_batch_strided(kblasHandle_t handle,
                                 char side, char uplo, char trans, char diag,
                                 const int m, const int n,
                                 const cuDoubleComplex alpha,
                                 const cuDoubleComplex* A, int lda, long strideA,
                                       cuDoubleComplex* B, int ldb, long strideB,
                                 int batchCount);
#ifdef __cplusplus
}
#endif

//============================================================================
// batch POTRF
void kblas_potrf_batch_wsquery(kblasHandle_t handle, const int n, int batchCount);
void kblas_potrf_batch_strided_wsquery(kblasHandle_t handle, const int n, int batchCount);

#ifdef __cplusplus
    //------------------------------------------------------------------------------
    // Non-Strided
    int kblas_potrf_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          float** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_potrf_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          double** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_potrf_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuFloatComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_potrf_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuDoubleComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    //------------------------------------------------------------------------------
    // Strided
    int kblas_potrf_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          float* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_potrf_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          double* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_potrf_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuFloatComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_potrf_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuDoubleComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);
#endif

#ifdef __cplusplus
extern "C" {
#endif
    //------------------------------------------------------------------------------
    // Non-Strided
    int kblasSpotrf_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          float** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblasDpotrf_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          double** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblasCpotrf_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuFloatComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblasZpotrf_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuDoubleComplex** A, int lda,
                          int batchCount,
                          int *info_array);
    //------------------------------------------------------------------------------
    // Strided
    int kblasSpotrf_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  float* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    int kblasDpotrf_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  double* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    int kblasCpotrf_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  cuFloatComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    int kblasZpotrf_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  cuDoubleComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);
#ifdef __cplusplus
}
#endif


//============================================================================
// batch LAUUM

void kblas_lauum_batch_wsquery(kblasHandle_t handle, const int n, int batchCount);
void kblas_lauum_batch_strided_wsquery(kblasHandle_t handle, const int n, int batchCount);

#ifdef __cplusplus
    //------------------------------------------------------------------------------
    // Non-Strided
    int kblas_lauum_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          float** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_lauum_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          double** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_lauum_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuFloatComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_lauum_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuDoubleComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    //------------------------------------------------------------------------------
    // Strided
    int kblas_lauum_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          float* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_lauum_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          double* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_lauum_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuFloatComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_lauum_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuDoubleComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);
#endif

#ifdef __cplusplus
extern "C" {
#endif
    //------------------------------------------------------------------------------
    // Non-Strided
    int kblasSlauum_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          float** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblasDlauum_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          double** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblasClauum_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuFloatComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblasZlauum_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuDoubleComplex** A, int lda,
                          int batchCount,
                          int *info_array);
    //------------------------------------------------------------------------------
    // Strided
    int kblasSlauum_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  float* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    int kblasDlauum_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  double* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    int kblasClauum_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  cuFloatComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    int kblasZlauum_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  cuDoubleComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);
#ifdef __cplusplus
}
#endif

//============================================================================
// batch TRTRI

void kblas_trtri_batch_wsquery(kblasHandle_t handle, const int n, int batchCount);
void kblas_trtri_batch_strided_wsquery(kblasHandle_t handle, const int n, int batchCount);

#ifdef __cplusplus
    //------------------------------------------------------------------------------
    // Non-Strided
    int kblas_trtri_batch(kblasHandle_t handle,
                          char uplo, char diag,
                          const int n,
                          float** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_trtri_batch(kblasHandle_t handle,
                          char uplo, char diag,
                          const int n,
                          double** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_trtri_batch(kblasHandle_t handle,
                          char uplo, char diag,
                          const int n,
                          cuFloatComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_trtri_batch(kblasHandle_t handle,
                          char uplo, char diag,
                          const int n,
                          cuDoubleComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    //------------------------------------------------------------------------------
    // Strided
    int kblas_trtri_batch(kblasHandle_t handle,
                          char uplo, char diag,
                          const int n,
                          float* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_trtri_batch(kblasHandle_t handle,
                          char uplo, char diag,
                          const int n,
                          double* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_trtri_batch(kblasHandle_t handle,
                          char uplo, char diag,
                          const int n,
                          cuFloatComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_trtri_batch(kblasHandle_t handle,
                          char uplo, char diag,
                          const int n,
                          cuDoubleComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);
#endif

#ifdef __cplusplus
extern "C" {
#endif
    //------------------------------------------------------------------------------
    // Non-Strided
    int kblasStrtri_batch(kblasHandle_t handle,
                          char uplo, char diag,
                          const int n,
                          float** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblasDtrtri_batch(kblasHandle_t handle,
                          char uplo, char diag,
                          const int n,
                          double** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblasCtrtri_batch(kblasHandle_t handle,
                          char uplo, char diag,
                          const int n,
                          cuFloatComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblasZtrtri_batch(kblasHandle_t handle,
                          char uplo, char diag,
                          const int n,
                          cuDoubleComplex** A, int lda,
                          int batchCount,
                          int *info_array);
    //------------------------------------------------------------------------------
    // Strided
    int kblasStrtri_batch_strided(kblasHandle_t handle,
                                  char uplo, char diag,
                                  const int n,
                                  float* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    int kblasDtrtri_batch_strided(kblasHandle_t handle,
                                  char uplo, char diag,
                                  const int n,
                                  double* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    int kblasCtrtri_batch_strided(kblasHandle_t handle,
                                  char uplo, char diag,
                                  const int n,
                                  cuFloatComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    int kblasZtrtri_batch_strided(kblasHandle_t handle,
                                  char uplo, char diag,
                                  const int n,
                                  cuDoubleComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);
#ifdef __cplusplus
}
#endif


//============================================================================
// batch POTRS
void kblas_potrs_batch_wsquery(kblasHandle_t handle, const int m, const int n, int batchCount);
void kblas_potrs_batch_strided_wsquery(kblasHandle_t handle, const int m, const int n, int batchCount);

#ifdef __cplusplus
    //------------------------------------------------------------------------------
    // Non-Strided
    int kblas_potrs_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          const float** A, int lda,
                                float** B, int ldb,
                          int batchCount);

    int kblas_potrs_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          const double** A, int lda,
                                double** B, int ldb,
                          int batchCount);

    int kblas_potrs_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          const cuFloatComplex** A, int lda,
                                cuFloatComplex** B, int ldb,
                          int batchCount);

    int kblas_potrs_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          const cuDoubleComplex** A, int lda,
                                cuDoubleComplex** B, int ldb,
                          int batchCount);

    //------------------------------------------------------------------------------
    // Strided
    int kblas_potrs_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          const float* A, int lda, long strideA,
                                float* B, int ldb, long strideB,
                          int batchCount);

    int kblas_potrs_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          const double* A, int lda, long strideA,
                                double* B, int ldb, long strideB,
                          int batchCount);

    int kblas_potrs_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          const cuFloatComplex* A, int lda, long strideA,
                                cuFloatComplex* B, int ldb, long strideB,
                          int batchCount);

    int kblas_potrs_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          const cuDoubleComplex* A, int lda, long strideA,
                                cuDoubleComplex* B, int ldb, long strideB,
                          int batchCount);
#endif

#ifdef __cplusplus
extern "C" {
#endif
    //------------------------------------------------------------------------------
    // Non-Strided
    int kblasSpotrs_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          const float** A, int lda,
                                float** B, int ldb,
                          int batchCount);

    int kblasDpotrs_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          const double** A, int lda,
                                double** B, int ldb,
                          int batchCount);

    int kblasCpotrs_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          const cuFloatComplex** A, int lda,
                                cuFloatComplex** B, int ldb,
                          int batchCount);

    int kblasZpotrs_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          const cuDoubleComplex** A, int lda,
                                cuDoubleComplex** B, int ldb,
                          int batchCount);

    //------------------------------------------------------------------------------
    // Strided
    int kblasSpotrs_batch_strided(kblasHandle_t handle,
                                  char side, char uplo,
                                  const int m, const int n,
                                  const float* A, int lda, long strideA,
                                        float* B, int ldb, long strideB,
                                  int batchCount);

    int kblasDpotrs_batch_strided(kblasHandle_t handle,
                                  char side, char uplo,
                                  const int m, const int n,
                                  const double* A, int lda, long strideA,
                                        double* B, int ldb, long strideB,
                                  int batchCount);

    int kblasCpotrs_batch_strided(kblasHandle_t handle,
                                  char side, char uplo,
                                  const int m, const int n,
                                  const cuFloatComplex* A, int lda, long strideA,
                                        cuFloatComplex* B, int ldb, long strideB,
                                  int batchCount);

    int kblasZpotrs_batch_strided(kblasHandle_t handle,
                                  char side, char uplo,
                                  const int m, const int n,
                                  const cuDoubleComplex* A, int lda, long strideA,
                                        cuDoubleComplex* B, int ldb, long strideB,
                                  int batchCount);
#ifdef __cplusplus
}
#endif

//============================================================================
// batch POTRI
void kblas_potri_batch_wsquery(kblasHandle_t handle, const int n, int batchCount);
void kblas_potri_batch_strided_wsquery(kblasHandle_t handle, const int n, int batchCount);

#ifdef __cplusplus
    //------------------------------------------------------------------------------
    // Non-Strided
    int kblas_potri_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          float** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_potri_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          double** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_potri_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuFloatComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_potri_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuDoubleComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    //------------------------------------------------------------------------------
    // Strided
    int kblas_potri_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          float* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_potri_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          double* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_potri_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuFloatComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_potri_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuDoubleComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);
#endif

#ifdef __cplusplus
extern "C" {
#endif
    //------------------------------------------------------------------------------
    // Non-Strided
    int kblasSpotri_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          float** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblasDpotri_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          double** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblasCpotri_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuFloatComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblasZpotri_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuDoubleComplex** A, int lda,
                          int batchCount,
                          int *info_array);
    //------------------------------------------------------------------------------
    // Strided
    int kblasSpotri_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  float* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    int kblasDpotri_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  double* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    int kblasCpotri_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  cuFloatComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    int kblasZpotri_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  cuDoubleComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);
#ifdef __cplusplus
}
#endif

//============================================================================
// batch POTRI
void kblas_poti_batch_wsquery(kblasHandle_t handle, const int n, int batchCount);
void kblas_poti_batch_strided_wsquery(kblasHandle_t handle, const int n, int batchCount);

#ifdef __cplusplus
    //------------------------------------------------------------------------------
    // Non-Strided
    int kblas_poti_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          float** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_poti_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          double** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_poti_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuFloatComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_poti_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuDoubleComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    //------------------------------------------------------------------------------
    // Strided
    int kblas_poti_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          float* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_poti_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          double* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_poti_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuFloatComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_poti_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuDoubleComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);
#endif

#ifdef __cplusplus
extern "C" {
#endif
    //------------------------------------------------------------------------------
    // Non-Strided
    int kblasSpoti_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          float** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblasDpoti_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          double** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblasCpoti_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuFloatComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblasZpoti_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          cuDoubleComplex** A, int lda,
                          int batchCount,
                          int *info_array);
    //------------------------------------------------------------------------------
    // Strided
    int kblasSpoti_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  float* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    int kblasDpoti_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  double* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    int kblasCpoti_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  cuFloatComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    int kblasZpoti_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  cuDoubleComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);
#ifdef __cplusplus
}
#endif

#endif // _KBLAS_BATCH_H_
