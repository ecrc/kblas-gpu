 /**
 -- (C) Copyright 2013 King Abdullah University of Science and Technology
  Authors:
  Ali Charara (ali.charara@kaust.edu.sa)
  David Keyes (david.keyes@kaust.edu.sa)
  Hatem Ltaief (hatem.ltaief@kaust.edu.sa)

  Redistribution  and  use  in  source and binary forms, with or without
  modification,  are  permitted  provided  that the following conditions
  are met:

  * Redistributions  of  source  code  must  retain  the above copyright
    notice,  this  list  of  conditions  and  the  following  disclaimer.
  * Redistributions  in  binary  form must reproduce the above copyright
    notice,  this list of conditions and the following disclaimer in the
    documentation  and/or other materials provided with the distribution.
  * Neither  the  name of the King Abdullah University of Science and
    Technology nor the names of its contributors may be used to endorse
    or promote products derived from this software without specific prior
    written permission.

  THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
  LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
    /*
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



#endif // _KBLAS_BATCH_H_