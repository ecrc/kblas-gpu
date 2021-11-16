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
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef _KBLAS_BATCH_H_
#define _KBLAS_BATCH_H_

//############################################################################
//BATCH GEMM routines
//wrappers around cuBLAS / MAGMA batch GEMM routines
//############################################################################


/**
 * @ingroup WSQUERY
 * @brief Workspace query for batch strided GEMM routines.
 *
 * @param[in,out] handle KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     batchCount Number of matrices to be processed.
 */
void kblas_gemm_batch_strided_wsquery(kblasHandle_t handle, int batchCount);
void kblas_gemm_batch_nonuniform_wsquery(kblasHandle_t handle);

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
     * @brief Non-Strided uniform-size single precision batched GEMM wrapper function to cuBLAS / MAGMA corresponding routines
     *
     * @param[in] handle  KBLAS handle
     * @param[in] transA  specifies the form of op( A ) to be used in
     *                    the matrix multiplication as follows:
     *                    = KBLAS_NoTrans:   op( A ) = A.
     *                    = KBLAS_Trans:     op( A ) = A**T.
     * @param[in] transB  specifies the form of op( B ) to be used in
     *                    the matrix multiplication as follows:
     *                    = KBLAS_NoTrans:   op( B ) = B.
     *                    = KBLAS_Trans:     op( B ) = B**T.
     * @param[in] m       specifies the number of rows of the matrices
     *                    op( A ) and of the matrices C. Must be at least zero.
     * @param[in] n       specifies the number of columns of the matrices
     *                    op( B ) and of the matrices C. Must be at least zero.
     * @param[in] k             specifies the number of columns of each matrix
     *                          op( A ) and the number of rows of each matrix op( B ). Must
     *                          be at least  zero.
     * @param[in] alpha         Specifies the scalar alpha.
     * @param[in] A_array       Host pointer to array of device pointers to device buffers.
     *                          Each buffer contians a matrix of dimension (lda, ka), where ka
     *                          is k when transA is KBLAS_NoTrans, m otherwise. Leading (m, ka)
     *                          of each buffer must contain the data of A's.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] B_array       Host pointer to array of device pointers to device buffers.
     *                          Each buffer contians a matrix of dimension (ldb, kb), where kb
     *                          is n when transB is KBLAS_NoTrans, k otherwise. Leading (k, kb)
     *                          of each buffer must contain the data of B's.
     * @param[in] ldb           Leading dimension of each matrix of B.
     * @param[in] beta          Specifies the scalar beta.
     * @param[in, out] C_array  Host pointer to array of device pointers to device buffers.
     *                          Each buffer contians a matrix of dimension (ldc, n). Leading (m, n)
     *                          of each buffer must contain the data of C's.
     * @param[in] ldc           Leading dimension of each matrix of C.
     * @param[in] batchCount    Number of matrices to be batch processed
     *
     * Workspace needed: none
     *
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
                          const hipFloatComplex alpha,
                          const hipFloatComplex** A, int lda,
                          const hipFloatComplex** B, int ldb,
                          const hipFloatComplex beta,
                                hipFloatComplex** C, int ldc,
                          int batchCount);

    /**
     * @brief Non-Strided uniform-size double-complex precision batched GEMM
     */
    int kblas_gemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int m, const int n, const int k,
                          const hipDoubleComplex alpha,
                          const hipDoubleComplex** A, int lda,
                          const hipDoubleComplex** B, int ldb,
                          const hipDoubleComplex beta,
                                hipDoubleComplex** C, int ldc,
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
                          const hipFloatComplex alpha,
                          const hipFloatComplex* A, int lda, long strideA,
                          const hipFloatComplex* B, int ldb, long strideB,
                          const hipFloatComplex beta,
                                hipFloatComplex* C, int ldc, long strideC,
                          int batchCount);

    /**
     * @brief Strided uniform-size double-complex precision batched GEMM
     */
    int kblas_gemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int m, const int n, const int k,
                          const hipDoubleComplex alpha,
                          const hipDoubleComplex* A, int lda, long strideA,
                          const hipDoubleComplex* B, int ldb, long strideB,
                          const hipDoubleComplex beta,
                                hipDoubleComplex* C, int ldc, long strideC,
                          int batchCount);
    //@}

    /**
     * @name TODO Non-uniform batch gemm,
     * if all maximum m/n/k are passed 0, they will be recomputed
     */
    //@{

    /**
     * @brief TODO
     */
    int kblas_gemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          int* m, int* n, int* k,
                          int max_m, int max_n, int max_k,
                          const float alpha,
                          const float** A, int* lda,
                          const float** B, int* ldb,
                          const float beta,
                                float** C, int* ldc,
                          int batchCount );
    /**
     * @brief TODO
     */
    int kblas_gemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          int* m, int* n, int* k,
                          int max_m, int max_n, int max_k,
                          const double alpha,
                          const double** A, int* lda,
                          const double** B, int* ldb,
                          const double beta,
                                double** C, int* ldc,
                          int batchCount );
    /**
     * @brief TODO
     */
    int kblas_gemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          int* m, int* n, int* k,
                          int max_m, int max_n, int max_k,
                          const hipFloatComplex alpha,
                          const hipFloatComplex** A, int* lda,
                          const hipFloatComplex** B, int* ldb,
                          const hipFloatComplex beta,
                                hipFloatComplex** C, int* ldc,
                          int batchCount );
    /**
     * @brief TODO
     */
    int kblas_gemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          int* m, int* n, int* k,
                          int max_m, int max_n, int max_k,
                          const hipDoubleComplex alpha,
                          const hipDoubleComplex** A, int* lda,
                          const hipDoubleComplex** B, int* ldb,
                          const hipDoubleComplex beta,
                                hipDoubleComplex** C, int* ldc,
                          int batchCount );
    //@}

#endif
/** @} */

#ifdef __cplusplus
extern "C" {
#endif

/** @addtogroup C_API
*  @{
*/

    /**
     * @name Uniform-size batched GEMM routines
     *
     * Wrappers around cuBLAS / MAGMA batch GEMM routines
     * @{
     */
    /**
     * @brief Non-Strided uniform-size single precision batched GEMM wrapper function to cuBLAS / MAGMA corresponding routines
     *
     * @param[in] handle  KBLAS handle
     * @param[in] transA  Specifies the form of op( A ) to be used in
     *                    the matrix multiplication as follows:
     *                    - KBLAS_NoTrans:   op( A ) = A.
     *                    - KBLAS_Trans:     op( A ) = A**T.
     * @param[in] transB  Specifies the form of op( B ) to be used in
     *                    the matrix multiplication as follows:
     *                    - KBLAS_NoTrans:   op( B ) = B.
     *                    - KBLAS_Trans:     op( B ) = B**T.
     * @param[in] m       Specifies the number of rows of the matrices
     *                    op( A ) and of the matrices C. Must be at least zero.
     * @param[in] n       Specifies the number of columns of the matrices
     *                    op( B ) and of the matrices C. Must be at least zero.
     * @param[in] k             Specifies the number of columns of each matrix
     *                          op( A ) and the number of rows of each matrix op( B ). Must
     *                          be at least  zero.
     * @param[in] alpha         Specifies the scalar alpha.
     * @param[in] A_array       Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (lda, ka), where ka
     *                          is k when transA is KBLAS_NoTrans, m otherwise. Leading (m, ka)
     *                          of each buffer must contain the data of A's.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] B_array       Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (ldb, kb), where kb
     *                          is n when transB is KBLAS_NoTrans, k otherwise. Leading (k, kb)
     *                          of each buffer must contain the data of B's.
     * @param[in] ldb           Leading dimension of each matrix of B.
     * @param[in] beta          Specifies the scalar beta.
     * @param[in, out] C_array  Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (ldc, n). Leading (m, n)
     *                          of each buffer must contain the data of C's.
     * @param[in] ldc           Leading dimension of each matrix of C.
     * @param[in] batchCount    Number of matrices to be batch processed
     *
     * Workspace needed: none
     *
     */
    int kblasSgemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int m, const int n, const int k,
                          const float alpha,
                          const float** A_array, int lda,
                          const float** B_array, int ldb,
                          const float beta,
                                float** C_array, int ldc,
                          int batchCount);

    /**
     * @brief Non-Strided uniform-size double precision batched GEMM wrapper function to cuBLAS / MAGMA corresponding routines
     *
     * @see kblasSgemm_batch() for details about params.
     */
    int kblasDgemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int m, const int n, const int k,
                          const double alpha,
                          const double** A_array, int lda,
                          const double** B_array, int ldb,
                          const double beta,
                                double** C_array, int ldc,
                          int batchCount);

    /**
     * @brief Non-Strided uniform-size single-complex precision batched GEMM wrapper function to cuBLAS / MAGMA corresponding routines
     *
     * @see kblasSgemm_batch() for details about params.
     */
    int kblasCgemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int m, const int n, const int k,
                          const hipFloatComplex alpha,
                          const hipFloatComplex** A_array, int lda,
                          const hipFloatComplex** B_array, int ldb,
                          const hipFloatComplex beta,
                                hipFloatComplex** C_array, int ldc,
                          int batchCount);

    /**
     * @brief Non-Strided uniform-size double-complex precision batched GEMM wrapper function to cuBLAS / MAGMA corresponding routines
     *
     * @see kblasSgemm_batch() for details about params.
     */
    int kblasZgemm_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int m, const int n, const int k,
                          const hipDoubleComplex alpha,
                          const hipDoubleComplex** A_array, int lda,
                          const hipDoubleComplex** B_array, int ldb,
                          const hipDoubleComplex beta,
                                hipDoubleComplex** C_array, int ldc,
                          int batchCount);

    //------------------------------------------------------------------------------
    /**
     * @brief Strided uniform-size single precision batched GEMM wrapper function to cuBLAS / MAGMA corresponding routines
     *
     * @param[in] handle  KBLAS handle
     * @param[in] transA  Specifies the form of op( A ) to be used in
     *                    the matrix multiplication as follows:
     *                    - KBLAS_NoTrans:   op( A ) = A.
     *                    - KBLAS_Trans:     op( A ) = A**T.
     * @param[in] transB  Specifies the form of op( B ) to be used in
     *                    the matrix multiplication as follows:
     *                    - KBLAS_NoTrans:   op( B ) = B.
     *                    - KBLAS_Trans:     op( B ) = B**T.
     * @param[in] m       Specifies the number of rows of the matrices.
     *                    op( A )'s and C's. Must be at least zero.
     * @param[in] n       Specifies the number of columns of the matrices.
     *                    op( B )'s and C's. Must be at least zero.
     * @param[in] k           Specifies the number of columns of each matrix.
     *                        op( A ) and the number of rows of each matrix op( B ).
     *                        Must be at least zero.
     * @param[in] alpha       Specifies the scalar alpha.
     * @param[in] A           Host pointer to device buffer.
     *                        Buffer contains a strided set of matrices each of dimension (lda, ka), where ka
     *                        is k when transA is KBLAS_NoTrans, m otherwise. Leading (m, ka)
     *                        of each matrix must contain the data of A's.
     * @param[in] lda         Leading dimension of each matrix of A.
     * @param[in] strideA     Stride in elements between consecutive matrices of A. Must be at least (lda * ka).
     * @param[in] B           Host pointer to device buffer.
     *                        Buffer contains a strided set of matrices each of dimension (ldb, kb), where kb
     *                        is n when transB is KBLAS_NoTrans, k otherwise. Leading (k, kb).
     *                        of each matrix must contain the data of B's.
     * @param[in] ldb         Leading dimension of each matrix of B.
     * @param[in] strideB     Stride in elements between consecutive matrices of B. Must be at least (ldb * kb).
     * @param[in] beta        Specifies the scalar beta.
     * @param[in, out] C      Host pointer to device buffer.
     *                        Buffer contains a strided set of matrices each of dimension (ldc, n). Leading (m, n).
     *                        of each buffer must contain the data of C's.
     * @param[in] ldc         Leading dimension of each matrix of C.
     * @param[in] strideC     Stride in elements between consecutive matrices of C. Must be at least (ldc * n).
     * @param[in] batchCount  Number of matrices to be batch processed.
     *
     * Workspace needed: query with kblas_gemm_batch_strided_wsquery().
     *
     */
    int kblasSgemm_batch_strided( kblasHandle_t handle,
                                  char transA, char transB,
                                  const int m, const int n, const int k,
                                  const float alpha,
                                  const float* A, int lda, long strideA,
                                  const float* B, int ldb, long strideB,
                                  const float beta,
                                        float* C, int ldc, long strideC,
                                  int batchCount);

    /**
     * @brief Strided uniform-size double precision batched GEMM wrapper function to cuBLAS / MAGMA corresponding routines
     *
     * @see kblasSgemm_batch_strided() for details about params.
     */
    int kblasDgemm_batch_strided( kblasHandle_t handle,
                                  char transA, char transB,
                                  const int m, const int n, const int k,
                                  const double alpha,
                                  const double* A, int lda, long strideA,
                                  const double* B, int ldb, long strideB,
                                  const double beta,
                                        double* C, int ldc, long strideC,
                                  int batchCount);

    /**
     * @brief Strided uniform-size single-complex precision batched GEMM wrapper function to cuBLAS / MAGMA corresponding routines
     *
     * @see kblasSgemm_batch_strided() for details about params.
     */
    int kblasCgemm_batch_strided( kblasHandle_t handle,
                                  char transA, char transB,
                                  const int m, const int n, const int k,
                                  const hipFloatComplex alpha,
                                  const hipFloatComplex* A, int lda, long strideA,
                                  const hipFloatComplex* B, int ldb, long strideB,
                                  const hipFloatComplex beta,
                                        hipFloatComplex* C, int ldc, long strideC,
                                  int batchCount);

    /**
     * @brief Strided uniform-size double-complex precision batched GEMM wrapper function to cuBLAS / MAGMA corresponding routines
     *
     * @see kblasSgemm_batch_strided() for details about params.
     */
    int kblasZgemm_batch_strided( kblasHandle_t handle,
                                  char transA, char transB,
                                  const int m, const int n, const int k,
                                  const hipDoubleComplex alpha,
                                  const hipDoubleComplex* A, int lda, long strideA,
                                  const hipDoubleComplex* B, int ldb, long strideB,
                                  const hipDoubleComplex beta,
                                        hipDoubleComplex* C, int ldc, long strideC,
                                  int batchCount);
/** @} */
#ifdef __cplusplus
}
#endif
/** @} */

//############################################################################
// KBLAS BATCH routines
//############################################################################

//============================================================================
// batch SYRK


/**
 * @ingroup WSQUERY
 * @brief Workspace query for batched SYRK routines.
 *
 * @param[in,out] handle      KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     m           Order of matrix C.
 * @param[in]     batchCount  Number of matrices to be processed.
 * @see kblasSsyrk_batch() for details of params.
 */
void kblas_syrk_batch_wsquery(kblasHandle_t handle, const int m, int batchCount);

void kblas_syrk_batch_nonuniform_wsquery(kblasHandle_t handle);

#ifdef __cplusplus

    //------------------------------------------------------------------------------
    // Non-Strided Uniform
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
                          const hipFloatComplex alpha, const hipFloatComplex** A, int lda,
                          const hipFloatComplex beta,        hipFloatComplex** B, int ldb,
                          int batchCount);

    int kblas_syrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          const int m, const int n,
                          const hipDoubleComplex alpha, const hipDoubleComplex** A, int lda,
                          const hipDoubleComplex beta,        hipDoubleComplex** B, int ldb,
                          int batchCount);

    //------------------------------------------------------------------------------
    // Non-Strided Non-uniform
    // if all maximum m/n are passed 0, they will be recomputed
    int kblas_syrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          int* m, int* n,
                          int max_m, int max_n,
                          float alpha, float** A, int* lda,
                          float beta,  float** B, int* ldb,
                          int batchCount);
    int kblas_syrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          int* m, int* n,
                          int max_m, int max_n,
                          double alpha, double** A, int* lda,
                          double beta,  double** B, int* ldb,
                          int batchCount);
    int kblas_syrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          int* m, int* n,
                          int max_m, int max_n,
                          hipFloatComplex alpha, hipFloatComplex** A, int* lda,
                          hipFloatComplex beta,  hipFloatComplex** B, int* ldb,
                          int batchCount);
    int kblas_syrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          int* m, int* n,
                          int max_m, int max_n,
                          hipDoubleComplex alpha, hipDoubleComplex** A, int* lda,
                          hipDoubleComplex beta,  hipDoubleComplex** B, int* ldb,
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
                          const hipFloatComplex alpha, const hipFloatComplex* A, int lda, long strideA,
                          const hipFloatComplex beta,        hipFloatComplex* B, int ldb, long strideB,
                          int batchCount);

    int kblas_syrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          const int m, const int n,
                          const hipDoubleComplex alpha, const hipDoubleComplex* A, int lda, long strideA,
                          const hipDoubleComplex beta,        hipDoubleComplex* B, int ldb, long strideB,
                          int batchCount);
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** @addtogroup C_API
*  @{
*/

    /**
     * @name Uniform-size batched SYRK routines
     * @{
     */
    //------------------------------------------------------------------------------
    // Non-Strided

    /**
     * @brief Non-Strided uniform-size single precision batched SYRK routine.
     *
     * @param[in] handle  KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] uplo    Specifies whether the upper or lower
     *                    triangular part of the matrices B is to be referenced as
     *                    follows:
     *                    - KBLAS_Lower: Only the lower triangular part of B is to be referenced.
     *                    - KBLAS_Upper: Only the upper triangular part of B is to be referenced.
     * @param[in] trans   Specifies the operation to be performed as follows:
     *                    - KBLAS_NoTrans:    B := alpha*A*A**T + beta*B.
     *                    - KBLAS_Trans:      B := alpha*A**T*A + beta*B.
     * @param[in] m       Specifies the number of rows and columns of the matrices B. Must be at least zero.
     * @param[in] n       Specifies the number of columns (trans = KBLAS_Trans, rows otherwise) of the matrices A.
     *                    Must be at least zero.
     * @param[in] alpha         Specifies the scalar alpha.
     * @param[in] A_array       Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (lda, ka), where ka
     *                          is n when trans is KBLAS_NoTrans, m otherwise. Leading (m, n)
     *                          (with trans = KBLAS_Trans, (n, m) otherwise)
     *                          of each buffer must contain the data of A's.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] beta          Specifies the scalar beta.
     * @param[in, out] B_array  Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (ldb, m). Leading (m, m)
     *                          of each buffer must contain the data of B's.
     * @param[in] ldb           Leading dimension of each matrix of B.
     * @param[in] batchCount    Number of matrices to be batch processed.
     *
     * Workspace needed: query with kblas_syrk_batch_wsquery().
     *
     */
    int kblasSsyrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          const int m, const int n,
                          const float alpha, const float** A_array, int lda,
                          const float beta,        float** B_array, int ldb,
                          int batchCount);

    /**
     * @brief Non-Strided uniform-size double precision batched SYRK.
     *
     * @see kblasSsyrk_batch() for details about params and workspace.
     */
    int kblasDsyrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          const int m, const int n,
                          const double alpha, const double** A_array, int lda,
                          const double beta,        double** B_array, int ldb,
                          int batchCount);

    /**
     * @brief Non-Strided uniform-size single-complex precision batched SYRK.
     *
     * @see kblasSsyrk_batch() for details about params and workspace.
     */
    int kblasCsyrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          const int m, const int n,
                          const hipFloatComplex alpha, const hipFloatComplex** A_array, int lda,
                          const hipFloatComplex beta,        hipFloatComplex** B_array, int ldb,
                          int batchCount);

    /**
     * @brief Non-Strided uniform-size double-complex precision batched SYRK.
     *
     * @see kblasSsyrk_batch() for details about params and workspace.
     */
    int kblasZsyrk_batch( kblasHandle_t handle,
                          char uplo, char trans,
                          const int m, const int n,
                          const hipDoubleComplex alpha, const hipDoubleComplex** A_array, int lda,
                          const hipDoubleComplex beta,        hipDoubleComplex** B_array, int ldb,
                          int batchCount);

    //------------------------------------------------------------------------------
    // Strided

    /**
     * @brief Strided uniform-size single precision batched SYRK routine.
     *
     * @param[in] handle  KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] uplo    Specifies whether the upper or lower
     *                    triangular part of the matrices B is to be referenced as
     *                    follows:
     *                    - KBLAS_Lower: Only the lower triangular part of B is to be referenced.
     *                    - KBLAS_Upper: Only the upper triangular part of B is to be referenced.
     * @param[in] trans   Specifies the operation to be performed as follows:
     *                    - KBLAS_NoTrans:    B := alpha*A*A**T + beta*B.
     *                    - KBLAS_Trans:      B := alpha*A**T*A + beta*B.
     * @param[in] m       Specifies the number of rows and columns of the matrices B. Must be at least zero.
     * @param[in] n       Specifies the number of columns (trans = KBLAS_Trans, rows otherwise) of the matrices A.
     *                    Must be at least zero.
     * @param[in] alpha         Specifies the scalar alpha.
     * @param[in] A             Host pointer to device buffer.
     *                          Buffer contains a strided set of matrices each of dimension (lda, ka), where ka
     *                          is n when trans is KBLAS_NoTrans, m otherwise. Leading (m, n)
     *                          (with trans = KBLAS_Trans, (n, m) otherwise)
     *                          of each buffer must contain the data of A's.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] strideA       Stride in elements between consecutive matrices of A. Must be at least (lda * ka).
     * @param[in] beta          Specifies the scalar beta.
     * @param[in, out] B        Host pointer to device buffer.
     *                          Buffer contains a strided set of matrices each of dimension (ldb, m). Leading (m, m)
     *                          of each buffer must contain the data of B's.
     * @param[in] ldb           Leading dimension of each matrix of B.
     * @param[in] strideB       Stride in elements between consecutive matrices of B. Must be at least (ldb * m).
     * @param[in] batchCount    Number of matrices to be batch processed.
     *
     * Workspace needed: query with kblas_syrk_batch_wsquery().
     *
     */
    int kblasSsyrk_batch_strided( kblasHandle_t handle,
                                  char uplo, char trans,
                                  const int m, const int n,
                                  const float alpha, const float* A, int lda, long strideA,
                                  const float beta,        float* B, int ldb, long strideB,
                                  int batchCount);

    /**
     * @brief Strided uniform-size double precision batched SYRK.
     *
     * @see kblasSsyrk_batch_strided() for details about params and workspace.
     */
    int kblasDsyrk_batch_strided( kblasHandle_t handle,
                                  char uplo, char trans,
                                  const int m, const int n,
                                  const double alpha, const double* A, int lda, long strideA,
                                  const double beta,        double* B, int ldb, long strideB,
                                  int batchCount);

    /**
     * @brief Strided uniform-size single-complex precision batched SYRK.
     *
     * @see kblasSsyrk_batch_strided() for details about params and workspace.
     */
    int kblasCsyrk_batch_strided( kblasHandle_t handle,
                                  char uplo, char trans,
                                  const int m, const int n,
                                  const hipFloatComplex alpha, const hipFloatComplex* A, int lda, long strideA,
                                  const hipFloatComplex beta,        hipFloatComplex* B, int ldb, long strideB,
                                  int batchCount);

    /**
     * @brief Strided uniform-size double-complex precision batched SYRK.
     *
     * @see kblasSsyrk_batch_strided() for details about params and workspace.
     */
    int kblasZsyrk_batch_strided( kblasHandle_t handle,
                                  char uplo, char trans,
                                  const int m, const int n,
                                  const hipDoubleComplex alpha, const hipDoubleComplex* A, int lda, long strideA,
                                  const hipDoubleComplex beta,        hipDoubleComplex* B, int ldb, long strideB,
                                  int batchCount);
/** @} */
/** @} */
#ifdef __cplusplus
}
#endif

//============================================================================
// batch TRSM

/**
 * @ingroup WSQUERY
 * @brief Workspace query for batched Non-strided TRSM routines.
 *
 * @param[in,out] handle      KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     side        KBLAS_Left/KBLAS_Right sided TRSM operation.
 * @param[in]     m           Number of rows of the matrices B.
 * @param[in]     n           Number of columns of the matrices B.
 * @param[in]     batchCount  Number of matrices to be processed.
 * @see kblasStrsm_batch() for details of params.
 */
void kblas_trsm_batch_wsquery(kblasHandle_t handle, char side, int m, int n, int batchCount);

/**
 * @ingroup WSQUERY
 * @brief Workspace query for batched Strided TRSM routines.
 *
 * @param[in,out] handle      KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     side        KBLAS_Left/KBLAS_Right sided TRSM operation.
 * @param[in]     m           Number of rows of the matrices B.
 * @param[in]     n           Number of columns of the matrices B.
 * @param[in]     batchCount  Number of matrices to be processed.
 * @see kblasStrsm_batch_strided() for details of params.
 */
void kblas_trsm_batch_strided_wsquery(kblasHandle_t handle, char side, int m, int n, int batchCount);

void kblas_trsm_batch_nonuniform_wsquery(kblasHandle_t handle);

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
                         const hipFloatComplex alpha,
                         const hipFloatComplex** A, int lda,
                               hipFloatComplex** B, int ldb,
                        int batchCount);

    int kblas_trsm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const hipDoubleComplex alpha,
                         const hipDoubleComplex** A, int lda,
                               hipDoubleComplex** B, int ldb,
                        int batchCount);

    //------------------------------------------------------------------------------
    // Non-Strided Non-uniform
    // if all maximum m/n are passed 0, they will be recomputed
    int kblas_trsm_batch( kblasHandle_t handle,
                          char side, char uplo, char trans, char diag,
                          int *m, int *n,
                          int max_m, int max_n,
                          float alpha,
                          float** A, int* lda,
                          float** B, int* ldb,
                          int batchCount);

    int kblas_trsm_batch( kblasHandle_t handle,
                          char side, char uplo, char trans, char diag,
                          int *m, int *n,
                          int max_m, int max_n,
                          double alpha,
                          double** A, int* lda,
                          double** B, int* ldb,
                          int batchCount);

    int kblas_trsm_batch( kblasHandle_t handle,
                          char side, char uplo, char trans, char diag,
                          int *m, int *n,
                          int max_m, int max_n,
                          hipFloatComplex alpha,
                          hipFloatComplex** A, int* lda,
                          hipFloatComplex** B, int* ldb,
                          int batchCount);

    int kblas_trsm_batch( kblasHandle_t handle,
                          char side, char uplo, char trans, char diag,
                          int *m, int *n,
                          int max_m, int max_n,
                          hipDoubleComplex alpha,
                          hipDoubleComplex** A, int* lda,
                          hipDoubleComplex** B, int* ldb,
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
                         const hipFloatComplex alpha,
                         const hipFloatComplex* A, int lda, long strideA,
                               hipFloatComplex* B, int ldb, long strideB,
                         int batchCount);

    int kblas_trsm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const hipDoubleComplex alpha,
                         const hipDoubleComplex* A, int lda, long strideA,
                               hipDoubleComplex* B, int ldb, long strideB,
                         int batchCount);
#endif

#ifdef __cplusplus
extern "C" {
#endif
/** @addtogroup C_API
*  @{
*/

    /**
     * @name Uniform-size batched TRSM routines
     * @{
     */
    //------------------------------------------------------------------------------
    // Non-Strided

    /**
     * @brief Non-Strided uniform-size single precision batched TRSM routine.
     *
     * @param[in] handle  KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] side    Specifies specifies whether op( A ) appears on the left or right of X as follows:
     *                    - KBLAS_Left: op( A )*X = alpha*B.
     *                    - KBLAS_Right: X*op( A ) = alpha*B.
     * @param[in] uplo    Specifies whether the upper or lower
     *                    triangular part of the matrices A is to be referenced as
     *                    follows:
     *                    - KBLAS_Lower: Only the lower triangular part of A is to be referenced.
     *                    - KBLAS_Upper: Only the upper triangular part of A is to be referenced.
     * @param[in] trans   Specifies the form of op( A ) to be used in the matrix multiplication as follows:
     *                    - KBLAS_NoTrans:    op( A ) = A.
     *                    - KBLAS_Trans:      op( A ) = A**T.
     * @param[in] diag    Specifies whether or not A is unit triangular as follows:
     *                    - KBLAS_NonUnit:  A is not assumed to be unit triangular.
     *                    - KBLAS_Unit:     A is assumed to be unit triangular.
     * @param[in] m       Specifies the number of rows of the matrices B. Must be at least zero.
     * @param[in] n       Specifies the number of columns of the matrices B. Must be at least zero.
     * @param[in] alpha         Specifies the scalar alpha.
     * @param[in] A_array       Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (lda, ka), where ka
     *                          is m when side is KBLAS_Left, n otherwise. Leading (ka, ka)
     *                          of each buffer must contain the data of A's.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in, out] B_array  Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (ldb, n). Leading (m, n)
     *                          of each buffer must contain the data of B's.
     * @param[in] ldb           Leading dimension of each matrix of B.
     * @param[in] batchCount    Number of matrices to be batch processed.
     *
     * Workspace needed: query with kblas_trsm_batch_wsquery().
     */
    int kblasStrsm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const float alpha,
                         const float** A_array, int lda,
                               float** B_array, int ldb,
                        int batchCount);

    /**
     * @brief Non-Strided uniform-size double precision batched TRSM routine.
     *
     * @see kblasStrsm_batch() for details about params and workspace.
     */
    int kblasDtrsm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const double alpha,
                         const double** A, int lda,
                               double** B, int ldb,
                        int batchCount);

    /**
     * @brief Non-Strided uniform-size single-complex precision batched TRSM routine.
     *
     * @see kblasStrsm_batch() for details about params and workspace.
     */
    int kblasCtrsm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const hipFloatComplex alpha,
                         const hipFloatComplex** A, int lda,
                               hipFloatComplex** B, int ldb,
                        int batchCount);

    /**
     * @brief Non-Strided uniform-size double-complex precision batched TRSM routine.
     *
     * @see kblasStrsm_batch() for details about params and workspace.
     */
    int kblasZtrsm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const hipDoubleComplex alpha,
                         const hipDoubleComplex** A, int lda,
                               hipDoubleComplex** B, int ldb,
                        int batchCount);

    //------------------------------------------------------------------------------
    // Strided

    /**
     * @brief Strided uniform-size single precision batched TRSM routine.
     *
     * @param[in] handle  KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] side    Specifies specifies whether op( A ) appears on the left or right of X as follows:
     *                    - KBLAS_Left: op( A )*X = alpha*B.
     *                    - KBLAS_Right: X*op( A ) = alpha*B.
     * @param[in] uplo    Specifies whether the upper or lower
     *                    triangular part of the matrices A is to be referenced as
     *                    follows:
     *                    - KBLAS_Lower: Only the lower triangular part of A is to be referenced.
     *                    - KBLAS_Upper: Only the upper triangular part of A is to be referenced.
     * @param[in] trans   Specifies the form of op( A ) to be used in the matrix multiplication as follows:
     *                    - KBLAS_NoTrans:    op( A ) = A.
     *                    - KBLAS_Trans:      op( A ) = A**T.
     * @param[in] diag    Specifies whether or not A is unit triangular as follows:
     *                    - KBLAS_NonUnit:  A is not assumed to be unit triangular.
     *                    - KBLAS_Unit:     A is assumed to be unit triangular.
     * @param[in] m       Specifies the number of rows of the matrices B. Must be at least zero.
     * @param[in] n       Specifies the number of columns of the matrices B. Must be at least zero.
     * @param[in] alpha         Specifies the scalar alpha.
     * @param[in] A             Host pointer to device buffer.
     *                          Buffer contains a strided set of matrices each of dimension (lda, ka), where ka
     *                          is m when side is KBLAS_Left, n otherwise. Leading (ka, ka)
     *                          of each buffer must contain the data of A's.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] strideA       Stride in elements between consecutive matrices of A. Must be at least (lda * ka).
     * @param[in, out] B        Host pointer to device buffer.
     *                          Buffer contains a strided set of matrices each of dimension (ldb, n). Leading (m, n)
     *                          of each buffer must contain the data of B's.
     * @param[in] ldb           Leading dimension of each matrix of B.
     * @param[in] strideB       Stride in elements between consecutive matrices of B. Must be at least (ldb * n).
     * @param[in] batchCount    Number of matrices to be batch processed.
     *
     * Workspace needed: query with kblas_trsm_batch_strided_wsquery().
     */
    int kblasStrsm_batch_strided(kblasHandle_t handle,
                                 char side, char uplo, char trans, char diag,
                                 const int m, const int n,
                                 const float alpha,
                                 const float* A, int lda, long strideA,
                                       float* B, int ldb, long strideB,
                                 int batchCount);

    /**
     * @brief Strided uniform-size double precision batched TRSM routine.
     *
     * @see kblasStrsm_batch_strided() for details about params and workspace.
     */
    int kblasDtrsm_batch_strided(kblasHandle_t handle,
                                 char side, char uplo, char trans, char diag,
                                 const int m, const int n,
                                 const double alpha,
                                 const double* A, int lda, long strideA,
                                       double* B, int ldb, long strideB,
                                 int batchCount);

    /**
     * @brief Strided uniform-size single-complex precision batched TRSM routine.
     *
     * @see kblasStrsm_batch_strided() for details about params and workspace.
     */
    int kblasCtrsm_batch_strided(kblasHandle_t handle,
                                 char side, char uplo, char trans, char diag,
                                 const int m, const int n,
                                 const hipFloatComplex alpha,
                                 const hipFloatComplex* A, int lda, long strideA,
                                       hipFloatComplex* B, int ldb, long strideB,
                                 int batchCount);

    /**
     * @brief Strided uniform-size double-complex precision batched TRSM routine.
     *
     * @see kblasStrsm_batch_strided() for details about params and workspace.
     */
    int kblasZtrsm_batch_strided(kblasHandle_t handle,
                                 char side, char uplo, char trans, char diag,
                                 const int m, const int n,
                                 const hipDoubleComplex alpha,
                                 const hipDoubleComplex* A, int lda, long strideA,
                                       hipDoubleComplex* B, int ldb, long strideB,
                                 int batchCount);
/** @} */
/** @} */
#ifdef __cplusplus
}
#endif

//============================================================================
// batch TRMM

/**
 * @ingroup WSQUERY
 * @brief Workspace query for batched Non-strided TRMM routines.
 *
 * @param[in,out] handle      KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     side        KBLAS_Left/KBLAS_Right sided TRMM operation.
 * @param[in]     m           Number of rows of the matrices B.
 * @param[in]     n           Number of columns of the matrices B.
 * @param[in]     batchCount  Number of matrices to be processed.
 * @see kblasStrmm_batch() for details of params.
 */
void kblas_trmm_batch_wsquery(kblasHandle_t handle, char side, int m, int n, int batchCount);

/**
 * @ingroup WSQUERY
 * @brief Workspace query for batched Strided TRMM routines.
 *
 * @param[in,out] handle      KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     side        KBLAS_Left/KBLAS_Right sided TRSM operation.
 * @param[in]     m           Number of rows of the matrices B.
 * @param[in]     n           Number of columns of the matrices B.
 * @param[in]     batchCount  Number of matrices to be processed.
 * @see kblasStrmm_batch_strided() for details of params.
 */
void kblas_trmm_batch_strided_wsquery(kblasHandle_t handle, char side, int m, int n, int batchCount);

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
                         const hipFloatComplex alpha,
                         const hipFloatComplex** A, int lda,
                               hipFloatComplex** B, int ldb,
                        int batchCount);

    int kblas_trmm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const hipDoubleComplex alpha,
                         const hipDoubleComplex** A, int lda,
                               hipDoubleComplex** B, int ldb,
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
                         const hipFloatComplex alpha,
                         const hipFloatComplex* A, int lda, long strideA,
                               hipFloatComplex* B, int ldb, long strideB,
                         int batchCount);

    int kblas_trmm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const hipDoubleComplex alpha,
                         const hipDoubleComplex* A, int lda, long strideA,
                               hipDoubleComplex* B, int ldb, long strideB,
                         int batchCount);
#endif

#ifdef __cplusplus
extern "C" {
#endif
/** @addtogroup C_API
*  @{
*/

    /**
     * @name Uniform-size batched TRMM routines
     * @{
     */
    //------------------------------------------------------------------------------
    // Non-Strided

    /**
     * @brief Non-Strided uniform-size single precision batched TRMM routine.
     *
     * @param[in] handle  KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] side    Specifies specifies whether op( A ) appears on the left or right of X as follows:
     *                    - KBLAS_Left:     B := alpha*op( A )*B.
     *                    - KBLAS_Right:    B := alpha*B*op( A ).
     * @param[in] uplo    Specifies whether the upper or lower triangular part of the matrices A is to be referenced as follows:
     *                    - KBLAS_Lower: Only the lower triangular part of A is to be referenced.
     *                    - KBLAS_Upper: Only the upper triangular part of A is to be referenced.
     * @param[in] trans   Specifies the form of op( A ) to be used in the matrix multiplication as follows:
     *                    - KBLAS_NoTrans:    op( A ) = A.
     *                    - KBLAS_Trans:      op( A ) = A**T.
     * @param[in] diag    Specifies whether or not A is unit triangular as follows:
     *                    - KBLAS_NonUnit:  A is not assumed to be unit triangular.
     *                    - KBLAS_Unit:     A is assumed to be unit triangular.
     * @param[in] m       Specifies the number of rows of the matrices B. Must be at least zero.
     * @param[in] n       Specifies the number of columns of the matrices B. Must be at least zero.
     * @param[in] alpha         Specifies the scalar alpha.
     * @param[in] A_array       Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (lda, ka), where ka
     *                          is m when side is KBLAS_Left, n otherwise. Leading (ka, ka)
     *                          of each buffer must contain the data of A's.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in, out] B_array  Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (ldb, n). Leading (m, n)
     *                          of each buffer must contain the data of B's.
     * @param[in] ldb           Leading dimension of each matrix of B.
     * @param[in] batchCount    Number of matrices to be batch processed.
     *
     * Workspace needed: query with kblas_trmm_batch_wsquery().
     */
    int kblasStrmm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const float alpha,
                         const float** A_array, int lda,
                               float** B_array, int ldb,
                        int batchCount);

    /**
     * @brief Non-Strided uniform-size double precision batched TRMM routine.
     *
     * @see kblasStrmm_batch() for details about params and workspace.
     */
    int kblasDtrmm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const double alpha,
                         const double** A_array, int lda,
                               double** B_array, int ldb,
                        int batchCount);

    /**
     * @brief Non-Strided uniform-size single-complex precision batched TRMM routine.
     *
     * @see kblasStrmm_batch() for details about params and workspace.
     */
    int kblasCtrmm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const hipFloatComplex alpha,
                         const hipFloatComplex** A_array, int lda,
                               hipFloatComplex** B_array, int ldb,
                        int batchCount);

    /**
     * @brief Non-Strided uniform-size double-complex precision batched TRMM routine.
     *
     * @see kblasStrmm_batch() for details about params and workspace.
     */
    int kblasZtrmm_batch(kblasHandle_t handle,
                         char side, char uplo, char trans, char diag,
                         const int m, const int n,
                         const hipDoubleComplex alpha,
                         const hipDoubleComplex** A_array, int lda,
                               hipDoubleComplex** B_array, int ldb,
                        int batchCount);

    //------------------------------------------------------------------------------
    // Strided

    /**
     * @brief Non-Strided uniform-size single precision batched TRMM routine.
     *
     * @param[in] handle  KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] side    Specifies specifies whether op( A ) appears on the left or right of X as follows:
     *                    - KBLAS_Left:     B := alpha*op( A )*B.
     *                    - KBLAS_Right:    B := alpha*B*op( A ).
     * @param[in] uplo    Specifies whether the upper or lower triangular part of the matrices A is to be referenced as follows:
     *                    - KBLAS_Lower: Only the lower triangular part of A is to be referenced.
     *                    - KBLAS_Upper: Only the upper triangular part of A is to be referenced.
     * @param[in] trans   Specifies the form of op( A ) to be used in the matrix multiplication as follows:
     *                    - KBLAS_NoTrans:    op( A ) = A.
     *                    - KBLAS_Trans:      op( A ) = A**T.
     * @param[in] diag    Specifies whether or not A is unit triangular as follows:
     *                    - KBLAS_NonUnit:  A is not assumed to be unit triangular.
     *                    - KBLAS_Unit:     A is assumed to be unit triangular.
     * @param[in] m       Specifies the number of rows of the matrices B. Must be at least zero.
     * @param[in] n       Specifies the number of columns of the matrices B. Must be at least zero.
     * @param[in] alpha         Specifies the scalar alpha.
     * @param[in] A             Host pointer to device buffer.
     *                          Buffer contains a strided set of matrices each of dimension (lda, ka), where ka
     *                          is m when side is KBLAS_Left, n otherwise. Leading (ka, ka)
     *                          of each buffer must contain the data of A's.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] strideA       Stride in elements between consecutive matrices of A. Must be at least (lda * ka).
     * @param[in, out] B        Host pointer to device buffer.
     *                          Buffer contains a strided set of matrices each of dimension (ldb, n). Leading (m, n)
     *                          of each buffer must contain the data of B's.
     * @param[in] ldb           Leading dimension of each matrix of B.
     * @param[in] strideB       Stride in elements between consecutive matrices of B. Must be at least (ldb * n).
     * @param[in] batchCount    Number of matrices to be batch processed.
     *
     * Workspace needed: query with kblas_trmm_batch_strided_wsquery().
     */
    int kblasStrmm_batch_strided(kblasHandle_t handle,
                                 char side, char uplo, char trans, char diag,
                                 const int m, const int n,
                                 const float alpha,
                                 const float* A, int lda, long strideA,
                                       float* B, int ldb, long strideB,
                                 int batchCount);

    /**
     * @brief Strided uniform-size double precision batched TRMM routine.
     *
     * @see kblasStrmm_batch_strided() for details about params and workspace.
     */
    int kblasDtrmm_batch_strided(kblasHandle_t handle,
                                 char side, char uplo, char trans, char diag,
                                 const int m, const int n,
                                 const double alpha,
                                 const double* A, int lda, long strideA,
                                       double* B, int ldb, long strideB,
                                 int batchCount);

    /**
     * @brief Strided uniform-size single-complex precision batched TRMM routine.
     *
     * @see kblasStrmm_batch_strided() for details about params and workspace.
     */
    int kblasCtrmm_batch_strided(kblasHandle_t handle,
                                 char side, char uplo, char trans, char diag,
                                 const int m, const int n,
                                 const hipFloatComplex alpha,
                                 const hipFloatComplex* A, int lda, long strideA,
                                       hipFloatComplex* B, int ldb, long strideB,
                                 int batchCount);

    /**
     * @brief Strided uniform-size double-complex precision batched TRMM routine.
     *
     * @see kblasStrmm_batch_strided() for details about params and workspace.
     */
    int kblasZtrmm_batch_strided(kblasHandle_t handle,
                                 char side, char uplo, char trans, char diag,
                                 const int m, const int n,
                                 const hipDoubleComplex alpha,
                                 const hipDoubleComplex* A, int lda, long strideA,
                                       hipDoubleComplex* B, int ldb, long strideB,
                                 int batchCount);
/** @} */
/** @} */
#ifdef __cplusplus
}
#endif

//============================================================================
// batch POTRF

/**
 * @ingroup WSQUERY
 * @brief Workspace query for batched Non-strided POTRF routines.
 *
 * @param[in,out] handle      KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     n           Number of rows and columns of the matrices A.
 * @param[in]     batchCount  Number of matrices to be processed.
 * @see kblasSpotrf_batch() for details of params.
 */
void kblas_potrf_batch_wsquery(kblasHandle_t handle, const int n, int batchCount);

/**
 * @ingroup WSQUERY
 * @brief Workspace query for batched Strided POTRF routines.
 *
 * @param[in,out] handle      KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     n           Number of rows and columns of the matrices A.
 * @param[in]     batchCount  Number of matrices to be processed.
 * @see kblasSpotrf_batch_strided() for details of params.
 */
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
                          hipFloatComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_potrf_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          hipDoubleComplex** A, int lda,
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
                          hipFloatComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_potrf_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          hipDoubleComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);
#endif

#ifdef __cplusplus
extern "C" {
#endif
/** @addtogroup C_API
*  @{
*/

    /**
     * @name Uniform-size batched POTRF routines
     * @{
     */
    //------------------------------------------------------------------------------
    // Non-Strided

    /**
     * @brief Non-Strided uniform-size single precision batched POTRF routine.
     *
     * @param[in] handle        KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] uplo          Specifies whether the upper or lower triangular part of the matrices A is to be referenced as follows:
     *                          - KBLAS_Lower: Only the lower triangular part of A is to be referenced.
     *                          - KBLAS_Upper: Only the upper triangular part of A is to be referenced.
     * @param[in] n             Specifies the number of rows and columns of the matrices A. Must be at least zero.
     * @param[in,out] A_array   Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (lda, n). Leading (n, n) of each buffer must contain the data of A's.
     *                          On exit, each buffer contains the Lower / Upper factor of corresponding matrix A.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] batchCount    Number of matrices to be batch processed.
     * @param[out] info_array   returns success / failure of each operation (0=Success, -i=ith parameter is wrong, i=ith leading minor is not positive dfinite).
     *
     * Workspace needed: query with kblas_potrf_batch_wsquery().
     */
    int kblasSpotrf_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          float** A_array, int lda,
                          int batchCount,
                          int *info_array);

    /**
     * @brief Non-Strided uniform-size double precision batched POTRF routine.
     *
     * @see kblasSpotrf_batch() for details about params and workspace.
     */
    int kblasDpotrf_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          double** A_array, int lda,
                          int batchCount,
                          int *info_array);

    /**
     * @brief Non-Strided uniform-size single-complex precision batched POTRF routine.
     *
     * @see kblasSpotrf_batch() for details about params and workspace.
     */
    int kblasCpotrf_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          hipFloatComplex** A_array, int lda,
                          int batchCount,
                          int *info_array);

    /**
     * @brief Non-Strided uniform-size double-complex precision batched POTRF routine.
     *
     * @see kblasSpotrf_batch() for details about params and workspace.
     */
    int kblasZpotrf_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          hipDoubleComplex** A_array, int lda,
                          int batchCount,
                          int *info_array);
    //------------------------------------------------------------------------------
    // Strided

    /**
     * @brief Strided uniform-size single precision batched POTRF routine.
     *
     * @param[in] handle        KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] uplo          Specifies whether the upper or lower triangular part of the matrices A is to be referenced as follows:
     *                          - KBLAS_Lower: Only the lower triangular part of A is to be referenced.
     *                          - KBLAS_Upper: Only the upper triangular part of A is to be referenced.
     * @param[in] n             Specifies the number of rows and columns of the matrices A. Must be at least zero.
     * @param[in,out] A         Host pointer to device buffer.
     *                          Buffer contains a strided set of matrices each of dimension (lda, n). Leading (n, n) of each buffer must contain the data of A's.
     *                          On exit, each buffer contains the Lower / Upper factor of corresponding matrix A.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] strideA       Stride in elements between consecutive matrices of A. Must be at least (lda * n).
     * @param[in] batchCount    Number of matrices to be batch processed.
     * @param[out] info_array   returns success / failure of each operation (0=Success, -i=ith parameter is wrong, i=ith leading minor is not positive dfinite).
     *
     * Workspace needed: query with kblas_potrf_batch_strided_wsquery().
     */
    int kblasSpotrf_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  float* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    /**
     * @brief Strided uniform-size double precision batched POTRF routine.
     *
     * @see kblasSpotrf_batch_strided() for details about params and workspace.
     */
    int kblasDpotrf_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  double* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    /**
     * @brief Strided uniform-size single-complex precision batched POTRF routine.
     *
     * @see kblasSpotrf_batch_strided() for details about params and workspace.
     */
    int kblasCpotrf_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  hipFloatComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    /**
     * @brief Strided uniform-size double-complex precision batched POTRF routine.
     *
     * @see kblasSpotrf_batch_strided() for details about params and workspace.
     */
    int kblasZpotrf_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  hipDoubleComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

/** @} */
/** @} */
#ifdef __cplusplus
}
#endif


//============================================================================
// batch LAUUM

/**
 * @ingroup WSQUERY
 * @brief Workspace query for batched Non-strided LAUUM routines.
 *
 * @param[in,out] handle      KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     n           Number of rows and columns of the matrices A.
 * @param[in]     batchCount  Number of matrices to be processed.
 * @see kblasSlauum_batch() for details of params.
 */
void kblas_lauum_batch_wsquery(kblasHandle_t handle, const int n, int batchCount);

/**
 * @ingroup WSQUERY
 * @brief Workspace query for batched Strided LAUUM routines.
 *
 * @param[in,out] handle      KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     n           Number of rows and columns of the matrices A.
 * @param[in]     batchCount  Number of matrices to be processed.
 * @see kblasSlauum_batch_strided() for details of params.
 */
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
                          hipFloatComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_lauum_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          hipDoubleComplex** A, int lda,
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
                          hipFloatComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_lauum_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          hipDoubleComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);
#endif

#ifdef __cplusplus
extern "C" {
#endif
/** @addtogroup C_API
*  @{
*/

    /**
     * @name Uniform-size batched LAUUM routines
     * @{
     */
    //------------------------------------------------------------------------------
    // Non-Strided

    /**
     * @brief Non-Strided uniform-size single precision batched LAUUM routine.
     *
     * @param[in] handle        KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] uplo          Specifies whether the upper or lower triangular part of the matrices A is to be referenced as follows:
     *                          - KBLAS_Lower: Only the lower triangular part of A is to be referenced.
     *                          - KBLAS_Upper: Only the upper triangular part of A is to be referenced.
     * @param[in] n             Specifies the number of rows and columns of the matrices A. Must be at least zero.
     * @param[in,out] A_array   Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (lda, n). Leading (n, n) of each buffer must contain the data of A's.
     *                          On exit, each buffer contains the Lower / Upper factor of corresponding L*L**T / U**T*U of matrix A.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] batchCount    Number of matrices to be batch processed.
     * @param[out] info_array   returns success / failure of each operation.
     *
     * Workspace needed: query with kblas_lauum_batch_wsquery().
     */
    int kblasSlauum_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          float** A_array, int lda,
                          int batchCount,
                          int *info_array);

    /**
     * @brief Non-Strided uniform-size double precision batched LAUUM routine.
     *
     * @see kblasSlauum_batch() for details about params and workspace.
     */
    int kblasDlauum_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          double** A_array, int lda,
                          int batchCount,
                          int *info_array);

    /**
     * @brief Non-Strided uniform-size single-complex precision batched LAUUM routine.
     *
     * @see kblasSlauum_batch() for details about params and workspace.
     */
    int kblasClauum_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          hipFloatComplex** A_array, int lda,
                          int batchCount,
                          int *info_array);

    /**
     * @brief Non-Strided uniform-size double-complex precision batched LAUUM routine.
     *
     * @see kblasSlauum_batch() for details about params and workspace.
     */
    int kblasZlauum_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          hipDoubleComplex** A_array, int lda,
                          int batchCount,
                          int *info_array);
    //------------------------------------------------------------------------------
    // Strided

    /**
     * @brief Strided uniform-size single precision batched LAUUM routine.
     *
     * @param[in] handle        KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] uplo          Specifies whether the upper or lower triangular part of the matrices A is to be referenced as follows:
     *                          - KBLAS_Lower: Only the lower triangular part of A is to be referenced.
     *                          - KBLAS_Upper: Only the upper triangular part of A is to be referenced.
     * @param[in] n             Specifies the number of rows and columns of the matrices A. Must be at least zero.
     * @param[in,out] A         Host pointer to device buffer.
     *                          Buffer contains a strided set of matrices each of dimension (lda, n). Leading (n, n) of each buffer must contain the data of A's.
     *                          On exit, each buffer contains the Lower / Upper factor of corresponding L*L**T / U**T*U of matrix A.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] strideA       Stride in elements between consecutive matrices of A. Must be at least (lda * n).
     * @param[in] batchCount    Number of matrices to be batch processed.
     * @param[out] info_array   returns success / failure of each operation.
     *
     * Workspace needed: query with kblas_lauum_batch_strided_wsquery().
     */
    int kblasSlauum_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  float* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    /**
     * @brief Strided uniform-size double precision batched LAUUM routine.
     *
     * @see kblasSlauum_batch_strided() for details about params and workspace.
     */
    int kblasDlauum_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  double* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    /**
     * @brief Strided uniform-size single-complex precision batched LAUUM routine.
     *
     * @see kblasSlauum_batch_strided() for details about params and workspace.
     */
    int kblasClauum_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  hipFloatComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    /**
     * @brief Strided uniform-size double-complex precision batched LAUUM routine.
     *
     * @see kblasSlauum_batch_strided() for details about params and workspace.
     */
    int kblasZlauum_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  hipDoubleComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);
/** @} */
/** @} */
#ifdef __cplusplus
}
#endif

//============================================================================
// batch TRTRI

/**
 * @ingroup WSQUERY
 * @brief Workspace query for batched Non-strided TRTRI routines.
 *
 * @param[in,out] handle      KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     n           Number of rows and columns of the matrices A.
 * @param[in]     batchCount  Number of matrices to be processed.
 * @see kblasStrtri_batch() for details of params.
 */
void kblas_trtri_batch_wsquery(kblasHandle_t handle, const int n, int batchCount);

/**
 * @ingroup WSQUERY
 * @brief Workspace query for batched Strided TRTRI routines.
 *
 * @param[in,out] handle      KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     n           Number of rows and columns of the matrices A.
 * @param[in]     batchCount  Number of matrices to be processed.
 * @see kblasStrtri_batch_strided() for details of params.
 */
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
                          hipFloatComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_trtri_batch(kblasHandle_t handle,
                          char uplo, char diag,
                          const int n,
                          hipDoubleComplex** A, int lda,
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
                          hipFloatComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_trtri_batch(kblasHandle_t handle,
                          char uplo, char diag,
                          const int n,
                          hipDoubleComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);
#endif

#ifdef __cplusplus
extern "C" {
#endif
/** @addtogroup C_API
*  @{
*/

    /**
     * @name Uniform-size batched TRTRI routines
     * @{
     */
    //------------------------------------------------------------------------------
    // Non-Strided

    /**
     * @brief Non-Strided uniform-size single precision batched TRTRI routine.
     *
     * @param[in] handle        KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] uplo          Specifies whether the upper or lower triangular part of the matrices A is to be referenced as follows:
     *                            - KBLAS_Lower: Only the lower triangular part of A is to be referenced.
     *                            - KBLAS_Upper: Only the upper triangular part of A is to be referenced.
     * @param[in] diag          Specifies whether or not A is unit triangular as follows:
     *                            - KBLAS_NonUnit:  A is not assumed to be unit triangular.
     *                            - KBLAS_Unit:     A is assumed to be unit triangular.
     * @param[in] n             Specifies the number of rows and columns of the matrices A. Must be at least zero.
     * @param[in,out] A_array   Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (lda, n). Leading (n, n) of each buffer must contain the data of A's.
     *                          On exit, each buffer contains the (triangular) inverse of the corresponding original matrix A, in the same storage format.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] batchCount    Number of matrices to be batch processed.
     * @param[out] info_array   returns success / failure of each operation.
     *
     * Workspace needed: query with kblas_trtri_batch_wsquery().
     */
    int kblasStrtri_batch(kblasHandle_t handle,
                          char uplo, char diag,
                          const int n,
                          float** A_array, int lda,
                          int batchCount,
                          int *info_array);

    /**
     * @brief Non-Strided uniform-size double precision batched TRTRI routine.
     *
     * @see kblasStrtri_batch() for details about params and workspace.
     */
    int kblasDtrtri_batch(kblasHandle_t handle,
                          char uplo, char diag,
                          const int n,
                          double** A_array, int lda,
                          int batchCount,
                          int *info_array);

    /**
     * @brief Non-Strided uniform-size single-complex precision batched TRTRI routine.
     *
     * @see kblasStrtri_batch() for details about params and workspace.
     */
    int kblasCtrtri_batch(kblasHandle_t handle,
                          char uplo, char diag,
                          const int n,
                          hipFloatComplex** A_array, int lda,
                          int batchCount,
                          int *info_array);

    /**
     * @brief Non-Strided uniform-size double-complex precision batched TRTRI routine.
     *
     * @see kblasStrtri_batch() for details about params and workspace.
     */
    int kblasZtrtri_batch(kblasHandle_t handle,
                          char uplo, char diag,
                          const int n,
                          hipDoubleComplex** A_array, int lda,
                          int batchCount,
                          int *info_array);
    //------------------------------------------------------------------------------
    // Strided

    /**
     * @brief Non-Strided uniform-size single precision batched TRTRI routine.
     *
     * @param[in] handle        KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] uplo          Specifies whether the upper or lower triangular part of the matrices A is to be referenced as follows:
     *                            - KBLAS_Lower: Only the lower triangular part of A is to be referenced.
     *                            - KBLAS_Upper: Only the upper triangular part of A is to be referenced.
     * @param[in] diag          Specifies whether or not A is unit triangular as follows:
     *                            - KBLAS_NonUnit:  A is not assumed to be unit triangular.
     *                            - KBLAS_Unit:     A is assumed to be unit triangular.
     * @param[in] n             Specifies the number of rows and columns of the matrices A. Must be at least zero.
     * @param[in,out] A         Host pointer to device buffer.
     *                          Buffer contains a strided set of matrices each of dimension (lda, n). Leading (n, n) of each buffer must contain the data of A's.
     *                          On exit, each buffer contains the (triangular) inverse of the corresponding original matrix A, in the same storage format.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] strideA       Stride in elements between consecutive matrices of A. Must be at least (lda * n).
     * @param[in] batchCount    Number of matrices to be batch processed.
     * @param[out] info_array   returns success / failure of each operation.
     *
     * Workspace needed: query with kblas_trtri_batch_strided_wsquery().
     */
    int kblasStrtri_batch_strided(kblasHandle_t handle,
                                  char uplo, char diag,
                                  const int n,
                                  float* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    /**
     * @brief Strided uniform-size double precision batched TRTRI routine.
     *
     * @see kblasStrtri_batch_strided() for details about params and workspace.
     */
    int kblasDtrtri_batch_strided(kblasHandle_t handle,
                                  char uplo, char diag,
                                  const int n,
                                  double* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    /**
     * @brief Strided uniform-size single-complex precision batched TRTRI routine.
     *
     * @see kblasStrtri_batch_strided() for details about params and workspace.
     */
    int kblasCtrtri_batch_strided(kblasHandle_t handle,
                                  char uplo, char diag,
                                  const int n,
                                  hipFloatComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);

    /**
     * @brief Strided uniform-size double-complex precision batched TRTRI routine.
     *
     * @see kblasStrtri_batch_strided() for details about params and workspace.
     */
    int kblasZtrtri_batch_strided(kblasHandle_t handle,
                                  char uplo, char diag,
                                  const int n,
                                  hipDoubleComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);
/** @} */
/** @} */
#ifdef __cplusplus
}
#endif


//============================================================================
// batch POTRS

/**
 * @ingroup WSQUERY
 * @brief Workspace query for batched Non-strided POTRS routines.
 *
 * @param[in,out] handle      KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     m           Number of rows of the matrices B.
 * @param[in]     n           Number of columns of the matrices B.
 * @param[in]     batchCount  Number of matrices to be processed.
 * @see kblasSpotrs_batch() for details of params.
 */
void kblas_potrs_batch_wsquery(kblasHandle_t handle, const int m, const int n, int batchCount);

/**
 * @ingroup WSQUERY
 * @brief Workspace query for batched Strided POTRS routines.
 *
 * @param[in,out] handle      KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     m           Number of rows of the matrices B.
 * @param[in]     n           Number of columns of the matrices B.
 * @param[in]     batchCount  Number of matrices to be processed.
 * @see kblasSpotrs_batch_strided() for details of params.
 */
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
                          const hipFloatComplex** A, int lda,
                                hipFloatComplex** B, int ldb,
                          int batchCount);

    int kblas_potrs_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          const hipDoubleComplex** A, int lda,
                                hipDoubleComplex** B, int ldb,
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
                          const hipFloatComplex* A, int lda, long strideA,
                                hipFloatComplex* B, int ldb, long strideB,
                          int batchCount);

    int kblas_potrs_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          const hipDoubleComplex* A, int lda, long strideA,
                                hipDoubleComplex* B, int ldb, long strideB,
                          int batchCount);
#endif

#ifdef __cplusplus
extern "C" {
#endif
/** @addtogroup C_API
*  @{
*/

    /**
     * @name Uniform-size batched POTRS routines
     * @{
     */
    //------------------------------------------------------------------------------
    // Non-Strided

    /**
     * @brief Non-Strided uniform-size single precision batched POTRS routine.
     *
     * @param[in] handle        KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] side          Specifies specifies whether A appears on the left or right of X as follows:
     *                            - KBLAS_Left:     B := A*X.
     *                            - KBLAS_Right:    B := X*A.
     * @param[in] uplo          Specifies whether the upper or lower triangular part of the matrices A is to be referenced as follows:
     *                            - KBLAS_Lower: Only the lower triangular part of A is to be referenced.
     *                            - KBLAS_Upper: Only the upper triangular part of A is to be referenced.
     * @param[in] m             Specifies the number of rows of the matrices B. Must be at least zero.
     * @param[in] n             Specifies the number of columns of the matrices B. Must be at least zero.
     * @param[in] A_array       Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (lda, n). Leading (n, n) of each buffer must contain the data of A's.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in,out] B_array   Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (ldb, n). Leading (m, n) of each buffer must contain the data of B's.
     *                          On exit, each buffer contains the solution of the corresponding system of linear equations, in the same storage format.
     * @param[in] ldb           Leading dimension of each matrix of B.
     * @param[in] batchCount    Number of matrices to be batch processed.
     *
     * Workspace needed: query with kblas_potrs_batch_wsquery().
     */
    int kblasSpotrs_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          const float** A_array, int lda,
                                float** B_array, int ldb,
                          int batchCount);

    /**
     * @brief Non-Strided uniform-size double precision batched POTRS routine.
     *
     * @see kblasSpotrs_batch() for details about params and workspace.
     */
    int kblasDpotrs_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          const double** A_array, int lda,
                                double** B_array, int ldb,
                          int batchCount);

    /**
     * @brief Non-Strided uniform-size single-complex precision batched POTRS routine.
     *
     * @see kblasSpotrs_batch() for details about params and workspace.
     */
    int kblasCpotrs_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          const hipFloatComplex** A_array, int lda,
                                hipFloatComplex** B_array, int ldb,
                          int batchCount);

    /**
     * @brief Non-Strided uniform-size double-complex precision batched POTRS routine.
     *
     * @see kblasSpotrs_batch() for details about params and workspace.
     */
    int kblasZpotrs_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          const hipDoubleComplex** A_array, int lda,
                                hipDoubleComplex** B_array, int ldb,
                          int batchCount);

    //------------------------------------------------------------------------------
    // Strided

    /**
     * @brief Strided uniform-size single precision batched POTRS routine.
     *
     * @param[in] handle        KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] side          Specifies specifies whether A appears on the left or right of X as follows:
     *                            - KBLAS_Left:     B := A*X.
     *                            - KBLAS_Right:    B := X*A.
     * @param[in] uplo          Specifies whether the upper or lower triangular part of the matrices A is to be referenced as follows:
     *                            - KBLAS_Lower: Only the lower triangular part of A is to be referenced.
     *                            - KBLAS_Upper: Only the upper triangular part of A is to be referenced.
     * @param[in] m             Specifies the number of rows of the matrices B. Must be at least zero.
     * @param[in] n             Specifies the number of columns of the matrices B. Must be at least zero.
     * @param[in] A             Host pointer to device buffer.
     *                          Buffer contains a strided set of dimension (lda, n). Leading (n, n) of each buffer must contain the data of A's.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] strideA       Stride in elements between consecutive matrices of A. Must be at least (lda * n).
     * @param[in,out] B         Host pointer to device buffer.
     *                          Buffer contains a strided set of dimension (ldb, n). Leading (m, n) of each buffer must contain the data of B's.
     *                          On exit, each buffer contains the solution of the corresponding system of linear equations, in the same storage format.
     * @param[in] ldb           Leading dimension of each matrix of B.
     * @param[in] strideB       Stride in elements between consecutive matrices of B. Must be at least (ldb * n).
     * @param[in] batchCount    Number of matrices to be batch processed.
     *
     * Workspace needed: query with kblas_potrs_batch_wsquery().
     */
    int kblasSpotrs_batch_strided(kblasHandle_t handle,
                                  char side, char uplo,
                                  const int m, const int n,
                                  const float* A, int lda, long strideA,
                                        float* B, int ldb, long strideB,
                                  int batchCount);

    /**
     * @brief Strided uniform-size double precision batched POTRS routine.
     *
     * @see kblasSpotrs_batch_strided() for details about params and workspace.
     */
    int kblasDpotrs_batch_strided(kblasHandle_t handle,
                                  char side, char uplo,
                                  const int m, const int n,
                                  const double* A, int lda, long strideA,
                                        double* B, int ldb, long strideB,
                                  int batchCount);

    /**
     * @brief Strided uniform-size single-complex precision batched POTRS routine.
     *
     * @see kblasSpotrs_batch_strided() for details about params and workspace.
     */
    int kblasCpotrs_batch_strided(kblasHandle_t handle,
                                  char side, char uplo,
                                  const int m, const int n,
                                  const hipFloatComplex* A, int lda, long strideA,
                                        hipFloatComplex* B, int ldb, long strideB,
                                  int batchCount);

    /**
     * @brief Strided uniform-size double-complex precision batched POTRS routine.
     *
     * @see kblasSpotrs_batch_strided() for details about params and workspace.
     */
    int kblasZpotrs_batch_strided(kblasHandle_t handle,
                                  char side, char uplo,
                                  const int m, const int n,
                                  const hipDoubleComplex* A, int lda, long strideA,
                                        hipDoubleComplex* B, int ldb, long strideB,
                                  int batchCount);
/** @} */
/** @} */
#ifdef __cplusplus
}
#endif

//============================================================================
// batch POTRI

/**
 * @ingroup WSQUERY
 * @brief Workspace query for batched Non-strided POTRI routines.
 *
 * @param[in,out] handle      KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     n           Number of rows and columns of the matrices A.
 * @param[in]     batchCount  Number of matrices to be processed.
 * @see kblasSpotri_batch() for details of params.
 */
void kblas_potri_batch_wsquery(kblasHandle_t handle, const int n, int batchCount);

/**
 * @ingroup WSQUERY
 * @brief Workspace query for batched Strided POTRI routines.
 *
 * @param[in,out] handle      KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     n           Number of rows and columns of the matrices A.
 * @param[in]     batchCount  Number of matrices to be processed.
 * @see kblasSpotri_batch_strided() for details of params.
 */
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
                          hipFloatComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_potri_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          hipDoubleComplex** A, int lda,
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
                          hipFloatComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_potri_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          hipDoubleComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);
#endif

#ifdef __cplusplus
extern "C" {
#endif
/** @addtogroup C_API
*  @{
*/

    /**
     * @name Uniform-size batched POTRI routines
     * @{
     */
    //------------------------------------------------------------------------------
    // Non-Strided

    /**
     * @brief Non-Strided uniform-size single precision batched POTRI routine.
     *
     * @param[in] handle        KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] uplo          Specifies whether the upper or lower triangular part of the matrices A is to be referenced as follows:
     *                          - KBLAS_Lower: Only the lower triangular part of A is to be referenced.
     *                          - KBLAS_Upper: Only the upper triangular part of A is to be referenced.
     * @param[in] n             Specifies the number of rows and columns of the matrices A. Must be at least zero.
     * @param[in,out] A_array   Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (lda, n). Leading (n, n) of each buffer must contain the data of A's.
     *                          On exit, each buffer contains the Lower / Upper inverse factor of corresponding matrix A.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] batchCount    Number of matrices to be batch processed.
     * @param[out] info_array   returns success / failure of each operation (0=Success, -i=ith parameter is wrong, i=ith leading minor is not positive dfinite).
     *
     * Workspace needed: query with kblas_potri_batch_wsquery().
     */
    int kblasSpotri_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          float** A_array, int lda,
                          int batchCount,
                          int *info_array);

    /**
     * @brief Non-Strided uniform-size double precision batched POTRI routine.
     *
     * @see kblasSpotri_batch() for details about params and workspace.
     */
    int kblasDpotri_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          double** A_array, int lda,
                          int batchCount,
                          int *info_array);

    /**
     * @brief Non-Strided uniform-size single-complex precision batched POTRI routine.
     *
     * @see kblasSpotri_batch() for details about params and workspace.
     */
    int kblasCpotri_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          hipFloatComplex** A_array, int lda,
                          int batchCount,
                          int *info_array);

    /**
     * @brief Non-Strided uniform-size double-complex precision batched POTRI routine.
     *
     * @see kblasSpotri_batch() for details about params and workspace.
     */
    int kblasZpotri_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          hipDoubleComplex** A_array, int lda,
                          int batchCount,
                          int *info_array);
    //------------------------------------------------------------------------------
    // Strided

    /**
     * @brief Strided uniform-size single precision batched POTRI routine.
     *
     * @param[in] handle        KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] uplo          Specifies whether the upper or lower triangular part of the matrices A is to be referenced as follows:
     *                          - KBLAS_Lower: Only the lower triangular part of A is to be referenced.
     *                          - KBLAS_Upper: Only the upper triangular part of A is to be referenced.
     * @param[in] n             Specifies the number of rows and columns of the matrices A. Must be at least zero.
     * @param[in,out] A         Host pointer to device buffer.
     *                          Buffer contains a strided set of matrices each of dimension (lda, n). Leading (n, n) of each buffer must contain the data of A's.
     *                          On exit, each buffer contains the Lower / Upper inverse factor of corresponding matrix A.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] strideA       Stride in elements between consecutive matrices of A. Must be at least (lda * n).
     * @param[in] batchCount    Number of matrices to be batch processed.
     * @param[out] info_array   returns success / failure of each operation (0=Success, -i=ith parameter is wrong, i=ith leading minor is not positive dfinite).
     *
     * Workspace needed: query with kblas_potri_batch_strided_wsquery().
     */
    int kblasSpotri_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  float* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);
    /**
     * @brief Strided uniform-size double precision batched POTRI routine.
     *
     * @see kblasSpotri_batch_strided() for details about params and workspace.
     */
    int kblasDpotri_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  double* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);
    /**
     * @brief Strided uniform-size single-complex precision batched POTRI routine.
     *
     * @see kblasSpotri_batch() for details about params and workspace.
     */
    int kblasCpotri_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  hipFloatComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);
    /**
     * @brief Strided uniform-size double-complex precision batched POTRI routine.
     *
     * @see kblasSpotri_batch_strided() for details about params and workspace.
     */
    int kblasZpotri_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  hipDoubleComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);
/** @} */
/** @} */
#ifdef __cplusplus
}
#endif

//============================================================================
// batch POTRI

/**
 * @ingroup WSQUERY
 * @brief Workspace query for batched Non-strided POTI routines.
 *
 * @param[in,out] handle      KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     n           Number of rows and columns of the matrices A.
 * @param[in]     batchCount  Number of matrices to be processed.
 * @see kblasSpoti_batch() for details of params.
 */
void kblas_poti_batch_wsquery(kblasHandle_t handle, const int n, int batchCount);

/**
 * @ingroup WSQUERY
 * @brief Workspace query for batched Strided POTI routines.
 *
 * @param[in,out] handle      KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     n           Number of rows and columns of the matrices A.
 * @param[in]     batchCount  Number of matrices to be processed.
 * @see kblasSpoti_batch_strided() for details of params.
 */
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
                          hipFloatComplex** A, int lda,
                          int batchCount,
                          int *info_array);

    int kblas_poti_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          hipDoubleComplex** A, int lda,
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
                          hipFloatComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);

    int kblas_poti_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          hipDoubleComplex* A, int lda, long strideA,
                          int batchCount,
                          int *info_array);
#endif

#ifdef __cplusplus
extern "C" {
#endif
/** @addtogroup C_API
*  @{
*/

    /**
     * @name Uniform-size batched POTI routines
     * @{
     */
    //------------------------------------------------------------------------------
    // Non-Strided

    /**
     * @brief Non-Strided uniform-size single precision batched POTI routine.
     *
     * @param[in] handle        KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] uplo          Specifies whether the upper or lower triangular part of the matrices A is to be referenced as follows:
     *                          - KBLAS_Lower: Only the lower triangular part of A is to be referenced.
     *                          - KBLAS_Upper: Only the upper triangular part of A is to be referenced.
     * @param[in] n             Specifies the number of rows and columns of the matrices A. Must be at least zero.
     * @param[in,out] A_array   Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (lda, n). Leading (n, n) of each buffer must contain the data of A's.
     *                          On exit, each buffer contains the Lower / Upper inverse factor of corresponding matrix A.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] batchCount    Number of matrices to be batch processed.
     * @param[out] info_array   returns success / failure of each operation (0=Success, -i=ith parameter is wrong, i=ith leading minor is not positive dfinite).
     *
     * Workspace needed: query with kblas_poti_batch_wsquery().
     */
    int kblasSpoti_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          float** A_array, int lda,
                          int batchCount,
                          int *info_array);
    /**
     * @brief Non-Strided uniform-size double precision batched POTI routine.
     *
     * @see kblasSpoti_batch() for details about params and workspace.
     */
    int kblasDpoti_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          double** A_array, int lda,
                          int batchCount,
                          int *info_array);
    /**
     * @brief Non-Strided uniform-size single-complex precision batched POTI routine.
     *
     * @see kblasSpoti_batch() for details about params and workspace.
     */
    int kblasCpoti_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          hipFloatComplex** A_array, int lda,
                          int batchCount,
                          int *info_array);
    /**
     * @brief Non-Strided uniform-size double-complex precision batched POTI routine.
     *
     * @see kblasSpoti_batch() for details about params and workspace.
     */
    int kblasZpoti_batch(kblasHandle_t handle,
                          char uplo,
                          const int n,
                          hipDoubleComplex** A_array, int lda,
                          int batchCount,
                          int *info_array);
    //------------------------------------------------------------------------------
    // Strided

    /**
     * @brief Strided uniform-size single precision batched POTI routine.
     *
     * @param[in] handle        KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] uplo          Specifies whether the upper or lower triangular part of the matrices A is to be referenced as follows:
     *                          - KBLAS_Lower: Only the lower triangular part of A is to be referenced.
     *                          - KBLAS_Upper: Only the upper triangular part of A is to be referenced.
     * @param[in] n             Specifies the number of rows and columns of the matrices A. Must be at least zero.
     * @param[in,out] A         Host pointer to device buffer.
     *                          Buffer contains a strided set of matrices each of dimension (lda, n). Leading (n, n) of each buffer must contain the data of A's.
     *                          On exit, each buffer contains the Lower / Upper inverse factor of corresponding matrix A.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] strideA       Stride in elements between consecutive matrices of A. Must be at least (lda * n).
     * @param[in] batchCount    Number of matrices to be batch processed.
     * @param[out] info_array   returns success / failure of each operation (0=Success, -i=ith parameter is wrong, i=ith leading minor is not positive dfinite).
     *
     * Workspace needed: query with kblas_poti_batch_strided_wsquery().
     */
    int kblasSpoti_batch_strided( kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  float* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);
    /**
     * @brief Strided uniform-size double precision batched POTI routine.
     *
     * @see kblasSpoti_batch_strided() for details about params and workspace.
     */
    int kblasDpoti_batch_strided( kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  double* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);
    /**
     * @brief Strided uniform-size single-complex precision batched POTI routine.
     *
     * @see kblasSpoti_batch_strided() for details about params and workspace.
     */
    int kblasCpoti_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  hipFloatComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);
    /**
     * @brief Strided uniform-size double-complex precision batched POTI routine.
     *
     * @see kblasSpoti_batch_strided() for details about params and workspace.
     */
    int kblasZpoti_batch_strided(kblasHandle_t handle,
                                  char uplo,
                                  const int n,
                                  hipDoubleComplex* A, int lda, long strideA,
                                  int batchCount,
                                  int *info_array);
/** @} */
/** @} */
#ifdef __cplusplus
}
#endif

//============================================================================
// batch POSV

/**
 * @ingroup WSQUERY
 * @brief Workspace query for batched Non-strided POSV routines.
 *
 * @param[in,out] handle      KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     side        KBLAS_Left/KBLAS_Right sided POSV operation.
 * @param[in]     m           Number of rows of the matrices B.
 * @param[in]     n           Number of columns of the matrices B.
 * @param[in]     batchCount  Number of matrices to be processed.
 * @see kblasSposv_batch() for details of params.
 */
void kblas_posv_batch_wsquery(kblasHandle_t handle, char side, const int m, const int n, int batchCount);

/**
 * @ingroup WSQUERY
 * @brief Workspace query for batched Strided POSV routines.
 *
 * @param[in,out] handle      KBLAS handle. On return, stores the needed workspace size in corresponding data field.
 * @param[in]     side        KBLAS_Left/KBLAS_Right sided POSV operation.
 * @param[in]     m           Number of rows of the matrices B.
 * @param[in]     n           Number of columns of the matrices B.
 * @param[in]     batchCount  Number of matrices to be processed.
 * @see kblasSposv_batch_strided() for details of params.
 */
void kblas_posv_batch_strided_wsquery(kblasHandle_t handle, char side, const int m, const int n, int batchCount);

#ifdef __cplusplus
    //------------------------------------------------------------------------------
    // Non-Strided
    int kblas_posv_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          float** A, int lda,
                          float** B, int ldb,
                          int batchCount,
                          int *info_array);

    int kblas_posv_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          double** A, int lda,
                          double** B, int ldb,
                          int batchCount,
                          int *info_array);

    int kblas_posv_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          hipFloatComplex** A, int lda,
                          hipFloatComplex** B, int ldb,
                          int batchCount,
                          int *info_array);

    int kblas_posv_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          hipDoubleComplex** A, int lda,
                          hipDoubleComplex** B, int ldb,
                          int batchCount,
                          int *info_array);

    //------------------------------------------------------------------------------
    // Strided
    int kblas_posv_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          float* A, int lda, long strideA,
                          float* B, int ldb, long strideB,
                          int batchCount,
                          int *info_array);

    int kblas_posv_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          double* A, int lda, long strideA,
                          double* B, int ldb, long strideB,
                          int batchCount,
                          int *info_array);

    int kblas_posv_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          hipFloatComplex* A, int lda, long strideA,
                          hipFloatComplex* B, int ldb, long strideB,
                          int batchCount,
                          int *info_array);

    int kblas_posv_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          hipDoubleComplex* A, int lda, long strideA,
                          hipDoubleComplex* B, int ldb, long strideB,
                          int batchCount,
                          int *info_array);
#endif

#ifdef __cplusplus
extern "C" {
#endif
/** @addtogroup C_API
*  @{
*/

    /**
     * @name Uniform-size batched POSV routines
     * @{
     */
    //------------------------------------------------------------------------------
    // Non-Strided

    /**
     * @brief Non-Strided uniform-size single precision batched POSV routine.
     *
     * @param[in] handle        KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] side          Specifies specifies whether A appears on the left or right of X as follows:
     *                            - KBLAS_Left:     B := A*X.
     *                            - KBLAS_Right:    B := X*A.
     * @param[in] uplo          Specifies whether the upper or lower triangular part of the matrices A is to be referenced as follows:
     *                            - KBLAS_Lower: Only the lower triangular part of A is to be referenced.
     *                            - KBLAS_Upper: Only the upper triangular part of A is to be referenced.
     * @param[in] m             Specifies the number of rows of the matrices B. Must be at least zero.
     * @param[in] n             Specifies the number of columns of the matrices B. Must be at least zero.
     * @param[in,out] A_array   Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (lda, n). Leading (n, n) of each buffer must contain the data of A's.
     *                          On exit, each buffer contains the Lower / Upper factor of corresponding matrix A.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in,out] B_array   Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (ldb, n). Leading (m, n) of each buffer must contain the data of B's.
     *                          On exit, each buffer contains the solution of the corresponding system of linear equations, in the same storage format.
     * @param[in] ldb           Leading dimension of each matrix of B.
     * @param[in] batchCount    Number of matrices to be batch processed.
     * @param[out] info_array   returns success / failure of each operation (0=Success, -i=ith parameter is wrong, i=ith leading minor is not positive dfinite).
     *
     * Workspace needed: query with kblas_posv_batch_wsquery().
     */
    int kblasSposv_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          float** A_array, int lda,
                          float** B_array, int ldb,
                          int batchCount,
                          int *info_array);
    /**
     * @brief Non-Strided uniform-size double precision batched POSV routine.
     *
     * @see kblasSposv_batch() for details about params and workspace.
     */
    int kblasDposv_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          double** A_array, int lda,
                          double** B_array, int ldb,
                          int batchCount,
                          int *info_array);
    /**
     * @brief Non-Strided uniform-size single-complex precision batched POSV routine.
     *
     * @see kblasSposv_batch() for details about params and workspace.
     */
    int kblasCposv_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          hipFloatComplex** A_array, int lda,
                          hipFloatComplex** B_array, int ldb,
                          int batchCount,
                          int *info_array);
    /**
     * @brief Non-Strided uniform-size double-complex precision batched POSV routine.
     *
     * @see kblasSposv_batch() for details about params and workspace.
     */
    int kblasZposv_batch(kblasHandle_t handle,
                          char side, char uplo,
                          const int m, const int n,
                          hipDoubleComplex** A_array, int lda,
                          hipDoubleComplex** B_array, int ldb,
                          int batchCount,
                          int *info_array);

    //------------------------------------------------------------------------------
    // Strided

    /**
     * @brief Strided uniform-size single precision batched POTRS routine.
     *
     * @param[in] handle        KBLAS handle, must hold enough workspace for successful operation.
     * @param[in] side          Specifies specifies whether A appears on the left or right of X as follows:
     *                            - KBLAS_Left:     B := A*X.
     *                            - KBLAS_Right:    B := X*A.
     * @param[in] uplo          Specifies whether the upper or lower triangular part of the matrices A is to be referenced as follows:
     *                            - KBLAS_Lower: Only the lower triangular part of A is to be referenced.
     *                            - KBLAS_Upper: Only the upper triangular part of A is to be referenced.
     * @param[in] m             Specifies the number of rows of the matrices B. Must be at least zero.
     * @param[in] n             Specifies the number of columns of the matrices B. Must be at least zero.
     * @param[in,out] A         Host pointer to device buffer.
     *                          Buffer contains a strided set of dimension (lda, n). Leading (n, n) of each buffer must contain the data of A's.
     *                          On exit, the buffer contains the Lower / Upper factor of each corresponding matrix A.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] strideA       Stride in elements between consecutive matrices of A. Must be at least (lda * n).
     * @param[in,out] B         Host pointer to device buffer.
     *                          Buffer contains a strided set of dimension (ldb, n). Leading (m, n) of each buffer must contain the data of B's.
     *                          On exit, the buffer contains the solution of each corresponding system of linear equations, in the same storage format.
     * @param[in] ldb           Leading dimension of each matrix of B.
     * @param[in] strideB       Stride in elements between consecutive matrices of B. Must be at least (ldb * n).
     * @param[in] batchCount    Number of matrices to be batch processed.
     * @param[out] info_array   returns success / failure of factorization of each operation (0=Success, -i=ith parameter is wrong, i=ith leading minor is not positive dfinite).
     *
     * Workspace needed: query with kblas_posv_batch_strided_wsquery().
     */
    int kblasSposv_batch_strided(kblasHandle_t handle,
                                  char side, char uplo,
                                  const int m, const int n,
                                  float* A, int lda, long strideA,
                                  float* B, int ldb, long strideB,
                                  int batchCount,
                                  int *info_array);
    /**
     * @brief Strided uniform-size double precision batched POSV routine.
     *
     * @see kblasSposv_batch_strided() for details about params and workspace.
     */
    int kblasDposv_batch_strided(kblasHandle_t handle,
                                  char side, char uplo,
                                  const int m, const int n,
                                  double* A, int lda, long strideA,
                                  double* B, int ldb, long strideB,
                                  int batchCount,
                                  int *info_array);
    /**
     * @brief Strided uniform-size single-complex precision batched POSV routine.
     *
     * @see kblasSposv_batch_strided() for details about params and workspace.
     */
    int kblasCposv_batch_strided(kblasHandle_t handle,
                                  char side, char uplo,
                                  const int m, const int n,
                                  hipFloatComplex* A, int lda, long strideA,
                                  hipFloatComplex* B, int ldb, long strideB,
                                  int batchCount,
                                  int *info_array);
    /**
     * @brief Strided uniform-size double-complex precision batched POSV routine.
     *
     * @see kblasSposv_batch_strided() for details about params and workspace.
     */
    int kblasZposv_batch_strided(kblasHandle_t handle,
                                  char side, char uplo,
                                  const int m, const int n,
                                  hipDoubleComplex* A, int lda, long strideA,
                                  hipDoubleComplex* B, int ldb, long strideB,
                                  int batchCount,
                                  int *info_array);
/** @} */
/** @} */
#ifdef __cplusplus
}
#endif

#endif // _KBLAS_BATCH_H_
