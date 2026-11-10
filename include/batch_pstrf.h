/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file include/batch_pstrf.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Wajih Halim Boukaram
 * @date 2020-12-10
 **/
 
#ifndef __BATCH_PSTRF_H__
#define __BATCH_PSTRF_H__

//############################################################################
//BATCH PSTRF routines
//############################################################################

/** @addtogroup C_API
*  @{
*/
#ifdef __cplusplus
extern "C" {
#endif
/**
 * @name Uniform-size batched PSTRF routines
 */
//@{
//------------------------------------------------------------------------------
// Strided interface
/**
 * @brief Strided uniform-size double precision batched pivoted Cholesky of matrices A: P^T A P = R^t * R
 *
 * @param[in] handle KBLAS handle 
 * @param[in] n Dimension of the matrix A. (m >= 0, m <= 64)
 * @param[in, out] A_strided Pointer to the batch of matrices A stored in strided format. On exit, the upper [n x n]
 * submatrix of each matrix A stores the upper triangular factor R
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] stride_a Stride of each matrix in A_strided. A[i] = A_strided + stride_a * i
 * @param[out] piv_strided Pointer to the batch of arrays piv stored in strided format. On exit, each array piv contains 
 * the pivot indices used during the factorization
 * @param[in] stride_piv Stride of each array in piv_strided. piv[i] = piv_strided + stride_piv * i; (stride_tau >= cols)
 * @param[out] ranks Pointer to array which will hold the detected ranks of the matrices (i.e., where the factorization breaks down due to rank deficiency)
 * @param[in] num_ops The size of the batched operation. 
 */
int kblasDpstrf_batch_strided(kblasHandle_t handle, int n, double* A_strided, int lda, int stride_a, int* piv_strided, int stride_piv, int* ranks, int num_ops);

/**
 * @brief Strided uniform-size single precision batched pivoted Cholesky of matrices A: P^T A P = R^t * R
 *
 * @param[in] handle KBLAS handle 
 * @param[in] n Dimension of the matrix A. (m >= 0, m <= 64)
 * @param[in, out] A_strided Pointer to the batch of matrices A stored in strided format. On exit, the upper [n x n]
 * submatrix of each matrix A stores the upper triangular factor R
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] stride_a Stride of each matrix in A_strided. A[i] = A_strided + stride_a * i
 * @param[out] piv_strided Pointer to the batch of arrays piv stored in strided format. On exit, each array piv contains 
 * the pivot indices used during the factorization
 * @param[in] stride_piv Stride of each array in piv_strided. piv[i] = piv_strided + stride_piv * i; (stride_tau >= cols)
 * @param[out] ranks Pointer to array which will hold the detected ranks of the matrices (i.e., where the factorization breaks down due to rank deficiency)
 * @param[in] num_ops The size of the batched operation. 
 */
int kblasSpstrf_batch_strided(kblasHandle_t handle, int n, float* A_strided, int lda, int stride_a, int* piv_strided, int stride_piv, int* ranks, int num_ops);

//------------------------------------------------------------------------------
// Array of pointers interface
/**
 * @brief Array of pointers uniform-size double precision batched pivoted Cholesky of matrices A: P^T A P = R^t * R
 * 
 * Array of pointers interface taking similar arguments to the strided interface; however, matrices and 
 * arrays are stored in arrays of pointers instead of strided access, so A[i] = A_array[i] and piv[i] = piv_array[i]
 * @see kblasDpstrf_batch_strided
 */
int kblasDpstrf_batch(kblasHandle_t handle, int n, double** A_array, int lda, int** piv_array, int* ranks, int num_ops);

//------------------------------------------------------------------------------
// Array of pointers interface
/**
 * @brief Array of pointers uniform-size single precision batched pivoted Cholesky of matrices A: P^T A P = R^t * R
 * 
 * Array of pointers interface taking similar arguments to the strided interface; however, matrices and 
 * arrays are stored in arrays of pointers instead of strided access, so A[i] = A_array[i] and piv[i] = piv_array[i]
 * @see kblasSpstrf_batch_strided
 */
int kblasSpstrf_batch(kblasHandle_t handle, int n, float** A_array, int lda, int** piv_array, int* ranks, int num_ops);

//@}
#ifdef __cplusplus
}
#endif
/** @} */

/** @addtogroup CPP_API
*  @{
*/
#ifdef __cplusplus
/**
 * @name Uniform-size batched PSTRF routines
 */
//@{
/**
 * @brief Array of pointers uniform-size double precision batched pivoted Cholesky of matrices A: P^T A P = R^t * R
 */
inline int kblas_pstrf_batch(kblasHandle_t handle, int n, double** A_array, int lda, int** piv_array, int* ranks, int num_ops)
{ return kblasDpstrf_batch(handle, n, A_array, lda, piv_array, ranks, num_ops); }
/**
 * @brief Array of pointers uniform-size single precision batched pivoted Cholesky of matrices A: P^T A P = R^t * R
 */
inline int kblas_pstrf_batch(kblasHandle_t handle, int n, float** A_array, int lda, int** piv_array, int* ranks, int num_ops)
{ return kblasSpstrf_batch(handle, n, A_array, lda, piv_array, ranks, num_ops); }
/**
* @brief Strided uniform-size double precision batched pivoted Cholesky of matrices A: P^T A P = R^t * R
*/
inline int kblas_pstrf_batch(kblasHandle_t handle, int n, double* A_strided, int lda, int stride_a, int* piv_strided, int stride_piv, int* ranks, int num_ops)
{ return kblasDpstrf_batch_strided(handle, n, A_strided, lda, stride_a, piv_strided, stride_piv, ranks, num_ops); }
/**
* @brief Strided uniform-size single precision batched pivoted Cholesky of matrices A: P^T A P = R^t * R
*/
inline int kblas_pstrf_batch(kblasHandle_t handle, int n, float* A_strided, int lda, int stride_a, int* piv_strided, int stride_piv, int* ranks, int num_ops)
{ return kblasSpstrf_batch_strided(handle, n, A_strided, lda, stride_a, piv_strided, stride_piv, ranks, num_ops); }
//@}

#endif
//@}

#endif
