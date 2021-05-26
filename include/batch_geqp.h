/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file include/batch_geqp.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Wajih Halim Boukaram
 * @date 2020-12-10
 **/
 
 #ifndef __BATCH_GEQP_H__
 #define __BATCH_GEQP_H__
//############################################################################
//BATCH pivoted QR routines
//############################################################################

/** @addtogroup C_API
*  @{
*/
#ifdef __cplusplus
extern "C" {
#endif
/**
 * @name Uniform-size batched QR routines
 */
//@{
//------------------------------------------------------------------------------
// Strided interface
/**
 * @brief Strided uniform-size double precision batched pivoted QR of matrices A: A * P = Q * R
 *
 * @param[in] handle KBLAS handle 
 * @param[in] m Rows of the matrix A. (m >= 0, m <= 512)
 * @param[in] n Columns of the matrix A. (n >= 0)
 * @param[in, out] A_strided Pointer to the batch of matrices A stored in strided format. On exit, the [min(m,n) x n] upper 
 * submatrix of each matrix A stores the factor R and the elementary reflectors V are stored below the diagonal. 
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] stride_a Stride of each matrix in A_strided. A[i] = A_strided + stride_a * i
 * @param[out] tau_strided Pointer to the batch of arrays tau stored in strided format. On exit, each array tau contains 
 * the scalar factors of the elementary reflectors V. Used with the reflectors to compute the factor Q in an ORGQR routine.
 * @param[in] stride_tau Stride of each array in tau_strided. tau[i] = tau_strided + stride_tau * i; (stride_tau >= n)
 * @param[out] piv_strided Pointer to the batch of arrays piv stored in strided format. On exit, each array piv contains 
 * the indexes of the columns pivoted during the QR decomposition
 * @param[in] stride_piv Stride of each array in piv_strided. piv[i] = piv_strided + stride_piv * i; (stride_piv >= n)
 * @param[out] ranks Pointer to array which will hold the detected ranks of the matrices (i.e., where the factorization breaks down due to rank deficiency or due to the tolerance being reached)
 * @param[in] tol The absolute tolerance of the factorization. Factorization stops when norm(A[:, k]) * sqrt(n - k) < tol so that the first k columns of Q form an approximate basis of the columns of A 
 * @param[in] num_ops The size of the batched operation. 
 */
int kblasDgeqp2_batch_strided(
	kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, 
	double* tau_strided, int stride_tau, int* piv_strided, int stride_piv, 
	int* ranks, double tol, int num_ops
);
/**
 * @brief Strided uniform-size single precision batched pivoted QR of matrices A: A * P = Q * R
 *
 * @param[in] handle KBLAS handle 
 * @param[in] m Rows of the matrix A. (m >= 0, m <= 512)
 * @param[in] n Columns of the matrix A. (n >= 0)
 * @param[in, out] A_strided Pointer to the batch of matrices A stored in strided format. On exit, the [min(m,n) x n] upper 
 * submatrix of each matrix A stores the factor R and the elementary reflectors V are stored below the diagonal. 
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] stride_a Stride of each matrix in A_strided. A[i] = A_strided + stride_a * i
 * @param[out] tau_strided Pointer to the batch of arrays tau stored in strided format. On exit, each array tau contains 
 * the scalar factors of the elementary reflectors V. Used with the reflectors to compute the factor Q in an ORGQR routine.
 * @param[in] stride_tau Stride of each array in tau_strided. tau[i] = tau_strided + stride_tau * i; (stride_tau >= n)
 * @param[out] piv_strided Pointer to the batch of arrays piv stored in strided format. On exit, each array piv contains 
 * the indexes of the columns pivoted during the QR decomposition
 * @param[in] stride_piv Stride of each array in piv_strided. piv[i] = piv_strided + stride_piv * i; (stride_piv >= n)
 * @param[out] ranks Pointer to array which will hold the detected ranks of the matrices (i.e., where the factorization breaks down due to rank deficiency or due to the tolerance being reached)
 * @param[in] tol The absolute tolerance of the factorization. Factorization stops when norm(A[:, k]) * sqrt(n - k) < tol so that the first k columns of Q form an approximate basis of the columns of A 
 * @param[in] num_ops The size of the batched operation. 
 */
int kblasSgeqp2_batch_strided(
	kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, 
	float* tau_strided, int stride_tau, int* piv_strided, int stride_piv, 
	int* ranks, float tol, int num_ops
);

// Array of pointers interface
/**
 * @brief Array of pointers uniform-size double precision batched pivoted QR of matrices A: A * P = Q * R
 * 
 * Array of pointers interface taking similar arguments to the strided interface; however, matrices and 
 * arrays are stored in arrays of pointers instead of strided access, so A[i] = A_array[i] and tau[i] = tau_array[i]
 * @see kblasDgeqp2_batch_strided
 */
int kblasDgeqp2_batch(
	kblasHandle_t handle, int m, int n, double** A_array, int lda, 
	double** tau_ptrs, int** piv_ptrs, int* ranks, double tol, int num_ops
);
/**
 * @brief Array of pointers uniform-size single precision batched pivoted QR of matrices A: A * P = Q * R
 * 
 * Array of pointers interface taking similar arguments to the strided interface; however, matrices and 
 * arrays are stored in arrays of pointers instead of strided access, so A[i] = A_array[i] and tau[i] = tau_array[i]
 * @see kblasSgeqp2_batch_strided
 */
int kblasSgeqp2_batch(
	kblasHandle_t handle, int m, int n, float** A_array, int lda, 
	float** tau_ptrs, int** piv_ptrs, int* ranks, float tol, int num_ops
);
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
 * @name Uniform-size batched pivoted QR routines
 */
//@{
/**
 * @brief Strided uniform-size double precision batched pivoted Cholesky of matrices A: A = R^t * R
 */
inline int kblas_geqp2_batch(
	kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, 
	double* tau_strided, int stride_tau, int* piv_strided, int stride_piv, 
	int* ranks, double tol, int num_ops
) { return kblasDgeqp2_batch_strided(handle, m, n, A_strided, lda, stride_a, tau_strided, stride_tau, piv_strided, stride_piv, ranks, tol, num_ops); }
/**
 * @brief Strided uniform-size single precision batched pivoted Cholesky of matrices A: A = R^t * R
 */
inline int kblas_geqp2_batch(
	kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, 
	float* tau_strided, int stride_tau, int* piv_strided, int stride_piv, 
	int* ranks, float tol, int num_ops
) { return kblasSgeqp2_batch_strided(handle, m, n, A_strided, lda, stride_a, tau_strided, stride_tau, piv_strided, stride_piv, ranks, tol, num_ops); }

/**
 * @brief Array of pointers uniform-size double precision batched pivoted QR of matrices A: A * P = Q * R
 */
inline int kblas_geqp2_batch(
	kblasHandle_t handle, int m, int n, double** A_array, int lda, 
	double** tau_ptrs, int** piv_ptrs, int* ranks, double tol, int num_ops
) { return kblasDgeqp2_batch(handle, m, n, A_array, lda, tau_ptrs, piv_ptrs, ranks, tol, num_ops); }

/**
 * @brief Array of pointers uniform-size single precision batched pivoted QR of matrices A: A * P = Q * R
 */
inline int kblas_geqp2_batch(
	kblasHandle_t handle, int m, int n, float** A_array, int lda, 
	float** tau_ptrs, int** piv_ptrs, int* ranks, float tol, int num_ops
) { return kblasSgeqp2_batch(handle, m, n, A_array, lda, tau_ptrs, piv_ptrs, ranks, tol, num_ops); }
//@}
#endif

//@}

#endif 
