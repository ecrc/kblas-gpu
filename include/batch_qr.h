/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file include/batch_qr.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Wajih Halim Boukaram
 * @date 2018-11-14
 **/

#ifndef __BATCH_QR_H__
#define __BATCH_QR_H__

//############################################################################
//BATCH QR routines
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
 * @brief Strided uniform-size double precision batched QR decomposition of matrices A: A = Q * R
 *
 * @param[in] handle KBLAS handle 
 * @param[in] m Rows of the matrix A. (m >= 0, m <= 512)
 * @param[in] n Columns of the matrix A (n >= 0, n <= 512)
 * @param[in, out] A_strided Pointer to the batch of matrices A stored in strided format. On exit, the [min(m,n) x n] upper 
 * submatrix of each matrix A stores the factor R and the elementary reflectors V are stored below the diagonal. 
 * The reflectors can be expanded into the full matrix Q using an ORGQR batched routine.
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] stride_a Stride of each matrix in A_strided. A[i] = A_strided + stride_a * i;  (stride_a >= lda * cols)
 * @param[out] tau_strided Pointer to the batch of arrays tau stored in strided format. On exit, each array tau contains 
 * the scalar factors of the elementary reflectors V. Used with the reflectors to compute the factor Q in an ORGQR routine.
 * @param[in] stride_tau Stride of each array in tau_strided. tau[i] = tau_strided + stride_tau * i; (stride_tau >= cols)
 * @param[in] num_ops The size of the batched operation. 
 * @see kblasDorgqr_batch_strided
 */
int kblasDgeqrf_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* tau_strided, int stride_tau, int num_ops);

/**
 * @brief Strided uniform-size single precision batched QR decomposition of matrices A: A = Q * R
 *
 * @param[in] handle KBLAS handle 
 * @param[in] m Rows of the matrix A. (m >= 0, m <= 512)
 * @param[in] n Columns of the matrix A (n >= 0, n <= 512)
 * @param[in, out] A_strided Pointer to the batch of matrices A stored in strided format. On exit, the [min(m,n) x n] upper 
 * submatrix of each matrix A stores the factor R and the elementary reflectors V are stored below the diagonal. 
 * The reflectors can be expanded into the full matrix Q using an ORGQR batched routine.
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] stride_a Stride of each matrix in A_strided. A[i] = A_strided + stride_a * i;  (stride_a >= lda * cols)
 * @param[out] tau_strided Pointer to the batch of arrays tau stored in strided format. On exit, each array tau contains 
 * the scalar factors of the elementary reflectors V. Used with the reflectors to compute the factor Q in an ORGQR routine.
 * @param[in] stride_tau Stride of each array in tau_strided. tau[i] = tau_strided + stride_tau * i; (stride_tau >= cols)
 * @param[in] num_ops The size of the batched operation. 
 * @see kblasSorgqr_batch_strided
 */
int kblasSgeqrf_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* tau_strided, int stride_tau, int num_ops);

/**
 * @brief Strided uniform-size double precision batched domino QR decomposition of matrices A: A = Q * R
 *
 * This batched routine is used when the matrix is tall and skinny and only the triangular factor is needed.
 * It is more efficient than the GEQRF variant for these matrices, but currently does not have the ability to compute the orthogonal 
 * factor. This will come in a future release.
 * @param[in] handle KBLAS handle 
 * @param[in] m Rows of the matrix A. (m >= 0)
 * @param[in] n Columns of the matrix A (n >= 0)
 * @param[in, out] A_strided Pointer to the batch of matrices A stored in strided format. On exit, the [min(m,n) x n] upper 
 * submatrix of each matrix A stores the factor R. 
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] stride_a Stride of each matrix in A_strided. A[i] = A_strided + stride_a * i;  (stride_a >= lda * cols)
 * @param[out] tau_strided Pointer to the batch of arrays tau stored in strided format.
 * @param[in] stride_tau Stride of each array in tau_strided. tau[i] = tau_strided + stride_tau * i; (stride_tau >= cols)
 * @param[in] num_ops The size of the batched operation. 
 */
int kblasDtsqrf_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* tau_strided, int stride_tau, int num_ops);

/**
 * @brief Strided uniform-size single precision batched domino QR decomposition of matrices A: A = Q * R
 * 
 * This batched routine is used when the matrix is tall and skinny and only the triangular factor is needed.
 * It is more efficient than the GEQRF variant for these matrices, but currently does not have the ability to compute the orthogonal 
 * factor. This will come in a future release.
 * @param[in] handle KBLAS handle 
 * @param[in] m Rows of the matrix A. (m >= 0)
 * @param[in] n Columns of the matrix A (n >= 0)
 * @param[in, out] A_strided Pointer to the batch of matrices A stored in strided format. On exit, the [min(m,n) x n] upper 
 * submatrix of each matrix A stores the factor R. 
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] stride_a Stride of each matrix in A_strided. A[i] = A_strided + stride_a * i;  (stride_a >= lda * cols)
 * @param[out] tau_strided Pointer to the batch of arrays tau stored in strided format.
 * @param[in] stride_tau Stride of each array in tau_strided. tau[i] = tau_strided + stride_tau * i; (stride_tau >= cols)
 * @param[in] num_ops The size of the batched operation. 
 */
int kblasStsqrf_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* tau_strided, int stride_tau, int num_ops);

/**
 * @brief Strided uniform-size double precision batched generation of the orthogonal factor Q of the QR decomposition formed by GEQRF
 * 
 * @param[in] handle KBLAS handle 
 * @param[in] m Rows of the matrix A. (m >= 0, m <= 512)
 * @param[in] n Columns of the matrix A (n >= 0, n <= 512)
 * @param[in, out] A_strided Pointer to the batch of matrices A stored in strided format. On input, the elementary reflectors
 * V as computed by a GEQRF routine are stored below the diagonal of each matrix A. On exit, contains the orthogonal factors Q.
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] stride_a Stride of each matrix in A_strided. A[i] = A_strided + stride_a * i;  (stride_a >= lda * cols)
 * @param[in] tau_strided Pointer to the batch of arrays tau stored in strided format. Each array tau contains 
 * the scalar factors of the elementary reflectors V as computed by a GEQRF routine.
 * @param[in] stride_tau Stride of each array in tau_strided. tau[i] = tau_strided + stride_tau * i; (stride_tau >= cols)
 * @param[in] num_ops The size of the batched operation. 
 * @see kblasDgeqrf_batch_strided
 */
int kblasDorgqr_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* tau_strided, int stride_tau, int num_ops);

/**
 * @brief Strided uniform-size single precision batched generation of the orthogonal factor Q of the QR decomposition formed by GEQRF
 * 
 * @param[in] handle KBLAS handle 
 * @param[in] m Rows of the matrix A. (m >= 0, m <= 512)
 * @param[in] n Columns of the matrix A (n >= 0, n <= 512)
 * @param[in, out] A_strided Pointer to the batch of matrices A stored in strided format. On input, the elementary reflectors
 * V as computed by a GEQRF routine are stored below the diagonal of each matrix A. On exit, contains the orthogonal factors Q.
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] stride_a Stride of each matrix in A_strided. A[i] = A_strided + stride_a * i;  (stride_a >= lda * cols)
 * @param[in] tau_strided Pointer to the batch of arrays tau stored in strided format. Each array tau contains 
 * the scalar factors of the elementary reflectors V as computed by a GEQRF routine.
 * @param[in] stride_tau Stride of each array in tau_strided. tau[i] = tau_strided + stride_tau * i; (stride_tau >= cols)
 * @param[in] num_ops The size of the batched operation. 
 * @see kblasSgeqrf_batch_strided
 */
int kblasSorgqr_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* tau_strided, int stride_tau, int num_ops);


/**
 * @brief Strided uniform-size double precision batched copy of the upper triangular factor R
 * 
 * Use this routine to copy the triangular factor R to a separate array before expanding the reflectors
 * into the orthogonal factor, which will overwrite the matrix (including the factor stored in the upper triangular portion).
 * @param[in] handle KBLAS handle 
 * @param[in] m Rows of the matrix A. (m >= 0)
 * @param[in] n Columns of the matrix A (n >= 0)
 * @param[in] A_strided Pointer to the batch of matrices A stored in strided format.
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] stride_a Stride of each matrix in A_strided. A[i] = A_strided + stride_a * i;  (stride_a >= lda * cols)
 * @param[out] R_strided Pointer to the batch of matrices Rstored in strided format. On exit, each matrix R will contain a copy of the 
 * upper triangular part of A.
 * @param[in] ldr Leading dimension of R (ldr >= min(m,n))
 * @param[in] stride_R Stride of each matrix in R_strided. R[i] = R_strided + stride_R * i; (stride_R >= ldr * cols)
 * @param[in] num_ops The size of the batched operation. 
 */
 
int kblasDcopy_upper_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* R_strided, int ldr, int stride_R, int num_ops);
/**
 * @brief Strided uniform-size single precision batched copy of the upper triangular factor R
 * 
 * Use this routine to copy the triangular factor R to a separate array before expanding the reflectors
 * into the orthogonal factor, which will overwrite the matrix (including the factor stored in the upper triangular portion).
 * @param[in] handle KBLAS handle 
 * @param[in] m Rows of the matrix A. (m >= 0)
 * @param[in] n Columns of the matrix A (n >= 0)
 * @param[in] A_strided Pointer to the batch of matrices A stored in strided format.
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] stride_a Stride of each matrix in A_strided. A[i] = A_strided + stride_a * i;  (stride_a >= lda * cols)
 * @param[out] R_strided Pointer to the batch of matrices Rstored in strided format. On exit, each matrix R will contain a copy of the 
 * upper triangular part of A.
 * @param[in] ldr Leading dimension of R (ldr >= min(m,n))
 * @param[in] stride_R Stride of each matrix in R_strided. R[i] = R_strided + stride_R * i; (stride_R >= ldr * cols)
 * @param[in] num_ops The size of the batched operation. 
 */
int kblasScopy_upper_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* R_strided, int ldr, int stride_R, int num_ops);

//------------------------------------------------------------------------------
// Array of pointers interface
/**
 * @brief Array of pointers uniform-size double precision batched QR decomposition of matrices A: A = Q * R
 * 
 * Array of pointers interface taking similar arguments to the strided interface; however, matrices and 
 * arrays are stored in arrays of pointers instead of strided access, so A[i] = A_array[i] and tau[i] = tau_array[i]
 * @see kblasDgeqrf_batch_strided
 */
int kblasDgeqrf_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** tau_array, int num_ops);

/**
 * @brief Array of pointers uniform-size single precision batched QR decomposition of matrices A: A = Q * R
 * 
 * Array of pointers interface taking similar arguments to the strided interface; however, matrices and 
 * arrays are stored in arrays of pointers instead of strided access, so A[i] = A_array[i] and tau[i] = tau_array[i]
 * @see kblasSgeqrf_batch_strided
 */
int kblasSgeqrf_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** tau_array, int num_ops);

/**
 * @brief Array of pointers uniform-size double precision batched domino QR decomposition of matrices A: A = Q * R
 * 
 * Array of pointers interface taking similar arguments to the strided interface; however, matrices and 
 * arrays are stored in arrays of pointers instead of strided access, so A[i] = A_array[i] and tau[i] = tau_array[i]
 * @see kblasDtsqrf_batch_strided
 */
int kblasDtsqrf_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** tau_array, int num_ops);

/**
 * @brief Array of pointers uniform-size single precision batched domino QR decomposition of matrices A: A = Q * R
 * 
 * Array of pointers interface taking similar arguments to the strided interface; however, matrices and 
 * arrays are stored in arrays of pointers instead of strided access, so A[i] = A_array[i] and tau[i] = tau_array[i]
 * @see kblasStsqrf_batch_strided
 */
int kblasStsqrf_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** tau_array, int num_ops);

/**
 * @brief Array of pointers uniform-size double precision batched generation of the orthogonal factor Q of the QR decomposition formed by GEQRF
 * 
 * Array of pointers interface taking similar arguments to the strided interface; however, matrices and 
 * arrays are stored in arrays of pointers instead of strided access, so A[i] = A_array[i] and tau[i] = tau_array[i]
 * @see kblasDorgqr_batch_strided
 */
int kblasDorgqr_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** tau_array, int num_ops);

/**
 * @brief Array of pointers uniform-size single precision batched generation of the orthogonal factor Q of the QR decomposition formed by GEQRF
 * 
 * Array of pointers interface taking similar arguments to the strided interface; however, matrices and 
 * arrays are stored in arrays of pointers instead of strided access, so A[i] = A_array[i] and tau[i] = tau_array[i]
 * @see kblasSorgqr_batch_strided
 */
int kblasSorgqr_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** tau_array, int num_ops);

/**
 * @brief Array of pointers uniform-size double precision batched copy of the upper triangular factor R
 * 
 * Array of pointers interface taking similar arguments to the strided interface; however, matrices and 
 * arrays are stored in arrays of pointers instead of strided access, so A[i] = A_array[i] and R[i] = R_array[i]
 * @see kblasDcopy_upper_batch_strided
 */
int kblasDcopy_upper_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** R_array, int ldr, int num_ops);

/**
 * @brief Array of pointers uniform-size single precision batched copy of the upper triangular factor R
 * 
 * Array of pointers interface taking similar arguments to the strided interface; however, matrices and 
 * arrays are stored in arrays of pointers instead of strided access, so A[i] = A_array[i] and R[i] = R_array[i]
 * @see kblasScopy_upper_batch_strided
 */
int kblasScopy_upper_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** R_array, int ldr, int num_ops);

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
 * @name Uniform-size batched QR routines
 */
//@{
//------------------------------------------------------------------------------
// Strided interface

/**
 * @brief Strided uniform-size double precision batched QR decomposition of matrices A: A = Q * R
 */
inline int kblas_geqrf_batch(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* tau_strided, int stride_tau, int num_ops)
{ return kblasDgeqrf_batch_strided(handle, m, n, A_strided, lda, stride_a, tau_strided, stride_tau, num_ops); }
/**
 * @brief Strided uniform-size single precision batched QR decomposition of matrices A: A = Q * R
 */
inline int kblas_geqrf_batch(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* tau_strided, int stride_tau, int num_ops)
{ return kblasSgeqrf_batch_strided(handle, m, n, A_strided, lda, stride_a, tau_strided, stride_tau, num_ops); }
/**
 * @brief Strided uniform-size double precision batched domino QR decomposition of matrices A: A = Q * R
 */
inline int kblas_tsqrf_batch(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* tau_strided, int stride_tau, int num_ops)
{ return kblasDtsqrf_batch_strided(handle, m, n, A_strided, lda, stride_a, tau_strided, stride_tau, num_ops); }
/**
 * @brief Strided uniform-size single precision batched domino QR decomposition of matrices A: A = Q * R
 */
inline int kblas_tsqrf_batch(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* tau_strided, int stride_tau, int num_ops)
{ return kblasStsqrf_batch_strided(handle, m, n, A_strided, lda, stride_a, tau_strided, stride_tau, num_ops); }
/**
 * @brief Strided uniform-size double precision batched generation of the orthogonal factor Q of the QR decomposition formed by GEQRF
 */
inline int kblas_orgqr_batch(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* tau_strided, int stride_tau, int num_ops)
{ return kblasDorgqr_batch_strided(handle, m, n, A_strided, lda, stride_a, tau_strided, stride_tau, num_ops); }
/**
 * @brief Strided uniform-size single precision batched generation of the orthogonal factor Q of the QR decomposition formed by GEQRF
 */
inline int kblas_orgqr_batch(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* tau_strided, int stride_tau, int num_ops)
{ return kblasSorgqr_batch_strided(handle, m, n, A_strided, lda, stride_a, tau_strided, stride_tau, num_ops); }
/**
 * @brief Strided uniform-size double precision batched copy of the upper triangular factor R
 */
inline int kblas_copy_upper_batch(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* R_strided, int ldr, int stride_R, int num_ops)
{ return kblasDcopy_upper_batch_strided(handle, m, n, A_strided, lda, stride_a, R_strided, ldr, stride_R, num_ops); }
/**
 * @brief Strided uniform-size single precision batched copy of the upper triangular factor R
 */
inline int kblas_copy_upper_batch(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* R_strided, int ldr, int stride_R, int num_ops)
{ return kblasScopy_upper_batch_strided(handle, m, n, A_strided, lda, stride_a, R_strided, ldr, stride_R, num_ops); }

//------------------------------------------------------------------------------
// Array of pointers interface
/**
 * @brief Array of pointers uniform-size double precision batched QR decomposition of matrices A: A = Q * R
 */
inline int kblas_geqrf_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** tau_array, int num_ops)
{ return kblasDgeqrf_batch(handle, m, n, A_array, lda, tau_array, num_ops); }
/**
 * @brief Array of pointers uniform-size single precision batched QR decomposition of matrices A: A = Q * R
 */
inline int kblas_geqrf_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** tau_array, int num_ops)
{ return kblasSgeqrf_batch(handle, m, n, A_array, lda, tau_array, num_ops); }
/**
 * @brief Array of pointers uniform-size double precision batched domino QR decomposition of matrices A: A = Q * R
 */
inline int kblas_tsqrf_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** tau_array, int num_ops)
{ return kblasDtsqrf_batch(handle, m, n, A_array, lda, tau_array, num_ops); }
/**
 * @brief Array of pointers uniform-size single precision batched domino QR decomposition of matrices A: A = Q * R
 */
inline int kblas_tsqrf_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** tau_array, int num_ops)
{ return kblasStsqrf_batch(handle, m, n, A_array, lda, tau_array, num_ops); }
/**
 * @brief Array of pointers uniform-size double precision batched generation of the orthogonal factor Q of the QR decomposition formed by GEQRF
 */
inline int kblas_orgqr_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** tau_array, int num_ops)
{ return kblasDorgqr_batch(handle, m, n, A_array, lda, tau_array, num_ops); }
/**
 * @brief Array of pointers uniform-size single precision batched generation of the orthogonal factor Q of the QR decomposition formed by GEQRF
 */
inline int kblas_orgqr_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** tau_array, int num_ops)
{ return kblasSorgqr_batch(handle, m, n, A_array, lda, tau_array, num_ops); }
/**
 * @brief Array of pointers uniform-size double precision batched copy of the upper triangular factor R
 */
inline int kblas_copy_upper_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** R_array, int ldr, int num_ops)
{ return kblasDcopy_upper_batch(handle, m, n, A_array, lda, R_array, ldr, num_ops); }
/**
 * @brief Array of pointers uniform-size single precision batched copy of the upper triangular factor R
 */
inline int kblas_copy_upper_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** R_array, int ldr, int num_ops)
{ return kblasScopy_upper_batch(handle, m, n, A_array, lda, R_array, ldr, num_ops); }

//@}

// Common interface for array of pointers with dummy strides
template<class T>
inline int kblas_geqrf_batch(kblasHandle_t handle, int m, int n, T** A_array, int lda, int stride_a, T** tau_array, int stride_tau, int num_ops)
{ return kblas_geqrf_batch(handle, m, n, A_array, lda, tau_array, num_ops); }
template<class T>
inline int kblas_tsqrf_batch(kblasHandle_t handle, int m, int n, T** A_array, int lda, int stride_a, T** tau_array, int stride_tau, int num_ops)
{ return kblas_tsqrf_batch(handle, m, n, A_array, lda, tau_array, num_ops); }
template<class T>
inline int kblas_orgqr_batch(kblasHandle_t handle, int m, int n, T** A_array, int lda, int stride_a, T** tau_array, int stride_tau, int num_ops)
{ return kblas_orgqr_batch(handle, m, n, A_array, lda, tau_array, num_ops); }
template<class T>
inline int kblas_copy_upper_batch(kblasHandle_t handle, int m, int n, T** A_array, int lda, int stride_a, T** R_array, int ldr, int stride_R, int num_ops)
{ return kblas_copy_upper_batch(handle, m, n, A_array, lda, R_array, ldr, num_ops); }
#endif

/** @} */

#endif
