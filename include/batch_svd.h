/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file include/batch_svd.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 2.0.0
 * @author Wajih Halim Boukaram
 * @date 2017-11-13
 **/

#ifndef __BATCH_SVD_H__
#define __BATCH_SVD_H__

#define SHARED_SVD_DIM_LIMIT 64
#define RSVD_DEFAULT_SAMPLES 64

//############################################################################
//BATCH SVD routines
//############################################################################

/** @addtogroup C_API
*  @{
*/
#ifdef __cplusplus
extern "C" {
#endif
/**
 * @name Uniform-size batched SVD routines
 */
//@{
//------------------------------------------------------------------------------
// workspace query routines for both strided and array of pointers interface
/**
 * @brief Workspace query routine for double precision batched Jacobi SVD
 *
 * Calling this routine updates the handle internal workspace state. A call to kblasAllocateWorkspace
 * will then allocate the workspace if necessary.
 * @param[in] handle KBLAS handle
 * @param[in] m Rows of the matrix A. (m >= 0, m <= 512)
 * @param[in] n Columns of the matrix A (n >= 0, n <= 512)
 * @param[in] num_ops The size of the batched operation.
 * @see kblasAllocateWorkspace
 * @see kblasFreeWorkspace
 */
void kblasDgesvj_batch_wsquery(kblasHandle_t handle, int m, int n, int num_ops);

/**
 * @brief Workspace query routine for single precision batched Jacobi SVD
 *
 * Calling this routine updates the handle internal workspace state. A call to kblasAllocateWorkspace
 * will then allocate the workspace if necessary.
 * @param[in] handle KBLAS handle
 * @param[in] m Rows of the matrix A. (m >= 0, m <= 512)
 * @param[in] n Columns of the matrix A (n >= 0, n <= 512)
 * @param[in] num_ops The size of the batched operation.
 * @see kblasAllocateWorkspace
 * @see kblasFreeWorkspace
 */
void kblasSgesvj_batch_wsquery(kblasHandle_t handle, int m, int n, int num_ops);

/**
 * @brief Workspace query routine for double precision batched Gram matrix variant Jacobi SVD
 *
 * Calling this routine updates the handle internal workspace state. A call to kblasAllocateWorkspace
 * will then allocate the workspace if necessary. This variant can be faster than the regular Jacobi
 * if the matrix is known to be well-conditioned. It may not converge for ill-conditioned matrices.
 * @param[in] handle KBLAS handle
 * @param[in] m Rows of the matrix A. (m >= 0, m <= 512)
 * @param[in] n Columns of the matrix A (n >= 0, n <= 512)
 * @param[in] num_ops The size of the batched operation.
 * @see kblasAllocateWorkspace
 * @see kblasFreeWorkspace
 */
void kblasDgesvj_gram_batch_wsquery(kblasHandle_t handle, int m, int n, int num_ops);

/**
 * @brief Workspace query routine for single precision batched Gram matrix variant Jacobi SVD
 *
 * Calling this routine updates the handle internal workspace state. A call to kblasAllocateWorkspace
 * will then allocate the workspace if necessary. This variant can be faster than the regular Jacobi
 * if the matrix is known to be well-conditioned. It may not converge for ill-conditioned matrices.
 * @param[in] handle KBLAS handle
 * @param[in] m Rows of the matrix A. (m >= 0, m <= 512)
 * @param[in] n Columns of the matrix A (n >= 0, n <= 512)
 * @param[in] num_ops The size of the batched operation.
 * @see kblasAllocateWorkspace
 * @see kblasFreeWorkspace
 */
void kblasSgesvj_gram_batch_wsquery(kblasHandle_t handle, int m, int n, int num_ops);

/**
 * @brief Workspace query routine for double precision batched Randomized SVD
 *
 * Calling this routine updates the handle internal workspace state. A call to kblasAllocateWorkspace
 * will then allocate the workspace if necessary. Use this routine if you want the top _rank_ singular values
 * and vectors for a matrix that is known to be low rank or numerically low rank.
 * @param[in] handle KBLAS handle
 * @param[in] m Rows of the matrix A. (m >= 0, m <= 512)
 * @param[in] n Columns of the matrix A (n >= 0, n <= 512)
 * @param[in] rank The number of requested singular values (rank < n). If rank == n then the regular SVD is used.
 * @param[in] num_ops The size of the batched operation.
 * @see kblasAllocateWorkspace
 * @see kblasFreeWorkspace
 */
void kblasDrsvd_batch_wsquery(kblasHandle_t handle, int m, int n, int rank, int num_ops);

/**
 * @brief Workspace query routine for double precision batched Randomized SVD
 *
 * Calling this routine updates the handle internal workspace state. A call to kblasAllocateWorkspace
 * will then allocate the workspace if necessary. Use this routine if you want the top _rank_ singular values
 * and vectors for a matrix that is known to be low rank or numerically low rank.
 * @param[in] handle KBLAS handle
 * @param[in] m Rows of the matrix A. (m >= 0, m <= 512)
 * @param[in] n Columns of the matrix A (n >= 0, n <= 512)
 * @param[in] rank The number of requested singular values (rank < n). If rank == n then the regular SVD is used.
 * @param[in] num_ops The size of the batched operation.
 * @see kblasAllocateWorkspace
 * @see kblasFreeWorkspace
 */
void kblasSrsvd_batch_wsquery(kblasHandle_t handle, int m, int n, int rank, int num_ops);

//------------------------------------------------------------------------------
// Strided interface
/**
 * @brief Strided uniform-size double precision batched Jacobi SVD
 * @param[in] handle KBLAS handle
 * @param[in] m Rows of the matrix A. (m >= 0, m <= 512)
 * @param[in] n Columns of the matrix A (n >= 0, n <= 512)
 * @param[in, out] A_strided Pointer to the batch of matrices A stored in strided format. On exit, contains the
 * right singular vectors of A. Left singular values are not currently supported.
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] stride_a Stride of each matrix in A_strided. A[i] = A_strided + stride_a * i;  (stride_a >= lda * cols)
 * @param[out] S_strided Pointer to the batch of arrays S stored in strided format. On exit, each array S contains
 * the singular values of A.
 * @param[in] stride_s Stride of each array in S_strided. S[i] = S_strided + stride_s * i; (stride_s >= cols)
 * @param[in] num_ops The size of the batched operation.
 */
int kblasDgesvj_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops);

/**
 * @brief Strided uniform-size single precision batched Jacobi SVD
 * @param[in] handle KBLAS handle
 * @param[in] m Rows of the matrix A. (m >= 0, m <= 512)
 * @param[in] n Columns of the matrix A (n >= 0, n <= 512)
 * @param[in, out] A_strided Pointer to the batch of matrices A stored in strided format. On exit, contains the
 * right singular vectors of A. Left singular values are not currently supported.
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] stride_a Stride of each matrix in A_strided. A[i] = A_strided + stride_a * i;  (stride_a >= lda * cols)
 * @param[out] S_strided Pointer to the batch of arrays S stored in strided format. On exit, each array S contains
 * the singular values of A.
 * @param[in] stride_s Stride of each array in S_strided. S[i] = S_strided + stride_s * i; (stride_s >= cols)
 * @param[in] num_ops The size of the batched operation.
 */
int kblasSgesvj_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops);

/**
 * @brief Strided uniform-size double precision batched Gram matrix variant Jacobi SVD
 *
 * This variant can be faster than the regular Jacobi if the matrix is known to be well-conditioned.
 * It may not converge for ill-conditioned matrices.
 * @param[in] handle KBLAS handle
 * @param[in] m Rows of the matrix A. (m >= 0, m <= 512)
 * @param[in] n Columns of the matrix A (n >= 0, n <= 512)
 * @param[in, out] A_strided Pointer to the batch of matrices A stored in strided format. On exit, contains the
 * right singular vectors of A. Left singular values are not currently supported.
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] stride_a Stride of each matrix in A_strided. A[i] = A_strided + stride_a * i;  (stride_a >= lda * cols)
 * @param[out] S_strided Pointer to the batch of arrays S stored in strided format. On exit, each array S contains
 * the singular values of A.
 * @param[in] stride_s Stride of each array in S_strided. S[i] = S_strided + stride_s * i; (stride_s >= cols)
 * @param[in] num_ops The size of the batched operation.
 */
int kblasDgesvj_gram_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops);

/**
 * @brief Strided uniform-size single precision batched Gram matrix variant Jacobi SVD
 *
 * This variant can be faster than the regular Jacobi if the matrix is known to be well-conditioned.
 * It may not converge for ill-conditioned matrices.
 * @param[in] handle KBLAS handle
 * @param[in] m Rows of the matrix A. (m >= 0, m <= 512)
 * @param[in] n Columns of the matrix A (n >= 0, n <= 512)
 * @param[in, out] A_strided Pointer to the batch of matrices A stored in strided format. On exit, contains the
 * right singular vectors of A. Left singular values are not currently supported.
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] stride_a Stride of each matrix in A_strided. A[i] = A_strided + stride_a * i;  (stride_a >= lda * cols)
 * @param[out] S_strided Pointer to the batch of arrays S stored in strided format. On exit, each array S contains
 * the singular values of A.
 * @param[in] stride_s Stride of each array in S_strided. S[i] = S_strided + stride_s * i; (stride_s >= cols)
 * @param[in] num_ops The size of the batched operation.
 */
int kblasSgesvj_gram_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops);

/**
 * @brief Strided uniform-size double precision batched Randomized SVD
 *
 * Use this routine if you want the top _rank_ singular values and vectors for a matrix
 * that is known to be low rank or numerically low rank.
 * @param[in] handle KBLAS handle
 * @param[in] m Rows of the matrix A. (m >= 0, m <= 512)
 * @param[in] n Columns of the matrix A (n >= 0, n <= 512)
 * @param[in] rank The number of requested singular values (rank < n). If rank == n then the regular SVD is used.
 * @param[in, out] A_strided Pointer to the batch of matrices A stored in strided format. On exit, the first _rank_ columns
 * contains the right singular vectors of A corresponding to the top _rank_ singular values.
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] stride_a Stride of each matrix in A_strided. A[i] = A_strided + stride_a * i;  (stride_a >= lda * cols)
 * @param[out] S_strided Pointer to the batch of arrays S stored in strided format. On exit, each array S contains
 * the singular values of A.
 * @param[in] stride_s Stride of each array in S_strided. S[i] = S_strided + stride_s * i; (stride_s >= cols)
 * @param[in] num_ops The size of the batched operation.
 */
int kblasDrsvd_batch_strided(kblasHandle_t handle, int m, int n, int rank, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops);

/**
 * @brief Strided uniform-size single precision batched Randomized SVD
 *
 * Use this routine if you want the top _rank_ singular values and vectors for a matrix
 * that is known to be low rank or numerically low rank.
 * @param[in] handle KBLAS handle
 * @param[in] m Rows of the matrix A. (m >= 0, m <= 512)
 * @param[in] n Columns of the matrix A (n >= 0, n <= 512)
 * @param[in] rank The number of requested singular values (rank < n). If rank == n then the regular SVD is used.
 * @param[in, out] A_strided Pointer to the batch of matrices A stored in strided format. On exit, the first _rank_ columns
 * contains the right singular vectors of A corresponding to the top _rank_ singular values.
 * @param[in] lda Leading dimension of A (lda >= m)
 * @param[in] stride_a Stride of each matrix in A_strided. A[i] = A_strided + stride_a * i;  (stride_a >= lda * cols)
 * @param[out] S_strided Pointer to the batch of arrays S stored in strided format. On exit, each array S contains
 * the singular values of A.
 * @param[in] stride_s Stride of each array in S_strided. S[i] = S_strided + stride_s * i; (stride_s >= cols)
 * @param[in] num_ops The size of the batched operation.
 */
int kblasSrsvd_batch_strided(kblasHandle_t handle, int m, int n, int rank, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops);

//------------------------------------------------------------------------------
// Array of pointers interface
/**
 * @brief Strided uniform-size double precision batched Jacobi SVD
 *
 * Array of pointers interface taking similar arguments to the strided interface; however, matrices and
 * arrays are stored in arrays of pointers instead of strided access, so A[i] = A_ptrs[i] and S[i] = S_ptrs[i]
 * @see kblasDgesvj_batch_strided
 */
int kblasDgesvj_batch(kblasHandle_t handle, int m, int n, double** A_ptrs, int lda, double** S_ptrs, int num_ops);

/**
 * @brief Strided uniform-size single precision batched Jacobi SVD
 *
 * Array of pointers interface taking similar arguments to the strided interface; however, matrices and
 * arrays are stored in arrays of pointers instead of strided access, so A[i] = A_ptrs[i] and S[i] = S_ptrs[i]
 * @see kblasSgesvj_batch_strided
 */
int kblasSgesvj_batch(kblasHandle_t handle, int m, int n, float** A_ptrs, int lda, float** S_ptrs, int num_ops);

/**
 * @brief Strided uniform-size double precision batched Gram matrix variant Jacobi SVD
 *
 * Array of pointers interface taking similar arguments to the strided interface; however, matrices and
 * arrays are stored in arrays of pointers instead of strided access, so A[i] = A_ptrs[i] and S[i] = S_ptrs[i]
 * @see kblasDgesvj_gram_batch_strided
 */
int kblasDgesvj_gram_batch(kblasHandle_t handle, int m, int n, double** A_ptrs, int lda, double** S_ptrs, int num_ops);

/**
 * @brief Strided uniform-size double precision batched Gram matrix variant Jacobi SVD
 *
 * Array of pointers interface taking similar arguments to the strided interface; however, matrices and
 * arrays are stored in arrays of pointers instead of strided access, so A[i] = A_ptrs[i] and S[i] = S_ptrs[i]
 * @see kblasSgesvj_gram_batch_strided
 */
int kblasSgesvj_gram_batch(kblasHandle_t handle, int m, int n, float** A_ptrs, int lda, float** S_ptrs, int num_ops);

/**
 * @brief Strided uniform-size double precision batched Randomized SVD
 *
 * Array of pointers interface taking similar arguments to the strided interface; however, matrices and
 * arrays are stored in arrays of pointers instead of strided access, so A[i] = A_ptrs[i] and S[i] = S_ptrs[i]
 * @see kblasDrsvd_batch_strided
 */
int kblasDrsvd_batch(kblasHandle_t handle, int m, int n, int rank, double** A_ptrs, int lda, double** S_ptrs, int num_ops);

/**
 * @brief Strided uniform-size double precision batched Randomized SVD
 *
 * Array of pointers interface taking similar arguments to the strided interface; however, matrices and
 * arrays are stored in arrays of pointers instead of strided access, so A[i] = A_ptrs[i] and S[i] = S_ptrs[i]
 * @see kblasSrsvd_batch_strided
 */
int kblasSrsvd_batch(kblasHandle_t handle, int m, int n, int rank, float** A_ptrs, int lda, float** S_ptrs, int num_ops);

//@}
#ifdef __cplusplus
}
#endif
/** @} */

#ifdef __cplusplus

// workspace query routines for both strided and array of pointers interface
template<class T> inline void kblas_gesvj_batch_wsquery(kblasHandle_t handle, int m, int n, int ops);
template<> inline void kblas_gesvj_batch_wsquery<double>(kblasHandle_t handle, int m, int n, int ops)
{ kblasDgesvj_batch_wsquery(handle, m, n, ops); }
template<> inline void kblas_gesvj_batch_wsquery<float>(kblasHandle_t handle, int m, int n, int ops)
{ kblasSgesvj_batch_wsquery(handle, m, n, ops); }

template<class T> inline void kblas_gesvj_gram_batch_wsquery(kblasHandle_t handle, int m, int n, int ops);
template<> inline void kblas_gesvj_gram_batch_wsquery<double>(kblasHandle_t handle, int m, int n, int ops)
{ kblasDgesvj_gram_batch_wsquery(handle, m, n, ops); }
template<> inline void kblas_gesvj_gram_batch_wsquery<float>(kblasHandle_t handle, int m, int n, int ops)
{ kblasSgesvj_gram_batch_wsquery(handle, m, n, ops); }

template<class T> inline void kblas_rsvd_batch_wsquery(kblasHandle_t handle, int m, int n, int rank, int ops);
template<> inline void kblas_rsvd_batch_wsquery<double>(kblasHandle_t handle, int m, int n, int rank, int ops)
{ kblasDrsvd_batch_wsquery(handle, m, n, rank, ops); }
template<> inline void kblas_rsvd_batch_wsquery<float>(kblasHandle_t handle, int m, int n, int rank, int ops)
{ kblasSrsvd_batch_wsquery(handle, m, n, rank, ops); }

// Strided interface
inline int kblas_gesvj_batch(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops)
{ return kblasDgesvj_batch_strided(handle, m, n, A_strided, lda, stride_a, S_strided, stride_s, num_ops); }
inline int kblas_gesvj_batch(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops)
{ return kblasSgesvj_batch_strided(handle, m, n, A_strided, lda, stride_a, S_strided, stride_s, num_ops); }

inline int kblas_gesvj_gram_batch(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops)
{ return kblasDgesvj_gram_batch_strided(handle, m, n, A_strided, lda, stride_a, S_strided, stride_s, num_ops); }
inline int kblas_gesvj_gram_batch(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops)
{ return kblasSgesvj_gram_batch_strided(handle, m, n, A_strided, lda, stride_a, S_strided, stride_s, num_ops); }

inline int kblas_rsvd_batch(kblasHandle_t handle, int m, int n, int rank, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops)
{ return kblasDrsvd_batch_strided(handle, m, n, rank, A_strided, lda, stride_a, S_strided, stride_s, num_ops); }
inline int kblas_rsvd_batch(kblasHandle_t handle, int m, int n, int rank, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops)
{ return kblasSrsvd_batch_strided(handle, m, n, rank, A_strided, lda, stride_a, S_strided, stride_s, num_ops); }

// Array of pointers interface
inline int kblas_gesvj_batch(kblasHandle_t handle, int m, int n, double** A_ptrs, int lda, double** S_ptrs, int num_ops)
{ return kblasDgesvj_batch(handle, m, n, A_ptrs, lda, S_ptrs, num_ops); }
inline int kblas_gesvj_batch(kblasHandle_t handle, int m, int n, float** A_ptrs, int lda, float** S_ptrs, int num_ops)
{ return kblasSgesvj_batch(handle, m, n, A_ptrs, lda, S_ptrs, num_ops); }

inline int kblas_gesvj_gram_batch(kblasHandle_t handle, int m, int n, double** A_ptrs, int lda, double** S_ptrs, int num_ops)
{ return kblasDgesvj_gram_batch(handle, m, n, A_ptrs, lda, S_ptrs, num_ops); }
inline int kblas_gesvj_gram_batch(kblasHandle_t handle, int m, int n, float** A_ptrs, int lda, float** S_ptrs, int num_ops)
{ return kblasSgesvj_gram_batch(handle, m, n, A_ptrs, lda, S_ptrs, num_ops); }

inline int kblas_rsvd_batch(kblasHandle_t handle, int m, int n, int rank, double** A_ptrs, int lda, double** S_ptrs, int num_ops)
{ return kblasDrsvd_batch(handle, m, n, rank, A_ptrs, lda, S_ptrs, num_ops); }
inline int kblas_rsvd_batch(kblasHandle_t handle, int m, int n, int rank, float** A_ptrs, int lda, float** S_ptrs, int num_ops)
{ return kblasSrsvd_batch(handle, m, n, rank, A_ptrs, lda, S_ptrs, num_ops); }

#endif

#endif //__BATCH_SVD_H__
