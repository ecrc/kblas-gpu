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

#ifdef __cplusplus
extern "C" {
#endif

// workspace query routines for both strided and array of pointers interface
void kblasDgesvj_batch_wsquery(kblasHandle_t handle, int m, int n, int ops);
void kblasSgesvj_batch_wsquery(kblasHandle_t handle, int m, int n, int ops);

void kblasDgesvj_gram_batch_wsquery(kblasHandle_t handle, int m, int n, int ops);
void kblasSgesvj_gram_batch_wsquery(kblasHandle_t handle, int m, int n, int ops);

void kblasDrsvd_batch_batch_wsquery(kblasHandle_t handle, int m, int n, int rank, int ops);
void kblasSrsvd_batch_batch_wsquery(kblasHandle_t handle, int m, int n, int rank, int ops);

// Strided interface
int kblasDgesvj_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops);
int kblasSgesvj_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops);

int kblasDgesvj_gram_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops);
int kblasSgesvj_gram_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops);

int kblasDrsvd_batch_strided(kblasHandle_t handle, int m, int n, int rank, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops);
int kblasSrsvd_batch_strided(kblasHandle_t handle, int m, int n, int rank, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops);

// Array of pointers interface
int kblasDgesvj_batch(kblasHandle_t handle, int m, int n, double** A_ptrs, int lda, double** S_ptrs, int num_ops);
int kblasSgesvj_batch(kblasHandle_t handle, int m, int n, float** A_ptrs, int lda, float** S_ptrs, int num_ops);

int kblasDgesvj_gram_batch(kblasHandle_t handle, int m, int n, double** A_ptrs, int lda, double** S_ptrs, int num_ops);
int kblasSgesvj_gram_batch(kblasHandle_t handle, int m, int n, float** A_ptrs, int lda, float** S_ptrs, int num_ops);

int kblasDrsvd_batch(kblasHandle_t handle, int m, int n, int rank, double** A_ptrs, int lda, double** S_ptrs, int num_ops);
int kblasSrsvd_batch(kblasHandle_t handle, int m, int n, int rank, float** A_ptrs, int lda, float** S_ptrs, int num_ops);

#ifdef __cplusplus
}
#endif

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

template<class T> inline void kblas_rsvd_batch_batch_wsquery(kblasHandle_t handle, int m, int n, int rank, int ops);
template<> inline void kblas_rsvd_batch_batch_wsquery<double>(kblasHandle_t handle, int m, int n, int rank, int ops)
{ kblasDrsvd_batch_batch_wsquery(handle, m, n, rank, ops); }
template<> inline void kblas_rsvd_batch_batch_wsquery<float>(kblasHandle_t handle, int m, int n, int rank, int ops)
{ kblasSrsvd_batch_batch_wsquery(handle, m, n, rank, ops); }

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
