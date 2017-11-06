#ifndef __BATCH_QR_H__
#define __BATCH_QR_H__

#include "kblas.h"

#ifdef __cplusplus
extern "C" {
#endif

// Strided interface
int kblasDgeqrf_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* tau, int stride_tau, int num_ops);
int kblasSgeqrf_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* tau, int stride_tau, int num_ops);

int kblasDtsqrf_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* tau, int stride_tau, int num_ops);
int kblasStsqrf_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* tau, int stride_tau, int num_ops);

int kblasDorgqr_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* tau, int stride_tau, int num_ops);
int kblasSorgqr_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* tau, int stride_tau, int num_ops);

int kblasDcopy_upper_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* R_strided, int ldr, int stride_R, int num_ops);
int kblasScopy_upper_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* R_strided, int ldr, int stride_R, int num_ops);

// Array of pointers interface
int kblasDgeqrf_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** tau_array, int num_ops);
int kblasSgeqrf_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** tau_array, int num_ops);

int kblasDtsqrf_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** tau_array, int num_ops);
int kblasStsqrf_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** tau_array, int num_ops);

int kblasDorgqr_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** tau_array, int num_ops);
int kblasSorgqr_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** tau_array, int num_ops);

int kblasDcopy_upper_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** R_array, int ldr, int num_ops);
int kblasScopy_upper_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** R_array, int ldr, int num_ops);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
// Strided interface
inline int kblas_geqrf_batch(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* tau, int stride_tau, int num_ops)
{ return kblasDgeqrf_batch_strided(handle, m, n, A_strided, lda, stride_a, tau, stride_tau, num_ops); }
inline int kblas_geqrf_batch(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* tau, int stride_tau, int num_ops)
{ return kblasSgeqrf_batch_strided(handle, m, n, A_strided, lda, stride_a, tau, stride_tau, num_ops); }

inline int kblas_tsqrf_batch(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* tau, int stride_tau, int num_ops)
{ return kblasDtsqrf_batch_strided(handle, m, n, A_strided, lda, stride_a, tau, stride_tau, num_ops); }
inline int kblas_tsqrf_batch(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* tau, int stride_tau, int num_ops)
{ return kblasStsqrf_batch_strided(handle, m, n, A_strided, lda, stride_a, tau, stride_tau, num_ops); }

inline int kblas_orgqr_batch(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* tau, int stride_tau, int num_ops)
{ return kblasDorgqr_batch_strided(handle, m, n, A_strided, lda, stride_a, tau, stride_tau, num_ops); }
inline int kblas_orgqr_batch(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* tau, int stride_tau, int num_ops)
{ return kblasSorgqr_batch_strided(handle, m, n, A_strided, lda, stride_a, tau, stride_tau, num_ops); }

inline int kblas_copy_upper_batch(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* R_strided, int ldr, int stride_R, int num_ops)
{ return kblasDcopy_upper_batch_strided(handle, m, n, A_strided, lda, stride_a, R_strided, ldr, stride_R, num_ops); }
inline int kblas_copy_upper_batch(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* R_strided, int ldr, int stride_R, int num_ops)
{ return kblasScopy_upper_batch_strided(handle, m, n, A_strided, lda, stride_a, R_strided, ldr, stride_R, num_ops); }

// Array of pointers interface
inline int kblas_geqrf_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** tau_array, int num_ops)
{ return kblasDgeqrf_batch(handle, m, n, A_array, lda, tau_array, num_ops); }
inline int kblas_geqrf_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** tau_array, int num_ops)
{ return kblasSgeqrf_batch(handle, m, n, A_array, lda, tau_array, num_ops); }

inline int kblas_tsqrf_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** tau_array, int num_ops)
{ return kblasDtsqrf_batch(handle, m, n, A_array, lda, tau_array, num_ops); }
inline int kblas_tsqrf_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** tau_array, int num_ops)
{ return kblasStsqrf_batch(handle, m, n, A_array, lda, tau_array, num_ops); }

inline int kblas_orgqr_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** tau_array, int num_ops)
{ return kblasDorgqr_batch(handle, m, n, A_array, lda, tau_array, num_ops); }
inline int kblas_orgqr_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** tau_array, int num_ops)
{ return kblasSorgqr_batch(handle, m, n, A_array, lda, tau_array, num_ops); }

inline int kblas_copy_upper_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** R_array, int ldr, int num_ops)
{ return kblasDcopy_upper_batch(handle, m, n, A_array, lda, R_array, ldr, num_ops); }
inline int kblas_copy_upper_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** R_array, int ldr, int num_ops)
{ return kblasScopy_upper_batch(handle, m, n, A_array, lda, R_array, ldr, num_ops); }

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

#endif
