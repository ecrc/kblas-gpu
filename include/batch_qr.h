#ifndef __BATCH_QR_H__
#define __BATCH_QR_H__

//TODO provide C API (without function overloading)

// Strided interface
int kblas_dgeqrf_batched(int m, int n, double* A_strided, int lda, int stride_a, double* tau, int stride_tau, int num_ops, GPUBlasHandle& handle);
int kblas_sgeqrf_batched(int m, int n, float* A_strided, int lda, int stride_a, float* tau, int stride_tau, int num_ops, GPUBlasHandle& handle);

int kblas_dtsqrf_batched(int m, int n, double* A_strided, int lda, int stride_a, double* tau, int stride_tau, int num_ops, GPUBlasHandle& handle);
int kblas_stsqrf_batched(int m, int n, float* A_strided, int lda, int stride_a, float* tau, int stride_tau, int num_ops, GPUBlasHandle& handle);

int kblas_dorgqr_batched(int m, int n, double* A_strided, int lda, int stride_a, double* tau, int stride_tau, int num_ops, GPUBlasHandle& handle);
int kblas_sorgqr_batched(int m, int n, float* A_strided, int lda, int stride_a, float* tau, int stride_tau, int num_ops, GPUBlasHandle& handle);

int kblas_copy_upper_batched(int m, int n, double* A_strided, int lda, int stride_a, double* R_strided, int ldr, int stride_R, int num_ops, GPUBlasHandle& handle);
int kblas_copy_upper_batched(int m, int n, float* A_strided, int lda, int stride_a, float* R_strided, int ldr, int stride_R, int num_ops, GPUBlasHandle& handle);

// Array of pointers interface
int kblas_dgeqrf_batched(int m, int n, double** A_array, int lda, double** tau_array, int num_ops, GPUBlasHandle& handle);
int kblas_sgeqrf_batched(int m, int n, float** A_array, int lda, float** tau_array, int num_ops, GPUBlasHandle& handle);

int kblas_dtsqrf_batched(int m, int n, double** A_array, int lda, double** tau_array, int num_ops, GPUBlasHandle& handle);
int kblas_stsqrf_batched(int m, int n, float** A_array, int lda, float** tau_array, int num_ops, GPUBlasHandle& handle);

int kblas_dorgqr_batched(int m, int n, double** A_array, int lda, double** tau_array, int num_ops, GPUBlasHandle& handle);
int kblas_sorgqr_batched(int m, int n, float** A_array, int lda, float** tau_array, int num_ops, GPUBlasHandle& handle);

int kblas_copy_upper_batched(int m, int n, double** A_array, int lda, double** R_array, int ldr, int num_ops, GPUBlasHandle& handle);
int kblas_copy_upper_batched(int m, int n, float** A_array, int lda, float** R_array, int ldr, int num_ops, GPUBlasHandle& handle);

#ifdef __cplusplus
// Strided interface
inline int kblas_geqrf_batched(int m, int n, double* A_strided, int lda, int stride_a, double* tau, int stride_tau, int num_ops, GPUBlasHandle& handle)
{ return kblas_dgeqrf_batched(m, n, A_strided, lda, stride_a, tau, stride_tau, num_ops, handle); }
inline int kblas_geqrf_batched(int m, int n, float* A_strided, int lda, int stride_a, float* tau, int stride_tau, int num_ops, GPUBlasHandle& handle)
{ return kblas_sgeqrf_batched(m, n, A_strided, lda, stride_a, tau, stride_tau, num_ops, handle); }

inline int kblas_tsqrf_batched(int m, int n, double* A_strided, int lda, int stride_a, double* tau, int stride_tau, int num_ops, GPUBlasHandle& handle)
{ return kblas_dtsqrf_batched(m, n, A_strided, lda, stride_a, tau, stride_tau, num_ops, handle); }
inline int kblas_tsqrf_batched(int m, int n, float* A_strided, int lda, int stride_a, float* tau, int stride_tau, int num_ops, GPUBlasHandle& handle)
{ return kblas_stsqrf_batched(m, n, A_strided, lda, stride_a, tau, stride_tau, num_ops, handle); }

inline int kblas_orgqr_batched(int m, int n, double* A_strided, int lda, int stride_a, double* tau, int stride_tau, int num_ops, GPUBlasHandle& handle)
{ return kblas_dorgqr_batched(m, n, A_strided, lda, stride_a, tau, stride_tau, num_ops, handle); }
inline int kblas_orgqr_batched(int m, int n, float* A_strided, int lda, int stride_a, float* tau, int stride_tau, int num_ops, GPUBlasHandle& handle)
{ return kblas_sorgqr_batched(m, n, A_strided, lda, stride_a, tau, stride_tau, num_ops, handle); }

// Array of pointers interface
inline int kblas_geqrf_batched(int m, int n, double** A_array, int lda, double** tau_array, int num_ops, GPUBlasHandle& handle)
{ return kblas_dgeqrf_batched(m, n, A_array, lda, tau_array, num_ops, handle); }
inline int kblas_geqrf_batched(int m, int n, float** A_array, int lda, float** tau_array, int num_ops, GPUBlasHandle& handle)
{ return kblas_sgeqrf_batched(m, n, A_array, lda, tau_array, num_ops, handle); }

inline int kblas_tsqrf_batched(int m, int n, double** A_array, int lda, double** tau_array, int num_ops, GPUBlasHandle& handle)
{ return kblas_dtsqrf_batched(m, n, A_array, lda, tau_array, num_ops, handle); }
inline int kblas_tsqrf_batched(int m, int n, float** A_array, int lda, float** tau_array, int num_ops, GPUBlasHandle& handle)
{ return kblas_stsqrf_batched(m, n, A_array, lda, tau_array, num_ops, handle); }

inline int kblas_orgqr_batched(int m, int n, double** A_array, int lda, double** tau_array, int num_ops, GPUBlasHandle& handle)
{ return kblas_dorgqr_batched(m, n, A_array, lda, tau_array, num_ops, handle); }
inline int kblas_orgqr_batched(int m, int n, float** A_array, int lda, float** tau_array, int num_ops, GPUBlasHandle& handle)
{ return kblas_sorgqr_batched(m, n, A_array, lda, tau_array, num_ops, handle); }

// Common interface for array of pointers with dummy strides
template<class T>
inline int kblas_geqrf_batched(int m, int n, T** A_array, int lda, int stride_a, T** tau_array, int stride_tau, int num_ops, GPUBlasHandle& handle)
{ return kblas_geqrf_batched(m, n, A_array, lda, tau_array, num_ops, handle); }
template<class T>
inline int kblas_tsqrf_batched(int m, int n, T** A_array, int lda, int stride_a, T** tau_array, int stride_tau, int num_ops, GPUBlasHandle& handle)
{ return kblas_tsqrf_batched(m, n, A_array, lda, tau_array, num_ops, handle); }
template<class T>
inline int kblas_orgqr_batched(int m, int n, T** A_array, int lda, int stride_a, T** tau_array, int stride_tau, int num_ops, GPUBlasHandle& handle)
{ return kblas_orgqr_batched(m, n, A_array, lda, tau_array, num_ops, handle); }
template<class T>
inline int kblas_copy_upper_batched(int m, int n, T** A_array, int lda, int stride_a, T** R_array, int ldr, int stride_R, int num_ops, GPUBlasHandle& handle)
{ return kblas_copy_upper_batched(m, n, A_array, lda, R_array, ldr, num_ops, handle); }
#endif

#endif
