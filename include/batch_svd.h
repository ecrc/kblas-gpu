#ifndef __BATCH_SVD_H__
#define __BATCH_SVD_H__

//TODO provide C API (without function overloading)

#define SHARED_SVD_DIM_LIMIT 64
#define RSVD_DEFAULT_SAMPLES 64

////////////////////////////////////////////////////////////////////
// Strided interface
////////////////////////////////////////////////////////////////////
int kblas_gesvj_batched(int m, int n, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops, GPUBlasHandle& handle);
int kblas_gesvj_gram_batched(int m, int n, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops, GPUBlasHandle& handle);
int kblas_gesvj_qr_batched(int m, int n, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops, GPUBlasHandle& handle);
int kblas_rsvd_batched(int m, int n, int rank, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops, GPUBlasHandle& handle);

int kblas_gesvj_batched(int m, int n, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops, GPUBlasHandle& handle);
int kblas_gesvj_gram_batched(int m, int n, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops, GPUBlasHandle& handle);
int kblas_gesvj_qr_batched(int m, int n, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops, GPUBlasHandle& handle);
int kblas_rsvd_batched(int m, int n, int rank, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops, GPUBlasHandle& handle);

////////////////////////////////////////////////////////////////////
// Array of pointers interface
////////////////////////////////////////////////////////////////////
int kblas_gesvj_batched(int m, int n, float** A_array, int lda, float** S_array, int num_ops, GPUBlasHandle& handle);
int kblas_gesvj_gram_batched(int m, int n, float** A_array, int lda, float** S_array, int num_ops, GPUBlasHandle& handle);
int kblas_gesvj_qr_batched(int m, int n, float** A_array, int lda, float** S_array, int num_ops, GPUBlasHandle& handle);
int kblas_rsvd_batched(int m, int n, int rank, float** A_array, int lda, float** S_array, int num_ops, GPUBlasHandle& handle);

int kblas_gesvj_batched(int m, int n, double** A_array, int lda, double** S_array, int num_ops, GPUBlasHandle& handle);
int kblas_gesvj_gram_batched(int m, int n, double** A_array, int lda, double** S_array, int num_ops, GPUBlasHandle& handle);
int kblas_gesvj_qr_batched(int m, int n, double** A_array, int lda, double** S_array, int num_ops, GPUBlasHandle& handle);
int kblas_rsvd_batched(int m, int n, int rank, double** A_array, int lda, double** S_array, int num_ops, GPUBlasHandle& handle);

#endif

