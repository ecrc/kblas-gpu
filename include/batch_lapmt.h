/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file include/batch_lapmt.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Wajih Halim Boukaram
 * @date 2023-09-26
 **/
 
 #ifndef __BATCH_LAPMT_H__
 #define __BATCH_LAPMT_H__

//############################################################################
//BATCH lapmt routines
//############################################################################

#ifdef __cplusplus
extern "C" {
#endif

int kblasDlapmt_batch_strided(
	kblasHandle_t handle, int forward, int m, int n, double* A_strided, int lda, int stride_A,
    int *J_strided, int stride_J, int ops
);

int kblasSlapmt_batch_strided(
	kblasHandle_t handle, int forward, int m, int n, float* A_strided, int lda, int stride_A,
    int *J_strided, int stride_J, int ops
);

int kblasDlapmt_batch(
	kblasHandle_t handle, int forward, int m, int n, double** A_ptrs, int lda,
    int **J_ptrs, int ops
);

int kblasSlapmt_batch(
	kblasHandle_t handle, int forward, int m, int n, float** A_ptrs, int lda,
    int **J_ptrs, int ops
);

int kblasDlapmt_vbatch(
	kblasHandle_t handle, int forward, int *m, int *n, int max_m, int max_n,
    double** A_ptrs, int *lda, int **J_ptrs, int ops
);

int kblasSlapmt_vbatch(
	kblasHandle_t handle, int forward, int *m, int *n, int max_m, int max_n,
    float** A_ptrs, int *lda, int **J_ptrs, int ops
);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
inline int kblas_lapmt_batch(
	kblasHandle_t handle, int forward, int m, int n, double* A_strided, int lda, int stride_A,
    int *J_strided, int stride_J, int ops
) { return kblasDlapmt_batch_strided(handle, forward, m, n, A_strided, lda, stride_A, J_strided, stride_J, ops); }

inline int kblas_lapmt_batch(
	kblasHandle_t handle, int forward, int m, int n, float* A_strided, int lda, int stride_A,
    int *J_strided, int stride_J, int ops
) { return kblasSlapmt_batch_strided(handle, forward, m, n, A_strided, lda, stride_A, J_strided, stride_J, ops); }

inline int kblas_lapmt_batch(
	kblasHandle_t handle, int forward, int m, int n, double** A_ptrs, int lda,
    int **J_ptrs, int ops
) { return kblasDlapmt_batch(handle, forward, m, n, A_ptrs, lda, J_ptrs, ops); }

inline int kblas_lapmt_batch(
	kblasHandle_t handle, int forward, int m, int n, float** A_ptrs, int lda,
    int **J_ptrs, int ops
) { return kblasSlapmt_batch(handle, forward, m, n, A_ptrs, lda, J_ptrs, ops); }

inline int kblas_lapmt_vbatch(
	kblasHandle_t handle, int forward, int *m, int *n, int max_m, int max_n,
    double** A_ptrs, int *lda, int **J_ptrs, int ops
) { return kblasDlapmt_vbatch(handle, forward, m, n, max_m, max_n, A_ptrs, lda, J_ptrs, ops); }

inline int kblas_lapmt_vbatch(
	kblasHandle_t handle, int forward, int *m, int *n, int max_m, int max_n,
    float** A_ptrs, int *lda, int **J_ptrs, int ops
) { return kblasSlapmt_vbatch(handle, forward, m, n, max_m, max_n, A_ptrs, lda, J_ptrs, ops); }

#endif

#endif
