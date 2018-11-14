/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_svd/batch_transpose.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Wajih Halim Boukaram
 * @date 2018-11-14
 **/

#ifndef __BATCH_TRANSPOSE_H__
#define __BATCH_TRANSPOSE_H__

#ifdef __cplusplus
extern "C" {
#endif

// Strided interface
int kblasDtranspose_batch_strided(kblasHandle_t handle, int m, int n, double* matrix_strided, int ldm, int stride_m, double* transpose_strided, int ldt, int stride_t, int ops);
int kblasStranspose_batch_strided(kblasHandle_t handle, int m, int n, float* matrix_strided, int ldm, int stride_m, float* transpose_strided, int ldt, int stride_t, int ops);

// Array of pointers interface
int kblasDtranspose_batch(kblasHandle_t handle, int m, int n, double** matrix_ptrs, int ldm, double** transpose_ptrs, int ldt, int ops);
int kblasStranspose_batch(kblasHandle_t handle, int m, int n, float** matrix_ptrs, int ldm, float** transpose_ptrs, int ldt, int ops);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
// Strided interface
inline int kblas_transpose_batch(kblasHandle_t handle, int m, int n, double* matrix_strided, int ldm, int stride_m, double* transpose_strided, int ldt, int stride_t, int ops)
{ return kblasDtranspose_batch_strided(handle, m, n, matrix_strided, ldm, stride_m, transpose_strided, ldt, stride_t, ops); }
inline int kblas_transpose_batch(kblasHandle_t handle, int m, int n, float* matrix_strided, int ldm, int stride_m, float* transpose_strided, int ldt, int stride_t, int ops)
{ return kblasStranspose_batch_strided(handle, m, n, matrix_strided, ldm, stride_m, transpose_strided, ldt, stride_t, ops); }

// Array of pointers interface
inline int kblas_transpose_batch(kblasHandle_t handle, int m, int n, double** matrix_ptrs, int ldm, double** transpose_ptrs, int ldt, int ops)
{ return kblasDtranspose_batch(handle, m, n, matrix_ptrs, ldm, transpose_ptrs, ldt, ops); }
inline int kblas_transpose_batch(kblasHandle_t handle, int m, int n, float** matrix_ptrs, int ldm, float** transpose_ptrs, int ldt, int ops)
{ return kblasStranspose_batch(handle, m, n, matrix_ptrs, ldm, transpose_ptrs, ldt, ops); }
#endif

#endif // __BATCH_TRANSPOSE_H__
