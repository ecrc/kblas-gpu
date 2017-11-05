#ifndef __BATCH_TSQR_CHOL_H__
#define __BATCH_TSQR_CHOL_H__

#include <mini_blas_gpu.h>

void batch_tsqr_chol(float** M_ptrs, int ldm, float** R_ptrs, int ldr, int** piv, int* rank, int rows, int cols, int num_ops, GPUBlasHandle& handle);
void batch_tsqr_chol(double** M_ptrs, int ldm, double** R_ptrs, int ldr, int** piv, int* rank, int rows, int cols, int num_ops, GPUBlasHandle& handle);

#endif
