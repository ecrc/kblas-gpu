#ifndef __BATCH_MM_WRAPPERS_H__
#define __BATCH_MM_WRAPPERS_H__

#include <mini_blas_gpu.h>

void batch_gemm(int trans_a, int trans_b, int m, int n, int k, float alpha, float** dA_array, int lda, float** dB_array, int ldb, float beta, float** dC_array, int ldc, int batchCount, GPUBlasHandle& handle);
void batch_syrk(int uppper, int trans, int n, int k, float alpha, float** dA_array, int lda, float beta, float** dC_array, int ldc, int batchCount, GPUBlasHandle& handle);

void batch_gemm(int trans_a, int trans_b, int m, int n, int k, double alpha, double** dA_array, int lda, double** dB_array, int ldb, double beta, double** dC_array, int ldc, int batchCount, GPUBlasHandle& handle);
void batch_syrk(int uppper, int trans, int n, int k, double alpha, double** dA_array, int lda, double beta, double** dC_array, int ldc, int batchCount, GPUBlasHandle& handle);

#endif 
