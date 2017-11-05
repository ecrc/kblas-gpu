#ifndef __BATCH_TRANSPOSE_H__
#define __BATCH_TRANSPOSE_H__

#include <mini_blas_gpu.h>

void batch_transpose(int m, int n, double** matrix_data, int ldm, double** transpose_data, int ldt, int ops, GPUBlasHandle& handle);
void batch_transpose(int m, int n, float** matrix_data, int ldm, float** transpose_data, int ldt, int ops, GPUBlasHandle& handle);

#endif
