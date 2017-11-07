#ifndef __BATCH_RAND_H__
#define __BATCH_RAND_H__

#include <mini_blas_gpu.h>

void batch_rand_blocks(float** M_ptrs, int ldm, int rows, int cols, int num_ops, GPUBlasHandle& handle);
void batch_rand_blocks(double** M_ptrs, int ldm, int rows, int cols, int num_ops, GPUBlasHandle& handle);

#endif
