#ifndef __BATCH_BLOCK_SET_H__
#define __BATCH_BLOCK_SET_H__

#include <mini_blas_gpu.h>

void batchBlockSetIdentity(double** block_ptrs, int row_offset, int col_offset, int ld, int rows, int cols, int ops, GPUBlasHandle& handle);
void batchBlockSetZero(double** block_ptrs, int row_offset, int col_offset, int ld, int rows, int cols, int ops, GPUBlasHandle& handle);

void batchBlockSetIdentity(float** block_ptrs, int row_offset, int col_offset, int ld, int rows, int cols, int ops, GPUBlasHandle& handle);
void batchBlockSetZero(float** block_ptrs, int row_offset, int col_offset, int ld, int rows, int cols, int ops, GPUBlasHandle& handle);

#endif
