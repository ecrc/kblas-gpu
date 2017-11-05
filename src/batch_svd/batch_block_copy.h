#ifndef __BATCH_BLOCK_COPY_H__
#define __BATCH_BLOCK_COPY_H__

#include <mini_blas_gpu.h>

// Array of pointers interface
void batchCopyMatrixBlock(
	double** orig_array, int row_offset_orig, int col_offset_orig, int ld_orig, 
	double** copy_array, int row_offset_copy, int col_offset_copy, int ld_copy,
	int rows, int cols, int ops, GPUBlasHandle& handle
);

void batchCopyMatrixBlock(
	float** orig_array, int row_offset_orig, int col_offset_orig, int ld_orig, 
	float** copy_array, int row_offset_copy, int col_offset_copy, int ld_copy,
	int rows, int cols, int ops, GPUBlasHandle& handle
);

// Strided interface
void batchCopyMatrixBlock(
	double* orig_array, int row_offset_orig, int col_offset_orig, int ld_orig, int stride_orig, 
	double* copy_array, int row_offset_copy, int col_offset_copy, int ld_copy, int stride_copy, 
	int rows, int cols, int ops, GPUBlasHandle& handle
);

void batchCopyMatrixBlock(
	float* orig_array, int row_offset_orig, int col_offset_orig, int ld_orig, int stride_orig, 
	float* copy_array, int row_offset_copy, int col_offset_copy, int ld_copy, int stride_copy, 
	int rows, int cols, int ops, GPUBlasHandle& handle
);

// Common interface for array of pointers with dummy strides
template<class T>
inline void batchCopyMatrixBlock(
	T** orig_array, int row_offset_orig, int col_offset_orig, int ld_orig, int stride_orig, 
	T** copy_array, int row_offset_copy, int col_offset_copy, int ld_copy, int stride_copy, 
	int rows, int cols, int ops, GPUBlasHandle& handle
)
{
	batchCopyMatrixBlock(
		orig_array, row_offset_orig, col_offset_orig, ld_orig, 
		copy_array, row_offset_copy, col_offset_copy, ld_copy,
		rows, cols, ops, handle
	);
}


#endif
