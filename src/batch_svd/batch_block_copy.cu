/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_svd/batch_block_copy.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Wajih Halim Boukaram
 * @date 2018-11-14
 **/

#include <cublas_v2.h>

#include <stdio.h>

#include "kblas.h"
#include "kblas_struct.h"
#include "kblas_gpu_util.ch"
#include "batch_block_copy.h"

#define COLS_PER_THREAD	8
#define MAX_THREAD_Y	8

template<class T, class T_ptr, class Dim_Type>
__global__
void batchCopyMatrixBlockKernel(
	Dim_Type rows_batch, Dim_Type cols_batch,
	T_ptr dest_array, int row_offset_dest, int col_offset_dest, Dim_Type ld_dest_batch, int stride_dest,
	T_ptr src_array, int row_offset_src, int col_offset_src, Dim_Type ld_src_batch, int stride_src,
	int ops
)
{
    int op_id = blockIdx.z;
    if(op_id >= ops) return;

    int row_index = blockDim.x * blockIdx.x + threadIdx.x;
	int col_index = (blockDim.y * blockIdx.y + threadIdx.y) * COLS_PER_THREAD;
	
	int rows = getOperationDim(rows_batch, op_id);
	int cols = getOperationDim(cols_batch, op_id);
	
	if(row_index >= rows || col_index >= cols) 
		return;

	int ld_dest = getOperationDim(ld_dest_batch, op_id);
	int ld_src  = getOperationDim(ld_src_batch, op_id);

    T* dest_block = getOperationPtr<T>(dest_array, op_id, stride_dest) + row_offset_dest + col_offset_dest * ld_dest;
	T* src_block = getOperationPtr<T>(src_array, op_id, stride_src) + row_offset_src + col_offset_src * ld_src;
	
	dest_block += row_index + col_index * ld_dest;
	src_block  += row_index + col_index * ld_src;

	T reg_buffer[COLS_PER_THREAD];
	
	#pragma unroll 
    for(int j = 0; j < COLS_PER_THREAD; j++)
		if(j + col_index < cols)
			reg_buffer[j] = src_block[j * ld_src];

	#pragma unroll 
	for(int j = 0; j < COLS_PER_THREAD; j++)
		if(j + col_index < cols)
			dest_block[j * ld_dest] = reg_buffer[j];
}

template<class T, class T_ptr, class Dim_Type>
int batchCopyMatrixBlock(
	kblasHandle_t handle, Dim_Type rows_array, Dim_Type cols_array, int max_rows, int max_cols,
	T_ptr dest_array, int row_offset_dest, int col_offset_dest, Dim_Type ld_dest_array, int stride_dest,
	T_ptr src_array, int row_offset_src, int col_offset_src, Dim_Type ld_src_array, int stride_src, int ops
)
{
	if(ops == 0 || max_rows == 0 || max_cols == 0)
		return KBLAS_Success;
	
	int max_thread_y = MAX_THREAD_Y;
	
    int thread_x = WARP_SIZE, thread_y = kmin(max_thread_y, iDivUp(max_cols, COLS_PER_THREAD));
    int grid_x = iDivUp(max_rows, thread_x), grid_y = iDivUp(max_cols, thread_y * COLS_PER_THREAD);
	
	int op_increment = 65535;
	
	for(int op_start = 0; op_start < ops; op_start += op_increment)
	{
		int batch_size = kmin(op_increment, ops - op_start);
		
		T_ptr dest_batch = advanceOperationPtr(dest_array, op_start, stride_dest);
		T_ptr src_batch = advanceOperationPtr(src_array, op_start, stride_src);
		
		Dim_Type rows_batch = advanceOperationDim(rows_array, op_start);
		Dim_Type cols_batch = advanceOperationDim(cols_array, op_start);
		Dim_Type ld_dest_batch = advanceOperationDim(ld_dest_array, op_start);
		Dim_Type ld_src_batch  = advanceOperationDim(ld_src_array, op_start);
		
		dim3 dimBlock(thread_x, thread_y, 1);		
		dim3 dimGrid(grid_x, grid_y, batch_size);
		
		batchCopyMatrixBlockKernel<T, T_ptr><<< dimGrid, dimBlock, 0, handle->stream >>> (
			rows_batch, cols_batch, dest_batch, row_offset_dest, col_offset_dest, ld_dest_batch, stride_dest,
			src_batch, row_offset_src, col_offset_src, ld_src_batch, stride_src,
			batch_size	
		);
	}

    check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	return KBLAS_Success;
}

// Array of pointers interface
extern "C" int kblasDcopyBlock_batch(
	kblasHandle_t handle, int rows, int cols,
	double** dest_array, int row_offset_dest, int col_offset_dest, int ld_dest,
	double** src_array, int row_offset_src, int col_offset_src, int ld_src, int ops
)
{
	return batchCopyMatrixBlock<double, double**, int>(
		handle, rows, cols, rows, cols,
		dest_array, row_offset_dest, col_offset_dest, ld_dest, 0,
		src_array, row_offset_src, col_offset_src, ld_src, 0, ops
	);
}

extern "C" int kblasScopyBlock_batch(
	kblasHandle_t handle, int rows, int cols,
	float** dest_array, int row_offset_dest, int col_offset_dest, int ld_dest,
	float** src_array, int row_offset_src, int col_offset_src, int ld_src, int ops
)
{
	return batchCopyMatrixBlock<float, float**, int>(
		handle, rows, cols, rows, cols,
		dest_array, row_offset_dest, col_offset_dest, ld_dest, 0,
		src_array, row_offset_src, col_offset_src, ld_src, 0, ops
	);
}

// Variable batch interface 
extern "C" int kblasScopyBlock_vbatch(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, int max_rows, int max_cols,
	float** dest_array, int* ld_dest, float** src_array, int* ld_src, int ops
)
{
	return batchCopyMatrixBlock<float, float**, int*>(
		handle, rows_batch, cols_batch, max_rows, max_cols,
		dest_array, 0, 0, ld_dest, 0,
		src_array, 0, 0, ld_src, 0, ops
	);
}

extern "C" int kblasDcopyBlock_vbatch(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, int max_rows, int max_cols,
	double** dest_array, int* ld_dest, double** src_array, int* ld_src, int ops
)
{
	return batchCopyMatrixBlock<double, double**, int*>(
		handle, rows_batch, cols_batch, max_rows, max_cols,
		dest_array, 0, 0, ld_dest, 0,
		src_array, 0, 0, ld_src, 0, ops
	);
}

// Strided interface
extern "C" int kblasDcopyBlock_batch_strided(
	kblasHandle_t handle, int rows, int cols,
	double* dest_array, int row_offset_dest, int col_offset_dest, int ld_dest, int stride_dest,
	double* src_array, int row_offset_src, int col_offset_src, int ld_src, int stride_src, int ops
)
{
	return batchCopyMatrixBlock<double, double*, int>(
		handle, rows, cols, rows, cols,
		dest_array, row_offset_dest, col_offset_dest, ld_dest, stride_dest,
		src_array, row_offset_src, col_offset_src, ld_src, stride_src, ops
	);
}

extern "C" int kblasScopyBlock_batch_strided(
	kblasHandle_t handle, int rows, int cols,
	float* dest_array, int row_offset_dest, int col_offset_dest, int ld_dest, int stride_dest,
	float* src_array, int row_offset_src, int col_offset_src, int ld_src, int stride_src, int ops
)
{
	return batchCopyMatrixBlock<float, float*, int>(
		handle, rows, cols, rows, cols,
		dest_array, row_offset_dest, col_offset_dest, ld_dest, stride_dest,
		src_array, row_offset_src, col_offset_src, ld_src, stride_src, ops
	);
}
