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

template<class T, class T_ptr>
__global__
void batchCopyMatrixBlockKernel(
	int rows, int cols,
	T_ptr dest_array, int row_offset_dest, int col_offset_dest, int ld_dest, int stride_dest,
	T_ptr src_array, int row_offset_src, int col_offset_src, int ld_src, int stride_src,
	int ops
)
{
    int op_id = blockIdx.z;
    if(op_id >= ops) return;

    T* dest_block = getOperationPtr<T>(dest_array, op_id, stride_dest) + row_offset_dest + col_offset_dest * ld_dest;
	T* src_block = getOperationPtr<T>(src_array, op_id, stride_src) + row_offset_src + col_offset_src * ld_src;

    int row_index = blockDim.x * blockIdx.x + threadIdx.x;
	int col_index = (blockDim.y * blockIdx.y + threadIdx.y) * COLS_PER_THREAD;

	if(row_index >= rows || col_index >= cols) 
		return;

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

template<class T, class T_ptr>
int batchCopyMatrixBlock(
	kblasHandle_t handle, int rows, int cols,
	T_ptr dest_array, int row_offset_dest, int col_offset_dest, int ld_dest, int stride_dest,
	T_ptr src_array, int row_offset_src, int col_offset_src, int ld_src, int stride_src, int ops
)
{
	if(ops == 0 || rows == 0 || cols == 0)
		return KBLAS_Success;
	
	int max_thread_y = MAX_THREAD_Y;
	
    int thread_x = WARP_SIZE, thread_y = kmin(max_thread_y, iDivUp(cols, COLS_PER_THREAD));
    int grid_x = iDivUp(rows, thread_x), grid_y = iDivUp(cols, thread_y * COLS_PER_THREAD);

    dim3 dimBlock(thread_x, thread_y, 1);
    dim3 dimGrid(grid_x, grid_y, ops);

    batchCopyMatrixBlockKernel<T, T_ptr><<< dimGrid, dimBlock, 0, handle->stream >>> (
		rows, cols, dest_array, row_offset_dest, col_offset_dest, ld_dest, stride_dest,
		src_array, row_offset_src, col_offset_src, ld_src, stride_src,
		ops
	);

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
	return batchCopyMatrixBlock<double, double**>(
		handle, rows, cols,
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
	return batchCopyMatrixBlock<float, float**>(
		handle, rows, cols,
		dest_array, row_offset_dest, col_offset_dest, ld_dest, 0,
		src_array, row_offset_src, col_offset_src, ld_src, 0, ops
	);
}

// Strided interface
extern "C" int kblasDcopyBlock_batch_strided(
	kblasHandle_t handle, int rows, int cols,
	double* dest_array, int row_offset_dest, int col_offset_dest, int ld_dest, int stride_dest,
	double* src_array, int row_offset_src, int col_offset_src, int ld_src, int stride_src, int ops
)
{
	return batchCopyMatrixBlock<double, double*>(
		handle, rows, cols,
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
	return batchCopyMatrixBlock<float, float*>(
		handle, rows, cols,
		dest_array, row_offset_dest, col_offset_dest, ld_dest, stride_dest,
		src_array, row_offset_src, col_offset_src, ld_src, stride_src, ops
	);
}
