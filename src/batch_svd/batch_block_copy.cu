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
 * @version 2.0.0
 * @author Wajih Halim Boukaram
 * @date 2017-11-13
 **/

#include <cublas_v2.h>

#include "kblas.h"
#include "kblas_struct.h"
#include "kblas_gpu_util.ch"
#include "batch_block_copy.h"

template<class T, class T_ptr>
__global__
void batchCopyMatrixBlockKernel(
	int rows, int cols,
	T_ptr dest_array, int row_offset_dest, int col_offset_dest, int ld_dest, int stride_dest,
	T_ptr src_array, int row_offset_src, int col_offset_src, int ld_src, int stride_src,
	int ops, int rows_per_thread
)
{
    int op_id = blockIdx.x * blockDim.y + threadIdx.y;
    if(op_id >= ops) return;

    T* dest_block = getOperationPtr<T>(dest_array, op_id, stride_dest) + row_offset_dest + col_offset_dest * ld_dest;
	T* src_block = getOperationPtr<T>(src_array, op_id, stride_src) + row_offset_src + col_offset_src * ld_src;

    int tid = threadIdx.x;

    for(int j = 0; j < cols; j++)
    {
        for(int i = 0; i < rows_per_thread; i++)
        {
            int row_index = WARP_SIZE * i + tid;
            if(row_index < rows)
                dest_block[row_index + j * ld_dest] = src_block[row_index + j * ld_src];
        }
    }
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

    int ops_per_block = 8;
    int rows_per_thread = iDivUp(rows, WARP_SIZE);
    int blocks = iDivUp(ops, ops_per_block);

    dim3 dimBlock(WARP_SIZE, ops_per_block);
    dim3 dimGrid(blocks, 1);

    batchCopyMatrixBlockKernel<T, T_ptr><<< dimGrid, dimBlock, 0, handle->stream >>> (
		rows, cols, dest_array, row_offset_dest, col_offset_dest, ld_dest, stride_dest,
		src_array, row_offset_src, col_offset_src, ld_src, stride_src,
		ops, rows_per_thread
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
