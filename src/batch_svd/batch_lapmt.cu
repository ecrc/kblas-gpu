/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_svd/batch_lapmt.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Wajih Halim Boukaram
 * @date 2023-09-26
 **/

#include <cublas_v2.h>

#include <stdio.h>

#include "kblas.h"
#include "kblas_struct.h"
#include "kblas_gpu_util.ch"
#include "batch_lapmt.h"

template<class T>
__device__ void forwardLapmt(int tid, int row_index, T* g_A, int lda, int rows, int cols, int *s_J)
{
    // Forward permutation 
    for(int i = 0; i < cols; i++)
    {
        int si = s_J[i];
        __syncthreads();

        if(si > 0)
            continue;
        
        // Swap sign and sync to make sure all threads read the swapped version
        int j = i;
        if(tid == 0) 
            s_J[j] = -s_J[j];
        __syncthreads();

        int in = -si - 1;
        
        while(s_J[in] < 0)
        {
            // Swap column data 
            if(row_index < rows)
            {
                T t = g_A[row_index + j * lda];
                g_A[row_index + j * lda] = g_A[row_index + in * lda];
                g_A[row_index + in * lda] = t;
            }
            
            // Sync before swapping signs to make sure all threads have swapped 
            // their assigned column data 
            __syncthreads();

            if(tid == 0)
                s_J[in] = -s_J[in];
            // Sync after to make sure all threads have the updated sign
            __syncthreads();

            j = in;
            in = s_J[in] - 1;
        }
    } 
}

template<class T>
__device__ void backwardLapmt(int tid, int row_index, T* g_A, int lda, int rows, int cols, int *s_J)
{
    // Backward permutation 
    for(int i = 0; i < cols; i++)
    {
        int si = s_J[i];
        __syncthreads();

        if(si > 0)
            continue;
        
        // Swap sign and sync to make sure all threads read the swapped version
        if(tid == 0)
            s_J[i] = -s_J[i];
        __syncthreads();

        int j = -si - 1;
        
        while(j != i)
        {
            // Swap column data
            if(row_index < rows)
            {
                T t = g_A[row_index + i * lda];
                g_A[row_index + i * lda] = g_A[row_index + j * lda];
                g_A[row_index + j * lda] = t;
            }
            
            // Sync before swapping signs to make sure all threads have swapped 
            // their assigned column data 
            __syncthreads();
            if(tid == 0)
                s_J[j] = -s_J[j];

            // Sync after to make sure all threads have the updated sign
            __syncthreads();

            j = s_J[j] - 1;
        }
    } 
}

template<class T, class T_ptr, class I_ptr, class Dim_Type>
__global__ void batchLapmtKernel(
    int forward, Dim_Type rows_batch, Dim_Type cols_batch, T_ptr A_batch, Dim_Type lda_batch, int stride_A, 
    I_ptr J_batch, int stride_J, int ops
)
{
    extern __shared__ char sdata[];	

    int op_id = blockIdx.y;
    if(op_id >= ops) return;

    int tid = threadIdx.x;
    int row_index = blockDim.x * blockIdx.x + tid;

	int rows = getOperationDim(rows_batch, op_id);
    int cols = getOperationDim(cols_batch, op_id);
	int lda = getOperationDim(lda_batch, op_id);

	if(cols <= 0 || lda <= 0 || rows <= 0) 
		return;

    T* g_A = getOperationPtr<T>(A_batch, op_id, stride_A);
    int* J_ptr = getOperationPtr<int>(J_batch, op_id, stride_J);

    if(!g_A || !J_ptr)
        return;
    
    // g_A += row_index;

    // Store a sign flipped version of J in shared memory
    int* s_J = (int*)sdata;
    int J_index = tid;
    while(J_index < cols)
    {
        s_J[J_index] = -J_ptr[J_index];
        J_index += blockDim.x;
    }
    __syncthreads();
    
    if(forward == 1)
        forwardLapmt<T>(tid, row_index, g_A, lda, rows, cols, s_J);
    else
        backwardLapmt<T>(tid, row_index, g_A, lda, rows, cols, s_J);
}

// TODO: might be more efficient to assign multiple column elements to
// a single thread
template<class T, class T_ptr, class I_ptr, class Dim_Type>
int batch_lapmt(
	kblasHandle_t handle, int forward, Dim_Type rows_batch, Dim_Type cols_batch, int max_rows, int max_cols,
	T_ptr A_batch, Dim_Type lda_batch, int stride_A, I_ptr J_batch, int stride_J, int ops
)
{
	if(ops == 0 || max_rows == 0 || max_cols == 0)
		return KBLAS_Success;
	
	int max_thread_x = 512;
	
    int thread_x = kmin(max_thread_x, max_rows);
    int grid_x = iDivUp(max_rows, thread_x);
	
	int op_increment = 65535;
	size_t smem = max_cols * sizeof(int);

	for(int op_start = 0; op_start < ops; op_start += op_increment)
	{
		int batch_size = kmin(op_increment, ops - op_start);
		
		T_ptr A_sub_batch = advanceOperationPtr(A_batch, op_start, stride_A);
		I_ptr J_sub_batch = advanceOperationPtr(J_batch, op_start, stride_J);

		Dim_Type rows_sub_batch = advanceOperationDim(rows_batch, op_start);
		Dim_Type cols_sub_batch = advanceOperationDim(cols_batch, op_start);
		Dim_Type lda_sub_batch = advanceOperationDim(lda_batch, op_start);
		
		dim3 dimBlock(thread_x, 1, 1);		
		dim3 dimGrid(grid_x, batch_size, 1);
		
        batchLapmtKernel<T, T_ptr, I_ptr, Dim_Type><<< dimGrid, dimBlock, smem, handle->stream >>> (
            forward, rows_sub_batch, cols_sub_batch, A_sub_batch, lda_sub_batch, stride_A, 
            J_sub_batch, stride_J, batch_size	
        );
	}

    check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	return KBLAS_Success;
}

extern "C" int kblasDlapmt_batch_strided(
	kblasHandle_t handle, int forward, int m, int n, double* A_strided, int lda, int stride_A,
    int *J_strided, int stride_J, int ops
)
{
    return batch_lapmt<double, double*, int*, int>(
        handle, forward, m, n, m, n, A_strided, lda, stride_A, J_strided, stride_J, ops
    );
}

extern "C" int kblasSlapmt_batch_strided(
	kblasHandle_t handle, int forward, int m, int n, float* A_strided, int lda, int stride_A,
    int *J_strided, int stride_J, int ops
)
{
    return batch_lapmt<float, float*, int*, int>(
        handle, forward, m, n, m, n, A_strided, lda, stride_A, J_strided, stride_J, ops
    );
}

extern "C" int kblasDlapmt_batch(
	kblasHandle_t handle, int forward, int m, int n, double** A_ptrs, int lda,
    int **J_ptrs, int ops
)
{
    return batch_lapmt<double, double**, int**, int>(
        handle, forward, m, n, m, n, A_ptrs, lda, 0, J_ptrs, 0, ops
    );
}

extern "C" int kblasSlapmt_batch(
	kblasHandle_t handle, int forward, int m, int n, float** A_ptrs, int lda,
    int **J_ptrs, int ops
)
{
    return batch_lapmt<float, float**, int**, int>(
        handle, forward, m, n, m, n, A_ptrs, lda, 0, J_ptrs, 0, ops
    );
}

extern "C" int kblasDlapmt_vbatch(
	kblasHandle_t handle, int forward, int *m, int *n, int max_m, int max_n,
    double** A_ptrs, int *lda, int **J_ptrs, int ops
)
{
    return batch_lapmt<double, double**, int**, int*>(
        handle, forward, m, n, max_m, max_n, A_ptrs, lda, 0, J_ptrs, 0, ops
    );
}

extern "C" int kblasSlapmt_vbatch(
	kblasHandle_t handle, int forward, int *m, int *n, int max_m, int max_n,
    float** A_ptrs, int *lda, int **J_ptrs, int ops
)
{
    return batch_lapmt<float, float**, int**, int*>(
        handle, forward, m, n, max_m, max_n, A_ptrs, lda, 0, J_ptrs, 0, ops
    );
}
