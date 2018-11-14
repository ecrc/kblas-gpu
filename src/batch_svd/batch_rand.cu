#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

#include "kblas.h"
#include "kblas_struct.h"
#include "kblas_gpu_util.ch"
#include "batch_rand.h"

//------------------------------------------------------------------------------
// Random state structure and routines
struct KBlasRandState 
{
	curandState_t* states;
	size_t num_states;
};

__global__ void init_random_states(unsigned int seed, curandState_t* states, size_t num_states) 
{
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(thread_id > num_states)
		return;
	
	curand_init(seed, thread_id, 0, &states[thread_id]);
}

int kblasInitRandState(kblasHandle_t handle, kblasRandState_t* state, int num_states, unsigned int seed)
{
	*state = new KBlasRandState();
	(*state)->num_states = num_states;	
	
	check_error_ret( cudaMalloc((void**)&((*state)->states), (*state)->num_states * sizeof(curandState_t)), KBLAS_Error_Allocation );
	
	int block_threads = 256;
	int blocks = iDivUp(num_states, block_threads);
	
	dim3 dimBlock(block_threads, 1);
	dim3 dimGrid(blocks, 1, 1);
	
	init_random_states<<< dimGrid, dimBlock, 0, handle->stream >>>(seed, (*state)->states, (*state)->num_states);
	check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	
	return KBLAS_Success;
}

int kblasDestroyRandState(kblasRandState_t state)
{
	if(state && state->states)
	{
		check_error_ret( cudaFree(state->states), KBLAS_Error_Deallocation );
		delete state;
	}
	
	return KBLAS_Success;
}

//------------------------------------------------------------------------------
template<class T> __device__ __forceinline__ T tcurand_normal(curandState* state);
template<> __device__ __forceinline__  float tcurand_normal<float>(curandState* state) { return curand_normal(state); }
template<> __device__ __forceinline__  double tcurand_normal<double>(curandState* state) { return curand_normal_double(state); }

template<class T, class T_ptr>
__global__
void batch_rand_kernel2(
	int rows, int cols, T_ptr A_batch, int lda, int stride_a, curandState_t* states, 
	int num_ops, int padded_rows, int elements_per_thread
)
{
	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Cache the state in local memory
	curandState state = states[thread_index];
	
	int curr_op = -1;
	T* A_block = NULL;
	
	for(int e = 0; e < elements_per_thread; e++)
	{
		int global_linear_index = e * blockDim.x * gridDim.x + thread_index;
		int op_index = global_linear_index / (padded_rows * cols);
		if(op_index >= num_ops) break;
		
		if(op_index != curr_op)
		{
			curr_op = op_index;
			A_block = getOperationPtr<T>(A_batch, op_index, stride_a);
		}
		
		int matrix_linear_index = global_linear_index % (padded_rows * cols);
		int row_index = matrix_linear_index % padded_rows;
		int col_index = matrix_linear_index / padded_rows;
		
		if(row_index < rows && col_index < cols)
			A_block[row_index + col_index * lda] = tcurand_normal<T>(&state);
	}
	
	// Flush back to global memory
	states[thread_index] = state;
}

template<class T, class T_ptr>
__global__
void batch_rand_kernel(
	int rows, int cols, T_ptr A_batch, int lda, int stride_a, curandState_t* states, 
	int num_ops, int thread_block_rows 
)
{
	int row_start_index = blockIdx.x * blockDim.x + threadIdx.x;
	int state_index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	int row_increment = blockDim.x * gridDim.x;
	
	// Cache the state in local memory
	curandState state = states[state_index];
	
	for(int op = blockIdx.y; op < num_ops; op += gridDim.y)
	{
		T* A_block = getOperationPtr<T>(A_batch, op, stride_a);
		
		for(int j = 0; j < cols; j++)
		{
			for(int b = 0; b < thread_block_rows ; b++)
			{
				int row_index = row_start_index + b * row_increment;
				if(row_index < rows)
					A_block[row_index + j * lda] = tcurand_normal<T>(&state);
			}
		}
	}
		
	// Flush back to global memory
	states[state_index] = state;
}

template<class T, class T_ptr>
int batch_rand_template2(kblasHandle_t handle, int rows, int cols, T_ptr A_batch, int lda, int stride_a, kblasRandState_t state, int num_ops)
{	
	int block_x = 128;
	int grid_x  = state->num_states / block_x;
	
	int padded_rows = iDivUp(rows, block_x) * block_x;
	int total_entries = padded_rows * cols * num_ops;
	int elements_per_thread = iDivUp(total_entries, block_x * grid_x);

	dim3 dimBlock(block_x, 1, 1);
	dim3 dimGrid(grid_x, 1, 1);
	
	batch_rand_kernel<T, T_ptr><<< dimGrid, dimBlock, 0, handle->stream >>>(
		rows, cols, A_batch, lda, stride_a, state->states, 
		num_ops, padded_rows, elements_per_thread
	);

	return KBLAS_Success;
}

template<class T, class T_ptr>
int batch_rand_template(kblasHandle_t handle, int rows, int cols, T_ptr A_batch, int lda, int stride_a, kblasRandState_t state, int num_ops)
{	
	int block_x = 64;
	int grouped_states = state->num_states / block_x;
	
	int block_rows = iDivUp(rows, block_x);
	int grid_x = kmin(grouped_states, block_rows);
	int grid_y = grouped_states / grid_x;
	
	int thread_block_rows = iDivUp(block_rows, grid_x);
	dim3 dimBlock(block_x, 1, 1);
	dim3 dimGrid(grid_x, grid_y, 1);
	
	batch_rand_kernel<T, T_ptr><<< dimGrid, dimBlock, 0, handle->stream >>>(
		rows, cols, A_batch, lda, stride_a, state->states, 
		num_ops, thread_block_rows
	);

	return KBLAS_Success;
}

//------------------------------------------------------------------------------
// Array of pointers interface
int kblasDrand_batch(kblasHandle_t handle, int m, int n, double** A_ptrs, int lda, kblasRandState_t state, int num_ops)
{
	return batch_rand_template<double, double**>(handle, m, n, A_ptrs, lda, 0, state, num_ops);
}

int kblasSrand_batch(kblasHandle_t handle, int m, int n, float** A_ptrs, int lda, kblasRandState_t state, int num_ops)
{
	return batch_rand_template<float, float**>(handle, m, n, A_ptrs, lda, 0, state, num_ops);
}

//------------------------------------------------------------------------------
// Strided interface
int kblasDrand_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, kblasRandState_t state, int num_ops)
{
	return batch_rand_template<double, double*>(handle, m, n, A_strided, lda, stride_a, state, num_ops);
}

int kblasSrand_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, kblasRandState_t state, int num_ops)
{
	return batch_rand_template<float, float*>(handle, m, n, A_strided, lda, stride_a, state, num_ops);
}
