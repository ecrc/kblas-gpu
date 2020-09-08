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

template<class T, class T_ptr, class TDim>
__global__
void batch_rand_kernel(
	TDim rows_batch, TDim cols_batch, T_ptr A_batch, TDim lda_batch, int stride_a, curandState_t* states, int num_ops
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
		int rows_op = getOperationDim(rows_batch, op);
		int cols_op = getOperationDim(cols_batch, op);		
		int lda_op  = getOperationDim(lda_batch , op);
		
		if(A_block == NULL || rows_op == 0 || cols_op == 0)
			continue;
		
		int thread_block_rows = iDivUp(rows_op, row_increment);
		
		for(int j = 0; j < cols_op; j++)
		{
			for(int b = 0; b < thread_block_rows ; b++)
			{
				int row_index = row_start_index + b * row_increment;
				if(row_index < rows_op)
					A_block[row_index + j * lda_op] = tcurand_normal<T>(&state);
			}
		}
	}
		
	// Flush back to global memory
	states[state_index] = state;
}

template<class T, class T_ptr, class TDim>
int batch_rand_template(kblasHandle_t handle, TDim rows_batch, TDim cols_batch, T_ptr A_batch, TDim lda, int stride_a, int max_rows, kblasRandState_t state, int num_ops)
{	
	int block_x = 64;
	int grouped_states = state->num_states / block_x;
	
	int block_rows = iDivUp(max_rows, block_x);
	int grid_x = kmin(grouped_states, block_rows);
	int grid_y = grouped_states / grid_x;

	dim3 dimBlock(block_x, 1, 1);
	dim3 dimGrid(grid_x, grid_y, 1);
	
	batch_rand_kernel<T, T_ptr><<< dimGrid, dimBlock, 0, handle->stream >>>(
		rows_batch, cols_batch, A_batch, lda, stride_a, state->states, num_ops
	);
	check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	
	return KBLAS_Success;
}

//------------------------------------------------------------------------------
// Array of pointers interface
int kblasDrand_batch(kblasHandle_t handle, int m, int n, double** A_ptrs, int lda, kblasRandState_t state, int num_ops)
{
	return batch_rand_template<double, double**, int>(handle, m, n, A_ptrs, lda, 0, m, state, num_ops);
}

int kblasSrand_batch(kblasHandle_t handle, int m, int n, float** A_ptrs, int lda, kblasRandState_t state, int num_ops)
{
	return batch_rand_template<float, float**, int>(handle, m, n, A_ptrs, lda, 0, m, state, num_ops);
}

//------------------------------------------------------------------------------
// Strided interface
int kblasDrand_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, kblasRandState_t state, int num_ops)
{
	return batch_rand_template<double, double*, int>(handle, m, n, A_strided, lda, stride_a, m, state, num_ops);
}

int kblasSrand_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, kblasRandState_t state, int num_ops)
{
	return batch_rand_template<float, float*, int>(handle, m, n, A_strided, lda, stride_a, m, state, num_ops);
}

//------------------------------------------------------------------------------
// Non-uniform array of pointers interface
int kblasDrand_vbatch(kblasHandle_t handle, int* m, int* n, double** A_ptrs, int* lda, int max_m, kblasRandState_t state, int num_ops)
{
	return batch_rand_template<double, double**, int*>(handle, m, n, A_ptrs, lda, 0, max_m, state, num_ops);
}

int kblasSrand_vbatch(kblasHandle_t handle, int* m, int* n, float** A_ptrs, int* lda, int max_m, kblasRandState_t state, int num_ops)
{
	return batch_rand_template<float, float**, int*>(handle, m, n, A_ptrs, lda, 0, max_m, state, num_ops);
}

//------------------------------------------------------------------------------
// Non-uniform strided interface
int kblasDrand_vbatch_strided(kblasHandle_t handle, int* m, int* n, double* A_strided, int* lda, int stride_a, int max_m, kblasRandState_t state, int num_ops)
{
	return batch_rand_template<double, double*, int*>(handle, m, n, A_strided, lda, stride_a, max_m, state, num_ops);
}

int kblasSrand_vbatch_strided(kblasHandle_t handle, int* m, int* n, float* A_strided, int* lda, int stride_a, int max_m, kblasRandState_t state, int num_ops)
{
	return batch_rand_template<float, float*, int*>(handle, m, n, A_strided, lda, stride_a, max_m, state, num_ops);
}
