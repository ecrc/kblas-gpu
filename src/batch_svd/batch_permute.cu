#include <mini_blas_gpu.h>
#include <batch_permute.h>
#include <gpu_util.h>

#define COLS_PER_THREAD 16

template<class T>
__global__
void batch_permute_columns_kernel(int rows, int cols, T** M_ptrs, int ldm, int** perm, int* rank, int num_ops)
{
	extern __shared__ char sdata[];
	
	int op_id = blockIdx.y;
	if(op_id > num_ops) return;
	
	T* M_op = M_ptrs[op_id];
	int* perm_op = perm[op_id];
	int* perm_shared = (int*)sdata;
	int thread_index = threadIdx.x;
	int rank_op = (rank ? rank[op_id] : cols);
	int linear_tid = thread_index + threadIdx.y * blockDim.x;
	int block_threads = blockDim.x * blockDim.y;
	int row_index = WARP_SIZE * blockIdx.x + thread_index;
	int col_start = threadIdx.y * COLS_PER_THREAD;
	
	// Load in the permutations
	int perm_id = linear_tid;
	while(perm_id < cols) {perm_shared[perm_id] = perm_op[perm_id]; perm_id += block_threads;}
	__syncthreads();
	
	if(row_index < rows)
	{
		// Each thread stores COLS_PER_THREAD entries of a row
		T perm_row[COLS_PER_THREAD];
		#pragma unroll
		for(int i = 0; i < COLS_PER_THREAD; i++)
		{
			int column_index = col_start + i;
			if(column_index < rank_op)
				perm_row[i] = M_op[row_index + perm_shared[column_index] * ldm];
			else 
				perm_row[i] = 0;
		}
		__syncthreads();
		
		// Now flush the permuted rows back to global memory
		#pragma unroll
		for(int i = 0; i < COLS_PER_THREAD; i++)
		{
			int column_index = col_start + i;
			if(column_index < cols)
				M_op[row_index + column_index * ldm] = perm_row[i];
		}
	}
}

template<class T>
void batch_permute_columns(int rows, int cols, T** M_ptrs, int ldm, int** perm, int* rank, int num_ops)
{
	const int max_cols = 512;
	if(cols > max_cols) printf("kblas_permute_columns_batched cols > %d not supported\n", max_cols);
	
	int thread_x = WARP_SIZE;
	int thread_y = iDivUp(cols, COLS_PER_THREAD);
	int blocks_per_op = iDivUp(rows, thread_x);
	
	dim3 dimBlock(thread_x, thread_y);
	dim3 dimGrid(blocks_per_op, num_ops);
	size_t smem_needed = cols * sizeof(int);
	
	batch_permute_columns_kernel<T><<< dimGrid, dimBlock, smem_needed >>>(rows, cols, M_ptrs, ldm, perm, rank, num_ops);
}

void kblas_permute_columns_batched(int rows, int cols, float** M_ptrs, int ldm, int** perm, int* rank, int num_ops)
{
	batch_permute_columns<float>(rows, cols, M_ptrs, ldm, perm, rank, num_ops);
}

void kblas_permute_columns_batched(int rows, int cols, double** M_ptrs, int ldm, int** perm, int* rank, int num_ops)
{
	batch_permute_columns<double>(rows, cols, M_ptrs, ldm, perm, rank, num_ops);
}
