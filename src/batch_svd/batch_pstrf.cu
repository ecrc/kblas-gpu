#include "hip/hip_runtime.h"
#include <stdio.h>
#include <hipblas.h>
#include <algorithm>

#include "kblas.h"
#include "kblas_struct.h"
#include "kblas_gpu_util.ch"
#include "batch_pstrf.h"

template<class Key, class Value>
struct MyKeyValue {
	Key key;
	Value value;
	__host__ __device__ __forceinline__
    MyKeyValue() {}
	__host__ __device__ __forceinline__
	MyKeyValue(Key key, Value value) { this->key = key; this->value = value; }
};
template<class Key, class Value>
__host__ __device__ __forceinline__
bool operator >(const MyKeyValue<Key, Value> &a, const MyKeyValue<Key, Value> &b) { return a.value > b.value; }

template<class Key, class Value>
__host__ __device__ __forceinline__
MyKeyValue<Key, Value> __shfl_xor(const MyKeyValue<Key, Value> &a, int mask) { 
	return MyKeyValue<Key, Value>(
		__shfl_xor_sync(0xFFFFFFFF, a.key, mask), __shfl_xor_sync(0xFFFFFFFF, a.value, mask)
	); 
}

template<class Pair>
__host__ __device__ __forceinline__
void warp_max(Pair& a)
{
    #pragma unroll 
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
	{
		Pair b = __shfl_xor(a, mask);
        if(b > a) a = b;
	}
}

template<class T, int BLOCK_DIM>
__host__ __device__ __forceinline__
void getMaxDiagEntry(int tid, int k, volatile T* s_matrix, int shared_ld, int dim, int& index, T& val)
{
	// Pivot data: the diagonal entry and its index
	typedef MyKeyValue<int, T> PivotData;
	
	PivotData my_pivot_data(tid, (tid < k || tid >= dim ? -1 : s_matrix[tid + tid * shared_ld]));
	if(BLOCK_DIM != WARP_SIZE) 
	{
		int next_index = tid + WARP_SIZE;
		if(next_index >= k && next_index < dim)
		{
			T next_val = s_matrix[next_index + next_index * shared_ld];
			if(next_val > my_pivot_data.value)
			{
				my_pivot_data.value = next_val;
				my_pivot_data.key = next_index;
			}
		}
	}
	warp_max<PivotData>(my_pivot_data);
	index = my_pivot_data.key;
	val = my_pivot_data.value;
}

template<class T>
__host__ __device__ __forceinline__
void swap_values(T& a, T& b) { T temp = a; a = b; b = temp; }

template<class T, class T_ptr, class Ti_ptr, int BLOCK_DIM>
__global__
void batch_pstrf_shared_kernel(int dim, T_ptr M_batch, int ldm, int stride_m, Ti_ptr piv_batch, int stride_piv, int* ranks, int num_ops)
{
	extern __shared__ char sdata[];
	
	// Figure out who gets what work
	int op_id     	  = blockIdx.x;
	int tid       	  = threadIdx.x;
	int warp_id   	  = threadIdx.y;
	int shared_ld 	  = blockDim.x + 1;
	int num_warps     = blockDim.y;
	int cols_per_warp = iDivUp(dim, num_warps);
	
    if(op_id >= num_ops) return;
	
	// Grab global memory data pointers for this operation
    T* m_op = getOperationPtr(M_batch, op_id, stride_m);
    int* piv_op = getOperationPtr(piv_batch, op_id, stride_piv);
	
	// Store full matrix in shared memory 
	T* s_matrix = (T*)sdata;
	int* piv_shared = (int*)(s_matrix + dim * shared_ld);
	int* shared_pivot = piv_shared + dim;
	int rank_op = dim;
	
	// Each warp loads a column of the matrix 
	if(tid < dim) 
	{
		if(warp_id == 0) 
			piv_shared[tid] = tid;
		for(int i = 0; i < cols_per_warp; i++) 
		{
			int col_index = warp_id + i * num_warps;
			if(col_index < dim)
				s_matrix[tid + col_index * shared_ld] = m_op[tid + col_index * ldm];
		}
	}
	__syncthreads();
	
	// To avoid branching in the solve loop, we mirror the matrix to include the lower triangle
	if(tid < dim) 
	{
		for(int i = 0; i < cols_per_warp; i++) 
		{
			int col_index = warp_id + i * num_warps;
			if(col_index < dim && tid > col_index)
				s_matrix[tid + col_index * shared_ld] = s_matrix[col_index + tid * shared_ld];
		}
	}
	__syncthreads();

	// Stopping criteria tol = eps * dim * max(a_jj)
	T tol;
	if(tid < WARP_SIZE && warp_id == 0)
	{
		int index; T val;
		getMaxDiagEntry<T, BLOCK_DIM>(tid, 0, s_matrix, shared_ld, dim, index, val);
		tol = val * dim * KBlasEpsilon<T>::eps;
	}
	__syncthreads();

	for(int k = 0; k < dim; k++)
	{
		// Only warp 0 handles pivoting
		if(tid < WARP_SIZE && warp_id == 0)
		{
			// Find the pivot
			int index; T val;
			getMaxDiagEntry<T, BLOCK_DIM>(tid, k, s_matrix, shared_ld, dim, index, val);
			if(tid == 0) shared_pivot[0] = (val <= tol ? -1 : index);
			//if(tid == 0) printf("Max diag = %e\n", val);
		}
		__syncthreads();
			
		// Exit when we find out the matrix is semi-definite
		if(shared_pivot[0] == -1) { rank_op = k; break; }

		// Swap rows and columns if necessary 
		if(shared_pivot[0] != k)
		{
			int p = shared_pivot[0];
			// Swap pivot values
			if(warp_id == 0 && tid == 0) swap_values(piv_shared[k], piv_shared[p]);
			// Swap rows
			if(warp_id == 0 && tid < dim) { swap_values(s_matrix[k + tid * shared_ld], s_matrix[p + tid * shared_ld]); }
			if(BLOCK_DIM != WARP_SIZE) __syncthreads();
			// Swap Columns
			if(warp_id == 0 && tid < dim) { swap_values(s_matrix[tid + k * shared_ld], s_matrix[tid + p * shared_ld]); }
			if(BLOCK_DIM != WARP_SIZE) __syncthreads();
		}

		// Time to cholesky
		if(warp_id == 0 && tid == k) 
			s_matrix[k + k * shared_ld] = sqrt(s_matrix[k + k * shared_ld]);
		if(BLOCK_DIM != WARP_SIZE) __syncthreads();
		
		// Update the row
		if(warp_id == 0 && tid > k && tid < dim) 
			s_matrix[k + tid * shared_ld] /= s_matrix[k + k * shared_ld];
		__syncthreads();
		
		// Update the trailing submatrix
		if(tid > k && tid < dim) 
		{
			for(int i = 0; i < cols_per_warp; i++) 
			{
				int col_index = warp_id + i * num_warps;
				if(col_index < dim)
					s_matrix[tid + col_index * shared_ld] -= s_matrix[k + tid * shared_ld] * s_matrix[k + col_index * shared_ld];
			}
		}
		__syncthreads();
	}
	
	// Flush cached data to global memory
	if(warp_id == 0 && tid == 0) 
		ranks[op_id] = rank_op;
	
	if(tid < dim)
	{
		if(warp_id == 0) 
			piv_op[tid] = piv_shared[tid];
		for(int i = 0; i < cols_per_warp; i++) 
		{
			int col_index = warp_id + i * num_warps;
			if(col_index < dim)
				m_op[tid + col_index * ldm] = (tid > col_index ? 0 : s_matrix[tid + col_index * shared_ld]);
		}
	}
}

template<class T, class T_ptr, class Ti_ptr>
int batch_pstrf_template(kblasHandle_t handle, int dim, T_ptr M_batch, int ldm, int stride_m, Ti_ptr piv_batch, int stride_piv, int* ranks, int num_ops)
{
	if(dim > 64)
		return KBLAS_Error_WrongInput;
	
	int threads = iDivUp(dim, WARP_SIZE) * WARP_SIZE;
	dim3 dimBlock(threads, 8);
	dim3 dimGrid(num_ops, 1);
	size_t smem_needed = (dim * (threads + 1) * sizeof(T) + (dim + 1) * sizeof(int));
	
	if     (hipLaunchKernelGGL(HIP_KERNEL_NAME(dim<= 32) batch_pstrf_shared_kernel<T, T_ptr, Ti_ptr, 32>), dim3(dimGrid), dim3(dimBlock), smem_needed, handle->stream , dim, M_batch, ldm, stride_m, piv_batch, stride_piv, ranks, num_ops);
	else if(hipLaunchKernelGGL(HIP_KERNEL_NAME(dim<= 64) batch_pstrf_shared_kernel<T, T_ptr, Ti_ptr, 64>), dim3(dimGrid), dim3(dimBlock), smem_needed, handle->stream , dim, M_batch, ldm, stride_m, piv_batch, stride_piv, ranks, num_ops);
	
	check_error_ret( hipGetLastError(), KBLAS_UnknownError );
	
	return KBLAS_Success;
}

//------------------------------------------------------------------------------
// Array of pointers interface
int kblasDpstrf_batch(kblasHandle_t handle, int n, double** A_array, int lda, int** piv_array, int* ranks, int num_ops)
{
	return batch_pstrf_template<double, double**, int**>(handle, n, A_array, lda, 0, piv_array, 0, ranks, num_ops);
}

int kblasSpstrf_batch(kblasHandle_t handle, int n, float** A_array, int lda, int** piv_array, int* ranks, int num_ops)
{
	return batch_pstrf_template<float, float**, int**>(handle, n, A_array, lda, 0, piv_array, 0, ranks, num_ops);
}

//------------------------------------------------------------------------------
// Strided interface
int kblasDpstrf_batch_strided(kblasHandle_t handle, int n, double* A_strided, int lda, int stride_a, int* piv_strided, int stride_piv, int* ranks, int num_ops)
{
	return batch_pstrf_template<double, double*, int*>(handle, n, A_strided, lda, stride_a, piv_strided, stride_piv, ranks, num_ops);
}

int kblasSpstrf_batch_strided(kblasHandle_t handle, int n, float* A_strided, int lda, int stride_a, int* piv_strided, int stride_piv, int* ranks, int num_ops)
{
	return batch_pstrf_template<float, float*, int*>(handle, n, A_strided, lda, stride_a, piv_strided, stride_piv, ranks, num_ops);
}
