#include <mini_blas_gpu.h>
#include <batch_pstrf.h>
#include <gpu_util.h>

#include <cub/cub.cuh>

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
MyKeyValue<Key, Value> __shfl_xor(const MyKeyValue<Key, Value> &a, int mask) { return MyKeyValue<Key, Value>(__shfl_xor(a.key, mask), __shfl_xor(a.value, mask)); }

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
__global__
void batch_pstrf_init_kernel(T** M_ptrs, int ldm, int** piv_ptrs, int* r, int dim, T* diag_data, int num_ops)
{
	int op_id = blockIdx.x * blockDim.y + threadIdx.y;
	if(op_id >= num_ops) return;
	
	T* m_op = M_ptrs[op_id];
	T* d_op = diag_data + op_id * dim;
	int* piv_op = piv_ptrs[op_id];
	
	int tid = threadIdx.x;
	if(tid < dim) 
	{
		d_op[tid] = m_op[tid + tid * ldm];
		piv_op[tid] = tid;
	}
}

template<class T>
__host__ __device__ __forceinline__
void swap_values(T& a, T& b) { T temp = a; a = b; b = temp; }

template<class T, int BLOCK_DIM>
__global__
void batch_pstrf_shared_kernel(T** M_ptrs, int ldm, int** piv_ptrs, int* r, int dim, int num_ops)
{
	extern __shared__ char sdata[];
	
	// Figure out who gets what work
	int op_id     = blockIdx.x;
	int tid       = threadIdx.x;
	int shared_ld = blockDim.x + 1;
	
    if(op_id >= num_ops) return;
	
	// Grab global memory data pointers for this operation
    T* m_op = M_ptrs[op_id];
    int* piv_op = piv_ptrs[op_id];
	
	// Store full matrix in shared memory 
	T* s_matrix = (T*)sdata;
	int* piv_shared = (int*)(s_matrix + dim * shared_ld);
	int* shared_pivot = piv_shared + dim;
	int rank_op = dim;
	
	if(tid < dim) 
	{
		piv_shared[tid] = tid;
		for(int i = 0; i < dim; i++) 
			s_matrix[tid + i * shared_ld] = m_op[tid + i * ldm];
	}
	if(BLOCK_DIM != 32) __syncthreads();
	
	// To avoid branching in the solve loop, we mirror the matrix to include the lower triangle
	if(tid < dim) 
	{
		for(int i = 0; i < tid; i++) 
			s_matrix[tid + i * shared_ld] = s_matrix[i + tid * shared_ld];
	}
	if(BLOCK_DIM != 32) __syncthreads();
	
	// Setting CUB up for finding the max pivot
	typedef MyKeyValue<int, T> PivotData;
	typedef cub::BlockReduce<PivotData, BLOCK_DIM> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	
	// Stopping criteria tol = eps * dim * max(a_jj)
	// TODO: change this from a serial loop to a reduction - using CUB seems to whack
	T max_element = 0;
	for(int i = 0; i < dim; i++)
		if(max_element < s_matrix[i + i * shared_ld])
			max_element = s_matrix[i + i * shared_ld];
	T tol = max_element * dim * KBlasEpsilon<T>::eps;
	if(BLOCK_DIM != 32) __syncthreads();

	for(int k = 0; k < dim; k++)
	{
		// Find the pivot
		PivotData tpd(tid, (tid < k || tid >= dim ? -1 : s_matrix[tid + tid * shared_ld]));
		PivotData pivot = BlockReduce(temp_storage).Reduce(tpd, cub::Max());
		if(tid == 0) shared_pivot[0] = (pivot.value <= tol ? -1 : pivot.key);
		if(BLOCK_DIM != 32) __syncthreads();
		
		// Exit when we find out the matrix is semi-definite
		if(shared_pivot[0] == -1) { rank_op = k; break; }
		
		// Swap rows and columns if necessary 
		if(shared_pivot[0] != k)
		{
			int p = shared_pivot[0];
			// Swap pivot values
			if(tid == 0) swap_values(piv_shared[k], piv_shared[p]);
			// Swap rows
			if(tid < dim) { swap_values(s_matrix[k + tid * shared_ld], s_matrix[p + tid * shared_ld]); }
			if(BLOCK_DIM != 32) __syncthreads();
			// Swap Columns
			if(tid < dim) { swap_values(s_matrix[tid + k * shared_ld], s_matrix[tid + p * shared_ld]); }
			if(BLOCK_DIM != 32) __syncthreads();
		}

		// Time to cholesky
		if(tid == k) s_matrix[k + k * shared_ld] = sqrt(s_matrix[k + k * shared_ld]);
		if(BLOCK_DIM != 32) __syncthreads();
		
		// Update the row
		if(tid > k && tid < dim) s_matrix[k + tid * shared_ld] /= s_matrix[k + k * shared_ld];
		if(BLOCK_DIM != 32) __syncthreads();
		
		// Update the trailing submatrix
		if(tid > k && tid < dim) 
		{
			for(int i = k; i < dim; i++)
				s_matrix[tid + i * shared_ld] -= s_matrix[k + tid * shared_ld] * s_matrix[k + i * shared_ld];
		}
		if(BLOCK_DIM != 32) __syncthreads();
	}
	
	// Flush cached data to global memory
	if(tid == 0) r[op_id] = rank_op;
	if(tid < dim)
	{
		piv_op[tid] = piv_shared[tid];
		for(int i = 0; i < dim; i++) 
			m_op[tid + i * ldm] = (tid > i ? 0 : s_matrix[tid + i * shared_ld]);
	}
}

template<class T, int BLOCK_DIM>
__global__
void batch_pstrf_shared_kernel_v2(T** M_ptrs, int ldm, int** piv_ptrs, int* r, int dim, int num_ops)
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
    T* m_op = M_ptrs[op_id];
    int* piv_op = piv_ptrs[op_id];
	
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
		r[op_id] = rank_op;
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

template<class T, int THREADS_PER_OP, int OPS_PER_BLOCK, int NB_COLS>
__global__
void batch_pstrf_panel_kernel(T** M_ptrs, int ldm, int** piv_ptrs, int* r, int dim, int offset, T* diag_data, int num_ops)
{
	extern __shared__ char sdata[];
	
	// Figure out who gets what work
	int local_op_id = threadIdx.x / THREADS_PER_OP;
	int op_id 		= blockIdx.x * OPS_PER_BLOCK + local_op_id;
	int tid 		= threadIdx.x % THREADS_PER_OP;
	int rows 		= dim - offset;
	
    if(op_id >= num_ops) return;
	
	// Grab global memory data pointers for this operation
    T* m_op = M_ptrs[op_id];
	T* d_op = diag_data + op_id * dim + offset;
    int* piv_op = piv_ptrs[op_id] + offset;
	
	// Row for the panel is stored in registers
	//T row_data[NB_COLS];
	
	// Pivot data is stored in shared memory 
	T* d_shared 	 = (T*)sdata + local_op_id * THREADS_PER_OP;
	T* d_last 	 	 = (T*)sdata + OPS_PER_BLOCK * THREADS_PER_OP;
	int*  piv_shared = (int*)d_last + local_op_id * THREADS_PER_OP;

	d_shared[tid] 	= (tid < offset  || tid >= rows ? -1 : d_op[tid]);
	piv_shared[tid] = (tid < rows ? piv_op[tid] : -1);
	
	// Setting CUB up for finding the max pivot
	typedef MyKeyValue<int, T> PivotData;
	typedef cub::BlockReduce<PivotData, THREADS_PER_OP> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage[OPS_PER_BLOCK];
	
	#pragma unroll
	for(int k = 0; k < NB_COLS; k++)
	{
		// Find the pivot
		PivotData tpd(piv_shared[tid], d_shared[tid]);
		PivotData pivot = BlockReduce(temp_storage[local_op_id]).Reduce(tpd, cub::Max());
		
		// Swap rows and columns if necessary - has to be done in global memory
		int global_k = k + offset;
		if(pivot.key != global_k)
		{
			// Swap rows
			//if(tid < rows) { T temp = m_op[global_k + tid * ldm]; m_op[global_k + tid * ldm] = m_op[pivot.key + tid * ldm]; m_op[pivot.key + tid * ldm] = temp; }
			// Swap Columns
			if(tid < rows) { T temp = m_op[tid + global_k * ldm]; m_op[tid + global_k * ldm] = m_op[tid + pivot.key * ldm]; m_op[tid + pivot.key * ldm] = temp; }
			// Swap cached pivot data
			if(tid == 0)   
			{ 
				{ int temp = piv_shared[k]; piv_shared[k] = piv_shared[pivot.key]; piv_shared[pivot.key] = temp; }
				{ T temp = d_last[k]; d_last[k] = d_last[pivot.key]; d_last[pivot.key] = temp; }
			}
		}
		// Load in the pivot column from global memory
	}
}

template<class T>
void batch_pstrf_panel(T** M_ptrs, int ldm, int** piv, int* r, int dim, T* diag_data, int offset, int num_ops)
{
	const int threads_per_block = 512;
	if(dim > threads_per_block) {printf("PSTRF can only handle matrix dim <= %d!\n", threads_per_block); return;}
	
	int threads_per_op = iDivUp(dim, WARP_SIZE) * WARP_SIZE;
	int ops_per_block = threads_per_block / threads_per_op;
	int blocks = iDivUp(num_ops, ops_per_block);
	
	dim3 dimBlock(ops_per_block * threads_per_op, 1);
	dim3 dimGrid(blocks, 1);
	
	size_t smem_needed = ops_per_block * (threads_per_op * sizeof(T) + threads_per_op * sizeof(int));
    
    batch_pstrf_panel_kernel<T, 32, 16, 32><<< dimGrid, dimBlock, smem_needed >>>(M_ptrs, ldm, piv, r, dim, offset, diag_data, num_ops);
}

template<class T>
void batch_pstrf_init(T** M_ptrs, int ldm, int** piv, int* r, int dim, T* diag_data, int num_ops)
{
	const int threads_per_block = 512;
	if(dim > threads_per_block) {printf("PSTRF can only handle matrix dim <= %d!\n", threads_per_block); return;}
	
	// Initialize the pivot arrays
	int ops_per_block = threads_per_block / dim;
	int blocks = iDivUp(num_ops, ops_per_block);
	
	dim3 dimBlock(dim, ops_per_block);
	dim3 dimGrid(blocks, 1);
	
	batch_pstrf_init_kernel<T><<< dimGrid, dimBlock >>>(M_ptrs, ldm, piv, r, dim, diag_data, num_ops);
}

template<class T>
void batch_pstrf(T** M_ptrs, int ldm, int** piv, int* r, int dim, int num_ops)
{
	int threads = iDivUp(dim, WARP_SIZE) * WARP_SIZE;
	dim3 dimBlock(threads, 8);
	dim3 dimGrid(num_ops, 1);
	size_t smem_needed = (dim * (threads + 1) * sizeof(T) + (dim + 1) * sizeof(int));
	
	if     (dim <= 32) batch_pstrf_shared_kernel_v2<T, 32><<< dimGrid, dimBlock, smem_needed >>>(M_ptrs, ldm, piv, r, dim, num_ops);
	else if(dim <= 64) batch_pstrf_shared_kernel_v2<T, 64><<< dimGrid, dimBlock, smem_needed >>>(M_ptrs, ldm, piv, r, dim, num_ops);
	else printf("Invalid size!\n");
	gpuErrchk( cudaGetLastError() );
}

void kblas_pstrf_batched(float** M_ptrs, int ldm, int** piv, int* r, int dim, int num_ops)
{
	batch_pstrf<float>(M_ptrs, ldm, piv, r, dim, num_ops);
}

void kblas_pstrf_batched(double** M_ptrs, int ldm, int** piv, int* r, int dim, int num_ops)
{
	batch_pstrf<double>(M_ptrs, ldm, piv, r, dim, num_ops);
}
