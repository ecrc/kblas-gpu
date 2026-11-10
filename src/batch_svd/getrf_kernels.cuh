#ifndef __KBLAS_GETRF_KERNELS_CUH__
#define __KBLAS_GETRF_KERNELS_CUH__

#include "kblas_gpu_util.ch"

template<class T>
__global__ void kblas_getrf_smem_kernel(
	int *m_batch, int *n_batch, T **A_ptrs, int *lda_batch, 
	int **ipiv_ptrs, int *info_batch
)
{
	typedef typename kblas_base_type<T>::mag_type mT;
	
	int op_id = blockIdx.x;
	
	int op_m = m_batch[op_id];
	int op_n = n_batch[op_id];
	T* op_A = A_ptrs[op_id];
	int lda_op = lda_batch[op_id];
	int* ipiv_op = ipiv_ptrs[op_id];
	
	// Check if this is a valid operation 
	if(op_m <= 0 || op_n <= 0 || op_A == NULL)
		return;
	
	extern __shared__ char sdata[];
	T *sA = (T*)sdata;
	mT *shared_argmax_x = (mT*)(sA + (WARP_SIZE + 1) * op_n);
	int *shared_argmax_i = (int*)(shared_argmax_x + WARP_SIZE);
	int *shared_ipiv = shared_argmax_i + WARP_SIZE;
	
	int min_mn = (op_m < op_n ? op_m : op_n);
	T t_zero = make_zero<T>(), t_one = make_one<T>();
	const int tid = threadIdx.x;
	const int warp_id = threadIdx.y;
	int warps = blockDim.y;
	int info = 0;
	
	// initialize shared pivot array 
    if(warp_id == 0 && tid < min_mn) 
		shared_ipiv[tid] = 0;
	
	// Load the matrix into shared memory 
	int c = warp_id;
	while(c < op_n)
	{
		sA[tid + c * (WARP_SIZE + 1)] = (tid < op_m ? op_A[tid + c * lda_op] : t_zero);
		c += warps;
	}
	__syncthreads();
	
	for(int i = 0; i < min_mn; i++)
	{
        // Find the row pivot with the largest magnitude 
        if(warp_id == 0)
		{
			T val = sA[tid + i * (WARP_SIZE + 1)];
			shared_argmax_x[tid] = fabs(kblas_get_real(val)) + fabs(kblas_get_img(val));
			shared_argmax_i[tid] = tid;
			kblas_sync_warp();
			
			arg_max_warp(tid, shared_argmax_x + i, shared_argmax_i + i, op_m - i);
		}
		
		// sync before accessing the max data 
		__syncthreads();
		
		mT pivot_abs_value = shared_argmax_x[i];
        int pivot_index = shared_argmax_i[i];

		// If the pivot is zero, don't update the matrix and set the info to the 
		// first failed index 
		if(pivot_abs_value == (mT)0)
		{
			if(info == 0) 
				info = i + 1;
			if(warp_id == 0 && tid == 0)
				shared_ipiv[i] = i;
			__syncthreads();
		}
		else
		{
			// Swap the rows in shared memory and scale the column - warp 0 only 
			if(warp_id == 0)
			{
				if(tid == 0)
					shared_ipiv[i] = pivot_index;
				
				if(tid < op_n)
				{
					T t = sA[i + tid * (WARP_SIZE + 1)];
					sA[i + tid * (WARP_SIZE + 1)] = sA[pivot_index + tid * (WARP_SIZE + 1)];
					sA[pivot_index + tid * (WARP_SIZE + 1)] = t;
				}
				kblas_sync_warp();
			
				if(tid > i)
				{
					T column_scale = kblas_div(t_one, sA[i + i * (WARP_SIZE + 1)]);					
					sA[tid + i * (WARP_SIZE + 1)] *= column_scale;
				}
			}
			__syncthreads();
			
			// Rank update to the trailing submatrix 
			if(tid > i)
			{
				c = i + 1 + warp_id;
				while(c < op_n)
				{
					sA[tid + c * (WARP_SIZE + 1)] -= sA[tid + i * (WARP_SIZE + 1)] * sA[i + c * (WARP_SIZE + 1)];
					c += warps;
				}
			}
			__syncthreads();
		}
	}
	
	// store pivot array 
    if(warp_id == 0 && tid < min_mn) 
		ipiv_op[tid] = shared_ipiv[tid] + 1;
	
	// Save the matrix from shared to global memory 
	if(tid < op_m)
	{
		c = warp_id;
		while(c < op_n)
		{
			op_A[tid + c * lda_op] = sA[tid + c * (WARP_SIZE + 1)];
			c += warps;
		}
	}
}

template<class T, int BS>
__global__ void getrf_trsm_kernel(
	int* m_batch, int* n_batch, T** A_batch, int* lda_batch,
	T** B_batch, int* ldb_batch, int batchCount
)
{
	int op_id = blockDim.y * blockIdx.y + threadIdx.y;
	if(op_id >= batchCount) return;
	
	int row_index = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	
	// fetch all operation data
	int rows = m_batch[op_id];
	int cols = n_batch[op_id];
	int lda  = lda_batch[op_id];
	int ldb  = ldb_batch[op_id];

	if(rows <= 0 || cols <= 0) return;
	
	T* A = A_batch[op_id];
	T* B = B_batch[op_id];
	T t_zero = make_zero<T>();
	
	// Store the data for B in registers and the data for A in shared memory
	T B_reg[BS];
	extern __shared__ char sdata[];
	T* A_shared = (T*)sdata + threadIdx.y * WARP_SIZE * BS;
	
	if(tid < cols)
		for(int i = 0; i < cols; i++)
            A_shared[tid + i * WARP_SIZE] = A[tid + i * lda];
	__syncthreads();
	
	if(row_index >= rows) return;
	
	#pragma unroll 
	for(int i = 0; i < BS; i++)
		B_reg[i] = (i < cols ? B[row_index + i * ldb] : t_zero);
	
	// Now do the triangular solve
	#pragma unroll 
	for(int j = 0; j < BS; j++)
	{
		if(j >= cols) continue;
		#pragma unroll
		for(int k = 0; k < j; k++)
			B_reg[j] -= A_shared[k + j * WARP_SIZE] * B_reg[k];
	}
	
	// Flush data back to global memory
	#pragma unroll 
	for(int i = 0; i < BS; i++)
		if(i < cols)
			B[row_index + i * ldb] = B_reg[i];
}

template<class T, int PANEL_WIDTH, bool transpose>
__device__ void getf2_device(
	int m, int n, T* g_A, int lda, int *g_ipiv, 
	int *g_info, char* sdata, int j
)
{
	typedef typename kblas_base_type<T>::mag_type mT;
	
    const int tid = threadIdx.x;
    T row_data[PANEL_WIDTH];
	
	// keep track of the row index so that we don't have to swap 
	// thread registers when a pivot is found 
    int current_row_index = tid;
    int info = *g_info;
    int min_mn = (m < n ? m : n);
	T t_zero = make_zero<T>(), t_one = make_one<T>();
	
	// Shared memory for swapping rows...
    T *shared_row = (T*)(sdata);
	// ...for computing the pivot using argmax...
    mT *shared_argmax_x = (mT*)(shared_row + PANEL_WIDTH);
    int *shared_argmax_i = (int*)(shared_argmax_x + blockDim.x);
	// ...and storing the local pivots 
    int *shared_ipiv = shared_argmax_i + blockDim.x;

    // initialize shared pivot array 
    if(tid < PANEL_WIDTH && tid < min_mn) shared_ipiv[tid] = 0;

    // Read the matrix row data from global memory into registers 
	// making sure to pad with zeros where necessary 
    #pragma unroll
    for(int i = 0; i < PANEL_WIDTH; i++)
        row_data[i] = (tid >= m || i >= n ? t_zero : g_A[(transpose ? i + tid * lda : tid + i * lda)]);
	
    #pragma unroll
    for(int i = 0; i < PANEL_WIDTH; i++)
	{
		// Don't process the padded columns 
		if(i >= n) break;
		
        // Find the row pivot with the largest magnitude 
        shared_argmax_x[current_row_index] = fabs(kblas_get_real(row_data[i])) + fabs(kblas_get_img(row_data[i]));
        shared_argmax_i[tid] = tid;
        __syncthreads();

		arg_max(tid, shared_argmax_x + i, shared_argmax_i + i, m - i);
		mT pivot_abs_value = shared_argmax_x[i];
        int pivot_index = shared_argmax_i[i];

		// If the pivot is zero, don't update the matrix and set the info to the 
		// first failed index 
		if(pivot_abs_value == (mT)0)
		{
			if(info == 0) 
				info = j + i + 1;
			
			if(current_row_index == i)
				shared_ipiv[i] = i;
		
			__syncthreads();
		}
		else
		{
			// Let the thread with the pivot row write its data to shared memory 
			if(current_row_index == pivot_index)
			{
				shared_ipiv[i] = pivot_index;
				current_row_index = i;
				
				#pragma unroll
				for(int j = 0; j < PANEL_WIDTH; j++)
					shared_row[j] = row_data[j];
			}
			else if(current_row_index == i)
				current_row_index = pivot_index;
			__syncthreads();
	
			// Scale the column and update the trailing submatrix 
			if(current_row_index > i)
			{
				T column_scale = kblas_div(t_one, shared_row[i]);
				row_data[i] *= column_scale;
				
				#pragma unroll
				for(int j = i + 1; j < PANEL_WIDTH; j++)
					row_data[j] -= row_data[i] * shared_row[j];
			}
		}
    }

    // Flush the selected pivots to global memory 
    if(tid < PANEL_WIDTH && tid < min_mn)	
		g_ipiv[tid] = j + shared_ipiv[tid] + 1; 
		
	// Flush the rows from registers to global memory
	// Keep them in their original order since we'll be 
	// doing a column swap for the entire matrix after this 
	// This way the memory writes can be more efficient 
	if(tid < m)
	{
		#pragma unroll
		for(int i = 0; i < PANEL_WIDTH; i++)
			if(i < n)
				g_A[(transpose ? i + tid * lda : tid + i * lda)] = row_data[i];
	}
	
	// Save the factorization status to global memory 
    if(tid == 0) *g_info = info;
}

template<class T, int PANEL_WIDTH>
__global__ void getf2_batched_kernel(
	int *m_batch, int* n_batch, T **A_ptrs, int *lda_batch, 
	int **ipiv_ptrs, int j, int *info_batch
)
{
	int op_id = blockIdx.x;
	
	int op_m = m_batch[op_id];
	int op_n = n_batch[op_id];
	T* op_A = A_ptrs[op_id];
	int lda_op = lda_batch[op_id];
	int* ipiv_op = ipiv_ptrs[op_id] + j;
	
	// Check if this is a valid operation 
	if(op_m <= 0 || op_n <= 0 || op_A == NULL)
		return;

	extern __shared__ char sdata[];	
	
	getf2_device<T, PANEL_WIDTH, false>(
		op_m, op_n, op_A, lda_op, ipiv_op, 
		info_batch + op_id, sdata, j
	);
}

template<class T>
__global__ void swap_columns_batched_kernel(
	int *m_batch, int *n_batch, T **A_ptrs, int *lda_batch, int **ipiv_ptrs, 
	int start, int end
)
{
	int op_id = blockIdx.x;
	
	int m = m_batch[op_id];
	int n = n_batch[op_id];
	T* g_A = A_ptrs[op_id];
	int lda = lda_batch[op_id];
	int* ipiv = ipiv_ptrs[op_id];
	
	// Check if this is a valid operation 
	if(m <= 0 || n <= 0 || g_A == NULL)
		return;
	
	// Check if the start and end are within the limits of this operation
	if(start >= n)
		return; 

	// Exit early if this thread has nothing to do
	int tid = threadIdx.x;
	if(tid >= m)
		return; 

	int rank = (m < n ? m : n);
	if(end > rank)
		end = rank;
	
	#pragma unroll 
	for(int i = start; i < end; i++)
	{
		int j = ipiv[i] - 1;
		
		if(i == j) 
			continue;
		
		T temp = g_A[tid + i * lda];
		g_A[tid + i * lda] = g_A[tid + j * lda];
		g_A[tid + j * lda] = temp;
	}
}

#endif
