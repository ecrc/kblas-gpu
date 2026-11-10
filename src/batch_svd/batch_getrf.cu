#include <cublas_v2.h>

#include "kblas.h"
#include "kblas_common.h"
#include "kblas_struct.h"
#include "kblas_gpu_util.ch"
#include "kblas_operators.h"
#include "batch_transpose.h"
#include "batch_getrf.h"
#include "getrf_functors.h"
#include "getrf_kernels.cuh"

#include <thrust/system/omp/execution_policy.h>

#define LU_MAX_ROWS 1024
#define LU_PANEL_SIZE 32
#define LU_SMEM_MAX_DIM 32
#define LU_SMEM_THREAD_GROUPS 8

template<class T>
int kblas_getf2_column_vbatch(
	kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int max_m, int max_n, 
	T **A_ptrs, int *lda_batch, int **ipiv_ptrs, int j, int *info_batch, 
	int batchCount
)
{
    if(max_m < 0 || max_m > LU_MAX_ROWS || max_n < 0 || max_n > 32)
	{
		printf("Invalid input %d %d to kblas_getf2_column_vbatch\n", max_m, max_n);
		return -1;
	}
	
	size_t smem_size = (max_n + max_m) * (sizeof(T) + sizeof(int));
    dim3 gridDim(batchCount, 1, 1);
    dim3 blockDim(max_m, 1);
	cudaStream_t stream = kblasGetStream(kblas_handle);

	switch(max_n)
	{
		case  1: getf2_batched_kernel<T,  1><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case  2: getf2_batched_kernel<T,  2><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case  3: getf2_batched_kernel<T,  3><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case  4: getf2_batched_kernel<T,  4><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case  5: getf2_batched_kernel<T,  5><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case  6: getf2_batched_kernel<T,  6><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case  7: getf2_batched_kernel<T,  7><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case  8: getf2_batched_kernel<T,  8><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case  9: getf2_batched_kernel<T,  9><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 10: getf2_batched_kernel<T, 10><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 11: getf2_batched_kernel<T, 11><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 12: getf2_batched_kernel<T, 12><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 13: getf2_batched_kernel<T, 13><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 14: getf2_batched_kernel<T, 14><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 15: getf2_batched_kernel<T, 15><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 16: getf2_batched_kernel<T, 16><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 17: getf2_batched_kernel<T, 17><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 18: getf2_batched_kernel<T, 18><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 19: getf2_batched_kernel<T, 19><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 20: getf2_batched_kernel<T, 20><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 21: getf2_batched_kernel<T, 21><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 22: getf2_batched_kernel<T, 22><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 23: getf2_batched_kernel<T, 23><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 24: getf2_batched_kernel<T, 24><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 25: getf2_batched_kernel<T, 25><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 26: getf2_batched_kernel<T, 26><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 27: getf2_batched_kernel<T, 27><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 28: getf2_batched_kernel<T, 28><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 29: getf2_batched_kernel<T, 29><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 30: getf2_batched_kernel<T, 30><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 31: getf2_batched_kernel<T, 31><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
		case 32: getf2_batched_kernel<T, 32><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch); break;
	}
	
	check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	return KBLAS_Success;
}

template<class T>
int kblas_swap_columns_vbatch(
	kblasHandle_t kblas_handle, int *m_batch, int *n_batch, int max_m, int max_n, 
	T **A_ptrs, int *lda_batch, int **ipiv_ptrs, int start, int end, int batchCount
)
{
	if(max_m < 0 || max_m > LU_MAX_ROWS || max_n < 0)
	{
		printf("Invalid input %d %d to kblas_swap_columns_vbatch\n", max_m, max_n);
		return -1;
	}
	
	if(start < 0 || end < 0 || start > end || end > max_n)
	{
		printf("Invalid limits %d %d to kblas_swap_columns_vbatch\n", start, end);
		return -1;
	}
	
    dim3 gridDim(batchCount, 1, 1);
    dim3 blockDim(max_m, 1);
	cudaStream_t stream = kblasGetStream(kblas_handle);
	
	swap_columns_batched_kernel<T><<<gridDim, blockDim, 0, stream>>>(
		m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, start, end
	);
	
	check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	return KBLAS_Success;
}

template<class T>
int kblas_getrf_trsm_vbatch(
	kblasHandle_t kblas_handle, int* m_batch, int* n_batch, 
	int max_m, int max_n, T** A_batch, int* lda_batch, 
	T** B_batch, int* ldb_batch, int batchCount
)
{
	const int max_block_threads = 256;
	
    if(max_m <= 0 || max_n <= 0 || batchCount <= 0) 
        return -1;
	if(max_n > 32)
		return -1;
	
	cudaStream_t stream = kblasGetStream(kblas_handle);
	
	int block_thread_x = max_block_threads;
	int block_thread_y = 1;
	
	if(block_thread_x > max_m)
	{
		block_thread_x = max_m;
		block_thread_y = max_block_threads / max_m;
        
        // Don't go overboard with the number of ops per block
        // Since we might not have enough shared memory
        if (block_thread_y > 3) 
			block_thread_y = 3;
		if(block_thread_y > batchCount)
			block_thread_y = batchCount;
	}
    
    // Make sure we have enough threads to load the triangular block
    if (block_thread_x < max_n)
        block_thread_x = max_n;
	
	int grid_x = iDivUp(max_m, block_thread_x);
	int grid_y = iDivUp(batchCount , block_thread_y);
	int smem_size = block_thread_y * WARP_SIZE * max_n * sizeof(T);
	
	dim3 blockDim(block_thread_x, block_thread_y);
	dim3 gridDim(grid_x, grid_y);
	
	switch(max_n)
	{
		case  1: getrf_trsm_kernel<T,  1><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case  2: getrf_trsm_kernel<T,  2><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case  3: getrf_trsm_kernel<T,  3><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case  4: getrf_trsm_kernel<T,  4><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case  5: getrf_trsm_kernel<T,  5><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case  6: getrf_trsm_kernel<T,  6><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case  7: getrf_trsm_kernel<T,  7><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case  8: getrf_trsm_kernel<T,  8><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case  9: getrf_trsm_kernel<T,  9><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 10: getrf_trsm_kernel<T, 10><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 11: getrf_trsm_kernel<T, 11><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 12: getrf_trsm_kernel<T, 12><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 13: getrf_trsm_kernel<T, 13><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 14: getrf_trsm_kernel<T, 14><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 15: getrf_trsm_kernel<T, 15><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 16: getrf_trsm_kernel<T, 16><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 17: getrf_trsm_kernel<T, 17><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 18: getrf_trsm_kernel<T, 18><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 19: getrf_trsm_kernel<T, 19><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 20: getrf_trsm_kernel<T, 20><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 21: getrf_trsm_kernel<T, 21><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 22: getrf_trsm_kernel<T, 22><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 23: getrf_trsm_kernel<T, 23><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 24: getrf_trsm_kernel<T, 24><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 25: getrf_trsm_kernel<T, 25><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 26: getrf_trsm_kernel<T, 26><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 27: getrf_trsm_kernel<T, 27><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 28: getrf_trsm_kernel<T, 28><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 29: getrf_trsm_kernel<T, 29><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 30: getrf_trsm_kernel<T, 30><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 31: getrf_trsm_kernel<T, 31><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
		case 32: getrf_trsm_kernel<T, 32><<<gridDim, blockDim, smem_size, stream>>>(m_batch, n_batch, A_batch, lda_batch, B_batch, ldb_batch, batchCount); break;
	}
	
	check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	return KBLAS_Success;
}

template<class T>
size_t kblas_getrf_vbatch_workspace_template(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount)
{
	cudaStream_t stream = kblasGetStream(kblas_handle);
	size_t ws_bytes = 0;

	// Workspace for pointers 
	ws_bytes += 4 * batchCount * sizeof(T*);
	
	// Workspace for scan
	ws_bytes = aligned_bytes<size_t>(ws_bytes);
	ws_bytes += batchCount * sizeof(size_t);
	
	// Workspace for dim data 
	ws_bytes = aligned_bytes<int>(ws_bytes);
	ws_bytes += 3 * batchCount * sizeof(int);
	
	// Workspace for all the transpose data 
	ws_bytes = aligned_bytes<T>(ws_bytes);
	
	size_t dim_product_sum = thrust::transform_reduce(
		thrust::cuda::par.on(stream), 
		thrust::make_zip_iterator(thrust::make_tuple(m_batch, n_batch)),
		thrust::make_zip_iterator(thrust::make_tuple(m_batch, n_batch)) + batchCount,
		dim_product_functor(), 0, thrust::plus<size_t>()
	);

	ws_bytes += dim_product_sum * sizeof(T);
	
	return ws_bytes;
}

template<class T>
size_t kblas_getrf_vbatch_workspace_host_template(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount)
{
	size_t ws_bytes = 0;

	// Workspace for pointers 
	ws_bytes += 4 * batchCount * sizeof(T*);
	
	// Workspace for scan
	ws_bytes = aligned_bytes<size_t>(ws_bytes);
	ws_bytes += batchCount * sizeof(size_t);
	
	// Workspace for dim data 
	ws_bytes = aligned_bytes<int>(ws_bytes);
	ws_bytes += 3 * batchCount * sizeof(int);
	
	// Workspace for all the transpose data 
	ws_bytes = aligned_bytes<T>(ws_bytes);
	
	size_t dim_product_sum = thrust::transform_reduce(
		thrust::system::omp::par, 
		thrust::make_zip_iterator(thrust::make_tuple(m_batch, n_batch)),
		thrust::make_zip_iterator(thrust::make_tuple(m_batch, n_batch)) + batchCount,
		dim_product_functor(), 0, thrust::plus<size_t>()
	);

	ws_bytes += dim_product_sum * sizeof(T);
	
	return ws_bytes;
}

template<class T>
int kblas_getrf_vbatch_set_workspace(
	kblasHandle_t kblas_handle, int *m_batch, int* n_batch, void* ws_ptr, size_t ws_bytes, 
	T **&transpose_ptrs, int *&m_ws_batch, int *&n_ws_batch, int *&k_ws_batch, 
	T **&A_ptr_ws, T **&B_ptr_ws, T **&C_ptr_ws, int batchCount
)
{
	cudaStream_t stream = kblasGetStream(kblas_handle);
	size_t allocated_ws_bytes = 0;

	// Workspace for pointers 
	setWSPtr<T*>(ws_ptr, batchCount, &transpose_ptrs, allocated_ws_bytes);
	setWSPtr<T*>(ws_ptr, batchCount, &A_ptr_ws, allocated_ws_bytes);
	setWSPtr<T*>(ws_ptr, batchCount, &B_ptr_ws, allocated_ws_bytes);
	setWSPtr<T*>(ws_ptr, batchCount, &C_ptr_ws, allocated_ws_bytes);

	// Workspace for scan
	size_t* product_scan;
	setWSPtr<size_t>(ws_ptr, batchCount, &product_scan, allocated_ws_bytes);

	// Workspace for dim data 
	setWSPtr<int>(ws_ptr, batchCount, &m_ws_batch, allocated_ws_bytes);
	setWSPtr<int>(ws_ptr, batchCount, &n_ws_batch, allocated_ws_bytes);
	setWSPtr<int>(ws_ptr, batchCount, &k_ws_batch, allocated_ws_bytes);

	// Workspace for all the transpose data 
	thrust::transform_exclusive_scan(
		thrust::cuda::par.on(stream), 
		thrust::make_zip_iterator(thrust::make_tuple(m_batch, n_batch)),
		thrust::make_zip_iterator(thrust::make_tuple(m_batch, n_batch)) + batchCount,
		product_scan, dim_product_functor(), 0, thrust::plus<size_t>() 
	);

	// Align bytes and get base pointer 
	allocated_ws_bytes = aligned_bytes<T>(allocated_ws_bytes);
	T* transpose_base_ptr = (T*)((KBlasWorkspace::WS_Byte*)ws_ptr + allocated_ws_bytes);
	
	// Assign transpose pointers 
	thrust::for_each(
		thrust::cuda::par.on(stream), thrust::counting_iterator<size_t>(0),
		thrust::counting_iterator<size_t>(batchCount),
		PointerScanAssign<T>(product_scan, transpose_ptrs, transpose_base_ptr)
	);
	
	check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	return KBLAS_Success;
}

template<class T>
int kblas_getrf_column_vbatch(
	kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int max_m, int max_n, 
	T **A_ptrs, int *lda_batch, int **ipiv_ptrs, T **ws_ptrs, int *ws_ld_batch, 
	int *m_ws_batch, int *n_ws_batch, int *k_ws_batch, 
	T **A_ptr_ws, T **B_ptr_ws, T **C_ptr_ws,
	int *info_batch, int batchCount
)
{
	int min_mn = std::min(max_m, max_n);
	int bs = LU_PANEL_SIZE;
	cudaStream_t stream = kblasGetStream(kblas_handle);
	
	T **panel_ptrs = A_ptr_ws;
	int *panel_m_batch = m_ws_batch;
	int *panel_n_batch = n_ws_batch;
	T one = make_one<T>();
	
	for(int j = 0; j < min_mn; j += bs)
	{
		int panel_width = std::min(min_mn - j, bs);
		
		/////////////////////////////////////////////////////////////////////
		// Panel factorization 
		/////////////////////////////////////////////////////////////////////
		// Set the panel pointers and sizes 
		GetrfPanelSet_Functor<T> panel_set(
			A_ptrs, lda_batch, m_batch, n_batch, panel_ptrs, 
			panel_m_batch, panel_n_batch, panel_width, j
		);
		thrust::for_each(
			thrust::system::cuda::par.on(stream), thrust::counting_iterator<size_t>(0),
			thrust::counting_iterator<size_t>(batchCount), panel_set
		);
		
		// Transpose the panel into the workspace 
		check_error_forward( kblas_transpose_vbatch(
			kblas_handle, panel_m_batch, panel_n_batch, panel_width, max_n - j, 
			panel_ptrs, lda_batch, ws_ptrs, ws_ld_batch, batchCount
		) );
		
		// Factorize the panel stored in the workspace
		check_error_forward( kblas_getf2_column_vbatch<T>(
			kblas_handle, panel_n_batch, panel_m_batch, max_n - j, panel_width, 
			ws_ptrs, ws_ld_batch, ipiv_ptrs, j, info_batch, batchCount
		) );
		
		// Transpose back into the full matrix 
		check_error_forward( kblas_transpose_vbatch(
			kblas_handle, panel_n_batch, panel_m_batch, max_n - j, panel_width, 
			ws_ptrs, ws_ld_batch, panel_ptrs, lda_batch, batchCount
		) );
		
		/////////////////////////////////////////////////////////////////////
		// Swap columns according to ipiv 
		/////////////////////////////////////////////////////////////////////
		check_error_forward( kblas_swap_columns_vbatch<T>(
			kblas_handle, m_batch, n_batch, max_m, max_n, A_ptrs, lda_batch, 
			ipiv_ptrs, j, j + panel_width, batchCount
		) );
		
		if(j + panel_width < max_m)
		{
			/////////////////////////////////////////////////////////////////////
			// Triangular solve
			/////////////////////////////////////////////////////////////////////
			GetrfTrsmSet_Functor<T> trsm_set(
				A_ptrs, lda_batch, m_batch, n_batch, B_ptr_ws, 
				panel_m_batch, panel_n_batch, panel_width, j
			);
			thrust::for_each(
				thrust::system::cuda::par.on(stream), thrust::counting_iterator<size_t>(0),
				thrust::counting_iterator<size_t>(batchCount), trsm_set
			);
			check_error_forward( kblas_getrf_trsm_vbatch<T>(
				kblas_handle, panel_m_batch, panel_n_batch, max_m - j - panel_width, panel_width, 
				panel_ptrs, lda_batch, B_ptr_ws, lda_batch, batchCount
			) );
			
			// /////////////////////////////////////////////////////////////////////
			// // Update the trailing submatrix 
			// /////////////////////////////////////////////////////////////////////
			// B_ptr_ws has the lower panel pointers, now we need trailing submatrix 
			// pointers and upper panel pointers 
			GetrfUpdateSet_Functor<T> update_set(
				A_ptrs, lda_batch, m_batch, n_batch, A_ptr_ws, C_ptr_ws, 
				m_ws_batch, n_ws_batch, k_ws_batch, panel_width, j
			);
			thrust::for_each(
				thrust::system::cuda::par.on(stream), thrust::counting_iterator<size_t>(0),
				thrust::counting_iterator<size_t>(batchCount), update_set
			);
			check_error_forward( kblas_gemm_batch(
				kblas_handle, KBLAS_NoTrans, KBLAS_NoTrans, m_ws_batch, n_ws_batch, k_ws_batch,
				max_m - j - panel_width, max_n - j - panel_width, panel_width, -one, 
				(const T**)B_ptr_ws, lda_batch, (const T**)A_ptr_ws, lda_batch, one, 
				C_ptr_ws, lda_batch, batchCount 
			) );
		}
	}
	
	return KBLAS_Success;
}

template<class T>
int kblas_getrf_smem_vbatch(
	kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int max_m, int max_n, 
	T **A_ptrs, int *lda_batch, int **ipiv_ptrs, void* workspace, size_t ws_bytes,
	int *info_batch, int batchCount
)
{
	cudaStream_t stream = kblasGetStream(kblas_handle);
	
	int block_thread_x = WARP_SIZE;
	int block_thread_y = LU_SMEM_THREAD_GROUPS;
	
	if(block_thread_y > max_n)
		block_thread_y = max_n;

	int smem_size = (WARP_SIZE + 1) * max_n * sizeof(T); // shared matrix A
	smem_size += kmin(max_m, max_n) * sizeof(int); // shared pivot array
	smem_size += WARP_SIZE * sizeof(T) + WARP_SIZE * sizeof(int); // reduction workspace
	
	dim3 blockDim(block_thread_x, block_thread_y);
	dim3 gridDim(batchCount, 1);
	
	kblas_getrf_smem_kernel<T><<<gridDim, blockDim, smem_size, stream>>>(
		m_batch, n_batch, A_ptrs, lda_batch, ipiv_ptrs, info_batch
	); 
	
	check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	return KBLAS_Success;
}

template<class T>
int kblas_getrf_vbatch(
	kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int max_m, int max_n, 
	T **A_ptrs, int *lda_batch, int **ipiv_ptrs, void* workspace, size_t ws_bytes,
	int *info_batch, int batchCount
)
{
	if(max_m <= LU_SMEM_MAX_DIM && max_n <= LU_SMEM_MAX_DIM)
	{
		int ops_per_kernel = MAX_GRID_DIM_Z;
		int op_start = 0;
		
		while(op_start < batchCount)
		{
			int batch_subcount = kmin(ops_per_kernel, batchCount - op_start);
		
			check_error_forward( kblas_getrf_smem_vbatch<T>(
				kblas_handle, m_batch + op_start, n_batch + op_start, max_m, max_n, 
				A_ptrs + op_start, lda_batch + op_start, ipiv_ptrs + op_start, 
				workspace, ws_bytes, info_batch, batch_subcount
			) );
			op_start += ops_per_kernel;
		}
	
	}
	else
	{
		T **transpose_ptrs, **A_ptr_ws, **B_ptr_ws, **C_ptr_ws;
		int *m_ws_batch, *n_ws_batch, *k_ws_batch; 
		
		check_error_forward( kblas_getrf_vbatch_set_workspace<T>(
			kblas_handle, m_batch, n_batch, workspace, ws_bytes, 
			transpose_ptrs, m_ws_batch, n_ws_batch, k_ws_batch, 
			A_ptr_ws, B_ptr_ws, C_ptr_ws, batchCount
		) );
		
		// Reset the info array 
		thrust::fill(
			thrust::cuda::par.on(kblasGetStream(kblas_handle)), 
			info_batch, info_batch + batchCount, 0
		);
		
		// Transpose first 
		check_error_forward( kblas_transpose_vbatch(
			kblas_handle, m_batch, n_batch, max_m, max_n, 
			(T**)A_ptrs, lda_batch, transpose_ptrs, n_batch, batchCount
		) );
		
		int ops_per_kernel = MAX_GRID_DIM_Z;
		int op_start = 0;
		
		while(op_start < batchCount)
		{
			int batch_subcount = kmin(ops_per_kernel, batchCount - op_start);
			
			// Column pivoted LU 
			check_error_forward( kblas_getrf_column_vbatch<T>(
				kblas_handle, n_batch + op_start, m_batch + op_start, max_n, max_m, 
				transpose_ptrs + op_start, n_batch + op_start, ipiv_ptrs + op_start, 
				(T**)A_ptrs + op_start, lda_batch + op_start, m_ws_batch + op_start, 
				n_ws_batch + op_start, k_ws_batch + op_start, A_ptr_ws + op_start, 
				B_ptr_ws + op_start, C_ptr_ws + op_start, info_batch + op_start, 
				batch_subcount
			) );
			
			op_start += ops_per_kernel;
		}
		
		// Transpose the results to get row pivoted LU as the result 
		check_error_forward( kblas_transpose_vbatch(
			kblas_handle, n_batch, m_batch, max_n, max_m, 
			transpose_ptrs, n_batch, (T**)A_ptrs, lda_batch, batchCount
		) );
	}
	
	return KBLAS_Success;
}

/////////////////////////////////////////////////////////
// Workspace interface 
/////////////////////////////////////////////////////////
extern "C" size_t kblasSgetrf_vbatch_workspace(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount)
{ return kblas_getrf_vbatch_workspace_template<float>(kblas_handle, m_batch, n_batch, batchCount); }

extern "C" size_t kblasDgetrf_vbatch_workspace(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount)
{ return kblas_getrf_vbatch_workspace_template<double>(kblas_handle, m_batch, n_batch, batchCount); }

extern "C" size_t kblasCgetrf_vbatch_workspace(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount)
{ return kblas_getrf_vbatch_workspace_template<cuFloatComplex>(kblas_handle, m_batch, n_batch, batchCount); }

extern "C" size_t kblasZgetrf_vbatch_workspace(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount)
{ return kblas_getrf_vbatch_workspace_template<cuDoubleComplex>(kblas_handle, m_batch, n_batch, batchCount); }

extern "C" size_t kblasSgetrf_vbatch_workspace_host(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount)
{ return kblas_getrf_vbatch_workspace_host_template<float>(kblas_handle, m_batch, n_batch, batchCount); }

extern "C" size_t kblasDgetrf_vbatch_workspace_host(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount)
{ return kblas_getrf_vbatch_workspace_host_template<double>(kblas_handle, m_batch, n_batch, batchCount); }

extern "C" size_t kblasCgetrf_vbatch_workspace_host(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount)
{ return kblas_getrf_vbatch_workspace_host_template<cuFloatComplex>(kblas_handle, m_batch, n_batch, batchCount); }

extern "C" size_t kblasZgetrf_vbatch_workspace_host(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount)
{ return kblas_getrf_vbatch_workspace_host_template<cuDoubleComplex>(kblas_handle, m_batch, n_batch, batchCount); }

/////////////////////////////////////////////////////////
// Batched LU interface 
/////////////////////////////////////////////////////////
extern "C" int kblasSgetrf_vbatch(
	kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int max_m, int max_n, 
	float **A_ptrs, int *lda_batch, int **ipiv_ptrs, void* workspace, size_t ws_bytes,
	int *info_batch, int batchCount
)
{ return kblas_getrf_vbatch<float>(kblas_handle, m_batch, n_batch, max_m, max_n, A_ptrs, lda_batch, ipiv_ptrs, workspace, ws_bytes, info_batch, batchCount); }

extern "C" int kblasDgetrf_vbatch(
	kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int max_m, int max_n, 
	double **A_ptrs, int *lda_batch, int **ipiv_ptrs, void* workspace, size_t ws_bytes,
	int *info_batch, int batchCount
)
{ return kblas_getrf_vbatch<double>(kblas_handle, m_batch, n_batch, max_m, max_n, A_ptrs, lda_batch, ipiv_ptrs, workspace, ws_bytes, info_batch, batchCount); }

extern "C" int kblasCgetrf_vbatch(
	kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int max_m, int max_n, 
	cuFloatComplex **A_ptrs, int *lda_batch, int **ipiv_ptrs, void* workspace, size_t ws_bytes,
	int *info_batch, int batchCount
)
{ return kblas_getrf_vbatch<cuFloatComplex>(kblas_handle, m_batch, n_batch, max_m, max_n, A_ptrs, lda_batch, ipiv_ptrs, workspace, ws_bytes, info_batch, batchCount); }

extern "C" int kblasZgetrf_vbatch(
	kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int max_m, int max_n, 
	cuDoubleComplex **A_ptrs, int *lda_batch, int **ipiv_ptrs, void* workspace, size_t ws_bytes,
	int *info_batch, int batchCount
)
{ return kblas_getrf_vbatch<cuDoubleComplex>(kblas_handle, m_batch, n_batch, max_m, max_n, A_ptrs, lda_batch, ipiv_ptrs, workspace, ws_bytes, info_batch, batchCount); }

////////// Binned code. 
/*
// thrust::device_vector<int> op_order(batchCount_gpu), op_keys(batchCount_gpu);
// thrust::device_vector<int> op_counts(batchCount_gpu), op_blocks(batchCount_gpu);
// thrust::constant_iterator<int> d_one(1);
// 
// thrust::sequence(op_order.begin(), op_order.end(), 0);
// thrust::transform(d_rows_batch[g], d_rows_batch[g] + batchCount, op_keys.begin(), dim_keygen<GETRF_BIN_SIZE>());
// thrust::sort_by_key(op_keys.begin(), op_keys.end(), op_order.begin(), thrust::greater<int>());
// auto new_end = thrust::reduce_by_key(op_keys.begin(), op_keys.end(), d_one, op_blocks.begin(), op_counts.begin());
// thrust::host_vector<int> batch_blocks(op_blocks.begin(), new_end.first), batch_size(op_counts.begin(), new_end.second);
// 
// printf("%zu batches!\n", batch_blocks.size());
// for(size_t i = 0; i < batch_blocks.size(); i++)
// 	printf("%d %d\n", batch_blocks[i] * GETRF_BIN_SIZE, batch_size[i]);
// printf("\n");
// 
// thrust::device_vector<int> binned_m_batch(batchCount_gpu), binned_ldm_batch(batchCount_gpu);
// thrust::device_vector<int*> binned_ipiv_ptrs(batchCount_gpu);
// thrust::device_vector<Scalar*> binned_A_ptrs(batchCount_gpu);
// 
// GetrfBinHelper<Scalar> bin_helper;
// bin_helper.binBatchDims(d_rows_batch[g], batchCount_gpu);
// size_t ws_bytes = bin_helper.getWorkspaceSize();
// printf("Needs %zu workspace bytes\n", ws_bytes);

// kblas_getrf_vbatch_binned<Scalar>(
// 	kblas_handles[g], d_rows_batch[g], max_rows, d_M_ptrs[g], d_ldm_batch[g], d_ipiv_ptrs[g], 
// 	vec_ptr(binned_m_batch), vec_ptr(binned_A_ptrs), vec_ptr(binned_ldm_batch), vec_ptr(binned_ipiv_ptrs),
// 	d_Mt_ptrs[g], d_ldmt_batch[g], 
// 	d_dim_ws[g], d_dim2_ws[g], d_dim3_ws[g], d_A_ptr_ws[g], d_B_ptr_ws[g], 
// 	d_C_ptr_ws[g], d_info_batch[g], vec_ptr(op_order), vec_ptr(batch_blocks), 
// 	vec_ptr(batch_size), batch_blocks.size(), batchCount_gpu
// );

#define GETRF_BIN_SIZE 32

template<class T, int PANEL_WIDTH>
__global__ void getf2_column_panel_batched_kernel(
	int *m_batch, T **A_ptrs, int *lda_batch, 
	int **ipiv_ptrs, int j, int *info_batch
)
{
	int op_id = blockIdx.x;
	
	int op_cols = m_batch[op_id] - j;
	int op_rows = (op_cols > PANEL_WIDTH ? PANEL_WIDTH : op_cols);
	
	int lda_op = lda_batch[op_id];
	int* ipiv_op = ipiv_ptrs[op_id] + j;
	T* op_A = A_ptrs[op_id];
	
	if(op_A == NULL || op_cols <= 0 || op_rows <= 0)
		return;
	
	op_A += j + j * lda_op;
	
	// Check if this operation has failed already at some point 
	if(j != 0 && info_batch[op_id] != 0)
		return;
	
	extern __shared__ char sdata[];	
	
	getf2_device<T, PANEL_WIDTH, true>(
		op_cols, op_rows, op_A, lda_op, ipiv_op, 
		info_batch + op_id, sdata, j
	);
}

template<class T>
int kblas_getf2_column_binned(
	cudaStream_t stream, int *m_batch, int bin_dim, T **A_ptrs, int *lda_batch, 
	int **ipiv_ptrs, int j, int *info_batch, int batchCount
)
{
	const int bs = GETRF_BIN_SIZE;
	int panel_cols = bin_dim - j;
	
    if(panel_cols < 0 || panel_cols > LU_MAX_ROWS)
	{
		printf("Invalid panel_cols %d to kblas_getf2_column_binned\n", panel_cols);
		return -1;
	}
	
	size_t smem_size = (panel_cols + bs) * (sizeof(T) + sizeof(int));
    dim3 gridDim(batchCount, 1, 1);
    dim3 blockDim(panel_cols, 1);
	
	getf2_column_panel_batched_kernel<T, bs><<<gridDim, blockDim, smem_size, stream>>>(
		m_batch, A_ptrs, lda_batch, ipiv_ptrs, j, info_batch);
	
	return 0;
}

template<class T>
struct BinnedCopy
{
	int *m_batch, *lda_batch, **ipiv_ptrs;
	T **A_ptrs, **binned_A_ptrs;
	int *binned_m_batch, *binned_lda_batch, **binned_ipiv_ptrs;
	int *op_order;
	
	BinnedCopy(
		int *m_batch, T **A_ptrs, int *lda_batch, int **ipiv_ptrs,
		int *binned_m_batch, T **binned_A_ptrs, int *binned_lda_batch, 
		int **binned_ipiv_ptrs, int *op_order 
	)
	{
		this->m_batch = m_batch;
		this->A_ptrs = A_ptrs;
		this->lda_batch = lda_batch;
		this->ipiv_ptrs = ipiv_ptrs;
		this->binned_m_batch = binned_m_batch;
		this->binned_A_ptrs = binned_A_ptrs;
		this->binned_lda_batch = binned_lda_batch;
		this->binned_ipiv_ptrs = binned_ipiv_ptrs;
		this->op_order = op_order;
	}
	
	inline __host__ __device__ 
	void operator()(size_t i)
	{
		int op_index = op_order[i];
		binned_m_batch[i] = m_batch[op_index];
		binned_A_ptrs[i] = A_ptrs[op_index];
		binned_lda_batch[i] = lda_batch[op_index];
		binned_ipiv_ptrs[i] = ipiv_ptrs[op_index];
	}
};

template <class T> 
thrust::host_vector<T> copyGPUBlockT(T **data_ptrs, int index, int ld, int rows, int cols)
{
    thrust::device_ptr<T *> dev_ptr(data_ptrs + index);
    T *dev_ptr_val = *dev_ptr;
    thrust::device_ptr<T> dev_data(dev_ptr_val);
    thrust::host_vector<T> host_data(rows * cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            host_data[i + j * rows] = *(dev_data + i + j * ld);
    return host_data;
}

// TODO: replace max_ld by proper workspace allocations 
template<class T, int N>
struct BlockOffsetPointer
{
	int offsets[N];
	T **original_ptrs[N], **offset_ptrs[N];
	
	BlockOffsetPointer(
		T** (offset_ptrs)[N], T** (original_ptrs)[N], 
		int (offsets) [N])
	{
		#pragma unroll 
		for(int i = 0; i < N; i++)
		{
			this->offset_ptrs[i] = offset_ptrs[i];
			this->original_ptrs[i] = original_ptrs[i];
			this->offsets[i] = offsets[i];
		}
	}
	
	__host__ __device__ 
	inline void operator()(const unsigned int& thread_id) const 
	{ 
		#pragma unroll
		for(int i = 0; i < N; i++)
			offset_ptrs[i][thread_id] = original_ptrs[i][thread_id] + offsets[i]; 
	}
};

template<class T, int NP>
inline void block_offset_pointers(
	T** (offset_ptrs) [NP], T** (original_ptrs) [NP], 
	int (offsets) [NP], int num_ops, cudaStream_t stream
)
{
	thrust::for_each(
		thrust::cuda::par.on(stream), thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(num_ops), BlockOffsetPointer<T, NP>(
			offset_ptrs, original_ptrs, offsets
		)
	);
}

template<class T>
struct GetrfWorkspace
{
	int* binned_m_batch, *binned_ldm_batch, *ws_ld_ptrs, **binned_ipiv_ptrs;
	T** binned_A_ptrs, *transpose_strided_data;
	int *m_ws_batch, *n_ws_batch; 
	T **A_ptr_ws, **B_ptr_ws, **C_ptr_ws;
};

template<class T>
inline void getGetrfBinWorkspaceSize(int bin_blocks, int bin_size, size_t& ws_bytes, void* ws_ptr = NULL, GetrfWorkspace<T>* ws = NULL)
{
	// dimension and ld workspace
	setWSPtr<int>(ws_ptr, bin_size, KBLAS_GET_WS_FIELD(ws, binned_m_batch), ws_bytes);
	setWSPtr<int>(ws_ptr, bin_size, KBLAS_GET_WS_FIELD(ws, binned_ldm_batch), ws_bytes);
	setWSPtr<int>(ws_ptr, bin_size, KBLAS_GET_WS_FIELD(ws, ws_ld_ptrs), ws_bytes);
	setWSPtr<int>(ws_ptr, bin_size, KBLAS_GET_WS_FIELD(ws, m_ws_batch), ws_bytes);
	setWSPtr<int>(ws_ptr, bin_size, KBLAS_GET_WS_FIELD(ws, n_ws_batch), ws_bytes);

	// binned piv array workspace
	setWSPtr<int*>(ws_ptr, bin_size, KBLAS_GET_WS_FIELD(ws, binned_ipiv_ptrs), ws_bytes);

	// binned pointer workspace
	setWSPtr<T*>(ws_ptr, bin_size, KBLAS_GET_WS_FIELD(ws, binned_A_ptrs), ws_bytes);
	setWSPtr<T*>(ws_ptr, bin_size, KBLAS_GET_WS_FIELD(ws, A_ptr_ws), ws_bytes);
	setWSPtr<T*>(ws_ptr, bin_size, KBLAS_GET_WS_FIELD(ws, B_ptr_ws), ws_bytes);
	setWSPtr<T*>(ws_ptr, bin_size, KBLAS_GET_WS_FIELD(ws, C_ptr_ws), ws_bytes);

	// transpose workspace
	int n = bin_blocks * GETRF_BIN_SIZE;
	setWSPtr<T>(ws_ptr, n * n * bin_size, KBLAS_GET_WS_FIELD(ws, transpose_strided_data), ws_bytes);
}

template<class T>
void kblas_getrf_binned(
	kblasHandle_t kblas_handle,  
	int *m_batch, int bin_dim, T **A_ptrs, int *lda_batch, int **ipiv_ptrs,
	T **ws_ptrs, int *ws_ld_batch, int *m_ws_batch, int *n_ws_batch, int *k_ws_batch, 
	T **A_ptr_ws, T **B_ptr_ws, T **C_ptr_ws, int *info_batch, int batchCount, int max_ld
)
{
	const int bs = GETRF_BIN_SIZE;
	cudaStream_t stream = kblasGetStream(kblas_handle);
	cublasHandle_t cublas_handle = kblasGetCublasHandle(kblas_handle);
	
	printf("Processing %d ops in bin of size %d\n", batchCount, bin_dim);
	
	for(int j = 0; j < bin_dim; j += bs)
	{
		/////////////////////////////////////////////////////////////////////
		// Panel factorization 
		/////////////////////////////////////////////////////////////////////
		// Set the panel pointers and sizes 
		T **panel_ptrs = A_ptr_ws;
		int *panel_m_batch = m_ws_batch;
		int *panel_n_batch = n_ws_batch;
		
		thrust::for_each(
			thrust::system::cuda::par.on(stream), thrust::counting_iterator<size_t>(0),
			thrust::counting_iterator<size_t>(batchCount), GetrfPanelSet_Functor<T>(
				A_ptrs, lda_batch, m_batch, m_batch, panel_ptrs, 
				panel_m_batch, panel_n_batch, bs, j
			)
		);
		
		// Transpose the panel into the workspace 
		kblas_transpose_vbatch(
			kblas_handle, panel_m_batch, panel_n_batch, bs, bin_dim - j, 
			panel_ptrs, lda_batch, ws_ptrs, ws_ld_batch, batchCount
		);
		
		// Factorize the panel stored in the workspace
		kblas_getf2_column_vbatch<T>(
			kblas_handle, panel_n_batch, panel_m_batch, bin_dim - j, bs, 
			ws_ptrs, ws_ld_batch, ipiv_ptrs, j, info_batch, batchCount
		);
		
		// Transpose back into the full matrix 
		kblas_transpose_vbatch(
			kblas_handle, panel_n_batch, panel_m_batch, bin_dim - j, bs, 
			ws_ptrs, ws_ld_batch, panel_ptrs, lda_batch, batchCount
		);
		
		/////////////////////////////////////////////////////////////////////
		// Swap columns according to ipiv 
		/////////////////////////////////////////////////////////////////////
		kblas_swap_columns_vbatch<T>(
			kblas_handle, m_batch, m_batch, bin_dim, bin_dim, A_ptrs, lda_batch, 
			ipiv_ptrs, j, j + bs, batchCount
		);
		
		/////////////////////////////////////////////////////////////////////
		// Triangular solve
		/////////////////////////////////////////////////////////////////////
		thrust::for_each(
			thrust::system::cuda::par.on(stream), thrust::counting_iterator<size_t>(0),
			thrust::counting_iterator<size_t>(batchCount), GetrfTrsmSet_Functor<T>(
				A_ptrs, lda_batch, m_batch, m_batch, B_ptr_ws, 
				panel_m_batch, panel_n_batch, bs, j
			)
		);
		kblas_getrf_trsm_vbatch<T>(
			kblas_handle, panel_m_batch, panel_n_batch, bin_dim - j, bs, 
			panel_ptrs, lda_batch, B_ptr_ws, lda_batch, batchCount
		);
		
		/////////////////////////////////////////////////////////////////////
		// Update the trailing submatrix 
		/////////////////////////////////////////////////////////////////////
		// B_ptr_ws has the lower panel pointers, now we need trailing submatrix 
		// pointers and upper panel pointers 
		int trailing_m = bin_dim - j - bs;
		if(trailing_m > 0)
		{
			T** off_ptrs[2] = {A_ptr_ws, C_ptr_ws}, **o_ptrs[2] = {A_ptrs, A_ptrs};
			int offsets[2] = {j + (j + bs) * max_ld, j + bs + (j + bs) * max_ld};
			block_offset_pointers<T, 2>(off_ptrs, o_ptrs, offsets, batchCount, stream);
			
			T alpha = -1, beta = 1;
			cublasXgemmBatched(
				cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, trailing_m, trailing_m, bs,
				&alpha, (const T**)B_ptr_ws, max_ld, (const T**)A_ptr_ws, max_ld,
				&beta, C_ptr_ws, max_ld, batchCount
			);
		}
	}
}

template<class T>
void kblas_getrf_vbatch_binned(
	kblasHandle_t kblas_handle, int *m_batch, int max_m, T **A_ptrs, int *lda_batch, 
	int **ipiv_ptrs, int *binned_m_batch, T **binned_A_ptrs, int *binned_lda_batch,
	int **binned_ipiv_ptrs, T **ws_ptrs, int *ws_ld_batch, int *m_ws_batch, int *n_ws_batch, 
	int *k_ws_batch, T **A_ptr_ws, T **B_ptr_ws, T **C_ptr_ws, int *info_batch,
	int *op_order, int *bin_blocks, int *bin_counts, int num_bins, int batchCount
)
{
	// TODO reorder info_batch at the end 	
	cudaStream_t stream = kblasGetStream(kblas_handle);
	
	// First reorder the input pointers into their bins 
	thrust::for_each(
		thrust::system::cuda::par.on(stream), thrust::counting_iterator<size_t>(0),
		thrust::counting_iterator<size_t>(batchCount), BinnedCopy<T>(
			m_batch, A_ptrs, lda_batch, ipiv_ptrs, binned_m_batch, 
			binned_A_ptrs, binned_lda_batch, binned_ipiv_ptrs, op_order
		)
	);
	
	// Transpose in the workspace (which is padded with zeros to match the binned
	// dimension if each operation)
	kblas_transpose_vbatch(
		kblas_handle, binned_m_batch, binned_m_batch, max_m, max_m, 
		binned_A_ptrs, binned_lda_batch, ws_ptrs, ws_ld_batch, 
		batchCount
	);

	// Figure out batch offsets for each bin 
	std::vector<int> bin_offset(num_bins, 0);
	for(int i = 1; i < num_bins; i++)
		bin_offset[i] = bin_offset[i - 1] + bin_counts[i - 1];
	
	// Process each bin 
	for(int bin = 0; bin < num_bins; bin++)
	{
		int bin_batch_count = bin_counts[bin];
		int bin_batch_dim   = bin_blocks[bin] * GETRF_BIN_SIZE;
		int bin_start       = bin_offset[bin];
		
		kblas_getrf_binned<T>(
			kblas_handle, binned_m_batch + bin_start, bin_batch_dim, ws_ptrs + bin_start, 
			ws_ld_batch + bin_start, binned_ipiv_ptrs + bin_start, binned_A_ptrs + bin_start, 
			binned_lda_batch + bin_start, m_ws_batch + bin_start, n_ws_batch + bin_start, 
			k_ws_batch + bin_start, A_ptr_ws + bin_start, B_ptr_ws + bin_start, C_ptr_ws + bin_start,
			info_batch + bin_start, bin_batch_count, max_m
		);
	}
	
	// Transpose out from the workspace
	kblas_transpose_vbatch(
		kblas_handle, binned_m_batch, binned_m_batch, max_m, max_m, 
		ws_ptrs, ws_ld_batch, binned_A_ptrs, binned_lda_batch, 
		batchCount
	);
}

template<int bs>
struct dim_keygen
{
	inline __host__ __device__
	int operator()(int x) const { return (x + bs - 1) / bs; }
};

template<class T>
struct GetrfBinHelper 
{
private:
	thrust::device_vector<int> op_order, op_keys, op_counts, op_blocks;
	thrust::host_vector<int> bin_blocks, bin_size, bin_offset;
	thrust::host_vector<size_t> ws_bin_offsets;
	void* ws_ptr;
	
public:
	void binBatchDims(int* m_batch, size_t n)
	{
		op_order.resize(n); op_keys.resize(n);
		op_counts.resize(n); op_blocks.resize(n);
		
		thrust::constant_iterator<int> d_one(1);
		thrust::sequence(op_order.begin(), op_order.end(), 0);
		thrust::transform(m_batch, m_batch + n, op_keys.begin(), dim_keygen<GETRF_BIN_SIZE>());
		thrust::sort_by_key(op_keys.begin(), op_keys.end(), op_order.begin(), thrust::greater<int>());
		auto new_end = thrust::reduce_by_key(op_keys.begin(), op_keys.end(), d_one, op_blocks.begin(), op_counts.begin());
		
		bin_blocks = thrust::host_vector<int>(op_blocks.begin(), new_end.first);
		bin_size = thrust::host_vector<int>(op_counts.begin(), new_end.second);
		bin_offset = thrust::host_vector<int>(bin_blocks.size(), 0);
		ws_bin_offsets = thrust::host_vector<int>(bin_blocks.size(), 0);
		
		for(size_t i = 1; i < bin_blocks.size(); i++)
			bin_offset[i] = bin_offset[i - 1] + bin_size[i - 1];
	}
	
	size_t getNumBins()
	{
		return bin_blocks.size();
	}
	
	int getBinBlocks(size_t bin_index)
	{
		return bin_blocks[bin_index];
	}
	
	int getBinSize(size_t bin_index)
	{
		return bin_size[bin_index];
	}
	
	size_t getWorkspaceSize()
	{
		size_t num_bins = getNumBins();
		
		if(num_bins == 0)
			return 0;
		
		size_t ws_bytes = 0;
		for(size_t i = 0; i < num_bins; i++)
			getGetrfBinWorkspaceSize<T>(getBinBlocks(i), getBinSize(i), ws_bytes);
		
		return ws_bytes;
	}
	
	void setWorkspace(void* ws_ptr, size_t bytes)
	{
		size_t num_bins = getNumBins();
		
		if(num_bins == 0)
			return;
		
		size_t ws_bytes = 0;
		for(size_t i = 0; i < num_bins; i++)
		{
			ws_bin_offsets[i] = ws_bytes;
			getGetrfBinWorkspaceSize<T>(getBinBlocks(i), getBinSize(i), ws_bytes);
		}
		assert(bytes == ws_bytes);
		this->ws_ptr = ws_ptr;
	}
	
	void* getBinWorkspace(size_t bin_index)
	{
		return (void*)((WS_Byte*)ws_ptr + ws_bin_offsets[bin_index]);
	}
	
	int* getBinOpStart(size_t bin_index)
	{
		return vec_ptr(op_order) + bin_offset[bin_index];
	}
};
*/