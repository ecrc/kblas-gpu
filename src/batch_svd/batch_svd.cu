#include <curand.h>
#include "kblas.h"
#include "kblas_struct.h"
#include "thrust_wrappers.h"
#include "batch_svd.h"
#include "batch_qr.h"
#include "batch_transpose.h"
#include "batch_block_copy.h"
#include <vector>

#include "svd_kernels.cuh"

#ifdef HLIB_PROFILING_ENABLED
#include "perf_counter.h"
#endif

#define OSBJ_BS		SHARED_SVD_DIM_LIMIT / 2
#define KBLAS_SVD_CHECK_RET(func)	{ if( (func) != KBLAS_Success ) return KBLAS_UnknownError; }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Workspace routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
void batch_svd_osbj_workspace(int rows, int cols, int num_ops, KBlasWorkspaceState& requested_ws, int top_level)
{
	if(cols <= SHARED_SVD_DIM_LIMIT && rows <= SHARED_SVD_DIM_LIMIT)
		return;
	
	int block_cols = iDivUp(cols, OSBJ_BS);

	requested_ws.d_data_bytes += (
		4 * OSBJ_BS * OSBJ_BS + // Gram matrix or R
		rows * 2 * OSBJ_BS    + // Temporary block column
		2 * OSBJ_BS           + // Temporary singular values
		2 * OSBJ_BS           + // tau
		1                       // Offdiagonal sum
	) * sizeof(T) * num_ops;

	requested_ws.d_ptrs_bytes += (block_cols + 7) * sizeof(T*) * num_ops;
}

template<class T>
void batch_tall_svd_workspace(int rows, int cols, int num_ops, KBlasWorkspaceState& requested_ws, int top_level)
{
	requested_ws.d_data_bytes += (
		cols * cols + // R
		rows * cols + // Q
		cols          // tau
	) * sizeof(T) * num_ops;

	requested_ws.d_ptrs_bytes += 4 * sizeof(T*) * num_ops;
}

template<class T>
void batch_wide_svd_workspace(int rows, int cols, int num_ops, KBlasWorkspaceState& requested_ws, int top_level)
{
	requested_ws.d_data_bytes += (
		rows * rows + // R
		cols * rows + // Q
		rows          // tau
	) * sizeof(T) * num_ops;

	requested_ws.d_ptrs_bytes += 3 * sizeof(T*) * num_ops;

	// Do we need to do osbj of the rows x rows matrix?
	if(!top_level && rows > SHARED_SVD_DIM_LIMIT)
		batch_svd_osbj_workspace<T>(rows, rows, num_ops, requested_ws, 0);
}

template<class T>
void batch_svd_randomized_workspace(int rows, int cols, int rank, int num_ops, KBlasWorkspaceState& requested_ws, int top_level)
{
	requested_ws.d_data_bytes += (
		cols * rank + // Omega
		rows * rank + // Y
		rank * cols + // B
		rank * rank + // R
		rank          // tau
	) * sizeof(T) * num_ops;

	requested_ws.d_ptrs_bytes += 5 * sizeof(T*) * num_ops;
	
	// Do we need to do osbj of the rank x rank matrix?
	if(!top_level && rank > SHARED_SVD_DIM_LIMIT)
		batch_svd_osbj_workspace<T>(rank, rank, num_ops, requested_ws, 0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Random matrix geenration using curand for the RSVD
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
inline void generateRandomMatrices(T* d_m, int rows, int cols, unsigned int seed, int num_ops, cudaStream_t stream = 0);

template<>
inline void generateRandomMatrices(float* d_m, int rows, int cols, unsigned int seed, int num_ops, cudaStream_t stream)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
	curandSetStream(gen, stream);

    curandGenerateNormal(gen, d_m, num_ops * rows * cols, 0, 1);

    curandDestroyGenerator(gen);
}

template<>
inline void generateRandomMatrices(double* d_m, int rows, int cols, unsigned int seed, int num_ops, cudaStream_t stream)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
	curandSetStream(gen, stream);

    curandGenerateNormalDouble(gen, d_m, num_ops * rows * cols, 0, 1);

    curandDestroyGenerator(gen);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Driver routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T, class T_ptr>
int batchSortSingularValues_small(T_ptr M, int ldm, int stride_m, T_ptr S, int stride_s, int rows, int cols, int num_ops, kblasHandle_t handle)
{
	int block_size = WARP_SIZE * iDivUp(rows, WARP_SIZE);

	dim3 dimBlock(block_size, 1);
	dim3 dimGrid(num_ops, 1);

	size_t smem_needed = (rows * cols) * sizeof(T) + cols * sizeof(int);

	switch(block_size)
	{
		case  32: batchSortSingularValuesSmallKernel<32, T, T_ptr><<< dimGrid, dimBlock, smem_needed, handle->stream >>>(M, ldm, stride_m, S, stride_s, rows, cols, num_ops); break;
		case  64: batchSortSingularValuesSmallKernel<64, T, T_ptr><<< dimGrid, dimBlock, smem_needed, handle->stream >>>(M, ldm, stride_m, S, stride_s, rows, cols, num_ops); break;
		default:  printf("batchSortSingularValues_small: Invalid block_size %d\n", block_size); break;
	}

	check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	return KBLAS_Success;
}

// Shared memory kernel limited to sizes [64 x 64]
template<class T, class T_ptr>
int batch_svd_small(T_ptr M, int ldm, int stride_m, T_ptr S, int stride_s, int rows, int cols, int normalize_cols, int num_ops, kblasHandle_t handle)
{
	double* svd_gflops = NULL;

	#ifdef HLIB_PROFILING_ENABLED
	handle->tic();
	check_error_ret( cudaMalloc(&svd_gflops, num_ops * sizeof(double)), KBLAS_Error_Allocation );
	#endif

	int rows_per_thread = iDivUp(rows, WARP_SIZE);
	int total_pairs = iDivUp(cols, 2);
	int srows = rows_per_thread * WARP_SIZE, scols = total_pairs * 2;

	dim3 dimBlock(WARP_SIZE, total_pairs);
	dim3 dimGrid(num_ops, 1);
	size_t smem_needed = (srows * scols + scols) * sizeof(T) + (2 * total_pairs) * sizeof(int);

	switch(rows_per_thread)
	{
		case 1: blockSVDKernel<1, T, T_ptr><<< dimGrid, dimBlock, smem_needed, handle->stream >>>(M, ldm, stride_m, S, stride_s, rows, cols, normalize_cols, num_ops, svd_gflops); break;
		case 2: blockSVDKernel<2, T, T_ptr><<< dimGrid, dimBlock, smem_needed, handle->stream >>>(M, ldm, stride_m, S, stride_s, rows, cols, normalize_cols, num_ops, svd_gflops); break;
		default: printf("batch_svd: Invalid rows_per_thread %d\n", rows_per_thread); return -1;
	}
	batchSortSingularValues_small<T, T_ptr>(M, ldm, stride_m, S, stride_s, rows, cols, num_ops, handle);
	check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	
	#ifdef HLIB_PROFILING_ENABLED
	double time_elapsed = handle->toc();
	double svd_gflop_sum = reduceSum(svd_gflops, num_ops, handle->stream);
	check_error_ret( cudaFree(svd_gflops), KBLAS_Error_Deallocation );
	PerformanceCounter::addOpCount(PerformanceCounter::SVD, 6 * rows * svd_gflop_sum);
    PerformanceCounter::addOpTime(PerformanceCounter::SVD, time_elapsed);
	#endif

	return KBLAS_Success;
}

template<class T, class T_ptr>
int batch_svd_small(T_ptr M, int ldm, int stride_m, T_ptr S, int stride_s, int rows, int cols, int num_ops, kblasHandle_t handle)
{
	return batch_svd_small<T, T_ptr>(M, ldm, stride_m, S, stride_s, rows, cols, 1, num_ops, handle);
}

// Normalize columns in global memory
template<class T, class T_ptr>
int batchNormalizeColumns(T_ptr M, int ldm, int stride_m, T_ptr S, int stride_s, int rows, int cols, int num_ops, kblasHandle_t handle)
{
    int warps_per_op = 16;
    int cols_per_thread = iDivUp(cols, warps_per_op);
    int rows_per_thread = iDivUp(rows, WARP_SIZE);

	dim3 dimBlock(WARP_SIZE, warps_per_op);
	dim3 dimGrid(num_ops, 1);

    batchNormalizeColumnsKernel<T, T_ptr><<< dimGrid, dimBlock, 0, handle->stream >>>
		(M, ldm, stride_m, S, stride_s, rows, cols, rows_per_thread, cols_per_thread, num_ops);
		
	check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	return KBLAS_Success;
}

template<class T, class T_ptr>
void batchMaxOffdiagonalSum(T_ptr M, int ldm, int stride_m, int rows, int cols, T* offdiagonal, int num_ops, kblasHandle_t handle)
{
    int ops_per_block = std::min(16, num_ops);
    int cols_per_thread = iDivUp(cols, WARP_SIZE);
    int blocks = iDivUp(num_ops, ops_per_block);

	dim3 dimBlock(WARP_SIZE, ops_per_block);
	dim3 dimGrid(blocks, 1);

    int smem_needed = ops_per_block * cols * sizeof(T);
    batchMaxOffdiagonalSumKernel<T, T_ptr>
		<<< dimGrid, dimBlock, smem_needed, handle->stream >>>(M, ldm, stride_m, rows, cols, cols_per_thread, offdiagonal, num_ops);
}

// Special routine to handle tall ( rows in [65, 512] and cols <= 64 at this point) matrix svd
template<class T, class T_ptr>
int batch_tall_svd(T_ptr M, int ldm, int stride_m, T_ptr S, int stride_s, int rows, int cols, int num_ops, kblasHandle_t handle)
{
	KBlasWorkspaceState ws_per_op, local_ws_per_op;
	batch_tall_svd_workspace<T>(rows, cols, 1, ws_per_op, 0);
	batch_tall_svd_workspace<T>(rows, cols, 1, local_ws_per_op, 1);

	KBlasWorkspaceState available_ws = handle->work_space.getAvailable();
	
	int op_increment_data = available_ws.d_data_bytes / ws_per_op.d_data_bytes;
	int op_increment_ptr  = available_ws.d_ptrs_bytes / ws_per_op.d_ptrs_bytes;
	int op_increment      = std::min(std::min(num_ops, op_increment_data), op_increment_ptr);
	
	if(op_increment == 0)
		return KBLAS_InsufficientWorkspace;

	T* R_strided   = (T*)handle->work_space.push_d_data(local_ws_per_op.d_data_bytes * op_increment);
	T* tau_strided = R_strided + cols * cols * op_increment;
	T* Q_strided   = tau_strided + cols * op_increment;

	T** Q_ptrs     = (T**)handle->work_space.push_d_ptrs(local_ws_per_op.d_ptrs_bytes * op_increment);
	T** R_ptrs     = Q_ptrs + op_increment;
	T** M_ptrs     = R_ptrs + op_increment;
	T** tau_ptrs   = M_ptrs + op_increment;

	// Generate the pointers for the workspace data only once
	generateArrayOfPointers(Q_strided, Q_ptrs, rows * cols, op_increment, handle->stream);
	generateArrayOfPointers(R_strided, R_ptrs, cols * cols, op_increment, handle->stream);
	generateArrayOfPointers(tau_strided, tau_ptrs, cols, op_increment, handle->stream);

	for(int op_start = 0; op_start < num_ops; op_start += op_increment)
	{
		int batch_size = std::min(op_increment, num_ops - op_start);

		T_ptr M_batch = advanceOperationPtr(M, op_start, stride_m);
		T_ptr S_batch = advanceOperationPtr(S, op_start, stride_s);

		// Generate pointers for the batch gemm routine
		generateArrayOfPointers(M_batch, M_ptrs, stride_m, batch_size, handle->stream);
	
		// [Q, R] = qr(M)
		kblas_geqrf_batch(handle, rows, cols, M_batch, ldm, stride_m, selectPointerData<T, T_ptr>(tau_strided, tau_ptrs), cols, batch_size);
		kblas_copy_upper_batch(handle, rows, cols, M_batch, ldm, stride_m, selectPointerData<T, T_ptr>(R_strided, R_ptrs), cols, cols * cols, batch_size);
		kblas_orgqr_batch(handle, rows, cols, M_batch, ldm, stride_m, selectPointerData<T, T_ptr>(tau_strided, tau_ptrs), cols, batch_size);
		kblas_copyBlock_batch(handle, rows, cols, selectPointerData<T, T_ptr>(Q_strided, Q_ptrs), 0, 0, rows, rows * cols, M_batch, 0, 0, ldm, stride_m, batch_size);
		
		// [U, S, ~] = svd(R)
		batch_svd_small<T, T_ptr>(selectPointerData<T, T_ptr>(R_strided, R_ptrs), cols, cols * cols, S_batch, stride_s, cols, cols, batch_size, handle);

		// M = Q * U
		kblas_gemm_batch(
			handle, KBLAS_NoTrans, KBLAS_NoTrans,
			rows, cols, cols, 1,
			(const T**)Q_ptrs, rows, (const T**)R_ptrs, cols,
			0, M_ptrs, ldm, batch_size
		);
	}
	
	handle->work_space.pop_d_ptrs(local_ws_per_op.d_ptrs_bytes * op_increment);
	handle->work_space.pop_d_data(local_ws_per_op.d_data_bytes * op_increment);
	
	return KBLAS_Success;
}

// One sided block jacobi - uses shared memory kernel
// TODO: Resolve CPU allocation for the block_col_ptrs
template<class T, class T_ptr>
int batch_svd_osbj(T_ptr M, int ldm, int stride_m, T_ptr S, int stride_s, int rows, int cols, int num_ops, int use_gram, kblasHandle_t handle)
{
	if(cols <= SHARED_SVD_DIM_LIMIT && rows <= SHARED_SVD_DIM_LIMIT)
		return batch_svd_small<T, T_ptr>(M, ldm, stride_m, S, stride_s, rows, cols, num_ops, handle);

    T tolerance = OSBJ_BS * KBlasEpsilon<T>::eps * cols * cols;
	int block_cols = iDivUp(cols, OSBJ_BS);
	
	KBlasWorkspaceState ws_per_op, local_ws_per_op;
	batch_svd_osbj_workspace<T>(rows, cols, 1, ws_per_op, 0);
	batch_svd_osbj_workspace<T>(rows, cols, 1, local_ws_per_op, 1);

	KBlasWorkspaceState available_ws = handle->work_space.getAvailable();
	
	int op_increment_data = available_ws.d_data_bytes / ws_per_op.d_data_bytes;
	int op_increment_ptr  = available_ws.d_ptrs_bytes / ws_per_op.d_ptrs_bytes;
	int op_increment      = std::min(std::min(num_ops, op_increment_data), op_increment_ptr);
	
	if(op_increment == 0)
		return KBLAS_InsufficientWorkspace;

	// Assign pointers from the workspace
	T* gram_strided  = (T*)handle->work_space.push_d_data(local_ws_per_op.d_data_bytes * op_increment);
	T* Aij_strided   = gram_strided  + 4 * OSBJ_BS * OSBJ_BS   * op_increment;
	T* svals_strided = Aij_strided   + 2 * OSBJ_BS * rows      * op_increment;
	T* tau_strided   = svals_strided + 2 * OSBJ_BS             * op_increment;
	T* offdiagonal   = tau_strided   + 2 * OSBJ_BS             * op_increment;

	T** gram_ii_ptrs = (T**)handle->work_space.push_d_ptrs(local_ws_per_op.d_ptrs_bytes * op_increment);
	T** gram_ij_ptrs = gram_ii_ptrs + op_increment;
	T** gram_ji_ptrs = gram_ij_ptrs + op_increment;
	T** gram_jj_ptrs = gram_ji_ptrs + op_increment;
	T** temp_i_ptrs  = gram_jj_ptrs + op_increment;
	T** temp_j_ptrs  = temp_i_ptrs  + op_increment;

	std::vector<T**> block_col_ptrs(block_cols);
	for(int i = 0; i < block_cols; i++)
        block_col_ptrs[i] = temp_j_ptrs + (i + 1) * op_increment;

    // Populate all the pointer arrays
	// Gram matrix sub-blocks
	generateArrayOfPointers(gram_strided, gram_ii_ptrs, (2 * OSBJ_BS) * (2 * OSBJ_BS), 0, op_increment, handle->stream);
    generateArrayOfPointers(gram_strided, gram_ij_ptrs, (2 * OSBJ_BS) * (2 * OSBJ_BS), (2 * OSBJ_BS) * OSBJ_BS, op_increment, handle->stream);
    generateArrayOfPointers(gram_strided, gram_ji_ptrs, (2 * OSBJ_BS) * (2 * OSBJ_BS), OSBJ_BS, op_increment, handle->stream);
    generateArrayOfPointers(gram_strided, gram_jj_ptrs, (2 * OSBJ_BS) * (2 * OSBJ_BS), (2 * OSBJ_BS) * OSBJ_BS + OSBJ_BS, op_increment, handle->stream);

    // Aij blocks
    generateArrayOfPointers(Aij_strided, temp_i_ptrs, rows * 2 * OSBJ_BS, 0, op_increment, handle->stream);
    generateArrayOfPointers(Aij_strided, temp_j_ptrs, rows * 2 * OSBJ_BS, rows * OSBJ_BS, op_increment, handle->stream);

	int result = KBLAS_Success;
	
	for(int op_start = 0; op_start < num_ops; op_start += op_increment)
	{
		int batch_size = std::min(op_increment, num_ops - op_start);
		int max_sweeps = std::max(5 * block_cols, 12);
		int sweeps = 0, converged = 0;

		T_ptr M_batch = advanceOperationPtr(M, op_start, stride_m);
		T_ptr S_batch = advanceOperationPtr(S, op_start, stride_s);

		// The block columns
		for(int i = 0; i < block_cols; i++)
			generateArrayOfPointers(M_batch, block_col_ptrs[i], stride_m, ldm * OSBJ_BS * i, batch_size, handle->stream);

		while(sweeps < max_sweeps && converged == 0)
		{
			// Reset offdiagonal terms
			fillArray(offdiagonal, batch_size, 0, handle->stream);

			for(int i = 0; i < block_cols - 1; i++)
			{
				for(int j = i + 1; j < block_cols; j++)
				{
					int i_cols = OSBJ_BS, j_cols = std::min(OSBJ_BS, cols - j * OSBJ_BS);
					// Form Aij since we need it anyway for the block rotations - this will allow us to
					// form the gram matrix in one gemm instead of 4
					int Aij_cols = i_cols + j_cols;

					kblas_copyBlock_batch(handle, rows, i_cols, temp_i_ptrs, 0, 0, rows, block_col_ptrs[i], 0, 0, ldm, batch_size);
					kblas_copyBlock_batch(handle, rows, j_cols, temp_j_ptrs, 0, 0, rows, block_col_ptrs[j], 0, 0, ldm, batch_size);

					if(use_gram)
					{
						// Get the gram matrix for the blocks i and j
						// G = [Gii Gij; Gji Gjj]
						kblas_gemm_batch(
							handle, KBLAS_Trans, KBLAS_NoTrans, Aij_cols, Aij_cols, rows, 1, 
							(const T**)temp_i_ptrs, rows, (const T**)temp_i_ptrs, rows, 0, 
							gram_ii_ptrs, 2 * OSBJ_BS, batch_size
						);

						// Get the sum of the offdiagonal terms of G
						batchMaxOffdiagonalSum<T, T*>(gram_strided, 2 * OSBJ_BS, 4 * OSBJ_BS * OSBJ_BS, Aij_cols, Aij_cols, offdiagonal, batch_size, handle);

						// Get the SVD of G
						batch_svd_small<T, T*>(gram_strided, 2 * OSBJ_BS, 4 * OSBJ_BS * OSBJ_BS, svals_strided, 2 * OSBJ_BS, Aij_cols, Aij_cols, batch_size, handle);
					}
					else
					{
						// Get the QR decomposition of the block column and store R in G
						kblas_geqrf_batch(handle, rows, Aij_cols, Aij_strided, rows, rows * 2 * OSBJ_BS, tau_strided, 2 * OSBJ_BS, batch_size);
						kblas_copy_upper_batch(handle, rows, Aij_cols, Aij_strided, rows, rows * 2 * OSBJ_BS, gram_strided, 2 * OSBJ_BS, 4 * OSBJ_BS * OSBJ_BS, batch_size);

						// Get the sum of the offdiagonal terms of G
						batchMaxOffdiagonalSum<T, T*>(gram_strided, 2 * OSBJ_BS, 4 * OSBJ_BS * OSBJ_BS, Aij_cols, Aij_cols, offdiagonal, batch_size, handle);

						// Get the SVD of G
						batch_svd_small<T, T*>(gram_strided, 2 * OSBJ_BS, 4 * OSBJ_BS * OSBJ_BS, svals_strided, 2 * OSBJ_BS, Aij_cols, Aij_cols, 0, batch_size, handle);

						// Get Q from the reflectors
						kblas_orgqr_batch(handle, rows, Aij_cols, Aij_strided, rows, rows * 2 * OSBJ_BS, tau_strided, 2 * OSBJ_BS, batch_size);
					}

					// Rotate the block columns
					kblas_gemm_batch(
						handle, KBLAS_NoTrans, KBLAS_NoTrans, rows, i_cols, i_cols, 1, 
						(const T**)temp_i_ptrs, rows, (const T**)gram_ii_ptrs, 2 * OSBJ_BS, 0, 
						block_col_ptrs[i], ldm, batch_size
					);
					kblas_gemm_batch(
						handle, KBLAS_NoTrans, KBLAS_NoTrans, rows, i_cols, j_cols, 1, 
						(const T**)temp_j_ptrs, rows, (const T**)gram_ji_ptrs, 2 * OSBJ_BS, 1, 
						block_col_ptrs[i], ldm, batch_size
					);

					kblas_gemm_batch(
						handle, KBLAS_NoTrans, KBLAS_NoTrans, rows, j_cols, j_cols, 1, 
						(const T**)temp_j_ptrs, rows, (const T**)gram_jj_ptrs, 2 * OSBJ_BS, 0, 
						block_col_ptrs[j], ldm, batch_size
					);
					kblas_gemm_batch(
						handle, KBLAS_NoTrans, KBLAS_NoTrans, rows, j_cols, i_cols, 1, 
						(const T**)temp_i_ptrs, rows, (const T**)gram_ij_ptrs, 2 * OSBJ_BS, 1, 
						block_col_ptrs[j], ldm, batch_size
					);
				}
			}
			T max_off_diag = getMaxElement(offdiagonal, batch_size, handle->stream);
			//printf("Max off diag = %e\n", max_off_diag);
			if(max_off_diag < tolerance) converged = 1;
			sweeps++;
		}

		// Normalize the columns of the matrix and compute the singular values
		batchNormalizeColumns<T, T_ptr>(M_batch, ldm, stride_m, S_batch, stride_s, rows, cols, batch_size, handle);
		
		if(converged != 1) 
			result = KBLAS_SVD_NoConvergence;
	}

	handle->work_space.pop_d_ptrs(local_ws_per_op.d_ptrs_bytes * op_increment);
	handle->work_space.pop_d_data(local_ws_per_op.d_data_bytes * op_increment);
	
	return result;
}

template<class T, class T_ptr>
int batch_svd_osbj_gram(T_ptr M, int ldm, int stride_m, T_ptr S, int stride_s, int rows, int cols, int num_ops, kblasHandle_t handle)
{
	return batch_svd_osbj<T, T_ptr>(M, ldm, stride_m, S, stride_s, rows, cols, num_ops, 1, handle);
}

template<class T, class T_ptr>
int batch_svd_osbj_qr(T_ptr M, int ldm, int stride_m, T_ptr S, int stride_s, int rows, int cols, int num_ops, kblasHandle_t handle)
{
	return batch_svd_osbj<T, T_ptr>(M, ldm, stride_m, S, stride_s, rows, cols, num_ops, 0, handle);
}

template<class T, class T_ptr>
int batch_svd_randomized(T_ptr M, int ldm, int stride_m, T_ptr S, int stride_s, int rows, int cols, int rank, int num_ops, kblasHandle_t handle)
{
	if(rank > cols || rank > rows)
	{
		printf("Requested rank must be smaller than or equal to the rank of the matrix\n");
		return KBLAS_Error_WrongInput;
	}

	if(cols == rank)
	{
		if(cols <= SHARED_SVD_DIM_LIMIT && rows <= SHARED_SVD_DIM_LIMIT)
			return batch_svd_small<T, T_ptr>(M, ldm, stride_m, S, stride_s, rows, cols, num_ops, handle);
		else
			return batch_svd_osbj_qr<T, T_ptr>(M, ldm, stride_m, S, stride_s, rows, cols, num_ops, handle);
	}

	KBlasWorkspaceState ws_per_op, local_ws_per_op;
	batch_svd_randomized_workspace<T>(rows, cols, rank, 1, ws_per_op, 0);
	batch_svd_randomized_workspace<T>(rows, cols, rank, 1, local_ws_per_op, 1);

	KBlasWorkspaceState available_ws = handle->work_space.getAvailable();
	
	int op_increment_data = available_ws.d_data_bytes / ws_per_op.d_data_bytes;
	int op_increment_ptr  = available_ws.d_ptrs_bytes / ws_per_op.d_ptrs_bytes;
	int op_increment      = std::min(std::min(num_ops, op_increment_data), op_increment_ptr);
	
	if(op_increment == 0)
		return KBLAS_InsufficientWorkspace;

	// Assign pointers from the workspace
	T* Omega_strided = (T*)handle->work_space.push_d_data(local_ws_per_op.d_data_bytes * op_increment);
	T* Y_strided     = Omega_strided + cols * rank * op_increment;
	T* B_strided     = Y_strided     + rows * rank * op_increment;
	T* R_strided     = B_strided     + rank * cols * op_increment;
	T* tau_strided   = R_strided     + rank * rank * op_increment;

	T** Omega_ptrs   = (T**)handle->work_space.push_d_ptrs(local_ws_per_op.d_ptrs_bytes * op_increment);
	T** M_ptrs       = Omega_ptrs + op_increment;
	T** Y_ptrs       = M_ptrs     + op_increment;
	T** B_ptrs       = Y_ptrs     + op_increment;
	T** R_ptrs       = B_ptrs     + op_increment;

	// Generate array of pointers for the intermediate data
	generateArrayOfPointers(Omega_strided, Omega_ptrs, cols * rank, op_increment, handle->stream);
	generateArrayOfPointers(Y_strided,     Y_ptrs,     rows * rank, op_increment, handle->stream);
	generateArrayOfPointers(B_strided,     B_ptrs,     cols * rank, op_increment, handle->stream);
	generateArrayOfPointers(R_strided,     R_ptrs,     rank * rank, op_increment, handle->stream);

	for(int op_start = 0; op_start < num_ops; op_start += op_increment)
	{
		int batch_size = std::min(op_increment, num_ops - op_start);
		// printf("RSVD Batch %d -> %d\n", op_start, op_start + batch_size);

		T_ptr M_batch = advanceOperationPtr(M, op_start, stride_m);
		T_ptr S_batch = advanceOperationPtr(S, op_start, stride_s);

		// generate the sampling matrices
		generateRandomMatrices(Omega_strided, cols, rank, op_start, batch_size, handle->stream);

		// generate the pointer arrays
		generateArrayOfPointers(M_batch, M_ptrs, stride_m, batch_size, handle->stream);

		// cublas handles
		T alpha = 1, beta = 0;

		// First form the sampled matrix Y = A * omega
		kblas_gemm_batch(
			handle, KBLAS_NoTrans, KBLAS_NoTrans, rows, rank, cols, alpha,
			(const T**)M_ptrs, ldm, (const T**)Omega_ptrs, cols, beta, 
			Y_ptrs, rows, batch_size
		);

		// Overwrite Y with Q of its QR decomposition
		kblas_geqrf_batch(handle, rows, rank, Y_strided, rows, rows * rank, tau_strided, rank, batch_size);
		kblas_geqrf_batch(handle, rows, rank, Y_strided, rows, rows * rank, tau_strided, rank, batch_size);

		// Form B = A' * Q_Y
		kblas_gemm_batch(
			handle, KBLAS_Trans, KBLAS_NoTrans, cols, rank, rows, alpha,
			(const T**)M_ptrs, ldm, (const T**)Y_ptrs, rows, beta, 
			B_ptrs, cols, batch_size
		);

		// Do the QR of B - we only need the Q of B if we want the right singular vectors
		kblas_geqrf_batch(
			handle, cols, rank, B_strided, cols, cols * rank,
			tau_strided, rank, batch_size
		);
		kblas_copy_upper_batch(
			handle, cols, rank, B_strided, cols, cols * rank,
			R_strided, rank, rank * rank, batch_size
		);

		// Transpose R so that we can get its right singular vectors when we
		// do the svd (since the svd gets the left singular vectors, which are
		// the right singular vectors of the transpose of the matrix). Store the
		// transpose in omega
		generateArrayOfPointers(Omega_strided, Omega_ptrs, rank * rank, batch_size, handle->stream);
		kblas_transpose_batch(handle, rank, rank, R_ptrs, rank, Omega_ptrs, rank, batch_size);

		// Now do the SVD of omega = R'
		if(rank <= SHARED_SVD_DIM_LIMIT)
			batch_svd_small<T, T_ptr>(selectPointerData<T, T_ptr>(Omega_strided, Omega_ptrs), rank, rank * rank, S_batch, stride_s, rank, rank, batch_size, handle);
		else
			batch_svd_osbj_qr<T, T_ptr>(selectPointerData<T, T_ptr>(Omega_strided, Omega_ptrs), rank, rank * rank, S_batch, stride_s, rank, rank, batch_size, handle);

		// Finally, we overwrite the matrix with left singular values of the
		// truncated SVD as the product U_A = Q_Y * V_B
		kblas_gemm_batch(
			handle, KBLAS_NoTrans, KBLAS_NoTrans, rows, rank, rank, alpha,
			(const T**)Y_ptrs, rows, (const T**)Omega_ptrs, rank, beta, 
			M_ptrs, ldm, batch_size
		);
	}
	
	handle->work_space.pop_d_ptrs(local_ws_per_op.d_ptrs_bytes * op_increment);
	handle->work_space.pop_d_data(local_ws_per_op.d_data_bytes * op_increment);
	
	return KBLAS_Success;
}

////////////////////////////////////////////////////////////////////
// workspace query routines for both strided and array of pointers interface
////////////////////////////////////////////////////////////////////
void kblasDgesvj_batch_wsquery(kblasHandle_t handle, int m, int n, int ops)
{
	if(m <= SHARED_SVD_DIM_LIMIT && n <= SHARED_SVD_DIM_LIMIT)
		return;
	else if(n <= SHARED_SVD_DIM_LIMIT)
		batch_tall_svd_workspace<double>(m, n, ops, handle->work_space.requested_ws_state, 0);
	else
		batch_svd_osbj_workspace<double>(m, n, ops, handle->work_space.requested_ws_state, 0);
}

void kblasSgesvj_batch_wsquery(kblasHandle_t handle, int m, int n, int ops)
{
	if(m <= SHARED_SVD_DIM_LIMIT && n <= SHARED_SVD_DIM_LIMIT)
		return;
	else if(n <= SHARED_SVD_DIM_LIMIT)
		batch_tall_svd_workspace<float>(m, n, ops, handle->work_space.requested_ws_state, 0);
	else
		batch_svd_osbj_workspace<float>(m, n, ops, handle->work_space.requested_ws_state, 0);
}

void kblasDgesvj_gram_batch_wsquery(kblasHandle_t handle, int m, int n, int ops)
{
	if(m <= SHARED_SVD_DIM_LIMIT && n <= SHARED_SVD_DIM_LIMIT)
		return;
	
	batch_svd_osbj_workspace<double>(m, n, ops, handle->work_space.requested_ws_state, 0);
}

void kblasSgesvj_gram_batch_wsquery(kblasHandle_t handle, int m, int n, int ops)
{
	if(m <= SHARED_SVD_DIM_LIMIT && n <= SHARED_SVD_DIM_LIMIT)
		return;
	
	batch_svd_osbj_workspace<double>(m, n, ops, handle->work_space.requested_ws_state, 0);
}

void kblasDrsvd_batch_batch_wsquery(kblasHandle_t handle, int m, int n, int rank, int ops)
{
	batch_svd_randomized_workspace<double>(m, n, rank, ops, handle->work_space.requested_ws_state, 0);
}

void kblasSrsvd_batch_batch_wsquery(kblasHandle_t handle, int m, int n, int rank, int ops)
{
	batch_svd_randomized_workspace<float>(m, n, rank, ops, handle->work_space.requested_ws_state, 0);
}

////////////////////////////////////////////////////////////////////
// Strided interface
////////////////////////////////////////////////////////////////////
int kblasDgesvj_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops)
{
	if(m < n)
		return KBLAS_NotImplemented;
	
	if(m <= SHARED_SVD_DIM_LIMIT && n <= SHARED_SVD_DIM_LIMIT)
		return batch_svd_small<double, double*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, num_ops, handle);
	else if(n <= SHARED_SVD_DIM_LIMIT)
		return batch_tall_svd<double, double*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, num_ops, handle);
	else
		return batch_svd_osbj_qr<double, double*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, num_ops, handle);
}

int kblasSgesvj_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops)
{
	if(m < n)
		return KBLAS_NotImplemented;
	
	if(m <= SHARED_SVD_DIM_LIMIT && n <= SHARED_SVD_DIM_LIMIT)
		return batch_svd_small<float, float*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, num_ops, handle);
	else if(n <= SHARED_SVD_DIM_LIMIT)
		return batch_tall_svd<float, float*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, num_ops, handle);
	else
		return batch_svd_osbj_qr<float, float*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, num_ops, handle);
}

int kblasDgesvj_gram_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops)
{
	if(m < n)
		return KBLAS_NotImplemented;
	
	return batch_svd_osbj_gram<double, double*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, num_ops, handle);
}

int kblasSgesvj_gram_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops)
{
	if(m < n)
		return KBLAS_NotImplemented;
	
	return batch_svd_osbj_gram<float, float*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, num_ops, handle);
}

int kblasDrsvd_batch_strided(kblasHandle_t handle, int m, int n, int rank, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops)
{
	return batch_svd_randomized<double, double*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, rank, num_ops, handle);
}

int kblasSrsvd_batch_strided(kblasHandle_t handle, int m, int n, int rank, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops)
{
	return batch_svd_randomized<float, float*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, rank, num_ops, handle);
}

////////////////////////////////////////////////////////////////////
// Array of pointers interface
////////////////////////////////////////////////////////////////////
int kblasDgesvj_batch(kblasHandle_t handle, int m, int n, double** A_ptrs, int lda, double** S_ptrs, int num_ops)
{
	if(m < n)
		return KBLAS_NotImplemented;
	
	if(m <= SHARED_SVD_DIM_LIMIT && n <= SHARED_SVD_DIM_LIMIT)
		return batch_svd_small<double, double**>(A_ptrs, lda, 0, S_ptrs, 0, m, n, num_ops, handle);
	else if(n <= SHARED_SVD_DIM_LIMIT)
		return batch_tall_svd<double, double**>(A_ptrs, lda, 0, S_ptrs, 0, m, n, num_ops, handle);
	else
		return batch_svd_osbj_qr<double, double**>(A_ptrs, lda, 0, S_ptrs, 0, m, n, num_ops, handle);
}

int kblasSgesvj_batch(kblasHandle_t handle, int m, int n, float** A_ptrs, int lda, float** S_ptrs, int num_ops)
{
	if(m < n)
		return KBLAS_NotImplemented;
	
	if(m <= SHARED_SVD_DIM_LIMIT && n <= SHARED_SVD_DIM_LIMIT)
		return batch_svd_small<float, float**>(A_ptrs, lda, 0, S_ptrs, 0, m, n, num_ops, handle);
	else if(n <= SHARED_SVD_DIM_LIMIT)
		return batch_tall_svd<float, float**>(A_ptrs, lda, 0, S_ptrs, 0, m, n, num_ops, handle);
	else
		return batch_svd_osbj_qr<float, float**>(A_ptrs, lda, 0, S_ptrs, 0, m, n, num_ops, handle);
}

int kblasDgesvj_gram_batch(kblasHandle_t handle, int m, int n, double** A_ptrs, int lda, double** S_ptrs, int num_ops)
{
	if(m < n)
		return KBLAS_NotImplemented;
	
	return batch_svd_osbj_gram<double, double**>(A_ptrs, lda, 0, S_ptrs, 0, m, n, num_ops, handle);
}

int kblasSgesvj_gram_batch(kblasHandle_t handle, int m, int n, float** A_ptrs, int lda, float** S_ptrs, int num_ops)
{
	if(m < n)
		return KBLAS_NotImplemented;
	
	return batch_svd_osbj_gram<float, float**>(A_ptrs, lda, 0, S_ptrs, 0, m, n, num_ops, handle);
}

int kblasDrsvd_batch(kblasHandle_t handle, int m, int n, int rank, double** A_ptrs, int lda, double** S_ptrs, int num_ops)
{
	return batch_svd_randomized<double, double**>(A_ptrs, lda, 0, S_ptrs, 0, m, n, rank, num_ops, handle);
}

int kblasSrsvd_batch(kblasHandle_t handle, int m, int n, int rank, float** A_ptrs, int lda, float** S_ptrs, int num_ops)
{
	return batch_svd_randomized<float, float**>(A_ptrs, lda, 0, S_ptrs, 0, m, n, rank, num_ops, handle);
}
