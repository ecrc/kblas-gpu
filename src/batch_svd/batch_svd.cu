
#include "kblas_struct.h"

#include "mini_blas_gpu.h"

#include <thrust_wrappers.h>
#include <batch_svd.h>
#include <batch_qr.h>
#include <batch_transpose.h>
#include <batch_block_copy.h>
#include <batch_mm_wrappers.h>
#include <vector>

#include "svd_kernels.cuh"

#define OSBJ_BS		SHARED_SVD_DIM_LIMIT / 2

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Workspace routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
unsigned int batch_svd_osbj_workspace(int rows, int cols, unsigned int* ws_sizes_per_op)
{
	int block_cols = iDivUp(cols, OSBJ_BS);

	ws_sizes_per_op[0] = (
		4 * OSBJ_BS * OSBJ_BS + // Gram matrix or R
		rows * 2 * OSBJ_BS    + // Temporary block column
		2 * OSBJ_BS           + // Temporary singular values
		2 * OSBJ_BS           + // tau
		1                       // Offdiagonal sum
	) * sizeof(T);

	ws_sizes_per_op[1] = (block_cols + 7) * sizeof(T*);

	return ws_sizes_per_op[0] + ws_sizes_per_op[1];
}

template<class T>
unsigned int batch_tall_svd_workspace(int rows, int cols, unsigned int* ws_sizes_per_op)
{
	ws_sizes_per_op[0] = (
		cols * cols + // R
		rows * cols + // Q
		cols          // tau
	) * sizeof(T);

	ws_sizes_per_op[1] = 4 * sizeof(T*);

	return ws_sizes_per_op[0] + ws_sizes_per_op[1];
}

template<class T>
unsigned int batch_wide_svd_workspace(int rows, int cols, unsigned int* ws_sizes_per_op)
{
	ws_sizes_per_op[0] = (
		rows * rows + // R
		cols * rows + // Q
		rows          // tau
	) * sizeof(T);

	ws_sizes_per_op[1] = 3 * sizeof(T*);
	ws_sizes_per_op[2] = 0;
	ws_sizes_per_op[3] = 0;

	// Do we need to do osbj of the rows x rows matrix?
	if(rows > SHARED_SVD_DIM_LIMIT)
		batch_svd_osbj_workspace<T>(rows, rows, ws_sizes_per_op + 2);

	return ws_sizes_per_op[0] + ws_sizes_per_op[1] + ws_sizes_per_op[2] + ws_sizes_per_op[3];
}

template<class T>
unsigned int batch_svd_randomized_workspace(int rows, int cols, int rank, unsigned int* ws_sizes_per_op)
{
	ws_sizes_per_op[0] = (
		cols * rank + // Omega
		rows * rank + // Y
		rank * cols + // B
		rank * rank + // R
		rank          // tau
	) * sizeof(T);

	ws_sizes_per_op[1] = 5 * sizeof(T*);
	ws_sizes_per_op[2] = 0;
	ws_sizes_per_op[3] = 0;

	// Do we need to do osbj of the rank x rank matrix?
	if(rank > SHARED_SVD_DIM_LIMIT)
		batch_svd_osbj_workspace<T>(rank, rank, ws_sizes_per_op + 2);

	return ws_sizes_per_op[0] + ws_sizes_per_op[1] + ws_sizes_per_op[2] + ws_sizes_per_op[3];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Driver routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T, class T_ptr>
void batchSortSingularValues_small(T_ptr M, int ldm, int stride_m, T_ptr S, int stride_s, int rows, int cols, int num_ops, GPUBlasHandle& handle)
{
	int block_size = WARP_SIZE * iDivUp(rows, WARP_SIZE);

	dim3 dimBlock(block_size, 1);
	dim3 dimGrid(num_ops, 1);

	size_t smem_needed = (rows * cols) * sizeof(T) + cols * sizeof(int);

	switch(block_size)
	{
		case  32: batchSortSingularValuesSmallKernel<32, T, T_ptr><<< dimGrid, dimBlock, smem_needed, handle.stream >>>(M, ldm, stride_m, S, stride_s, rows, cols, num_ops); break;
		case  64: batchSortSingularValuesSmallKernel<64, T, T_ptr><<< dimGrid, dimBlock, smem_needed, handle.stream >>>(M, ldm, stride_m, S, stride_s, rows, cols, num_ops); break;
		default:  printf("batchSortSingularValues_small: Invalid block_size %d\n", block_size); break;
	}

	gpuErrchk( cudaGetLastError() );

}

// Shared memory kernel limited to sizes [64 x 64]
template<class T, class T_ptr>
int batch_svd_small(T_ptr M, int ldm, int stride_m, T_ptr S, int stride_s, int rows, int cols, int normalize_cols, int num_ops, GPUBlasHandle& handle)
{
	double* svd_gflops = NULL;

	#ifdef HLIB_PROFILING_ENABLED
	handle.tic();
	gpuErrchk( cudaMalloc(&svd_gflops, num_ops * sizeof(double)) );
	#endif

	int rows_per_thread = iDivUp(rows, WARP_SIZE);
	int total_pairs = iDivUp(cols, 2);
	int srows = rows_per_thread * WARP_SIZE, scols = total_pairs * 2;

	dim3 dimBlock(WARP_SIZE, total_pairs);
	dim3 dimGrid(num_ops, 1);
	size_t smem_needed = (srows * scols + scols) * sizeof(T) + (2 * total_pairs) * sizeof(int);

	switch(rows_per_thread)
	{
		case 1: blockSVDKernel<1, T, T_ptr><<< dimGrid, dimBlock, smem_needed, handle.stream >>>(M, ldm, stride_m, S, stride_s, rows, cols, normalize_cols, num_ops, svd_gflops); break;
		case 2: blockSVDKernel<2, T, T_ptr><<< dimGrid, dimBlock, smem_needed, handle.stream >>>(M, ldm, stride_m, S, stride_s, rows, cols, normalize_cols, num_ops, svd_gflops); break;
		default: printf("batch_svd: Invalid rows_per_thread %d\n", rows_per_thread); return -1;
	}
	batchSortSingularValues_small<T, T_ptr>(M, ldm, stride_m, S, stride_s, rows, cols, num_ops, handle);
    gpuErrchk( cudaGetLastError() );

	#ifdef HLIB_PROFILING_ENABLED
	double time_elapsed = handle.toc();
	double svd_gflop_sum = reduceSum(svd_gflops, num_ops, handle.stream);
	gpuErrchk( cudaFree(svd_gflops) );
	PerformanceCounter::addOpCount(PerformanceCounter::SVD, 6 * rows * svd_gflop_sum);
    PerformanceCounter::addOpTime(PerformanceCounter::SVD, time_elapsed);
	#endif

	return 0;
}

template<class T, class T_ptr>
int batch_svd_small(T_ptr M, int ldm, int stride_m, T_ptr S, int stride_s, int rows, int cols, int num_ops, GPUBlasHandle& handle)
{
	return batch_svd_small<T, T_ptr>(M, ldm, stride_m, S, stride_s, rows, cols, 1, num_ops, handle);
}

// Normalize columns in global memory
template<class T, class T_ptr>
void batchNormalizeColumns(T_ptr M, int ldm, int stride_m, T_ptr S, int stride_s, int rows, int cols, int num_ops, GPUBlasHandle& handle)
{
    int warps_per_op = 16;
    int cols_per_thread = iDivUp(cols, warps_per_op);
    int rows_per_thread = iDivUp(rows, WARP_SIZE);

	dim3 dimBlock(WARP_SIZE, warps_per_op);
	dim3 dimGrid(num_ops, 1);

    batchNormalizeColumnsKernel<T, T_ptr><<< dimGrid, dimBlock, 0, handle.stream >>>
		(M, ldm, stride_m, S, stride_s, rows, cols, rows_per_thread, cols_per_thread, num_ops);
}

template<class T, class T_ptr>
void batchMaxOffdiagonalSum(T_ptr M, int ldm, int stride_m, int rows, int cols, T* offdiagonal, int num_ops, GPUBlasHandle& handle)
{
    int ops_per_block = std::min(16, num_ops);
    int cols_per_thread = iDivUp(cols, WARP_SIZE);
    int blocks = iDivUp(num_ops, ops_per_block);

	dim3 dimBlock(WARP_SIZE, ops_per_block);
	dim3 dimGrid(blocks, 1);

    int smem_needed = ops_per_block * cols * sizeof(T);
    batchMaxOffdiagonalSumKernel<T, T_ptr>
		<<< dimGrid, dimBlock, smem_needed, handle.stream >>>(M, ldm, stride_m, rows, cols, cols_per_thread, offdiagonal, num_ops);
}

// Special routine to handle tall ( rows in [65, 512] and cols <= 64 at this point) matrix svd
template<class T, class T_ptr>
int batch_tall_svd(T_ptr M, int ldm, int stride_m, T_ptr S, int stride_s, int rows, int cols, int num_ops, GPUBlasHandle& handle)
{
	unsigned int ws_bytes[2];
	unsigned int ws_per_op = batch_tall_svd_workspace<T>(rows, cols, &ws_bytes[0]);

	int op_increment = handle.workspace_bytes / ws_per_op;
	if(op_increment > num_ops) op_increment = num_ops;
	alignWorkspace(&ws_bytes[0], 2, handle.workspace_bytes, op_increment, 8);

	if(op_increment == 0)
	{
		printf("Batch Tall SVD: Insufficient workspace\n");
		return -1;
	}

	T* R_strided   = (T*)handle.workspace;
	T* tau_strided = R_strided + cols * cols * op_increment;
	T* Q_strided   = tau_strided + cols * op_increment;

	T** Q_ptrs     = (T**)((GPUBlasHandle::WS_Byte*)handle.workspace + ws_bytes[0]);
	T** R_ptrs     = Q_ptrs + op_increment;
	T** M_ptrs     = R_ptrs + op_increment;
	T** tau_ptrs   = M_ptrs + op_increment;

	// Generate the pointers for the workspace data only once
	generateArrayOfPointers(Q_strided, Q_ptrs, rows * cols, op_increment, handle.stream);
	generateArrayOfPointers(R_strided, R_ptrs, cols * cols, op_increment, handle.stream);
	generateArrayOfPointers(tau_strided, tau_ptrs, cols, op_increment, handle.stream);

	for(int op_start = 0; op_start < num_ops; op_start += op_increment)
	{
		int batch_size = std::min(op_increment, num_ops - op_start);

		T_ptr M_batch = advanceOperationPtr(M, op_start, stride_m);
		T_ptr S_batch = advanceOperationPtr(S, op_start, stride_s);

		// Generate pointers for the batch gemm routine
		generateArrayOfPointers(M_batch, M_ptrs, stride_m, batch_size, handle.stream);

		kblas_geqrf_batched(rows, cols, M_batch, ldm, stride_m, selectPointerData<T, T_ptr>(tau_strided, tau_ptrs), cols, batch_size, handle);
		kblas_copy_upper_batched(rows, cols, M_batch, ldm, stride_m, selectPointerData<T, T_ptr>(R_strided, R_ptrs), cols, cols * cols, batch_size, handle);
		kblas_orgqr_batched(rows, cols, M_batch, ldm, stride_m, selectPointerData<T, T_ptr>(tau_strided, tau_ptrs), cols, batch_size, handle);

		batchCopyMatrixBlock(M_batch, 0, 0, ldm, stride_m, selectPointerData<T, T_ptr>(Q_strided, Q_ptrs), 0, 0, rows, rows * cols, rows, cols, batch_size, handle);
		batch_svd_small<T, T_ptr>(selectPointerData<T, T_ptr>(R_strided, R_ptrs), cols, cols * cols, S_batch, stride_s, cols, cols, batch_size, handle);

		batch_gemm(
			0, 0, rows, cols, cols, 1,
			Q_ptrs, rows, R_ptrs, cols,
			0, M_ptrs, ldm, batch_size,
			handle
		);
	}

	return 0;
}

// One sided block jacobi - uses shared memory kernel
// TODO: Resolve CPU allocation for the block_col_ptrs
template<class T, class T_ptr>
int batch_svd_osbj(T_ptr M, int ldm, int stride_m, T_ptr S, int stride_s, int rows, int cols, int num_ops, int use_gram, GPUBlasHandle& handle)
{
	if(cols <= SHARED_SVD_DIM_LIMIT && rows <= SHARED_SVD_DIM_LIMIT)
		return batch_svd_small<T, T_ptr>(M, ldm, stride_m, S, stride_s, rows, cols, num_ops, handle);

    T tolerance = OSBJ_BS * KBlasEpsilon<T>::eps * cols * cols;
	int block_cols = iDivUp(cols, OSBJ_BS);
	unsigned int ws_bytes[2] = {0};
	unsigned int ws_per_op = batch_svd_osbj_workspace<T>(rows, cols, &ws_bytes[0]);

	int op_increment = handle.workspace_bytes / ws_per_op;
	if(op_increment > num_ops) op_increment = num_ops;
	alignWorkspace(&ws_bytes[0], 2, handle.workspace_bytes, op_increment, 8);

	if(op_increment == 0)
	{
		printf("Batch Block SVD: Insufficient workspace\n");
		return -1;
	}

	// Assign pointers from the workspace
	T* gram_strided  = (T*)handle.workspace;
	T* Aij_strided   = gram_strided  + 4 * OSBJ_BS * OSBJ_BS   * op_increment;
	T* svals_strided = Aij_strided   + 2 * OSBJ_BS * rows      * op_increment;
	T* tau_strided   = svals_strided + 2 * OSBJ_BS             * op_increment;
	T* offdiagonal   = tau_strided   + 2 * OSBJ_BS             * op_increment;

	T** gram_ii_ptrs = (T**)((GPUBlasHandle::WS_Byte*)handle.workspace + ws_bytes[0]);
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
	generateArrayOfPointers(gram_strided, gram_ii_ptrs, (2 * OSBJ_BS) * (2 * OSBJ_BS), 0, op_increment, handle.stream);
    generateArrayOfPointers(gram_strided, gram_ij_ptrs, (2 * OSBJ_BS) * (2 * OSBJ_BS), (2 * OSBJ_BS) * OSBJ_BS, op_increment, handle.stream);
    generateArrayOfPointers(gram_strided, gram_ji_ptrs, (2 * OSBJ_BS) * (2 * OSBJ_BS), OSBJ_BS, op_increment, handle.stream);
    generateArrayOfPointers(gram_strided, gram_jj_ptrs, (2 * OSBJ_BS) * (2 * OSBJ_BS), (2 * OSBJ_BS) * OSBJ_BS + OSBJ_BS, op_increment, handle.stream);

    // Aij blocks
    generateArrayOfPointers(Aij_strided, temp_i_ptrs, rows * 2 * OSBJ_BS, 0, op_increment, handle.stream);
    generateArrayOfPointers(Aij_strided, temp_j_ptrs, rows * 2 * OSBJ_BS, rows * OSBJ_BS, op_increment, handle.stream);

	for(int op_start = 0; op_start < num_ops; op_start += op_increment)
	{
		int batch_size = std::min(op_increment, num_ops - op_start);
		int max_sweeps = std::max(5 * block_cols, 12);
		int sweeps = 0, converged = 0;

		T_ptr M_batch = advanceOperationPtr(M, op_start, stride_m);
		T_ptr S_batch = advanceOperationPtr(S, op_start, stride_s);

		// The block columns
		for(int i = 0; i < block_cols; i++)
			generateArrayOfPointers(M_batch, block_col_ptrs[i], stride_m, ldm * OSBJ_BS * i, batch_size, handle.stream);

		while(sweeps < max_sweeps && converged == 0)
		{
			// Reset offdiagonal terms
			fillArray(offdiagonal, batch_size, 0, handle.stream);

			for(int i = 0; i < block_cols - 1; i++)
			{
				for(int j = i + 1; j < block_cols; j++)
				{
					int i_cols = OSBJ_BS, j_cols = std::min(OSBJ_BS, cols - j * OSBJ_BS);
					// Form Aij since we need it anyway for the block rotations - this will allow us to
					// form the gram matrix in one gemm instead of 4
					int Aij_cols = i_cols + j_cols;

					batchCopyMatrixBlock(block_col_ptrs[i], 0, 0, ldm, temp_i_ptrs, 0, 0, rows, rows, i_cols, batch_size, handle);
					batchCopyMatrixBlock(block_col_ptrs[j], 0, 0, ldm, temp_j_ptrs, 0, 0, rows, rows, j_cols, batch_size, handle);

					if(use_gram)
					{
						// Get the gram matrix for the blocks i and j
						// G = [Gii Gij; Gji Gjj]
						batch_gemm(1, 0, Aij_cols, Aij_cols, rows, 1, temp_i_ptrs, rows, temp_i_ptrs, rows, 0, gram_ii_ptrs, 2 * OSBJ_BS, batch_size, handle);

						// Get the sum of the offdiagonal terms of G
						batchMaxOffdiagonalSum<T, T*>(gram_strided, 2 * OSBJ_BS, 4 * OSBJ_BS * OSBJ_BS, Aij_cols, Aij_cols, offdiagonal, batch_size, handle);

						// Get the SVD of G
						batch_svd_small<T, T*>(gram_strided, 2 * OSBJ_BS, 4 * OSBJ_BS * OSBJ_BS, svals_strided, 2 * OSBJ_BS, Aij_cols, Aij_cols, batch_size, handle);
					}
					else
					{
						// Get the QR decomposition of the block column and store R in G
						kblas_geqrf_batched(rows, Aij_cols, Aij_strided, rows, rows * 2 * OSBJ_BS, tau_strided, 2 * OSBJ_BS, batch_size, handle);
						kblas_copy_upper_batched(rows, Aij_cols, Aij_strided, rows, rows * 2 * OSBJ_BS, gram_strided, 2 * OSBJ_BS, 4 * OSBJ_BS * OSBJ_BS, batch_size, handle);

						// Get the sum of the offdiagonal terms of G
						batchMaxOffdiagonalSum<T, T*>(gram_strided, 2 * OSBJ_BS, 4 * OSBJ_BS * OSBJ_BS, Aij_cols, Aij_cols, offdiagonal, batch_size, handle);

						// Get the SVD of G
						batch_svd_small<T, T*>(gram_strided, 2 * OSBJ_BS, 4 * OSBJ_BS * OSBJ_BS, svals_strided, 2 * OSBJ_BS, Aij_cols, Aij_cols, 0, batch_size, handle);

						// Get Q from the reflectors
						kblas_orgqr_batched(rows, Aij_cols, Aij_strided, rows, rows * 2 * OSBJ_BS, tau_strided, 2 * OSBJ_BS, batch_size, handle);
					}

					// Rotate the block columns
					batch_gemm(0, 0, rows, i_cols, i_cols, 1, temp_i_ptrs, rows, gram_ii_ptrs, 2 * OSBJ_BS, 0, block_col_ptrs[i], ldm, batch_size, handle);
					batch_gemm(0, 0, rows, i_cols, j_cols, 1, temp_j_ptrs, rows, gram_ji_ptrs, 2 * OSBJ_BS, 1, block_col_ptrs[i], ldm, batch_size, handle);

					batch_gemm(0, 0, rows, j_cols, j_cols, 1, temp_j_ptrs, rows, gram_jj_ptrs, 2 * OSBJ_BS, 0, block_col_ptrs[j], ldm, batch_size, handle);
					batch_gemm(0, 0, rows, j_cols, i_cols, 1, temp_i_ptrs, rows, gram_ij_ptrs, 2 * OSBJ_BS, 1, block_col_ptrs[j], ldm, batch_size, handle);
				}
			}
			T max_off_diag = getMaxElement(offdiagonal, batch_size, handle.stream);
			//printf("Max off diag = %e\n", max_off_diag);
			if(max_off_diag < tolerance) converged = 1;
			sweeps++;
		}

		if(converged != 1) printf("Block SVD did not converge!\n");

		// Normalize the columns of the matrix and compute the singular values
		batchNormalizeColumns<T, T_ptr>(M_batch, ldm, stride_m, S_batch, stride_s, rows, cols, batch_size, handle);
	}

	return 0;
}

template<class T, class T_ptr>
int batch_svd_osbj_gram(T_ptr M, int ldm, int stride_m, T_ptr S, int stride_s, int rows, int cols, int num_ops, GPUBlasHandle& handle)
{
	return batch_svd_osbj<T, T_ptr>(M, ldm, stride_m, S, stride_s, rows, cols, num_ops, 1, handle);
}

template<class T, class T_ptr>
int batch_svd_osbj_qr(T_ptr M, int ldm, int stride_m, T_ptr S, int stride_s, int rows, int cols, int num_ops, GPUBlasHandle& handle)
{
	return batch_svd_osbj<T, T_ptr>(M, ldm, stride_m, S, stride_s, rows, cols, num_ops, 0, handle);
}

template<class T, class T_ptr>
int batch_svd_randomized(T_ptr M, int ldm, int stride_m, T_ptr S, int stride_s, int rows, int cols, int rank, int num_ops, GPUBlasHandle& handle)
{
	if(rank > cols || rank > rows)
	{
		printf("Requested rank must be smaller than or equal to the rank of the matrix\n");
		return -1;
	}

	if(cols == rank)
	{
		if(cols <= SHARED_SVD_DIM_LIMIT && rows <= SHARED_SVD_DIM_LIMIT)
			return batch_svd_small<T, T_ptr>(M, ldm, stride_m, S, stride_s, rows, cols, num_ops, handle);
		else
			return batch_svd_osbj_qr<T, T_ptr>(M, ldm, stride_m, S, stride_s, rows, cols, num_ops, handle);
	}

	unsigned int ws_bytes[4] = {0};
	unsigned int ws_per_op = batch_svd_randomized_workspace<T>(rows, cols, rank, &ws_bytes[0]);

	int op_increment = handle.workspace_bytes / ws_per_op;
	if(op_increment > num_ops) op_increment = num_ops;
	alignWorkspace(&ws_bytes[0], 4, handle.workspace_bytes, op_increment, 8);

	if(op_increment == 0)
	{
		printf("Batch RSVD: Insufficient workspace\n");
		return -1;
	}

	// Assign pointers from the workspace
	T* Omega_strided = (T*)handle.workspace;
	T* Y_strided     = Omega_strided + cols * rank * op_increment;
	T* B_strided     = Y_strided     + rows * rank * op_increment;
	T* R_strided     = B_strided     + rank * cols * op_increment;
	T* tau_strided   = R_strided     + rank * rank * op_increment;

	T** Omega_ptrs   = (T**)((GPUBlasHandle::WS_Byte*)handle.workspace + ws_bytes[0]);
	T** M_ptrs       = Omega_ptrs + op_increment;
	T** Y_ptrs       = M_ptrs     + op_increment;
	T** B_ptrs       = Y_ptrs     + op_increment;
	T** R_ptrs       = B_ptrs     + op_increment;

	// Generate array of pointers for the intermediate data
	generateArrayOfPointers(Omega_strided, Omega_ptrs, cols * rank, op_increment, handle.stream);
	generateArrayOfPointers(Y_strided,     Y_ptrs,     rows * rank, op_increment, handle.stream);
	generateArrayOfPointers(B_strided,     B_ptrs,     cols * rank, op_increment, handle.stream);
	generateArrayOfPointers(R_strided,     R_ptrs,     rank * rank, op_increment, handle.stream);

	// Temporary removal of the used workspace
	push_workspace(handle, ws_bytes[1]);

	for(int op_start = 0; op_start < num_ops; op_start += op_increment)
	{
		int batch_size = std::min(op_increment, num_ops - op_start);
		// printf("RSVD Batch %d -> %d\n", op_start, op_start + batch_size);

		T_ptr M_batch = advanceOperationPtr(M, op_start, stride_m);
		T_ptr S_batch = advanceOperationPtr(S, op_start, stride_s);

		// generate the sampling matrices
		generateRandomMatrices(Omega_strided, cols, rank, op_start, batch_size, handle.stream);

		// generate the pointer arrays
		generateArrayOfPointers(M_batch, M_ptrs, stride_m, batch_size, handle.stream);

		// cublas handles
		T alpha = 1, beta = 0;

		// First form the sampled matrix Y = A * omega
		batch_gemm(
			0, 0, rows, rank, cols, alpha,
			M_ptrs, ldm,
			Omega_ptrs, cols,
			beta, Y_ptrs, rows, batch_size, handle
		);

		// Overwrite Y with Q of its QR decomposition
		kblas_geqrf_batched(rows, rank, Y_strided, rows, rows * rank, tau_strided, rank, batch_size, handle);
		kblas_orgqr_batched(rows, rank, Y_strided, rows, rows * rank, tau_strided, rank, batch_size, handle);

		// Form B = A' * Q_Y
		batch_gemm(
			1, 0, cols, rank, rows, alpha,
			M_ptrs, ldm,
			Y_ptrs, rows,
			beta, B_ptrs, cols, batch_size, handle
		);

		// Do the QR of B - we only need the Q of B if we want the right singular vectors
		kblas_geqrf_batched(
			cols, rank, B_strided, cols, cols * rank,
			tau_strided, rank, batch_size,
			handle
		);
		kblas_copy_upper_batched(
			cols, rank, B_strided, cols, cols * rank,
			R_strided, rank, rank * rank, batch_size,
			handle
		);

		// Transpose R so that we can get its right singular vectors when we
		// do the svd (since the svd gets the left singular vectors, which are
		// the right singular vectors of the transpose of the matrix). Store the
		// transpose in omega
		generateArrayOfPointers(Omega_strided, Omega_ptrs, rank * rank, batch_size, handle.stream);
		batch_transpose(rank, rank, R_ptrs, rank, Omega_ptrs, rank, batch_size, handle);

		// Now do the SVD of omega = R'
		if(rank <= SHARED_SVD_DIM_LIMIT)
			batch_svd_small<T, T_ptr>(selectPointerData<T, T_ptr>(Omega_strided, Omega_ptrs), rank, rank * rank, S_batch, stride_s, rank, rank, batch_size, handle);
		else
			batch_svd_osbj_qr<T, T_ptr>(selectPointerData<T, T_ptr>(Omega_strided, Omega_ptrs), rank, rank * rank, S_batch, stride_s, rank, rank, batch_size, handle);

		// Finally, we overwrite the matrix with left singular values of the
		// truncated SVD as the product U_A = Q_Y * V_B
		batch_gemm (
			0, 0, rows, rank, rank, alpha,
			Y_ptrs, rows,
			Omega_ptrs, rank,
			beta, M_ptrs, ldm, batch_size, handle
		);
	}

	// Restore the workspace
	pop_workspace(handle, ws_bytes[1]);

	return 0;
}

////////////////////////////////////////////////////////////////////
// Strided interface
////////////////////////////////////////////////////////////////////
int kblas_gesvj_batched(int m, int n, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops, GPUBlasHandle& handle)
{
	if(m <= SHARED_SVD_DIM_LIMIT && n <= SHARED_SVD_DIM_LIMIT)
		return batch_svd_small<float, float*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, num_ops, handle);
	else if(n <= SHARED_SVD_DIM_LIMIT)
		return batch_tall_svd<float, float*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, num_ops, handle);
	else
		return -1;
}

int kblas_gesvj_gram_batched(int m, int n, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops, GPUBlasHandle& handle)
{
	return batch_svd_osbj_gram<float, float*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, num_ops, handle);
}

int kblas_gesvj_qr_batched(int m, int n, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops, GPUBlasHandle& handle)
{
	return batch_svd_osbj_qr<float, float*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, num_ops, handle);
}

int kblas_rsvd_batched(int m, int n, int rank, float* A_strided, int lda, int stride_a, float* S_strided, int stride_s, int num_ops, GPUBlasHandle& handle)
{
	return batch_svd_randomized<float, float*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, rank, num_ops, handle);
}

int kblas_gesvj_batched(int m, int n, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops, GPUBlasHandle& handle)
{
	if(m <= SHARED_SVD_DIM_LIMIT && n <= SHARED_SVD_DIM_LIMIT)
		return batch_svd_small<double, double*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, num_ops, handle);
	else if(n <= SHARED_SVD_DIM_LIMIT)
		return batch_tall_svd<double, double*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, num_ops, handle);
	else
		return -1;
}

int kblas_gesvj_gram_batched(int m, int n, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops, GPUBlasHandle& handle)
{
	return batch_svd_osbj_gram<double, double*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, num_ops, handle);
}

int kblas_gesvj_qr_batched(int m, int n, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops, GPUBlasHandle& handle)
{
	return batch_svd_osbj_qr<double, double*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, num_ops, handle);
}

int kblas_rsvd_batched(int m, int n, int rank, double* A_strided, int lda, int stride_a, double* S_strided, int stride_s, int num_ops, GPUBlasHandle& handle)
{
	return batch_svd_randomized<double, double*>(A_strided, lda, stride_a, S_strided, stride_s, m, n, rank, num_ops, handle);
}

////////////////////////////////////////////////////////////////////
// Array of pointers interface
////////////////////////////////////////////////////////////////////
int kblas_gesvj_batched(int m, int n, float** A_array, int lda, float** S_array, int num_ops, GPUBlasHandle& handle)
{
	if(m <= SHARED_SVD_DIM_LIMIT && n <= SHARED_SVD_DIM_LIMIT)
		return batch_svd_small<float, float**>(A_array, lda, 0, S_array, 0, m, n, num_ops, handle);
	else if(n <= SHARED_SVD_DIM_LIMIT)
		return batch_tall_svd<float, float**>(A_array, lda, 0, S_array, 0, m, n, num_ops, handle);
	else
		return -1;
}

int kblas_gesvj_gram_batched(int m, int n, float** A_array, int lda, float** S_array, int num_ops, GPUBlasHandle& handle)
{
	return batch_svd_osbj_gram<float, float**>(A_array, lda, 0, S_array, 0, m, n, num_ops, handle);
}

int kblas_gesvj_qr_batched(int m, int n, float** A_array, int lda, float** S_array, int num_ops, GPUBlasHandle& handle)
{
	return batch_svd_osbj_qr<float, float**>(A_array, lda, 0, S_array, 0, m, n, num_ops, handle);
}

int kblas_rsvd_batched(int m, int n, int rank, float** A_array, int lda, float** S_array, int num_ops, GPUBlasHandle& handle)
{
	return batch_svd_randomized<float, float**>(A_array, lda, 0, S_array, 0, m, n, rank, num_ops, handle);
}

int kblas_gesvj_batched(int m, int n, double** A_array, int lda, double** S_array, int num_ops, GPUBlasHandle& handle)
{
	if(m <= SHARED_SVD_DIM_LIMIT && n <= SHARED_SVD_DIM_LIMIT)
		return batch_svd_small<double, double**>(A_array, lda, 0, S_array, 0, m, n, num_ops, handle);
	else if(n <= SHARED_SVD_DIM_LIMIT)
		return batch_tall_svd<double, double**>(A_array, lda, 0, S_array, 0, m, n, num_ops, handle);
	else
		return -1;
}

int kblas_gesvj_gram_batched(int m, int n, double** A_array, int lda, double** S_array, int num_ops, GPUBlasHandle& handle)
{
	return batch_svd_osbj_gram<double, double**>(A_array, lda, 0, S_array, 0, m, n, num_ops, handle);
}

int kblas_gesvj_qr_batched(int m, int n, double** A_array, int lda, double** S_array, int num_ops, GPUBlasHandle& handle)
{
	return batch_svd_osbj_qr<double, double**>(A_array, lda, 0, S_array, 0, m, n, num_ops, handle);
}

int kblas_rsvd_batched(int m, int n, int rank, double** A_array, int lda, double** S_array, int num_ops, GPUBlasHandle& handle)
{
	return batch_svd_randomized<double, double**>(A_array, lda, 0, S_array, 0, m, n, rank, num_ops, handle);
}
