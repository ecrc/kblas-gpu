#include "hip/hip_runtime.h"
/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_svd/svd_kernels.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Wajih Halim Boukaram
 * @date 2018-11-14
 **/

#ifndef __SVD_KERNELS_CUH__
#define __SVD_KERNELS_CUH__

#include "kblas_gpu_util.ch"
#include <hipcub/hipcub.hpp>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                  					SVD ALGORITHMS                                    							//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions for the parallel one sided jacobi SVD
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
__inline__ __device__
T sign(T x)
{
	if(x > 0) return 1;
	if(x < 0) return -1;
	return 0;
}

__inline__ __device__
void init_pairs(int* pairs_a, int* pairs_b, int total_pairs, int thread_x)
{
    if(thread_x >= total_pairs) return;

    pairs_a[thread_x] = 2 * thread_x;
    pairs_b[thread_x] = 2 * thread_x + 1;
}

__inline__ __device__
void cycle_pairs(int* pairs_a, int* pairs_b, int total_pairs, int thread_x)
{
    if(thread_x >= total_pairs) return;

    int old_a = pairs_a[thread_x], old_b = pairs_b[thread_x];

    if(thread_x < total_pairs - 1)  pairs_a[thread_x + 1] = old_a;
    if(thread_x > 0)                pairs_b[thread_x - 1] = old_b;
    if(thread_x == 0)               pairs_a[0] = old_b;
    if(thread_x == total_pairs - 1)
    {
        pairs_b[thread_x - 1] = old_a;
        pairs_b[thread_x] = old_b;
    }
}

template<int rows_per_thread, class T>
__inline__ __device__
void parNormalizeColumns(T* matrix, T* svals, int m, int n, int thread_x, int warp_id, int normalize_cols)
{
    #pragma unroll
    for(int rj = 0; rj < 2; rj++)
    {
        int j = warp_id + rj * blockDim.y;

        T sj = 0;
        #pragma unroll
        for(int ri = 0; ri < rows_per_thread; ri++)
        {
            int i = thread_x + ri * WARP_SIZE;
            sj += matrix[i + j * m] * matrix[i + j * m];
        }
        sj = sqrt(warpAllReduceSum(sj));

        if(sj != 0 && normalize_cols == 1)
		{
			#pragma unroll
			for(int ri = 0; ri < rows_per_thread; ri++)
			{
				int i = thread_x + ri * WARP_SIZE;
				matrix[i + j * m] /= sj;
			}
        }

        if(thread_x == 0)
            svals[j] = sj;
    }
    __syncthreads();
}

template<int rows_per_thread, class T>
__forceinline__ __device__
void multiRowGramMatrix(T* __restrict__ matrix, int m, int n, int i, int j, int thread_x, T& a, T& b, T& p)
{
    // Acuumulate the values in the columns within a warp before doing a reduction
    a = b = p = 0;
    T* mi_ptr = matrix + thread_x + i * m;
    T* mj_ptr = matrix + thread_x + j * m;
    #pragma unroll
    for(int rb = 0; rb < rows_per_thread; rb++)
    {
        a += mi_ptr[0] * mi_ptr[0];
        b += mj_ptr[0] * mj_ptr[0];
        p += mi_ptr[0] * mj_ptr[0];

        if(rb != rows_per_thread - 1)
        {
            mi_ptr += WARP_SIZE;
            mj_ptr += WARP_SIZE;
        }
    }
    a = warpAllReduceSum(a);
    b = warpAllReduceSum(b);
    p = warpAllReduceSum(p);
}

template<int rows_per_thread, class T>
__inline__ __device__
void multiRowRotateColumns(T* __restrict__ matrix, int m, int n, int i, int j, T c, T s, int thread_x)
{
    T* mi_ptr = matrix + thread_x + i * m;
    T* mj_ptr = matrix + thread_x + j * m;
    #pragma unroll
    for(int rb = 0; rb < rows_per_thread; rb++)
    {
        T U_i = mi_ptr[0], U_j = mj_ptr[0];

        mi_ptr[0] = c * U_i - s * U_j;
        mj_ptr[0] = s * U_i + c * U_j;

        if(rb != rows_per_thread - 1)
        {
            mi_ptr += WARP_SIZE;
            mj_ptr += WARP_SIZE;
        }
    }
}

template<int rows_per_thread, class T>
__inline__ __device__
void parSvdJacobi(T* __restrict__ matrix, T* __restrict__ svals, int m, int n, int thread_x, int warp_id, int* __restrict__ pairs_a, int* __restrict__ pairs_b, int total_pairs, int op_id, double* gflops)
{
    int sweep = 0;
    int sweep_max = 5 * n;
    if(sweep_max < 12) sweep_max = 12;

	T rel_tol = KBlasEpsilon<T>::eps;
	// T rel_tol2 = rel_tol * rel_tol;
	T tolerance = rel_tol * n;

	T offdiagonal = tolerance + 1;
#ifdef HLIB_PROFILING_ENABLED
    int rotations = 0, gram_products = 0;
#endif

	// Let the first warp initialize the pairs
	if(warp_id == 0) init_pairs(pairs_a, pairs_b, total_pairs, thread_x);
	__syncthreads();

	while(offdiagonal > tolerance && sweep < sweep_max)
    {
		offdiagonal = 0;

        for(int k = 0; k < n - 1; k++)
        {
            int i = pairs_a[warp_id];
            int j = pairs_b[warp_id];

            // compute [a p; p b]=(i,j) submatrix of A'*A
            T a, b, p;
            multiRowGramMatrix<rows_per_thread>(matrix, m, n, i, j, thread_x, a, b, p);

			#ifdef HLIB_PROFILING_ENABLED
            gram_products++;
			#endif
			
			if(a != 0 && b != 0 && p != 0)
			{
				T r = abs(p) / (sqrt(a) * sqrt(b));
				if(offdiagonal < r) offdiagonal = r;
				
				T zeta = (b - a) / (2 * p);
				T t = sign(zeta) / (abs(zeta) + sqrt(1 + zeta * zeta));
				T c = (T)1 / sqrt(1 + t * t);
				T s = c * t;
				
				// Rotate the two columns
				multiRowRotateColumns<rows_per_thread>(matrix, m, n, i, j, c, s, thread_x);

				#ifdef HLIB_PROFILING_ENABLED
				rotations++;
				#endif
			}
            // Synchronize the threads after processing their pairs of columns
            __syncthreads();

            // Let the first warp handle the cycling of the pairs
            if(warp_id == 0) cycle_pairs(pairs_a, pairs_b, total_pairs, thread_x);
            __syncthreads();
        }

        // Accumulate the off diagonal norms from all warps - use the svals array as temporary shared memory
        if(thread_x == 0) svals[warp_id] = offdiagonal;
        __syncthreads();
        offdiagonal = warpAllReduceMax(thread_x < total_pairs ? svals[thread_x] : 0);
        __syncthreads();
        sweep++;
		//if(thread_x == 0 && warp_id == 0)
		//	printf("op %d offdiagonal = %e \n", op_id, offdiagonal);
    }
	#ifdef HLIB_PROFILING_ENABLED
	if(thread_x == 0) svals[warp_id] = (double)(rotations + gram_products);
	__syncthreads();
	if(warp_id == 0)
	{
		T gops = warpAllReduceSum(thread_x < total_pairs ? svals[thread_x] : 0);
		if(thread_x == 0) gflops[op_id] = gops * 1e-9;
	}
	#endif
	
	//if(thread_x == 0 && warp_id == 0 && sweep > 15)
	//	printf("op %d Sweeps = %d \n", op_id, sweep);
}

template<int n, class T>
__inline__ __device__
void warpSvdJacobi_reg(T matrix_row[n], T* svals, int m, int lane_id, int op_id, double* gflops)
{
    int sweep = 0;
    int sweep_max = 5 * n;
    if(sweep_max < 12) sweep_max = 12;

	T rel_tol = m * KBlasEpsilon<T>::eps;
	T tolerance = rel_tol * n * n;

    T offdiagonal_norm = 1;

	#ifdef HLIB_PROFILING_ENABLED
	int rotations = 0, gram_products = 0;
	#endif

    while(offdiagonal_norm > tolerance && sweep < sweep_max)
    {
		offdiagonal_norm = 0;

        #pragma unroll
        for(int j = 0; j < n - 1; j++)
        {
            #pragma unroll
            for(int k = 0; k < n; k++)
            {
                if(k < j + 1) continue;

                // compute [a p; p b]=(i,j) submatrix of A'*A
                T a = warpAllReduceSum(matrix_row[j] * matrix_row[j]);
                T b = warpAllReduceSum(matrix_row[k] * matrix_row[k]);
                T p = warpAllReduceSum(matrix_row[k] * matrix_row[j]);

				#ifdef HLIB_PROFILING_ENABLED
				gram_products++;
				#endif

				// Check if the two columns are already orthogonalized
				T alpha = 2 * p;
				T a_norm = sqrt(a), b_norm = sqrt(b);

				if(abs(alpha) > rel_tol * a_norm * b_norm)
				{
					// Calculate the rotation matrix
					T beta = a - b;
					T gamma = hypot(alpha, beta);
					if(beta < 0) gamma *= -1;

					T c = sqrt((gamma + beta) / (2 * gamma));
					T s = alpha / (2 * gamma * c);

					// Apply the rotations to the matrix and its V factor
					T m_ij = matrix_row[j];

					matrix_row[j] = c * m_ij + s * matrix_row[k];
					matrix_row[k] = c * matrix_row[k] - s * m_ij;

					#ifdef HLIB_PROFILING_ENABLED
					rotations++;
					#endif

					offdiagonal_norm += 2 * p * p / (a * b);
				}
            }
        }
        sweep++;
    }

    #pragma unroll
    for(int j = 0; j < n; j++)
    {
        T sj = sqrt(warpAllReduceSum(matrix_row[j] * matrix_row[j]));
        if(sj != 0.0   ) matrix_row[j] /= sj;
        if(lane_id == 0) svals[j] = sj;
    }

	#ifdef HLIB_PROFILING_ENABLED
    double op_count = (double)(rotations + gram_products);
	if(lane_id == 0) gflops[op_id] = op_count* 1e-9;
	#endif
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernels
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<int cols, class T, class T_ptr>
__global__
void warpSVDKernel(T_ptr __restrict__ M, int ldm, int stride_m, T_ptr __restrict__ S, int stride_s, int rows, int num_ops, double* gflops)
{
    extern __shared__ char sdata[];

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int op_id = thread_id / WARP_SIZE;

    if(op_id >= num_ops) return;

    int local_tid = threadIdx.x % WARP_SIZE;
    int local_leaf_id = threadIdx.x / WARP_SIZE;

    T* shared_s = (T*)sdata + local_leaf_id * cols;
    T* matrix = getOperationPtr<T>(M, op_id, stride_m);
    T* s_op = getOperationPtr<T>(S, op_id, stride_s);

    // Load the data in
    T matrix_row[cols];
    #pragma unroll
    for(int i = 0; i < cols; i++)
        matrix_row[i] = (local_tid < rows ? matrix[local_tid + i * ldm] : 0);

    warpSvdJacobi_reg<cols, T>(matrix_row, shared_s, rows, local_tid, op_id, gflops);

    // Flush the data to memory
    if(local_tid < rows)
    {
        #pragma unroll
        for(int i = 0; i < cols; i++)
            matrix[local_tid + i * ldm] = matrix_row[i];
    }

	if(local_tid < cols)
        s_op[local_tid] = shared_s[local_tid];
}

template<int rows_per_thread, class T, class T_ptr>
__global__
void blockSVDKernel(T_ptr __restrict__ M, int ldm, int stride_m, T_ptr __restrict__ S, int stride_s, int rows, int cols, int normalize_cols, int num_ops, double* gflops = NULL)
{
    extern __shared__ char sdata[];

    int op_id = blockIdx.x;
    if(op_id >= num_ops) return;

    int thread_x = threadIdx.x;
    int warp_id  = threadIdx.y;
    int total_pairs = blockDim.y;

    int srows = rows_per_thread * WARP_SIZE, scols = total_pairs * 2;
    T* shared_matrix = (T*)sdata;
    T* shared_s      = (T*)&shared_matrix[srows * scols];
    int* pairs_a     = (int* )&shared_s[scols];
    int* pairs_b     = (int* )&pairs_a[total_pairs];

    T* matrix = getOperationPtr<T>(M, op_id, stride_m);
    T* s_op = getOperationPtr<T>(S, op_id, stride_s);

    // Load the data in and pad the rows if necessary
    #pragma unroll
    for(int ri = 0; ri < rows_per_thread; ri++)
    {
        int i = thread_x + ri * WARP_SIZE;
        int j1 = warp_id, j2 = warp_id + total_pairs;
        shared_matrix[i + j1 * srows] = (i < rows ? matrix[i + j1 * ldm] : 0);
        shared_matrix[i + j2 * srows] = (i < rows && j2 < cols ? matrix[i + j2 * ldm] : 0);
    }
    __syncthreads();

    parSvdJacobi<rows_per_thread, T>
		(shared_matrix, shared_s, srows, scols, thread_x, warp_id, pairs_a, pairs_b, total_pairs, op_id, gflops);
    parNormalizeColumns<rows_per_thread, T>
        (shared_matrix, shared_s, srows, scols, thread_x, warp_id, normalize_cols);

    // Flush the data to memory
    #pragma unroll
    for(int ri = 0; ri < rows_per_thread; ri++)
    {
        int i = thread_x + ri * WARP_SIZE;
        if(i < rows)
        {
            int j1 = warp_id, j2 = warp_id + total_pairs;
            matrix[i + j1 * ldm] = shared_matrix[i + j1 * srows];
            if(j2 < cols) matrix[i + j2 * ldm] = shared_matrix[i + j2 * srows];
        }
    }

	thread_x = thread_x + warp_id * WARP_SIZE;
    if(thread_x < cols)
		s_op[thread_x] = shared_s[thread_x];
}

template<int blockSize, class T, class T_ptr>
__global__
void batchSortSingularValuesSmallKernel(T_ptr __restrict__ M, int ldm, int stride_m, T_ptr __restrict__ S, int stride_s, int rows, int cols, int num_ops)
{
	extern __shared__ char sdata[];

	int op_id = blockIdx.x;
	if(op_id >= num_ops) return;

	int tid = threadIdx.x;

	T* shared_M = (T*)sdata;
	int* shared_index = (int*)(shared_M + rows * cols);

	T* matrix = getOperationPtr<T>(M, op_id, stride_m);
	T* s_op = getOperationPtr<T>(S, op_id, stride_s);

	if(tid < rows)
	{
		for(int i = 0; i < cols; i++)
			shared_M[tid + i * rows] = matrix[tid + i * ldm];
	}

	T key[1] = {(tid < cols ? s_op[tid] : -1)};
	int  val[1] = {tid};

	typedef hipcub::BlockRadixSort<T, blockSize, 1, int> BlockRadixSort;
	__shared__ typename BlockRadixSort::TempStorage temp_storage;
	BlockRadixSort(temp_storage).SortDescending(key, val);

	if(tid < cols)
	{
		s_op[tid] = key[0];
		shared_index[tid] = val[0];
	}

	__syncthreads();

	if(tid < rows)
	{
		for(int i = 0; i < cols; i++)
		{
			int sorted_index = shared_index[i];
			matrix[tid + i * ldm] = shared_M[tid + sorted_index * rows];
		}
	}
}

template<class T, class T_ptr>
__global__
void batchNormalizeColumnsKernel(T_ptr __restrict__ M, int ldm, int stride_m, T_ptr __restrict__ S, int stride_s, int rows, int cols, int rows_per_thread, int cols_per_thread, int num_ops)
{
    int op_id = blockIdx.x;
    if(op_id >= num_ops) return;

    int tid = threadIdx.x;
    int warp_id = threadIdx.y;
    int total_warps = blockDim.y;

    int col_index = warp_id;

	T* matrix = getOperationPtr<T>(M, op_id, stride_m);
	T* s_op = getOperationPtr<T>(S, op_id, stride_s);

    for(int j = 0; j < cols_per_thread; j++)
    {
        if(col_index >= cols) return;

        T s = 0;
        for(int i = 0; i < rows_per_thread; i++)
        {
            int row_index = tid + WARP_SIZE * i;
            if(row_index < rows)
                s += matrix[row_index + col_index * ldm] * matrix[row_index + col_index * ldm];
        }
        s = sqrt(warpAllReduceSum(s));
        if(s != 0.0)
        {
            for(int i = 0; i < rows_per_thread; i++)
            {
                int row_index = tid + WARP_SIZE * i;
                if(row_index < rows)
                    matrix[row_index + col_index * ldm] /= s;
            }
            if(tid == 0)
                s_op[col_index] = s;
        }
        col_index += total_warps;
    }
}

template<class T, class T_ptr>
__global__
void batchMaxOffdiagonalKernel(T_ptr __restrict__ M, int ldm, int stride_m, int rows, int cols, int cols_per_thread, T* offdiagonal, int num_ops)
{
    extern __shared__ char sdata[];

    int local_op_id = threadIdx.y;
    int op_id = blockIdx.x * blockDim.x + local_op_id;
    if(op_id >= num_ops) return;

    int tid = threadIdx.x;
    T* matrix = getOperationPtr<T>(M, op_id, stride_m);
    T* shared_diag = (T*)sdata + cols * local_op_id;

    // Extract the diagonal first
    int diag_entry = tid;
    while(diag_entry < cols)
    {
        shared_diag[diag_entry] = matrix[diag_entry + diag_entry * ldm];
        diag_entry += WARP_SIZE;
    }

    T off_diag = 0;
    for(int j = 0; j < cols; j++)
    {
        T a = abs(shared_diag[j]);
        if(a == 0) continue;

        for(int i = 0; i < cols_per_thread; i++)
        {
            int row_index = WARP_SIZE * i + tid;
            if(row_index < j)
            {
                T p = abs(matrix[row_index + j * ldm]);
                T b = abs(shared_diag[row_index]);
                if(b != 0)
				{
					T r = p / (sqrt(a) * sqrt(b));
                    if(off_diag < r) off_diag = r;
				}
            }
        }
    }
    off_diag = warpAllReduceMax(off_diag);

    if(offdiagonal[op_id] < off_diag)
        offdiagonal[op_id] = off_diag;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
