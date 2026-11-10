/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_svd/batch_qr.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Wajih Halim Boukaram
 * @date 2020-12-10
 **/

#include <stdio.h>
#include <cublas_v2.h>
#include <algorithm>

#include "kblas.h"
#include "kblas_struct.h"
#include "kblas_gpu_util.ch"

////////////////////////////////////////////////////////////
// Utility routines 
////////////////////////////////////////////////////////////

template<class T>
inline __device__ void d_swap(T& a, T& b) { T t = a; a = b; b = t; }

template<class T, int BLOCK_SIZE>
inline __device__ void larfg(
	T col_entry, T alpha, T& v, T& tau, T& beta,
	int warp_tid, int warp_id, T* temp_storage
)
{
	beta = sqrt( blockAllReduceSum<T, BLOCK_SIZE>(
		col_entry * col_entry, warp_tid, warp_id, temp_storage
	) );
	kblas_sync_threads<BLOCK_SIZE>();
	if(alpha >= 0) beta *= -1;
	tau = (beta != 0 ? (beta - alpha) / beta : 0);
	v   = (alpha - beta != 0 ? col_entry / (alpha - beta) : 0);
}

template<class Key, class Value>
struct MyKeyValue {
	Key key;
	Value value;
	__device__ __forceinline__
    MyKeyValue() {}
	__device__ __forceinline__
	MyKeyValue(Key key, Value value) { this->key = key; this->value = value; }
};
template<class Key, class Value>
__device__ __forceinline__
bool operator >(const MyKeyValue<Key, Value> &a, const MyKeyValue<Key, Value> &b) { return a.value > b.value; }

template<class Key, class Value>
__device__ __forceinline__
MyKeyValue<Key, Value> __shfl_xor(const MyKeyValue<Key, Value> &a, int mask) { 
	return MyKeyValue<Key, Value>(
		__shfl_xor_sync(0xFFFFFFFF, a.key, mask), __shfl_xor_sync(0xFFFFFFFF, a.value, mask)
	); 
}

template<class Pair>
__device__ __forceinline__
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
__device__ __forceinline__
void argMaxColNorm(int tid, int k, volatile T* scol_norms, int cols, int& index)
{
	// Pivot data: the diagonal entry and its index
	typedef MyKeyValue<int, T> PivotData;
	
	// Accumulate into first warp
	if(tid < WARP_SIZE)
	{
		int remaining_cols = cols - k;
		int col_index = k + tid;

		PivotData my_pivot_data(
			col_index, 
			(col_index >= cols ? -1 : scol_norms[col_index])
		);
		
		while(remaining_cols > WARP_SIZE)
		{
			col_index += WARP_SIZE;
			remaining_cols -= WARP_SIZE;
			
			if(col_index < cols)
			{
				T val = scol_norms[col_index];
				if(my_pivot_data.value < val)
				{
					my_pivot_data.key = col_index;
					my_pivot_data.value = val;
				}
			}
		}
		warp_max<PivotData>(my_pivot_data);
		if(tid == 0)
			index = my_pivot_data.key;
	}
}

////////////////////////////////////////////////////////////
//Main Kernel 
////////////////////////////////////////////////////////////

template<class T, class T_ptr, class I_ptr, class Dim_Type, int BLOCK_SIZE, int PANEL_WIDTH>
__global__ 
void batch_geqp2_kernel(
	Dim_Type m_batch, Dim_Type n_batch, T_ptr __restrict__ A_batch, Dim_Type lda_batch, 
    int stride, T_ptr tau_batch, int stride_tau, I_ptr piv_batch, int stride_piv, 
	int* ranks, T tol, int num_ops
) 
{   
	const int BLOCK_WARPS = BLOCK_SIZE / WARP_SIZE;
	
	extern __shared__ char sdata[];	
	__shared__ T temp_storage[BLOCK_WARPS], alpha;
	__shared__ int pivot_index;
	
    int tid = threadIdx.x;
	int warp_tid = tid % WARP_SIZE, warp_id = tid / WARP_SIZE;
    int op_id = blockIdx.x;
    
    if(op_id >= num_ops) return;
    
    int m = getOperationDim(m_batch, op_id);
    int n = getOperationDim(n_batch, op_id);
    int lda = getOperationDim(lda_batch, op_id);

    if(m <= 0 || n <= 0)
        return;

	// Column norms in shared memory 
    T* scol_norms = (T*)sdata;
	T* pivot_row  = scol_norms + n;
	T* mv_smem_ws = pivot_row + n;
	
	// Global memory matrix data
    T* g_A   = getOperationPtr<T>(A_batch, op_id, stride);
    T* g_tau = getOperationPtr<T>(tau_batch, op_id, stride_tau);
    int* g_piv = (piv_batch ? getOperationPtr<int>(piv_batch, op_id, stride_piv) : NULL);
	
    if(!g_A || ! g_tau)
        return;

	int mn = (m < n ? m : n);
 
	// Compute the column norms 
	for(int j = 0; j < n; j++)
	{
		T col_entry = (tid < m ? g_A[tid + j * lda] : 0);
		T val = blockAllReduceSum<T, BLOCK_SIZE>(col_entry * col_entry, warp_tid, warp_id, temp_storage);
        kblas_sync_threads<BLOCK_SIZE>();
		if(tid == 0) scol_norms[j] = sqrt(val);
	}

	// Initialize the pivot array 
	if(g_piv)
    {
        int piv_index = tid;
        while(piv_index < n)
		{
            g_piv[piv_index] = piv_index + 1;
            piv_index += blockDim.x;
        }
    }
	
	kblas_sync_threads<BLOCK_SIZE>();
	T matrix_panel[PANEL_WIDTH];
	
	// Main loop 
	int k = 0;
	for(; k < mn; k++)
	{
		////////////////////////////////////////////////
		// Find pivot column having largest norm 
		////////////////////////////////////////////////
		argMaxColNorm<T, BLOCK_SIZE>(tid, k, scol_norms, n, pivot_index);
		kblas_sync_threads<BLOCK_SIZE>();

		// Check for convergence 
		if(scol_norms[pivot_index] * sqrt((float)(n - k)) < tol)
			break;
        
		// swap columns and column norm 
		if(pivot_index != k)
		{
            // Make sure to sync before swapping the column norms so there's no conflict 
            // with the convergence check
            kblas_sync_threads<BLOCK_SIZE>();
			if(tid < m ) d_swap(g_A[tid + pivot_index * lda], g_A[tid + k * lda]);
			if(g_piv && tid == 0) { d_swap(g_piv[k], g_piv[pivot_index]); }
            if(tid == 0) scol_norms[pivot_index] = scol_norms[k];
		}
		kblas_sync_threads<BLOCK_SIZE>();
		
		////////////////////////////////////////////////
		// Now eliminate the column
		////////////////////////////////////////////////
		T x = (tid < k || tid >= m ? 0 : g_A[tid + k * lda]);
		if(tid == k) alpha = x;
		kblas_sync_threads<BLOCK_SIZE>();
		
		// Generate and save the reflector 
		T v, tau, beta;
		larfg<T, BLOCK_SIZE>(x, alpha, v, tau, beta, warp_tid, warp_id, temp_storage);
		if(tid == k) v = 1;
		if(tid >= k && tid < m ) g_A[tid + k * lda] = (tid == k ? beta : v);
		
		// Apply the reflector to the trailing submatrix
		if(tau != 0 && k < n)
		{
			// Apply the reflector in blocks to improve transpose matvec performance
			int panel_start = k + 1;
			int block_panels = iDivUp(n - panel_start, PANEL_WIDTH);
			
			for(int panel = 0; panel < block_panels; panel++)
			{
				// first store a panel of the global matrix in registers 
				#pragma unroll
				for(int i = 0; i < PANEL_WIDTH; i++)
					matrix_panel[i] = (panel_start + i < n && tid < m ? g_A[tid + (panel_start + i) * lda] : 0);
				
				// Now update the panel using the reflector v: 
				// First compute w in shared memory as: w = A(k1:k2,k1:k2)' * v
				multreduction_gemv<T, PANEL_WIDTH, 0, BLOCK_SIZE, PANEL_WIDTH>(matrix_panel, v, tid, warp_tid, warp_id, mv_smem_ws);
                
				// Now shared memory has the vector w in the first PANEL_WIDTH entries
				// so we can apply the rank 1 update: A(k1:k2, k1:k2) = A(k1:k2, k1:k2) - tau * v * w;
				T tv = tau * v;
				#pragma unroll
				for(int i = 0; i < PANEL_WIDTH; i++)
					matrix_panel[i] -= tv * mv_smem_ws[i];
				
				// Flush matrix data back to global memory 
				if(tid < m)
				{
					#pragma unroll
					for(int i = 0; i < PANEL_WIDTH; i++)
					{
						if(panel_start + i < n)
						{
							g_A[tid + (panel_start + i) * lda] = matrix_panel[i];
							
							// Save the pivot row data so we don't have to reload it from global memory 
							if(tid == k) pivot_row[panel_start + i] = matrix_panel[i];
						}
					}
				}
				kblas_sync_threads<BLOCK_SIZE>();
				
				panel_start += PANEL_WIDTH;
			}
			
			// Update column norms 
            int norm_index = tid + k + 1;
			while(norm_index < n)
            {
				T t = scol_norms[norm_index] * scol_norms[norm_index] - pivot_row[norm_index] * pivot_row[norm_index];
				scol_norms[norm_index] = (t <= 0 ? 0 : sqrt(t));
                norm_index += blockDim.x;
            }
            kblas_sync_threads<BLOCK_SIZE>();
		}
		
		// Save tau for this column 
		if(tid == 0) g_tau[k] = tau;
	}

	ranks[op_id] = k;
}

////////////////////////////////////////////////////////////
// Template Driver routines
////////////////////////////////////////////////////////////

template<class T, class T_ptr, class I_ptr, class Dim_Type>
int batch_geqp2_template(
	kblasHandle_t handle, Dim_Type m, Dim_Type n, int max_m, int max_n, T_ptr A_batch, Dim_Type lda, 
    int stride, T_ptr tau_batch, int stride_tau, I_ptr piv_batch, int stride_piv, 
	int* ranks, T tol, int num_ops
) 
{
	int warps = iDivUp(max_m, WARP_SIZE);
	int threads = warps * WARP_SIZE;
	const int PANEL_WIDTH = 8;
	
	dim3 dimBlock(threads, 1);
    dim3 dimGrid(num_ops, 1);
	
	int smem_reduction = warps * WARP_SIZE * PANEL_WIDTH + warps * PANEL_WIDTH;
	int smem_col_norms = max_n, smem_pivot_row = max_n;
    size_t smem = sizeof(T) * (smem_reduction + smem_col_norms + smem_pivot_row);
    
	cudaStream_t stream = kblasGetStream(handle);
	
	switch(threads)
	{
		case  32: batch_geqp2_kernel<T, T_ptr, I_ptr, Dim_Type,  32, PANEL_WIDTH><<<dimGrid, dimBlock, smem, stream>>>(m, n, A_batch, lda, stride, tau_batch, stride_tau, piv_batch, stride_piv, ranks, tol, num_ops); break;
		case  64: batch_geqp2_kernel<T, T_ptr, I_ptr, Dim_Type,  64, PANEL_WIDTH><<<dimGrid, dimBlock, smem, stream>>>(m, n, A_batch, lda, stride, tau_batch, stride_tau, piv_batch, stride_piv, ranks, tol, num_ops); break;
		case  96: batch_geqp2_kernel<T, T_ptr, I_ptr, Dim_Type,  96, PANEL_WIDTH><<<dimGrid, dimBlock, smem, stream>>>(m, n, A_batch, lda, stride, tau_batch, stride_tau, piv_batch, stride_piv, ranks, tol, num_ops); break;
		case 128: batch_geqp2_kernel<T, T_ptr, I_ptr, Dim_Type, 128, PANEL_WIDTH><<<dimGrid, dimBlock, smem, stream>>>(m, n, A_batch, lda, stride, tau_batch, stride_tau, piv_batch, stride_piv, ranks, tol, num_ops); break;
		case 160: batch_geqp2_kernel<T, T_ptr, I_ptr, Dim_Type, 160, PANEL_WIDTH><<<dimGrid, dimBlock, smem, stream>>>(m, n, A_batch, lda, stride, tau_batch, stride_tau, piv_batch, stride_piv, ranks, tol, num_ops); break;
		case 192: batch_geqp2_kernel<T, T_ptr, I_ptr, Dim_Type, 192, PANEL_WIDTH><<<dimGrid, dimBlock, smem, stream>>>(m, n, A_batch, lda, stride, tau_batch, stride_tau, piv_batch, stride_piv, ranks, tol, num_ops); break;
		case 224: batch_geqp2_kernel<T, T_ptr, I_ptr, Dim_Type, 224, PANEL_WIDTH><<<dimGrid, dimBlock, smem, stream>>>(m, n, A_batch, lda, stride, tau_batch, stride_tau, piv_batch, stride_piv, ranks, tol, num_ops); break;
		case 256: batch_geqp2_kernel<T, T_ptr, I_ptr, Dim_Type, 256, PANEL_WIDTH><<<dimGrid, dimBlock, smem, stream>>>(m, n, A_batch, lda, stride, tau_batch, stride_tau, piv_batch, stride_piv, ranks, tol, num_ops); break;
		case 288: batch_geqp2_kernel<T, T_ptr, I_ptr, Dim_Type, 288, PANEL_WIDTH><<<dimGrid, dimBlock, smem, stream>>>(m, n, A_batch, lda, stride, tau_batch, stride_tau, piv_batch, stride_piv, ranks, tol, num_ops); break;
		case 320: batch_geqp2_kernel<T, T_ptr, I_ptr, Dim_Type, 320, PANEL_WIDTH><<<dimGrid, dimBlock, smem, stream>>>(m, n, A_batch, lda, stride, tau_batch, stride_tau, piv_batch, stride_piv, ranks, tol, num_ops); break;
		case 352: batch_geqp2_kernel<T, T_ptr, I_ptr, Dim_Type, 352, PANEL_WIDTH><<<dimGrid, dimBlock, smem, stream>>>(m, n, A_batch, lda, stride, tau_batch, stride_tau, piv_batch, stride_piv, ranks, tol, num_ops); break;
		case 384: batch_geqp2_kernel<T, T_ptr, I_ptr, Dim_Type, 384, PANEL_WIDTH><<<dimGrid, dimBlock, smem, stream>>>(m, n, A_batch, lda, stride, tau_batch, stride_tau, piv_batch, stride_piv, ranks, tol, num_ops); break;
		case 416: batch_geqp2_kernel<T, T_ptr, I_ptr, Dim_Type, 416, PANEL_WIDTH><<<dimGrid, dimBlock, smem, stream>>>(m, n, A_batch, lda, stride, tau_batch, stride_tau, piv_batch, stride_piv, ranks, tol, num_ops); break;
		case 448: batch_geqp2_kernel<T, T_ptr, I_ptr, Dim_Type, 448, PANEL_WIDTH><<<dimGrid, dimBlock, smem, stream>>>(m, n, A_batch, lda, stride, tau_batch, stride_tau, piv_batch, stride_piv, ranks, tol, num_ops); break;
		case 480: batch_geqp2_kernel<T, T_ptr, I_ptr, Dim_Type, 480, PANEL_WIDTH><<<dimGrid, dimBlock, smem, stream>>>(m, n, A_batch, lda, stride, tau_batch, stride_tau, piv_batch, stride_piv, ranks, tol, num_ops); break;
		case 512: batch_geqp2_kernel<T, T_ptr, I_ptr, Dim_Type, 512, PANEL_WIDTH><<<dimGrid, dimBlock, smem, stream>>>(m, n, A_batch, lda, stride, tau_batch, stride_tau, piv_batch, stride_piv, ranks, tol, num_ops); break;		
		default: printf("Unsupported matrix dimension (%d x %d)\n", max_m, max_n); { return KBLAS_UnknownError; }
	}
    
	check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	return KBLAS_Success;
}

///////////////////////////////////////////////////////////////
// Strided routines
///////////////////////////////////////////////////////////////
extern "C" int kblasDgeqp2_batch_strided(
	kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, 
	double* tau_strided, int stride_tau, int* piv_strided, int stride_piv, 
	int* ranks, double tol, int num_ops
) 
{
	if(lda < m)
		return KBLAS_Error_WrongInput;
	
	return batch_geqp2_template<double, double*, int*, int>(
		handle, m, n, m, n, A_strided, lda, stride_a, tau_strided, stride_tau, 
		piv_strided, stride_piv, ranks, tol, num_ops
	);
}

extern "C" int kblasSgeqp2_batch_strided(
	kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, 
	float* tau_strided, int stride_tau, int* piv_strided, int stride_piv, 
	int* ranks, float tol, int num_ops
) 
{
	if(lda < m)
		return KBLAS_Error_WrongInput;
	
	return batch_geqp2_template<float, float*, int*, int>(
		handle, m, n, m, n, A_strided, lda, stride_a, tau_strided, stride_tau, 
		piv_strided, stride_piv, ranks, tol, num_ops
	);
}

///////////////////////////////////////////////////////////////
// Array of pointers routines
///////////////////////////////////////////////////////////////
extern "C" int kblasDgeqp2_batch(
	kblasHandle_t handle, int m, int n, double** A_array, int lda, 
	double** tau_ptrs, int** piv_ptrs, int* ranks, double tol, int num_ops
) 
{
	if(lda < m)
		return KBLAS_Error_WrongInput;
	
	return batch_geqp2_template<double, double**, int**, int>(
		handle, m, n, m, n, A_array, lda, 0, tau_ptrs, 0, 
		piv_ptrs, 0, ranks, tol, num_ops
	);
}

extern "C" int kblasSgeqp2_batch(
	kblasHandle_t handle, int m, int n, float** A_array, int lda, 
	float** tau_ptrs, int** piv_ptrs, int* ranks, float tol, int num_ops
) 
{
	if(lda < m)
		return KBLAS_Error_WrongInput;
	
	return batch_geqp2_template<float, float**, int**, int>(
		handle, m, n, m, n, A_array, lda, 0, tau_ptrs, 0, 
		piv_ptrs, 0, ranks, tol, num_ops
	);
}

extern "C" int kblasDgeqp2_vbatch(
	kblasHandle_t handle, int* m, int* n, int max_m, int max_n, double** A_array, int *lda, 
	double** tau_ptrs, int** piv_ptrs, int* ranks, double tol, int num_ops
) 
{
	return batch_geqp2_template<double, double**, int**, int*>(
		handle, m, n, max_m, max_n, A_array, lda, 0, tau_ptrs, 0, 
		piv_ptrs, 0, ranks, tol, num_ops
	);
}

extern "C" int kblasSgeqp2_vbatch(
	kblasHandle_t handle, int* m, int* n, int max_m, int max_n, float** A_array, int *lda, 
	float** tau_ptrs, int** piv_ptrs, int* ranks, float tol, int num_ops
) 
{
	return batch_geqp2_template<float, float**, int**, int*>(
		handle, m, n, max_m, max_n, A_array, lda, 0, tau_ptrs, 0, 
		piv_ptrs, 0, ranks, tol, num_ops
	);
}
