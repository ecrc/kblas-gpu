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
 * @version 3.0.0
 * @author Wajih Halim Boukaram
 * @date 2018-11-14
 **/

#include <stdio.h>
#include <cublas_v2.h>
#include <algorithm>

#include "kblas.h"
#include "kblas_struct.h"
#include "kblas_gpu_util.ch"
#include "qr_kernels.cuh"

#ifdef HLIB_PROFILING_ENABLED
#include "perf_counter.h"
#endif

#define QR_LOAD(m) 					__ldg(&(m))
#define KBLAS_QR_CHECK_RET(func)	{ if( (func) != KBLAS_Success ) return KBLAS_UnknownError; }

#define HLIB_R_COLS_PER_THREAD	8
#define HLIB_R_MAX_THREAD_Y		8

// Apply the generated householder vectors at the current panel to the trailing submatrix
template<class T, class T_ptr, int BLOCK_SIZE, int APPLY_FORWARD, class Dim_Type>
__global__
void batch_apply_hh_panel(
	T_ptr m_batch, Dim_Type ldm_batch, int stride, T_ptr tau_batch, int stride_tau,
	Dim_Type rows_batch, Dim_Type cols_batch, int row_offset, int col_offset,
	int smem_entries, int panel_rows, int num_ops
)
{
	extern __shared__ char sdata[];
    const int HH_CB = QR_Config<T>::HH_CB;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int op_id = thread_id / BLOCK_SIZE;

    if(op_id >= num_ops) return;

    int local_op_id = threadIdx.x / BLOCK_SIZE;
    int local_tid = threadIdx.x % BLOCK_SIZE;

    T* s_tau = (T*)sdata + smem_entries * local_op_id;
    T* smem = s_tau + HH_CB;

	int rows = getOperationDim(rows_batch, op_id);
	int cols = getOperationDim(cols_batch, op_id);
	int ldm  = getOperationDim(ldm_batch,  op_id);
	
	if(row_offset >= rows || col_offset >= cols)
		return;
	
	int trailing_blocks = cols - col_offset - HH_CB;
	if(trailing_blocks <= 0) return;
	trailing_blocks = iDivUp(trailing_blocks, HH_CB);
	
    T* m_panel = getOperationPtr<T>(m_batch, op_id, stride) + row_offset + col_offset * ldm;
    T* tau_panel = getOperationPtr<T>(tau_batch, op_id, stride_tau) + col_offset;
    T* trailing_panel = m_panel;

    int nopadding = (local_tid < panel_rows && local_tid + row_offset < rows);

    // Load in tau from global memory
	if(local_tid < HH_CB && local_tid + col_offset < cols)
		s_tau[local_tid] = QR_LOAD(tau_panel[local_tid]);

    if(BLOCK_SIZE != WARP_SIZE) __syncthreads();

    // Store the matrix panel in registers
    T matrix_row[HH_CB], panel_row[HH_CB];

    // Load the panel that we're applying
    #pragma unroll
    for(int i = 0; i < HH_CB; i++)
        panel_row[i] = (nopadding ? QR_LOAD(m_panel[local_tid + i * ldm]) : 0);

	int column_index = col_offset;
    for(int block = 0; block < trailing_blocks; block++)
    {
        trailing_panel += HH_CB * ldm;
		column_index   += HH_CB;

        // Load the trailing panel in
        #pragma unroll
        for(int i = 0; i < HH_CB; i++)
		{
			matrix_row[i] = 0;
			if(nopadding && column_index + i < cols)
				matrix_row[i] = QR_LOAD(trailing_panel[local_tid + i * ldm]);
		}

        if(APPLY_FORWARD)
        {
            #pragma unroll
            for(int i = 0; i < HH_CB; i++)
            {
                T v = (local_tid > i && nopadding ? panel_row[i] : 0);
                if(local_tid == i) v = 1;
                qr_apply_househoulder_panel<T, HH_CB, BLOCK_SIZE>(matrix_row, v, s_tau[i], local_tid, smem);
            }
        }
        else
        {
            #pragma unroll
            for(int i = HH_CB - 1; i >= 0; i--)
            {
                T v = (local_tid > i && nopadding ? panel_row[i] : 0);
                if(local_tid == i) v = 1;
                qr_apply_househoulder_panel<T, HH_CB, BLOCK_SIZE>(matrix_row, v, s_tau[i], local_tid, smem);
            }
        }

        // Flush the current panel so we can load in the next one
        if(nopadding)
        {
            #pragma unroll
            for(int i = 0; i < HH_CB; i++)
				if(column_index + i < cols)
					trailing_panel[local_tid + i * ldm] = matrix_row[i];
        }
    }
}

template<class T, class T_ptr, int BLOCK_SIZE, class Dim_Type>
__global__
void batch_qr_panel(
	T_ptr __restrict__ m_batch, Dim_Type ldm_batch, int stride, T_ptr tau_batch, int stride_tau,
	Dim_Type rows_batch, Dim_Type cols_batch, int row_offset, int col_offset,
	int smem_entries, int panel_rows, int num_ops
)
{
    extern __shared__ char sdata[];
    const int HH_CB = QR_Config<T>::HH_CB;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int op_id = thread_id / BLOCK_SIZE;

    if(op_id >= num_ops) return;
	
	int rows = getOperationDim(rows_batch, op_id);
	int cols = getOperationDim(cols_batch, op_id);
	int ldm  = getOperationDim(ldm_batch,  op_id);
	
	if(row_offset >= rows || col_offset >= cols)
		return;
	
    int local_op_id = threadIdx.x / BLOCK_SIZE;
    int local_tid = threadIdx.x % BLOCK_SIZE;

	// Shared memory
    T* s_tau = (T*)sdata + smem_entries * local_op_id;
    T* s_pivot = s_tau + HH_CB;
    T* smem_reduction = s_pivot + 1;

	// Global memory
    T* m_panel = getOperationPtr<T>(m_batch, op_id, stride) + row_offset + col_offset * ldm;
    T* tau_panel = getOperationPtr<T>(tau_batch, op_id, stride_tau) + col_offset;

    // Store the matrix panel in registers
    T matrix_row[HH_CB];

    // Threads with id beyond the remaining rows will be padded with zeros
	int nopadding = (local_tid < panel_rows && local_tid + row_offset < rows);
	
    // Load the current panel in
    #pragma unroll
    for(int i = 0; i < HH_CB; i++)
        matrix_row[i] = (nopadding && col_offset + i < cols ? m_panel[local_tid + i * ldm] : 0);

    // Factor the panel, generating the current block of R and reflectors
    qr_householder_panel<T, HH_CB, BLOCK_SIZE>(matrix_row, s_tau, local_tid, smem_reduction, s_pivot);

    // Flush the data to global memory
    if(nopadding)
    {
        #pragma unroll
        for(int i = 0; i < HH_CB; i++)
			if(col_offset + i < cols)
				m_panel[local_tid + i * ldm] = matrix_row[i];
    }

    if(local_tid < HH_CB && local_tid + col_offset < cols)
        tau_panel[local_tid] = s_tau[local_tid];
}

template<class T, class T_ptr, int BLOCK_SIZE>
__global__
void batch_unpackQ_panel(
	T_ptr __restrict__ m_batch, int ldm, int stride, T_ptr tau_batch, int stride_tau,
	int rows, int cols, int row_offset, int col_offset,
	int smem_entries, int panel_rows, int num_ops
)
{
    extern __shared__ char sdata[];
    const int HH_CB = QR_Config<T>::HH_CB;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int op_id = thread_id / BLOCK_SIZE;

    if(op_id >= num_ops) return;

    int local_op_id = threadIdx.x / BLOCK_SIZE;
    int local_tid = threadIdx.x % BLOCK_SIZE;

	// Shared memory
    T* s_tau = (T*)sdata + smem_entries * local_op_id;
    T* smem_reduction = s_tau + HH_CB;

	// Global memory
    T* m_panel = getOperationPtr<T>(m_batch, op_id, stride) + row_offset + col_offset * ldm;
    T* tau_panel = getOperationPtr<T>(tau_batch, op_id, stride_tau) + col_offset;

    // Store the matrix panel in registers
    T matrix_row[HH_CB];

    // Threads with id beyond the remaining rows will be padded with zeros
    int nopadding = (local_tid < panel_rows);

    // Load the current panel in
    #pragma unroll
    for(int i = 0; i < HH_CB; i++)
        matrix_row[i] = (nopadding && col_offset + i < cols ? m_panel[local_tid + i * ldm] : 0);
    if(local_tid < HH_CB && local_tid + col_offset < cols) s_tau[local_tid] = tau_panel[local_tid];

    if(BLOCK_SIZE != WARP_SIZE) __syncthreads();

    // Factor the panel, generating the current block of R and reflectors
    qr_unpackQ_panel<T, HH_CB, BLOCK_SIZE>(matrix_row, s_tau, local_tid, smem_reduction);

    // Flush the data to global memory
    if(nopadding)
    {
        #pragma unroll
        for(int i = 0; i < HH_CB; i++)
			if(col_offset + i < cols)
				m_panel[local_tid + i * ldm] = matrix_row[i];
    }
}

// Annihilate the block (A2) below the current panel (A1)
template<class T, class T_ptr, int BLOCK_SIZE, class Dim_Type>
__global__
void batch_dtsqrt_panel(
	T_ptr __restrict__ m_batch, Dim_Type ldm_batch, int stride, T_ptr tau_batch, int stride_tau,
	Dim_Type rows_batch, Dim_Type cols_batch, int A1_row_off, int A1_col_off, int A2_row_off, int A2_rows,
	int smem_entries_per_op, int num_ops
)
{
    extern __shared__ char sdata[];
    const int HH_CB = QR_Config<T>::HH_CB;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int op_id = thread_id / BLOCK_SIZE;

    if(op_id >= num_ops) return;
	
	int rows = getOperationDim(rows_batch, op_id);
	int cols = getOperationDim(cols_batch, op_id);
	int ldm  = getOperationDim(ldm_batch,  op_id);
	
	if(A1_row_off >= rows || A2_row_off >= rows || A1_col_off >= cols)
		return;
	
    int local_op_id = threadIdx.x / BLOCK_SIZE;
    int local_tid = threadIdx.x % BLOCK_SIZE;

    // Shared memory
    T* s_tau = (T*)sdata + smem_entries_per_op * local_op_id;
    T* A1_smem = s_tau + HH_CB;
    T* reduction_smem = A1_smem + HH_CB * HH_CB;

    // Global memory
    T* m_global = getOperationPtr<T>(m_batch, op_id, stride);
    T* tau_global = getOperationPtr<T>(tau_batch, op_id, stride_tau);
	T* A1_global = m_global + A1_row_off + A1_col_off * ldm;
    T* A2_global = m_global + A2_row_off + A1_col_off * ldm;

    // Store A2 in registers
    T A2_matrix_row[HH_CB];

    // Load the HH_CB x HH_CB A1 block into shared memory
    if(local_tid < HH_CB && A1_row_off + local_tid < rows)
	{
		#pragma unroll
        for(int i = 0; i < HH_CB; i++)
		{
			A1_smem[local_tid + i * HH_CB] = 0;
			if(A1_col_off + i < cols)
				A1_smem[local_tid + i * HH_CB] = A1_global[local_tid + i * ldm];
		}
	}
    __syncthreads();

	int nopadding = (local_tid < A2_rows && A2_row_off + local_tid < rows);
    // Load the A2_rows x HH_CB A2 block into registers
    #pragma unroll
    for(int i = 0; i < HH_CB; i++)
        A2_matrix_row[i] = (nopadding && A1_col_off + i < cols ? A2_global[local_tid + i * ldm] : 0);

    // Eliminate the A2 block
    dtsqrt_panel<T, HH_CB, BLOCK_SIZE>(A2_matrix_row, s_tau, local_tid, reduction_smem, A1_smem);

    // Dump everything back to global memory
    if(local_tid < HH_CB)
    {
		if(local_tid + A1_col_off < cols)
			tau_global[local_tid] = s_tau[local_tid];
		#pragma unroll
        for(int i = 0; i < HH_CB; i++)
            if(A1_col_off + i < cols)
				A1_global[local_tid + i * ldm] = A1_smem[local_tid + i * HH_CB];
    }
	
    if(nopadding)
    {
        #pragma unroll
        for(int i = 0; i < HH_CB; i++)
			if(A1_col_off + i < cols)
				A2_global[local_tid + i * ldm] = A2_matrix_row[i];
    }
}

// Apply the generated householder vectors that annihilated the block at (V) to the trailing block row (A2) and the block row (A1) from the panel above it
template<class T, class T_ptr, int BLOCK_SIZE, class Dim_Type>
__global__
void batch_apply_dtsqrt_panel(
	T_ptr __restrict__ m_batch, Dim_Type ldm_batch, int stride, T_ptr tau_batch, int stride_tau,
	Dim_Type rows_batch, Dim_Type cols_batch, int V_row_off, int V_col_off, int V_rows, int A1_row_off,
	int smem_entries_per_op, int num_ops
)
{
	extern __shared__ char sdata[];
    const int HH_CB = QR_Config<T>::HH_CB;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int op_id = thread_id / BLOCK_SIZE;

    if(op_id >= num_ops) return;
	
	int rows = getOperationDim(rows_batch, op_id);
	int cols = getOperationDim(cols_batch, op_id);
	int ldm  = getOperationDim(ldm_batch,  op_id);
	
	if(V_row_off >= rows || A1_row_off >= rows || V_col_off >= cols)
		return;
	
	int trailing_blocks = cols - V_col_off - HH_CB;
	if(trailing_blocks <= 0) return;
	trailing_blocks = iDivUp(trailing_blocks, HH_CB);
	
    int local_op_id = threadIdx.x / BLOCK_SIZE;
    int local_tid = threadIdx.x % BLOCK_SIZE;

    // Shared memory
    T* s_tau = (T*)sdata + smem_entries_per_op * local_op_id;
    T* A1_smem = s_tau + HH_CB;
    T* reduction_smem = A1_smem + HH_CB * HH_CB;

    // Global memory
    T* m_global = getOperationPtr<T>(m_batch, op_id, stride);
	T* tau_global = getOperationPtr<T>(tau_batch, op_id, stride_tau);

    T* V_global  = m_global + V_row_off + V_col_off * ldm;
    T* A1_global = m_global + A1_row_off + V_col_off * ldm;
    T* A2_global = V_global;

    // Store V and A2 in registers
    T A2_matrix_row[HH_CB], V_matrix_row[HH_CB];

    // Load tau entries into shared memory
    if(local_tid < HH_CB)
		s_tau[local_tid] = tau_global[local_tid];

    // Load the V_rows x HH_CB V block into registers
	int no_padding = (local_tid < V_rows && local_tid + V_row_off < rows);
    #pragma unroll
    for(int i = 0; i < HH_CB; i++)
        V_matrix_row[i] = (no_padding ? V_global[local_tid + i * ldm] : 0);

	int column_index = V_col_off;

    for(int b = 0; b < trailing_blocks; b++)
    {
        A1_global += HH_CB * ldm;
        A2_global += HH_CB * ldm;
        column_index += HH_CB;

        // Load A1 into shared memory
        if(local_tid < HH_CB)
            for(int i = 0; i < HH_CB; i++)
                A1_smem[local_tid + i * HH_CB] = (column_index + i < cols ? A1_global[local_tid + i * ldm] : 0);

        // Load the V_rows x HH_CB A2 block into registers
        #pragma unroll
        for(int i = 0; i < HH_CB; i++)
            A2_matrix_row[i] = (no_padding && column_index + i < cols ? A2_global[local_tid + i * ldm] : 0);
        __syncthreads();

        // Update the blocks
        #pragma unroll
        for(int i = 0; i < HH_CB; i++)
            dtsqrt_apply_panel<T, HH_CB, BLOCK_SIZE>(A2_matrix_row, V_matrix_row[i], s_tau[i], local_tid, reduction_smem, A1_smem, i);

        // Flush blocks back to global memory
        if(local_tid < HH_CB)
            for(int i = 0; i < HH_CB; i++)
                if(column_index + i < cols)
					A1_global[local_tid + i * ldm] = A1_smem[local_tid + i * HH_CB];

        if(no_padding)
        {
            #pragma unroll
            for(int i = 0; i < HH_CB; i++)
				if(column_index + i < cols)
					A2_global[local_tid + i * ldm] = A2_matrix_row[i];
        }
        __syncthreads();
    }
}

template<class T, class T_ptr>
__global__
void batch_qr_copy_R_kernel(T_ptr m_batch, int ldm, int stride_m, T_ptr r_batch, int ldr, int stride_r, int rows, int cols, int ops)
{
    int op_id = blockIdx.z;
    if(op_id >= ops) return;
	
	int R_rows = (rows > cols ? cols : rows);
	int R_cols = cols;
	
    int row_index = blockDim.x * blockIdx.x + threadIdx.x;
	int col_index = (blockDim.y * blockIdx.y + threadIdx.y) * HLIB_R_COLS_PER_THREAD;

	if(row_index >= R_rows || col_index >= R_cols) 
		return;

    T* m_global = getOperationPtr<T>(m_batch, op_id, stride_m);
    T* r_global = getOperationPtr<T>(r_batch, op_id, stride_r);

	m_global += row_index + col_index * ldm;
	r_global += row_index + col_index * ldr;
	
	T reg_buffer[HLIB_R_COLS_PER_THREAD];
	
	#pragma unroll 
    for(int j = 0; j < HLIB_R_COLS_PER_THREAD; j++)
		if(j + col_index < R_cols)
			reg_buffer[j] = (row_index > j + col_index ? 0 : m_global[j * ldm]);
    
	#pragma unroll 
	for(int j = 0; j < HLIB_R_COLS_PER_THREAD; j++)
		if(j + col_index < R_cols)
			r_global[j * ldr] = reg_buffer[j];
}

template<class T, class T_ptr>
__global__
void batch_qr_clear_R_kernel(T_ptr m_batch, int ldm, int stride, int rows, int cols, int ops)
{
    int op_id = blockIdx.z;
    if(op_id >= ops) return;
	
	int R_rows = (rows > cols ? cols : rows);
	int R_cols = cols;
	
    int row_index = blockDim.x * blockIdx.x + threadIdx.x;
	int col_index = (blockDim.y * blockIdx.y + threadIdx.y) * HLIB_R_COLS_PER_THREAD;

	if(row_index >= R_rows || col_index >= R_cols) 
		return;

    T* m_global = getOperationPtr<T>(m_batch, op_id, stride);
	
	m_global += row_index + col_index * ldm;
	
	#pragma unroll 
    for(int j = 0; j < HLIB_R_COLS_PER_THREAD; j++)
		if(j + col_index < R_cols && row_index < j + col_index)
			m_global[j * ldm] = 0;
}

template<class T, class T_ptr>
int batch_qr_clear_R(kblasHandle_t handle, T_ptr m_batch, int ldm, int stride, int rows, int cols, int ops)
{
	int R_rows = (rows > cols ? cols : rows);
	int R_cols = cols;
	
	int max_thread_y = HLIB_R_MAX_THREAD_Y;
	
    int thread_x = WARP_SIZE, thread_y = kmin(max_thread_y, iDivUp(R_cols, HLIB_R_COLS_PER_THREAD));
    int grid_x = iDivUp(R_rows, thread_x), grid_y = iDivUp(R_cols, thread_y * HLIB_R_COLS_PER_THREAD);

    dim3 dimBlock(thread_x, thread_y, 1);
    dim3 dimGrid(grid_x, grid_y, ops);

    batch_qr_clear_R_kernel<T, T_ptr> <<< dimGrid, dimBlock, 0, handle->stream >>> 
		(m_batch, ldm, stride, rows, cols, ops);

	check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	return KBLAS_Success;
}

template<class T, class T_ptr>
int batch_qr_copy_R(kblasHandle_t handle, T_ptr m_batch, int ldm, int stride_m, T_ptr r_batch, int ldr, int stride_r, int rows, int cols, int ops)
{
	int R_rows = (rows > cols ? cols : rows);
	int R_cols = cols;
	
	int max_thread_y = HLIB_R_MAX_THREAD_Y;
	
    int thread_x = WARP_SIZE, thread_y = kmin(max_thread_y, iDivUp(R_cols, HLIB_R_COLS_PER_THREAD));
    int grid_x = iDivUp(R_rows, thread_x), grid_y = iDivUp(R_cols, thread_y * HLIB_R_COLS_PER_THREAD);

    dim3 dimBlock(thread_x, thread_y, 1);
    dim3 dimGrid(grid_x, grid_y, ops);

    batch_qr_copy_R_kernel<T, T_ptr><<< dimGrid, dimBlock, 0, handle->stream >>>
		(m_batch, ldm, stride_m, r_batch, ldr, stride_r, rows, cols, ops);

	check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	return KBLAS_Success;
}

template<class T, class T_ptr, class Dim_Type>
int driver_hh_panel(
	kblasHandle_t handle, T_ptr m_batch, Dim_Type ldm, int stride, T_ptr tau_batch, int stride_tau, 
	Dim_Type rows, Dim_Type cols, int num_ops, int row_offset, int col_offset, int panel_rows,
	int max_rows, int max_cols
)
{
    int ops_per_block = OPS_PER_BLOCK;
	const int HH_CB = QR_Config<T>::HH_CB;

	if(panel_rows > 256) ops_per_block = 1;

    int blocks = iDivUp(num_ops, ops_per_block);
    int warps = iDivUp(panel_rows, WARP_SIZE);

    dim3 dimBlock( ops_per_block * warps * WARP_SIZE, 1 );
    dim3 dimGrid( blocks, 1 );

    int smem_reduction = warps * WARP_SIZE * HH_CB + warps * HH_CB;
    int smem_tau       = HH_CB;
    int smem_pivot     = 1;

    int smem_entries_per_op = smem_reduction + smem_tau + smem_pivot;
    int smem_per_block = sizeof(T) * smem_entries_per_op * ops_per_block;

    switch(warps)
    {
        case  1: batch_qr_panel<T, T_ptr,   32, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case  2: batch_qr_panel<T, T_ptr,   64, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case  3: batch_qr_panel<T, T_ptr,   96, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case  4: batch_qr_panel<T, T_ptr,  128, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case  5: batch_qr_panel<T, T_ptr,  160, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case  6: batch_qr_panel<T, T_ptr,  192, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case  7: batch_qr_panel<T, T_ptr,  224, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case  8: batch_qr_panel<T, T_ptr,  256, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case  9: batch_qr_panel<T, T_ptr,  288, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 10: batch_qr_panel<T, T_ptr,  320, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 11: batch_qr_panel<T, T_ptr,  352, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 12: batch_qr_panel<T, T_ptr,  384, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 13: batch_qr_panel<T, T_ptr,  416, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 14: batch_qr_panel<T, T_ptr,  448, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 15: batch_qr_panel<T, T_ptr,  480, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 16: batch_qr_panel<T, T_ptr,  512, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		#ifdef QR_SUPPORT_LARGE                                                                                                                                                                                         
		case 17: batch_qr_panel<T, T_ptr,  544, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 18: batch_qr_panel<T, T_ptr,  576, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 19: batch_qr_panel<T, T_ptr,  608, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 20: batch_qr_panel<T, T_ptr,  640, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 21: batch_qr_panel<T, T_ptr,  672, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 22: batch_qr_panel<T, T_ptr,  704, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 23: batch_qr_panel<T, T_ptr,  736, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 24: batch_qr_panel<T, T_ptr,  768, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 25: batch_qr_panel<T, T_ptr,  800, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 26: batch_qr_panel<T, T_ptr,  832, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 27: batch_qr_panel<T, T_ptr,  864, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 28: batch_qr_panel<T, T_ptr,  896, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 29: batch_qr_panel<T, T_ptr,  928, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 30: batch_qr_panel<T, T_ptr,  960, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 31: batch_qr_panel<T, T_ptr,  992, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 32: batch_qr_panel<T, T_ptr, 1024, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		#endif
        default: { printf("driver_hh_panel: Invalid row size %d\n", panel_rows); return KBLAS_UnknownError; }
    }

    check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	return KBLAS_Success;
}

template<class T, class T_ptr>
int driver_unpackQ_panel(kblasHandle_t handle, T_ptr m_batch, int ldm, int stride, T_ptr tau_batch, int stride_tau, int rows, int cols, int num_ops, int row_offset, int col_offset, int panel_rows)
{
    int ops_per_block = OPS_PER_BLOCK;
	const int HH_CB = QR_Config<T>::HH_CB;

	if(panel_rows > 256) ops_per_block = 1;

    int blocks = iDivUp(num_ops, ops_per_block);
    int warps = iDivUp(panel_rows, WARP_SIZE);

    dim3 dimBlock( ops_per_block * warps * WARP_SIZE, 1 );
    dim3 dimGrid( blocks, 1 );

    int smem_reduction = warps * WARP_SIZE * HH_CB + warps * HH_CB;
    int smem_tau       = HH_CB;

    int smem_entries_per_op = smem_reduction + smem_tau;
    int smem_per_block = sizeof(T) * smem_entries_per_op * ops_per_block;

    switch(warps)
    {
        case  1: batch_unpackQ_panel<T, T_ptr,   32><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case  2: batch_unpackQ_panel<T, T_ptr,   64><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case  3: batch_unpackQ_panel<T, T_ptr,   96><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case  4: batch_unpackQ_panel<T, T_ptr,  128><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case  5: batch_unpackQ_panel<T, T_ptr,  160><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case  6: batch_unpackQ_panel<T, T_ptr,  192><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case  7: batch_unpackQ_panel<T, T_ptr,  224><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case  8: batch_unpackQ_panel<T, T_ptr,  256><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case  9: batch_unpackQ_panel<T, T_ptr,  288><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 10: batch_unpackQ_panel<T, T_ptr,  320><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 11: batch_unpackQ_panel<T, T_ptr,  352><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 12: batch_unpackQ_panel<T, T_ptr,  384><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 13: batch_unpackQ_panel<T, T_ptr,  416><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 14: batch_unpackQ_panel<T, T_ptr,  448><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 15: batch_unpackQ_panel<T, T_ptr,  480><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 16: batch_unpackQ_panel<T, T_ptr,  512><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		#ifdef QR_SUPPORT_LARGE
		case 17: batch_unpackQ_panel<T, T_ptr,  544><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 18: batch_unpackQ_panel<T, T_ptr,  576><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 19: batch_unpackQ_panel<T, T_ptr,  608><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 20: batch_unpackQ_panel<T, T_ptr,  640><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 21: batch_unpackQ_panel<T, T_ptr,  672><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 22: batch_unpackQ_panel<T, T_ptr,  704><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 23: batch_unpackQ_panel<T, T_ptr,  736><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 24: batch_unpackQ_panel<T, T_ptr,  768><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 25: batch_unpackQ_panel<T, T_ptr,  800><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 26: batch_unpackQ_panel<T, T_ptr,  832><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 27: batch_unpackQ_panel<T, T_ptr,  864><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 28: batch_unpackQ_panel<T, T_ptr,  896><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 29: batch_unpackQ_panel<T, T_ptr,  928><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 30: batch_unpackQ_panel<T, T_ptr,  960><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 31: batch_unpackQ_panel<T, T_ptr,  992><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 32: batch_unpackQ_panel<T, T_ptr, 1024><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		#endif
        default: { printf("driver_unpackQ_panel: Invalid row size %d\n", panel_rows); return KBLAS_UnknownError; }
    }

    check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	return KBLAS_Success;
}

template<class T, class T_ptr, int APPLY_FORWARD, class Dim_Type>
int driver_apply_hh_panel(
	kblasHandle_t handle, T_ptr m_batch, Dim_Type ldm, int stride, T_ptr tau_batch, int stride_tau, 
	Dim_Type rows, Dim_Type cols, int num_ops, int row_offset, int col_offset, int panel_rows, 
	int max_rows, int max_cols
)
{
    int ops_per_block = OPS_PER_BLOCK;
	const int HH_CB = QR_Config<T>::HH_CB;

    if(panel_rows > 256) ops_per_block = 1;

    if(max_cols - col_offset <= HH_CB) return KBLAS_Success;

    int blocks = iDivUp(num_ops, ops_per_block);
    int warps = iDivUp(panel_rows, WARP_SIZE);

    dim3 dimBlock( ops_per_block * warps * WARP_SIZE, 1 );
    dim3 dimGrid( blocks, 1 );

    int smem_padding = warps * HH_CB;
    int smem_entries_per_op = warps * WARP_SIZE * HH_CB + smem_padding + HH_CB;
    int smem_per_block = sizeof(T) * smem_entries_per_op * ops_per_block;

    switch(warps)
    {
        case  1: batch_apply_hh_panel<T, T_ptr,   32, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case  2: batch_apply_hh_panel<T, T_ptr,   64, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case  3: batch_apply_hh_panel<T, T_ptr,   96, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case  4: batch_apply_hh_panel<T, T_ptr,  128, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case  5: batch_apply_hh_panel<T, T_ptr,  160, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case  6: batch_apply_hh_panel<T, T_ptr,  192, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case  7: batch_apply_hh_panel<T, T_ptr,  224, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case  8: batch_apply_hh_panel<T, T_ptr,  256, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case  9: batch_apply_hh_panel<T, T_ptr,  288, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 10: batch_apply_hh_panel<T, T_ptr,  320, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 11: batch_apply_hh_panel<T, T_ptr,  352, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 12: batch_apply_hh_panel<T, T_ptr,  384, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 13: batch_apply_hh_panel<T, T_ptr,  416, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 14: batch_apply_hh_panel<T, T_ptr,  448, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 15: batch_apply_hh_panel<T, T_ptr,  480, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 16: batch_apply_hh_panel<T, T_ptr,  512, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		#ifdef QR_SUPPORT_LARGE                                                                                                                                                                                                               
		case 17: batch_apply_hh_panel<T, T_ptr,  544, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 18: batch_apply_hh_panel<T, T_ptr,  576, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 19: batch_apply_hh_panel<T, T_ptr,  608, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 20: batch_apply_hh_panel<T, T_ptr,  640, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 21: batch_apply_hh_panel<T, T_ptr,  672, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 22: batch_apply_hh_panel<T, T_ptr,  704, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 23: batch_apply_hh_panel<T, T_ptr,  736, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 24: batch_apply_hh_panel<T, T_ptr,  768, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 25: batch_apply_hh_panel<T, T_ptr,  800, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 26: batch_apply_hh_panel<T, T_ptr,  832, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 27: batch_apply_hh_panel<T, T_ptr,  864, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
        case 28: batch_apply_hh_panel<T, T_ptr,  896, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 29: batch_apply_hh_panel<T, T_ptr,  928, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 30: batch_apply_hh_panel<T, T_ptr,  960, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 31: batch_apply_hh_panel<T, T_ptr,  992, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		case 32: batch_apply_hh_panel<T, T_ptr, 1024, APPLY_FORWARD, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, row_offset, col_offset, smem_entries_per_op, panel_rows, num_ops); break;
		#endif
        default: {printf("driver_apply_hh_panel: Invalid row size %d\n", panel_rows); return KBLAS_UnknownError;}
    }

	check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	return KBLAS_Success;
}

template<class T, class T_ptr, class Dim_Type>
int driver_dtsqrt_panel(
	kblasHandle_t handle, T_ptr m_batch, Dim_Type ldm, int stride, T_ptr tau_batch, int stride_tau, 
	Dim_Type rows, Dim_Type cols, int num_ops, int A1_row_off, int A1_col_off, int A2_row_off, int A2_rows, 
	int max_rows, int max_cols
)
{
    int ops_per_block = OPS_PER_BLOCK;
	const int HH_CB = QR_Config<T>::HH_CB;

    int blocks = iDivUp(num_ops, ops_per_block);
    int warps = iDivUp(A2_rows, WARP_SIZE);

    dim3 dimBlock( ops_per_block * warps * WARP_SIZE, 1 );
    dim3 dimGrid( blocks, 1 );

    int smem_reduction = warps * WARP_SIZE * HH_CB + warps * HH_CB;
    int smem_tau       = HH_CB;
    int smem_A1        = HH_CB * HH_CB;

    int smem_entries_per_op = smem_reduction + smem_tau + smem_A1;
    int smem_per_block = sizeof(T) * smem_entries_per_op * ops_per_block;

    switch(warps)
    {
        case 1: batch_dtsqrt_panel<T, T_ptr,  32, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, A1_row_off, A1_col_off, A2_row_off, A2_rows, smem_entries_per_op, num_ops); break;
        case 2: batch_dtsqrt_panel<T, T_ptr,  64, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, A1_row_off, A1_col_off, A2_row_off, A2_rows, smem_entries_per_op, num_ops); break;
        case 3: batch_dtsqrt_panel<T, T_ptr,  96, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, A1_row_off, A1_col_off, A2_row_off, A2_rows, smem_entries_per_op, num_ops); break;
        case 4: batch_dtsqrt_panel<T, T_ptr, 128, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, A1_row_off, A1_col_off, A2_row_off, A2_rows, smem_entries_per_op, num_ops); break;
        default: { printf("driver_dtsqrt_panel: Invalid row size %d\n", A2_rows); return KBLAS_UnknownError; }
    }

	check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	return KBLAS_Success;
}

template<class T, class T_ptr, class Dim_Type>
int driver_apply_dtsqrt_panel(
	kblasHandle_t handle, T_ptr m_batch, Dim_Type ldm, int stride, T_ptr tau_batch, int stride_tau, 
	Dim_Type rows, Dim_Type cols, int num_ops, int V_row_off, int V_col_off, int V_rows, int A1_row_off, 
	int max_rows, int max_cols
)
{
    int ops_per_block = OPS_PER_BLOCK;
    const int HH_CB = QR_Config<T>::HH_CB;

    if(max_cols - V_col_off <= HH_CB) return KBLAS_Success;

    int blocks = iDivUp(num_ops, ops_per_block);
    int warps = iDivUp(V_rows, WARP_SIZE);

    dim3 dimBlock( ops_per_block * warps * WARP_SIZE, 1 );
    dim3 dimGrid( blocks, 1 );

    int smem_reduction = warps * WARP_SIZE * HH_CB + warps * HH_CB;
    int smem_tau       = HH_CB;
    int smem_A1        = HH_CB * HH_CB;

    int smem_entries_per_op = smem_reduction + smem_tau + smem_A1;
    int smem_per_block = sizeof(T) * smem_entries_per_op * ops_per_block;

    switch(warps)
    {
        case 1: batch_apply_dtsqrt_panel<T, T_ptr,  32, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, V_row_off, V_col_off, V_rows, A1_row_off, smem_entries_per_op, num_ops); break;
        case 2: batch_apply_dtsqrt_panel<T, T_ptr,  64, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, V_row_off, V_col_off, V_rows, A1_row_off, smem_entries_per_op, num_ops); break;
        case 3: batch_apply_dtsqrt_panel<T, T_ptr,  96, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, V_row_off, V_col_off, V_rows, A1_row_off, smem_entries_per_op, num_ops); break;
        case 4: batch_apply_dtsqrt_panel<T, T_ptr, 128, Dim_Type><<< dimGrid, dimBlock, smem_per_block, handle->stream >>>(m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, V_row_off, V_col_off, V_rows, A1_row_off, smem_entries_per_op, num_ops); break;
		default: { printf("driver_apply_dtsqrt_panel: Invalid row size %d\n", V_rows); return KBLAS_UnknownError; }
    }

	check_error_ret( cudaGetLastError(), KBLAS_UnknownError );
	return KBLAS_Success;
}

template<class T, class T_ptr, class Dim_Type>
int batch_qr(kblasHandle_t handle, T_ptr m_batch, Dim_Type ldm, int stride, T_ptr tau_batch, int stride_tau, Dim_Type rows, Dim_Type cols, int num_ops, int block_rows, int max_rows, int max_cols)
{
	const int HH_CB = QR_Config<T>::HH_CB;

    int rows_per_block = (max_rows < block_rows ? max_rows : block_rows);
    int matrix_rank = (max_rows > max_cols ? max_cols : max_rows);

	for(int c = 0; c < matrix_rank; c += HH_CB)
    {
        int upper_panel_height = rows_per_block - c % rows_per_block;
        if(c + upper_panel_height > max_rows) 
			upper_panel_height = max_rows - c;

        KBLAS_QR_CHECK_RET( (driver_hh_panel<T, T_ptr, Dim_Type>(
			handle, m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, num_ops, 
			c, c, upper_panel_height, max_rows, max_cols
		)) );
        KBLAS_QR_CHECK_RET( (driver_apply_hh_panel<T, T_ptr, 1, Dim_Type>(
			handle, m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, num_ops, 
			c, c, upper_panel_height, max_rows, max_cols
		)) );

        int remaining_rows = max_rows - c - upper_panel_height;
        if(remaining_rows <= 0) continue;

        int remaining_row_blocks = iDivUp(remaining_rows, rows_per_block);

        for(int rb = 0; rb < remaining_row_blocks; rb++)
        {
            int A2_row_offset = c + upper_panel_height + rb * rows_per_block;
            int A2_rows = (A2_row_offset + rows_per_block > max_rows ? max_rows - A2_row_offset : rows_per_block);
			
            KBLAS_QR_CHECK_RET( (driver_dtsqrt_panel<T, T_ptr, Dim_Type>(
				handle, m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, num_ops, 
				c, c, A2_row_offset, A2_rows, max_rows, max_cols
			)) );
            KBLAS_QR_CHECK_RET( (driver_apply_dtsqrt_panel<T, T_ptr, Dim_Type>(
				handle, m_batch, ldm, stride, tau_batch, stride_tau, rows, cols, num_ops, 
				A2_row_offset, c, A2_rows, c, max_rows, max_cols
			)) );
        }
    }

	return KBLAS_Success;
}

template<class T, class T_ptr>
int batch_unpack_Q(kblasHandle_t handle, T_ptr m_batch, int ldm, int stride, T_ptr tau_batch, int stride_tau, int rows, int cols, int num_ops)
{
	const int HH_CB = QR_Config<T>::HH_CB;

	// Zero out the upper triangular part of the matrix
	int matrix_rank = (rows > cols ? cols : rows);
	KBLAS_QR_CHECK_RET( (batch_qr_clear_R<T, T_ptr>(handle, m_batch, ldm, stride, rows, matrix_rank, num_ops)) );

	int col_start = (matrix_rank % HH_CB == 0 ? matrix_rank - HH_CB : matrix_rank - matrix_rank % HH_CB);
    for(int c = col_start; c >= 0; c -= HH_CB)
    {
        int panel_rows = rows - c;
        KBLAS_QR_CHECK_RET( (driver_apply_hh_panel<T, T_ptr, 0, int>(
			handle, m_batch, ldm, stride, tau_batch, stride_tau, rows, matrix_rank, num_ops, 
			c, c, panel_rows, rows, cols
		)) );
        KBLAS_QR_CHECK_RET( (driver_unpackQ_panel<T, T_ptr>(
			handle, m_batch, ldm, stride, tau_batch, stride_tau, rows, matrix_rank, num_ops, 
			c, c, panel_rows
		)) );
    }

	return KBLAS_Success;
}

///////////////////////////////////////////////////////////////
// Strided routines
///////////////////////////////////////////////////////////////
extern "C" int kblasDgeqrf_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* tau, int stride_tau, int num_ops)
{
	if(m > QR_Config<double>::HH_MAX_ROWS || lda < m)
		return KBLAS_Error_WrongInput;
	else if(stride_a < m || stride_tau < kmin(m, n))
		return KBLAS_Error_WrongInput;
	if(m == 0 || n == 0)
		return KBLAS_Success;
	
	return batch_qr<double, double*, int>(handle, A_strided, lda, stride_a, tau, stride_tau, m, n, num_ops, m, m, n);
}

extern "C" int kblasSgeqrf_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* tau, int stride_tau, int num_ops)
{
	if(m > QR_Config<float>::HH_MAX_ROWS || lda < m)
		return KBLAS_Error_WrongInput;
	else if(stride_a < m || stride_tau < kmin(m, n))
		return KBLAS_Error_WrongInput;

	if(m == 0 || n == 0)
		return KBLAS_Success;
	
	return batch_qr<float, float*, int>(handle, A_strided, lda, stride_a, tau, stride_tau, m, n, num_ops, m, m, n);
}

extern "C" int kblasDtsqrf_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* tau, int stride_tau, int num_ops)
{
	if(lda < m)
		return KBLAS_Error_WrongInput;
	else if(stride_a < m || stride_tau < kmin(m, n))
		return KBLAS_Error_WrongInput;
	
	if(m == 0 || n == 0)
		return KBLAS_Success;
	
	return batch_qr<double, double*, int>(handle, A_strided, lda, stride_a, tau, stride_tau, m, n, num_ops, ROWS_PER_BLOCK, m, n);
}

extern "C" int kblasStsqrf_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* tau, int stride_tau, int num_ops)
{
	if(lda < m)
		return KBLAS_Error_WrongInput;
	else if(stride_a < m || stride_tau < kmin(m, n))
		return KBLAS_Error_WrongInput;
	
	if(m == 0 || n == 0)
		return KBLAS_Success;
	
	return batch_qr<float, float*, int>(handle, A_strided, lda, stride_a, tau, stride_tau, m, n, num_ops, ROWS_PER_BLOCK, m, n);
}

extern "C" int kblasDorgqr_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* tau, int stride_tau, int num_ops)
{
	if(m > QR_Config<double>::HH_MAX_ROWS || lda < m)
		return KBLAS_Error_WrongInput;
	else if(stride_a < m || stride_tau < kmin(m, n))
		return KBLAS_Error_WrongInput;	
	if(m == 0 || n == 0)
		return KBLAS_Success;
	
	return batch_unpack_Q<double, double*>(handle, A_strided, lda, stride_a, tau, stride_tau, m, n, num_ops);
}

extern "C" int kblasSorgqr_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* tau, int stride_tau, int num_ops)
{
	if(m > QR_Config<float>::HH_MAX_ROWS || lda < m)
		return KBLAS_Error_WrongInput;
	else if(stride_a < m || stride_tau < kmin(m, n))
		return KBLAS_Error_WrongInput;

	if(m == 0 || n == 0)
		return KBLAS_Success;
	
	return batch_unpack_Q<float, float*>(handle, A_strided, lda, stride_a, tau, stride_tau, m, n, num_ops);
}

extern "C" int kblasDcopy_upper_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, double* R_strided, int ldr, int stride_R, int num_ops)
{
	if(lda < m || ldr < kmin(m, n))
		return KBLAS_Error_WrongInput;
	else if(stride_a < m || stride_R < n)
		return KBLAS_Error_WrongInput;

	if(m == 0 || n == 0)
		return KBLAS_Success;
	
	return batch_qr_copy_R<double, double*>(handle, A_strided, lda, stride_a, R_strided, ldr, stride_R, m, n, num_ops);
}

extern "C" int kblasScopy_upper_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, float* R_strided, int ldr, int stride_R, int num_ops)
{
	if(lda < m || ldr < kmin(m, n))
		return KBLAS_Error_WrongInput;
	else if(stride_a < m || stride_R < n)
		return KBLAS_Error_WrongInput;
	
	if(m == 0 || n == 0)
		return KBLAS_Success;
	
	return batch_qr_copy_R<float, float*>(handle, A_strided, lda, stride_a, R_strided, ldr, stride_R, m, n, num_ops);
}

///////////////////////////////////////////////////////////////
// Array of pointers routines
///////////////////////////////////////////////////////////////
extern "C" int kblasDgeqrf_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** tau_array, int num_ops)
{
	if(m > QR_Config<double>::HH_MAX_ROWS || lda < m)
		return KBLAS_Error_WrongInput;

	if(m == 0 || n == 0)
		return KBLAS_Success;
	
	return batch_qr<double, double**, int>(handle, A_array, lda, 0, tau_array, 0, m, n, num_ops, m, m, n);
}

extern "C" int kblasSgeqrf_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** tau_array, int num_ops)
{
	if(m > QR_Config<float>::HH_MAX_ROWS || lda < m)
		return KBLAS_Error_WrongInput;

	if(m == 0 || n == 0)
		return KBLAS_Success;
	
	return batch_qr<float, float**, int>(handle, A_array, lda, 0, tau_array, 0, m, n, num_ops, m, m, n);
}

extern "C" int kblasDtsqrf_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** tau_array, int num_ops)
{
	if(lda < m)
		return KBLAS_Error_WrongInput;
	
	if(m == 0 || n == 0)
		return KBLAS_Success;
	
	return batch_qr<double, double**, int>(handle, A_array, lda, 0, tau_array, 0, m, n, num_ops, ROWS_PER_BLOCK, m, n);
}

extern "C" int kblasStsqrf_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** tau_array, int num_ops)
{
	if(lda < m)
		return KBLAS_Error_WrongInput;

	if(m == 0 || n == 0)
		return KBLAS_Success;
	
	return batch_qr<float, float**, int>(handle, A_array, lda, 0, tau_array, 0, m, n, num_ops, ROWS_PER_BLOCK, m, n);
}

extern "C" int kblasDorgqr_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** tau_array, int num_ops)
{
	if(m > QR_Config<double>::HH_MAX_ROWS || lda < m)
		return KBLAS_Error_WrongInput;
	
	if(m == 0 || n == 0)
		return KBLAS_Success;
	
	return batch_unpack_Q<double, double**>(handle, A_array, lda, 0, tau_array, 0, m, n, num_ops);
}

extern "C" int kblasSorgqr_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** tau_array, int num_ops)
{
	if(m > QR_Config<float>::HH_MAX_ROWS || lda < m)
		return KBLAS_Error_WrongInput;
	
	if(m == 0 || n == 0)
		return KBLAS_Success;
	
	return batch_unpack_Q<float, float**>(handle, A_array, lda, 0, tau_array, 0, m, n, num_ops);
}

extern "C" int kblasDcopy_upper_batch(kblasHandle_t handle, int m, int n, double** A_array, int lda, double** R_array, int ldr, int num_ops)
{
	if(lda < m || ldr < std::min(m, n))
		return KBLAS_Error_WrongInput;
	
	if(m == 0 || n == 0)
		return KBLAS_Success;
	
	return batch_qr_copy_R<double, double**>(handle, A_array, lda, 0, R_array, ldr, 0, m, n, num_ops);
}

extern "C" int kblasScopy_upper_batch(kblasHandle_t handle, int m, int n, float** A_array, int lda, float** R_array, int ldr, int num_ops)
{
	if(lda < m || ldr < std::min(m, n))
		return KBLAS_Error_WrongInput;
	
	if(m == 0 || n == 0)
		return KBLAS_Success;
	
	return batch_qr_copy_R<float, float**>(handle, A_array, lda, 0, R_array, ldr, 0, m, n, num_ops);
}

///////////////////////////////////////////////////////////////
// Array of pointers variable size routines
///////////////////////////////////////////////////////////////
extern "C" int kblasDtsqrf_vbatch(kblasHandle_t handle, int* m, int* n, int max_m, int max_n, double** A_array, int* lda, double** tau_array, int num_ops)
{
	return batch_qr<double, double**, int*>(handle, A_array, lda, 0, tau_array, 0, m, n, num_ops, ROWS_PER_BLOCK, max_m, max_n);
}

extern "C" int kblasStsqrf_vbatch(kblasHandle_t handle, int* m, int* n, int max_m, int max_n, float** A_array, int* lda, float** tau_array, int num_ops)
{
	return batch_qr<float, float**, int*>(handle, A_array, lda, 0, tau_array, 0, m, n, num_ops, ROWS_PER_BLOCK, max_m, max_n);
}
