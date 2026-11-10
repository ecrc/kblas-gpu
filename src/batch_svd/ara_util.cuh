#ifndef __ARA_UTIL_H__
#define __ARA_UTIL_H__

#include "kblas_operators.h"

//////////////////////////////////////////////////////////////////////////////////////
// ARA helpers
//////////////////////////////////////////////////////////////////////////////////////
template <typename T> struct AraGramType;
template <typename T> struct AraTransposeVal;

template <> struct AraGramType<double> {typedef double type;};
template <> struct AraGramType<float> {typedef double type;};
template <> struct AraGramType<cuDoubleComplex> {typedef cuDoubleComplex type;};
template <> struct AraGramType<cuComplex> {typedef cuDoubleComplex type;};

template <> struct AraTransposeVal<double> {static const char val = KBLAS_Trans;};
template <> struct AraTransposeVal<float>  {static const char val = KBLAS_Trans;};
template <> struct AraTransposeVal<cuDoubleComplex> {static const char val = KBLAS_ConjTrans;};
template <> struct AraTransposeVal<cuComplex> {static const char val = KBLAS_ConjTrans;};

template<typename out, typename in> __device__ inline out mp_cast(const in& val);
template<> __device__ inline cuComplex mp_cast<cuComplex, cuDoubleComplex> (const cuDoubleComplex& val) {return make_cuComplex(val.x, val.y); }
template<> __device__ inline cuDoubleComplex mp_cast<cuDoubleComplex, cuComplex> (const cuComplex& val) {return make_cuDoubleComplex(val.x, val.y); }
template<> __device__ inline float mp_cast<float, double> (const double& val) {return val; }
template<> __device__ inline double mp_cast<double, float> (const float& val) {return val; }
template<> __device__ inline double mp_cast<double, double> (const double& val) {return val; }
template<> __device__ inline cuDoubleComplex mp_cast<cuDoubleComplex, cuDoubleComplex> (const cuDoubleComplex& val) {return val; }

__device__ inline void setReal(cuDoubleComplex& val, double r) {val.x = r; val.y = 0;}
__device__ inline void setReal(double& val, double r) {val = r;}
__device__ inline void setReal(cuComplex& val, float r) {val.x = r; val.y = 0;}
__device__ inline void setReal(float& val, float r) {val = r;}

struct ARASampleSetter
{
	int* small_vectors, *max_ranks, *ranks;
	int bs, r;
	
	ARASampleSetter(int* small_vectors, int *ranks, int *max_ranks, int bs, int r) { 
		this->small_vectors = small_vectors; 
        this->ranks = ranks;
		this->max_ranks = max_ranks;
        this->bs = bs;
        this->r = r;
	}

	__host__ __device__ int operator()(const unsigned int& thread_id) const { 
		int samples = 0;
        
        if(small_vectors[thread_id] < r) 
        {
            samples = max_ranks[thread_id] - ranks[thread_id];
            if(bs < samples)
                samples = bs;
        }
        return samples;
	}
};

template<class T>
struct ARAOffsetPointer
{
	T** original_ptrs;
	int* ld_batch;
	int row_offset, col_offset;
	
	ARAOffsetPointer(T** original_ptrs, int* ld_batch, int row_offset, int col_offset)
	{
		this->original_ptrs = original_ptrs;
		this->ld_batch = ld_batch;
		this->row_offset = row_offset;
		this->col_offset = col_offset;
	}
	
	__host__ __device__ T* operator()(const unsigned int& thread_id) const { 
		return original_ptrs[thread_id] + row_offset + col_offset * ld_batch[thread_id];
	}
};

struct TRSM_Offset_Dims
{
	int *temp_n, *temp_k, *cols_batch;
	int offset;
	TRSM_Offset_Dims(int *temp_n, int *temp_k, int *cols_batch, int offset)
	{
		this->temp_n = temp_n;
		this->temp_k = temp_k;
		this->cols_batch = cols_batch;
		this->offset = offset;
	}
	__host__ __device__
	void operator()(int index)
	{
		int cols = cols_batch[index];
		temp_n[index] = (cols < offset ? cols : offset);
		temp_k[index] = (cols < offset ? 0 : cols - offset);
	}
};

template<class TR, class TY>
struct KBLAS_ARA_ConvergenceCheck
{
    typedef typename kblas_base_type<TY>::mag_type mag_type;
    
    TR* diag_R, *max_diag;
    TY** Y_batch;
    mag_type tol;
    const mag_type* tol_batch;
    int bs, *block_ranks, *ranks_batch, *ldy_batch, r, *small_vectors, relative, *info;

    KBLAS_ARA_ConvergenceCheck(
        TR* diag_R, int bs, int* block_ranks, int* ranks_batch, TR* max_diag, 
	    TY** Y_batch, int* ldy_batch, mag_type tol, int r, int* small_vectors, 
        int *info, int relative, const mag_type* tol_batch
    )
    {
        this->diag_R = diag_R;
        this->bs = bs;
        this->block_ranks = block_ranks;
        this->ranks_batch = ranks_batch;
        this->max_diag = max_diag;
        this->Y_batch = Y_batch;
        this->ldy_batch = ldy_batch;
        this->tol = tol;
        this->tol_batch = tol_batch;
        this->r = r;
        this->small_vectors = small_vectors;
        this->info = info;
        this->relative = relative;
    }
    
	__host__ __device__
	void operator()(int op_id)
	{
        int block_rank = block_ranks[op_id];
	    int small_vecs_op = small_vectors[op_id];

        if(block_rank == 0) 
        {
            small_vectors[op_id] = r;
            return;
        }
	
        TR* diag_op = diag_R + op_id * bs;
        mag_type max_op = kblas_get_real(max_diag[op_id]);
        int ldy = ldy_batch[op_id];
        int current_rank = ranks_batch[op_id];
        
        // Load the diagonal entries and update the max for the op if we need a relative threshold
        if(relative == 1)
        {
            for(int i = 0; i < block_rank; i++)
            {
                mag_type diag_entry = kblas_get_real(diag_op[i]);
                if(diag_entry > max_op)
                    max_op = diag_entry;
            }
        }

        // Update the small vector count and check for convergence
        const mag_type op_tol = tol_batch ? tol_batch[op_id] : tol;
        mag_type eps = (relative ? op_tol * max_op : op_tol);

        for(int i = 0; i < block_rank; i++)
        {
            mag_type diag_entry = kblas_get_real(diag_op[i]);
            current_rank++;

            if(diag_entry < eps)
			{
                small_vecs_op++;
				if(small_vecs_op == r)
				{
					small_vectors[op_id] = r;
					ranks_batch[op_id] = current_rank - r;
					info[op_id] = KBLAS_Success;
					return;
				}
			}
            else
                small_vecs_op = 0;

			if(diag_entry == 0)
			{
				small_vectors[op_id] = r;
				ranks_batch[op_id] = current_rank - 1;
				info[op_id] = KBLAS_Success;
				return;
			}
        }
        
        // We didn't converge so advance the pointers and update the ranks
        Y_batch[op_id] += block_rank * ldy;
        setReal(max_diag[op_id], max_op);
        ranks_batch[op_id] = current_rank;
        small_vectors[op_id] = small_vecs_op;
    }
};

struct SampleIsZeroPredicate
{
	__host__ __device__
	bool operator()(const int &x)
	{ return x == 0; }
};

template<class T>
inline void copyGPUArray(T* originalArray, T* copyArray, int num_ptrs, cudaStream_t stream)
{
	thrust::copy(
		thrust::cuda::par.on(stream),
		originalArray, originalArray + num_ptrs,
		copyArray
	);
	
	check_error( cudaGetLastError() );
}

inline void ara_trsm_set_offset_dims(
	int *temp_n, int *temp_k, int *cols_batch, int offset, 
	int num_ops, cudaStream_t stream
)
{
	thrust::for_each(
		thrust::cuda::par.on(stream),
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(num_ops),
		TRSM_Offset_Dims(temp_n, temp_k, cols_batch, offset)
	);
}

template<class T>
inline void ara_offset_pointers(
	T** offset_ptrs, T** original_ptrs, int* ld_batch, int row_offset, int col_offset, 
	int num_ops, cudaStream_t stream
)
{
	thrust::transform(
		thrust::cuda::par.on(stream),
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(num_ops),
		thrust::device_ptr<T*>(offset_ptrs),
		ARAOffsetPointer<T>(original_ptrs, ld_batch, row_offset, col_offset)
	);
}

int kblas_ara_batch_set_samples(
	int* op_samples, int* small_vectors, int *ranks_batch, int bs, int *max_ranks, 
    int r, int num_ops, cudaStream_t stream
)
{
	thrust::device_ptr<int> dev_data(op_samples);

	thrust::transform(
		thrust::cuda::par.on(stream),
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(num_ops),
		dev_data,
		ARASampleSetter(small_vectors, ranks_batch, max_ranks, bs, r)
	);

	check_error( cudaGetLastError() );
	
	return thrust::reduce(
        thrust::cuda::par.on(stream), op_samples, op_samples + num_ops, 0, 
        thrust::maximum<int>()
    );
}

//////////////////////////////////////////////////////////////////////////////////////
// ARA kernels
//////////////////////////////////////////////////////////////////////////////////////
template<class T, int N>
__device__ __forceinline__
T warp_max(T a)
{
    #pragma unroll 
    for (int mask = N / 2; mask > 0; mask /= 2)
	{
		T b = __shfl_xor_sync(0xFFFFFFFF, a, mask);
        if(b > a) a = b;
	}
	return a;
}

template<class T, int N>
__device__ __forceinline__
T warp_sum(T val)
{
    #pragma unroll
    for (int mask = N / 2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

// Mixed precision potrf
template<class inType, class outType, int BS>
__global__ void ara_fused_potrf_kernel(
	int* op_samples, inType** A_batch, int* lda_batch, outType** R_batch, 
    int* ldr_batch, inType* diag_R, int* block_ranks, int num_ops
)
{
	extern __shared__ char sdata[];
	typedef typename kblas_base_type<inType>::mag_type MagT;

	int thread_index = blockDim.x * blockIdx.x + threadIdx.x;
	int op_id = thread_index / BS;
	inType zero = make_zero<inType>();

	if(op_id >= num_ops) return;
	
	// Get the local thread data within the block
	int local_op = threadIdx.x / BS;
	int tid = threadIdx.x % BS;
	int warp_id = threadIdx.y;
	int num_warps = blockDim.y;
	int dim = op_samples[op_id];
	
	// Exit early if this block is empty
	if(dim <= 0) 
	{
		if(tid == 0 && warp_id == 0) 
			block_ranks[op_id] = 0;
		return;
	}
	
	int lda = lda_batch[op_id];
	int ldr = ldr_batch[op_id];
	
	int shared_ld = BS + 1;
	int cpt = iDivUp(dim, num_warps);
	
	inType* A = A_batch[op_id];
	outType* R = R_batch[op_id];
	inType* d = diag_R + BS * op_id;
	
	inType* shared_A = (inType*)sdata + local_op * (BS + BS * shared_ld);
	inType* shared_d = shared_A + BS * shared_ld;
	
	// Load the matrix into shared memory 
	if(tid < dim)
	{
		for(int i = 0; i < cpt; i++)
		{
			int col_index = warp_id + i * num_warps;
			if(col_index < dim)
				shared_A[tid + col_index * shared_ld] = A[tid + col_index * lda];
		}
	}
	__syncthreads();
	
	if(warp_id == 0)
		shared_d[tid] = d[tid];
	
	MagT tol = warp_max<MagT, BS>(tid < dim ? kblas_get_real(shared_A[tid + tid * shared_ld]) : 0) * KBlasEpsilon<MagT>::eps;
	__syncthreads();
	
	int k;	
	for(k = 0; k < dim; k++)
	{	
		// Check if the matrix is semi-definite 
        MagT diagMag = kblas_get_real(shared_A[k + k * shared_ld]);
		if(diagMag <= tol)
        {
            // if(tid == k && warp_id == 0) 
                // printf("Chol for op %d had diagonal %e at col %d\n", op_id, diagMag, k);
			break;
        }
		__syncthreads();
		diagMag = sqrt(diagMag);

		// Compute the diagonal and update the local copy
		if(tid == k && warp_id == 0) 
		{
			setReal(shared_A[k + k * shared_ld], diagMag);
			shared_d[k] *= shared_A[k + k * shared_ld];
		}
        
        __syncthreads();

		// Update the column
		if(tid > k && warp_id == 0)
            shared_A[tid + k * shared_ld] /= diagMag;
		__syncthreads();
		
		// Update the trailing submatrix
		if(tid > k)
		{
			cpt = iDivUp(dim - k - 1, num_warps);
			for(int i = 0; i < cpt; i++)
			{
				int col_index = k + 1 + warp_id + i * num_warps;
				if(col_index < dim)
				 	shared_A[tid + col_index * shared_ld] -= conjugate(shared_A[col_index + k * shared_ld]) * shared_A[tid + k * shared_ld];
			}
		}
		__syncthreads();
	}
	
	// Flush results to global memory
	if(warp_id == 0)
	{
		if(tid == 0) block_ranks[op_id] = k;
		d[tid] = (tid >= k ? zero : shared_d[tid]);
	}
	
	cpt = iDivUp(dim, num_warps);
	
	if(tid < dim)
	{
		for(int i = 0; i < cpt; i++)
		{
			int col_index = warp_id + i * num_warps;
			if(col_index < dim)
            {
				R[tid + col_index * ldr] = mp_cast<outType, inType>(
                    tid > col_index ? zero : conjugate(shared_A[col_index + tid * shared_ld])
                );
            }
		}
	}
}

template<class TR, class TY, int BS>
__global__ void ara_svec_count_kernel(	
	TR* diag_R, int* block_ranks, int* ranks_batch, TR* max_diag, 
	TY** Y_batch, int* ldy_batch, typename kblas_base_type<TY>::mag_type tol, 
    int r, int* small_vectors, int relative, int num_ops
)
{
    typedef typename kblas_base_type<TR>::mag_type MagT;

	int thread_index = blockDim.x * blockIdx.x + threadIdx.x;
	int op_id = thread_index / BS;
	
	if(op_id >= num_ops) return;
	
	int block_rank = block_ranks[op_id];
	int small_vecs_op = small_vectors[op_id];
	
	// if this block has rank 0, make sure the operation is set to
	// converged
	if(block_rank == 0) 
	{
		small_vectors[op_id] = r;
		return;
	}
	
	TR* diag_op = diag_R + op_id * BS;
	MagT max_op = kblas_get_real(max_diag[op_id]);
	int ldy = ldy_batch[op_id];
	int current_rank = ranks_batch[op_id];
	
	int tid = threadIdx.x % BS;
	
	// Load the diagonal entries and update the max for the op
	MagT diag_entry = kblas_get_real(diag_op[tid]);
	MagT bmax_entry = warp_max<MagT, BS>(diag_entry);
	if(bmax_entry > max_op) max_op = bmax_entry;
	
	// Update the small vector count
	MagT eps = (relative ? tol * max_op : tol);
	int entry_small = (diag_entry <= eps ? 1 : 0);
	int total_small_entries = small_vecs_op + warp_sum<int, BS>(entry_small);
	
	current_rank += BS;
	
	if(tid == 0)
	{
		// Consider the operation converged if we have enough small entries
		// or if the cholesky broke down and determined that the current block
		// is singular
		if(total_small_entries >= r || block_rank != BS)
		{
			small_vectors[op_id] = r;
			ranks_batch[op_id] = current_rank - total_small_entries;
			
			// printf("%f op %d converged to rank %d (block rank was %d) and %d small vectors\n", max_op, op_id, ranks_batch[op_id], block_rank, total_small_entries);
		}	
		else
		{
			// Advance the pointers for the orthogonal factor
			Y_batch[op_id] += BS * ldy;
			setReal(max_diag[op_id], max_op);
			ranks_batch[op_id] = current_rank;
			small_vectors[op_id] = total_small_entries;
			
			// printf("op %d had rank %d (block rank was %d) and %d small vectors\n", op_id, ranks_batch[op_id], block_rank, total_small_entries);
		}
	}
}

template<class T>
__global__ void ara_16x16_trsm_kernel(
	T** B_batch, int* ldb_batch, T** A_batch, int* lda_batch, 
	int* rows_batch, int* cols_batch, int offset, int num_ops
)
{
	const int BS = 16;
	
	int op_id = blockDim.y * blockIdx.y + threadIdx.y;
	if(op_id >= num_ops) return;
	
	int row_index = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	
	// fetch all operation data
	int rows = rows_batch[op_id];
	int cols = cols_batch[op_id];
	int lda  = lda_batch[op_id];
	int ldb  = ldb_batch[op_id];
	int trailing_cols = cols - offset;
    
	if(trailing_cols < 0) return;
	
    T zero = make_zero<T>();
	T* A = A_batch[op_id];
	T* B = B_batch[op_id];
	
	// Advance the pointer for A and B
	B += offset * ldb;
	A += offset + offset * lda;
	
	// Store the data for B in registers and the data for A in shared memory
	T B_reg[BS];
	extern __shared__ char sdata[];
	T* A_shared = (T*)sdata + threadIdx.y * BS * BS;
	
    int col_bs = (BS < trailing_cols ? BS : trailing_cols);
    
	if(tid < col_bs)
	{
		for(int i = 0; i < col_bs; i++)
            A_shared[tid + i * BS] = A[tid + i * lda];
	}
	__syncthreads();
	
	if(row_index >= rows) return;
	
	#pragma unroll 
	for(int i = 0; i < BS; i++)
		B_reg[i] = (i < trailing_cols ? B[row_index + i * ldb] : zero);
	
	// Now do the triangular solve
	#pragma unroll 
	for(int j = 0; j < BS; j++)
	{
		if(j >= trailing_cols) continue;
		#pragma unroll
		for(int k = 0; k < j; k++)
			B_reg[j] -= A_shared[k + j * BS] * B_reg[k];
		B_reg[j] = kblas_div(B_reg[j], A_shared[j + j * BS]);
	}
	
	// Flush data back to global memory
	#pragma unroll 
	for(int i = 0; i < BS; i++)
		if(i < trailing_cols)
			B[row_index + i * ldb] = B_reg[i];
}

//////////////////////////////////////////////////////////////////////////////////////
// ARA drivers
//////////////////////////////////////////////////////////////////////////////////////
// Assumptions: max(op_samples) <= bs
//				bs = 16 or 32
// 				TODO: optimized subwarp kernel for bs = 16
template<class inType, class outType>
inline int kblas_ara_fused_potrf_batch_template( 
	int* op_samples, inType** A_batch, int* lda_batch, outType** R_batch, int* ldr_batch, 
	inType* diag_R, int bs, int* block_ranks, int num_ops, 
	cudaStream_t stream
)
{
	if(bs == 32)
	{
		int ops_per_block = 1;
		if(ops_per_block > num_ops) ops_per_block = num_ops;
		
		int warps_per_op = 8;
		
		int threads_per_block = ops_per_block * bs;
		int smem_per_block = ops_per_block * (bs + bs * (bs + 1)) * sizeof(inType);
		int thread_blocks = iDivUp(num_ops, ops_per_block);
		
		dim3 dimBlock(threads_per_block, warps_per_op);
		dim3 dimGrid(thread_blocks, 1 );
	
		ara_fused_potrf_kernel<inType, outType, 32> <<<dimGrid, dimBlock, smem_per_block, stream>>> (
			op_samples, A_batch, lda_batch, R_batch, ldr_batch, diag_R, block_ranks, num_ops
		);
	
		check_error( cudaGetLastError() );
	}
	else if(bs == 16)
	{
		int ops_per_block = 2;
		if(ops_per_block > num_ops) ops_per_block = num_ops;
		
		int warps_per_op = 8;
		
		int threads_per_block = ops_per_block * bs;
		int smem_per_block = ops_per_block * (bs + bs * (bs + 1)) * sizeof(inType);
		int thread_blocks = iDivUp(num_ops, ops_per_block);
		
		dim3 dimBlock(threads_per_block, warps_per_op);
		dim3 dimGrid(thread_blocks, 1 );
	
		ara_fused_potrf_kernel<inType, outType, 16> <<<dimGrid, dimBlock, smem_per_block, stream>>> (
			op_samples, A_batch, lda_batch, R_batch, ldr_batch, diag_R, block_ranks, num_ops
		);
	
		check_error( cudaGetLastError() );
	}
	else 
		return KBLAS_NotImplemented;
	
	return KBLAS_Success;
}

template<class TR, class TY>
inline int kblas_ara_svec_count_batch_template(			
	TR* diag_R, int bs, int* op_samples, int* ranks_batch, TR* max_diag, 
	TY** Y_batch, int* ldy_batch, typename kblas_base_type<TY>::mag_type tol, 
    int r, int* small_vectors, int* info, int relative, int num_ops, cudaStream_t stream,
    const typename kblas_base_type<TY>::mag_type* tol_batch = nullptr
)
{
    thrust::for_each(
		thrust::cuda::par.on(stream),
		thrust::counting_iterator<int>(0),
		thrust::counting_iterator<int>(num_ops),
		KBLAS_ARA_ConvergenceCheck<TR, TY>(
            diag_R, bs, op_samples, ranks_batch, max_diag, Y_batch, ldy_batch, tol, 
            r, small_vectors, info, relative, tol_batch
        )
	);

	// int ops_per_block = 4;
	// int threads_per_block = ops_per_block * bs;
	// int thread_blocks = iDivUp(num_ops, ops_per_block);
	
	// dim3 dimBlock(threads_per_block, 1);
	// dim3 dimGrid(thread_blocks, 1 );
	
	// if(bs == 16)
	// 	ara_svec_count_kernel<TR, TY, 16> <<<dimGrid, dimBlock, 0, stream>>> (
	// 		diag_R, op_samples, ranks_batch, max_diag, Y_batch, ldy_batch, 
	// 		tol, r, small_vectors, relative, num_ops
	// 	);
	// else if(bs == 32)
	// 	ara_svec_count_kernel<TR, TY, 32> <<<dimGrid, dimBlock, 0, stream>>> (
	// 		diag_R, op_samples, ranks_batch, max_diag, Y_batch, ldy_batch, 
	// 		tol, r, small_vectors, relative, num_ops
	// 	);
	// else 
	// 	return KBLAS_NotImplemented;
	
	check_error( cudaGetLastError() );
	return KBLAS_Success;
}

template<class T>
int kblas_ara_16x16_trsm(
	T** B_batch, int* ldb_batch, T** A_batch, int* lda_batch, 
	int* rows_batch, int* cols_batch, int offset, int num_ops, 
	int max_rows, cudaStream_t stream
)
{
	const int max_block_threads = 128;
	const int BS = 16;
	
    if(max_rows <= 0) 
        return KBLAS_Success;
        
	int block_thread_x = max_block_threads;
	int block_thread_y = 1;
	
	if(block_thread_x > max_rows)
	{
		block_thread_x = max_rows;
		block_thread_y = max_block_threads / max_rows;
        
        // Don't go overboard with the number of ops per block
        // Since we might not have enough shared memory
        if (block_thread_y > 16)
            block_thread_y = 16;
	}
    
    // Make sure we have enough threads to load the triangular block
    if (block_thread_x < BS)
        block_thread_x = BS;
        
	int grid_x = iDivUp(max_rows, block_thread_x);
	int grid_y = iDivUp(num_ops , block_thread_y);
	
	dim3 dimBlock(block_thread_x, block_thread_y);
	dim3 dimGrid(grid_x, grid_y);
	
	int smem_per_block = block_thread_y * BS * BS * sizeof(T);
	
	ara_16x16_trsm_kernel<<<dimGrid, dimBlock, smem_per_block, stream>>> ( 
		B_batch, ldb_batch, A_batch, lda_batch, rows_batch, cols_batch, offset, num_ops
	);
	
	check_error( cudaGetLastError() );
	
	return KBLAS_Success;
}

template<class T>
inline void ara_trsm_batch_wsquery(KBlasWorkspaceState& requested_ws, int num_ops, int external_only)
{
	gemm_batch_nonuniform_wsquery_core(&requested_ws);
	
	if(!external_only)
	{
		// Align to sizeof(int) bytes
		requested_ws.d_data_bytes += requested_ws.d_data_bytes % sizeof(int);
		
		requested_ws.d_data_bytes += (
			num_ops + // temp_n
			num_ops   // temp_k
		) * sizeof(int);
		
		requested_ws.d_ptrs_bytes += (
			num_ops +  // temp_B
			num_ops    // temp_A
		) * sizeof(T*);
	}
}

// #define TESTING_MALLOC_DEV( ptr, T, size) check_error( cudaMalloc( (void**)&ptr, (size)*sizeof(T) ) )

template<class T>
inline int kblas_ara_trsm_batch_template(
	kblasHandle_t handle, T** B_batch, int* ldb_batch, T** A_batch, int* lda_batch, 
	int* rows_batch, int* cols_batch, int num_ops, int max_rows, int bs
)
{
    T one = make_one<T>();

	if(bs == 32)
	{
		int *temp_n, *temp_k;
		T **temp_B, **temp_A;
		
		///////////////////////////////////////////////////////////////////////////
		// Workspace allocation
		///////////////////////////////////////////////////////////////////////////
		KBlasWorkspaceState external_ws, total_ws;
		ara_trsm_batch_wsquery<T>(external_ws, num_ops, 1);
		ara_trsm_batch_wsquery<T>(total_ws, num_ops, 0);
		
		KBlasWorkspaceState available_ws = handle->work_space.getAvailable();
		if(!total_ws.isSufficient(&available_ws))
			return KBLAS_InsufficientWorkspace;
		
		// Align to sizeof(int) bytes
		external_ws.d_data_bytes += external_ws.d_data_bytes % sizeof(int);
		
		int* base_ws_ptr = (int*)((KBlasWorkspace::WS_Byte*)handle->work_space.d_data + external_ws.d_data_bytes);
		temp_n = base_ws_ptr; 
		temp_k = temp_n + num_ops;
		
		temp_B = (T**)((KBlasWorkspace::WS_Byte*)handle->work_space.d_ptrs + external_ws.d_ptrs_bytes);	
		temp_A = temp_B + num_ops;

		/*TESTING_MALLOC_DEV(temp_n, int, num_ops);
		TESTING_MALLOC_DEV(temp_k, int, num_ops);
		TESTING_MALLOC_DEV(temp_B, T*, num_ops);
		TESTING_MALLOC_DEV(temp_A, T*, num_ops);*/
		///////////////////////////////////////////////////////////////////////////
		const int BS = 16;
		cudaStream_t kblas_stream = kblasGetStream(handle);
		
		// Split into two 16x16 trsms and one gemm
		// First trsm of the top diagonal block: B1 = B1 * A11^{-1}
		// printDenseMatrixGPU(B_batch, ldb_batch, rows_batch, cols_batch, 0, 15, "B1");
		// printDenseMatrixGPU(A_batch, lda_batch, cols_batch, cols_batch, 0, 15, "R1");
		
		check_error( kblas_ara_16x16_trsm(
			B_batch, ldb_batch, A_batch, lda_batch, rows_batch, cols_batch, 
			0, num_ops, max_rows, kblas_stream
		) );
		
		ara_trsm_set_offset_dims(temp_n, temp_k, cols_batch, BS, num_ops, kblas_stream);
		ara_offset_pointers<T>(temp_B, B_batch, ldb_batch, 0, BS, num_ops, kblas_stream);
		ara_offset_pointers<T>(temp_A, A_batch, lda_batch, 0, BS, num_ops, kblas_stream);
		
		// printDenseMatrixGPU(B_batch, ldb_batch, rows_batch, cols_batch, 0, 15, "Q1");
		
		// Update the right block of B with a gemm 
		// B2 = B2 - B1 * A12
		check_error( kblas_gemm_batch(
			handle, KBLAS_NoTrans, KBLAS_NoTrans, rows_batch, temp_k, temp_n, 
			max_rows, BS, BS, -one, (const T**)B_batch, ldb_batch, 
			(const T**)temp_A, lda_batch, one, temp_B, ldb_batch, num_ops
		) );
		
		// Final trsm of the lower diagonal block: B2 = B2 * A22^{-1}
		check_error( kblas_ara_16x16_trsm(
			B_batch, ldb_batch, A_batch, lda_batch, rows_batch, cols_batch, 
			BS, num_ops, max_rows, kblas_stream
		) );
	}
	else if(bs == 16)
	{
		cudaStream_t kblas_stream = kblasGetStream(handle);
		
		check_error( kblas_ara_16x16_trsm(
			B_batch, ldb_batch, A_batch, lda_batch, rows_batch, cols_batch, 
			0, num_ops, max_rows, kblas_stream
		) );
	}
	else
		return KBLAS_NotImplemented;
	
	return KBLAS_Success;
}

//////////////////////////////////////////////////////////////////////////////////////
// Mixed precision syrks: specifically G = A'* A
// If A is double precision, then G only needs to be double (uses default kblas gemm)
// If A is single precision, then G needs to be double precision (uses my monstrosity)
//////////////////////////////////////////////////////////////////////////////////////
template<class inType, class outType, int ept, int DIM>
__global__ void ara_mp_syrk_batch_kernel(
	int* m_batch, int* n_batch, const inType** A_batch, int* lda_batch, 
	outType** B_batch, int* ldb_batch, int batchCount 
)
{
	__shared__ outType shared_A[DIM][DIM+1];
	
	outType B_reg[ept];
    outType zero = make_zero<outType>();

	int op_id = blockIdx.x;
	if(op_id >= batchCount) return;
	
	// Get operation parameters
	int rows_op = m_batch[op_id];
	int cols_op = n_batch[op_id];
	
	if(rows_op == 0 || cols_op == 0) return;
	
	int lda_op = lda_batch[op_id];
	int ldb_op = ldb_batch[op_id];
	
	const inType* A_op = A_batch[op_id];
	outType* B_op = B_batch[op_id];
	
	int tx = threadIdx.x, ty = threadIdx.y;
	int col_threads = blockDim.y;
	
	int row = tx;
	int block_rows = rows_op / DIM;
	int remaining_rows = rows_op % DIM;
	
	#pragma unroll 
	for(int j = 0; j < ept; j++)
		B_reg[j] = zero;

	for(int rb = 0; rb < block_rows; rb++)
	{
		// Load the current block into smem
		#pragma unroll 
		for(int j = 0; j < ept; j++)
		{
			int col = ty + j * col_threads;
			if(col < cols_op)
				shared_A[tx][col] = mp_cast<outType, inType>(A_op[row + col * lda_op]);
		}
		__syncthreads();
		
		// Now accumulate into registers
		#pragma unroll 
		for(int j = 0; j < ept; j++)
		{
			int col = ty + j * col_threads;
			if(col < cols_op)
			{
				#pragma unroll 
				for(int k = 0; k < DIM; k++)
					B_reg[j] += conjugate(shared_A[k][tx]) * shared_A[k][col];
			}
		}
		__syncthreads();
		row += DIM;
	}
	
	// handle the remaining rows
	if(remaining_rows != 0)
	{
		if(row < rows_op)
		{
			// Load the current block into smem
			#pragma unroll 
			for(int j = 0; j < ept; j++)
			{
				int col = ty + j * col_threads;
				if(col < cols_op)
					shared_A[tx][col] = mp_cast<outType, inType>(A_op[row + col * lda_op]);
			}
		}
		__syncthreads();
	
		// Now accumulate into registers
		#pragma unroll 
		for(int j = 0; j < ept; j++)
		{
			int col = ty + j * col_threads;
			if(col < cols_op)
			{
				for(int k = 0; k < remaining_rows; k++)
					B_reg[j] += conjugate(shared_A[k][tx]) * shared_A[k][col];
			}
		}
	}
	
	// output the values from registers to global memory
	if(tx < cols_op)
	{
		#pragma unroll 
		for(int j = 0; j < ept; j++)
		{
			int col = ty + j * col_threads;
			if(col < cols_op)
            {
				B_op[tx + col * ldb_op] = B_reg[j];
                // printf("%d %d %e %e\n", tx, col, kblas_get_real(B_reg[j]), kblas_get_img(B_reg[j]));
            }
		}
	}
}

inline int kblas_ara_mp_syrk_batch_template(
	kblasHandle_t handle, int* m, int* n, int max_m, int max_n, 
	const double** A, int* lda, double** B, int* ldb, int batchCount 
) 
{
	return kblas_gemm_batch(
		handle, KBLAS_Trans, KBLAS_NoTrans, n, n, m, 
		max_n, max_n, max_m, 1, A, lda, A, lda, 
		0, B, ldb, batchCount
	);
}

inline int kblas_ara_mp_syrk_batch_template(
	kblasHandle_t handle, int* m, int* n, int max_m, int max_n, 
	const cuDoubleComplex** A, int* lda, cuDoubleComplex** B, int* ldb, int batchCount 
) 
{
	return kblas_gemm_batch(
		handle, KBLAS_ConjTrans, KBLAS_NoTrans, n, n, m, 
		max_n, max_n, max_m, make_one<cuDoubleComplex>(), A, lda, A, lda, 
		make_zero<cuDoubleComplex>(), B, ldb, batchCount
	);
}

inline int kblas_ara_mp_syrk_batch_template(
	kblasHandle_t handle, int* m, int* n, int max_m, int max_n, 
	const float** A, int* lda, double** B, int* ldb, int batchCount 
) 
{
	if(max_n > 32) return KBLAS_NotImplemented;
	
	cudaStream_t kblas_stream = kblasGetStream(handle);
	
	if(max_n <= 16)
	{
		const int ept = 2;
		const int DIM = 16;
		
		int thread_x = DIM;
		int thread_y = iDivUp(max_n, ept);
		int grid_x = batchCount;
		
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(grid_x, 1);
			
		ara_mp_syrk_batch_kernel<float, double, ept, DIM> 
			<<<dimGrid, dimBlock, 0, kblas_stream>>> (m, n, A, lda, B, ldb, batchCount);
	}
	else
	{
		const int ept = 4;
		const int DIM = 32;
		
		int thread_x = DIM;
		int thread_y = iDivUp(max_n, ept);
		int grid_x = batchCount;
		
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(grid_x, 1);
			
		ara_mp_syrk_batch_kernel<float, double, ept, DIM> 
			<<<dimGrid, dimBlock, 0, kblas_stream>>> (m, n, A, lda, B, ldb, batchCount);
	}
	
	return KBLAS_Success;
}

inline int kblas_ara_mp_syrk_batch_template(
	kblasHandle_t handle, int* m, int* n, int max_m, int max_n, 
	const cuComplex** A, int* lda, cuDoubleComplex** B, int* ldb, int batchCount 
) 
{
	if(max_n > 32) return KBLAS_NotImplemented;
	
	cudaStream_t kblas_stream = kblasGetStream(handle);
	
	if(max_n <= 16)
	{
		const int ept = 2;
		const int DIM = 16;
		
		int thread_x = DIM;
		int thread_y = iDivUp(max_n, ept);
		int grid_x = batchCount;
		
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(grid_x, 1);
			
		ara_mp_syrk_batch_kernel<cuComplex, cuDoubleComplex, ept, DIM> 
			<<<dimGrid, dimBlock, 0, kblas_stream>>> (m, n, A, lda, B, ldb, batchCount);
	}
	else
	{
		const int ept = 4;
		const int DIM = 32;
		
		int thread_x = DIM;
		int thread_y = iDivUp(max_n, ept);
		int grid_x = batchCount;
		
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(grid_x, 1);
			
		ara_mp_syrk_batch_kernel<cuComplex, cuDoubleComplex, ept, DIM> 
			<<<dimGrid, dimBlock, 0, kblas_stream>>> (m, n, A, lda, B, ldb, batchCount);
	}
	
	return KBLAS_Success;
}

#endif
