#include <stdio.h>
#include <cublas_v2.h>

#include "kblas.h"
#include "kblas_struct.h"
#include "kblas_gpu_util.ch"
#include "batch_rand.h"

#include "batch_ara.h"

#include "thrust_wrappers.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/transform.h>
// #include <mkl.h>

#include "workspace_queries.ch"
#include "ara_util.cuh"

//////////////////////////////////////////////////////////////////////////////////////
// Debug routines
//////////////////////////////////////////////////////////////////////////////////////
template<class Real>
void printDenseMatrix(Real *matrix, int ldm, int m, int n, int digits, const char* name)
{
	char format[64];
	assert(ldm >= m);
	
    sprintf(format, "%%.%de ", digits);
    printf("%s = ([\n", name);
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++)
            printf(format, matrix[i + j * ldm]);
        printf(";\n");
    }
    printf("]);\n");
}

void printDenseMatrixGPU(float** block_ptrs, int* ld_batch, int* rows_batch, int* cols_batch, int index, int digits, const char* name)
{}

void printDenseMatrixGPU(double** block_ptrs, int* ld_batch, int* rows_batch, int* cols_batch, int index, int digits, const char* name)
{}

template<class T>
void printDenseMatrixGPU(T** block_ptrs, int* ld_batch, int* rows_batch, int* cols_batch, int index, int digits, const char* name)
{
	thrust::device_ptr<T*> dev_ptr(block_ptrs + index);
	T* dev_ptr_val = *dev_ptr;
	thrust::device_ptr<T> dev_data(dev_ptr_val);

	thrust::device_ptr<int> ld_ptr(ld_batch + index), rows_ptr(rows_batch + index), cols_ptr(cols_batch + index);
	int rows = *rows_ptr, cols = *cols_ptr, ld = *ld_ptr;

	char format[64];
	sprintf(format, "%%.%de + %%.%dei ", digits, digits);
    printf("%s = ([\n", name);
    for(int i = 0; i < rows; i++) 
	{
        for(int j = 0; j < cols; j++)
		{
			T entry = *(dev_data + i + j * ld);
            printf(format, entry.x, entry.y);
		}
        printf(";\n");
    }
    printf("]);\n");
}

template<class Real>
void copyDenseMatrix(
	Real** block_ptrs, int* ld_batch, int* rows_batch, int* cols_batch, int index, 
	thrust::host_vector<Real>& copy_block, int& rows, int& cols
)
{
	thrust::device_ptr<Real*> dev_ptr(block_ptrs + index);
	Real* dev_ptr_val = *dev_ptr;
	thrust::device_ptr<Real> dev_data(dev_ptr_val);
	
	thrust::device_ptr<int> ld_ptr(ld_batch + index), rows_ptr(rows_batch + index), cols_ptr(cols_batch + index);
	int ld = *ld_ptr;
	
	rows = *rows_ptr;
	cols = *cols_ptr;
	copy_block.resize(rows * cols);
    
	for(int i = 0; i < rows; i++) 
        for(int j = 0; j < cols; j++)
			copy_block[i + j * rows] = *(dev_data + i + j * ld);
}

//////////////////////////////////////////////////////////////////////////////////////
template<class T> 
void kblas_ara_batch_wsquery(KBlasWorkspaceState& requested_ws, int bs, int num_ops, int external_only)
{
    typedef typename AraGramType<T>::type GramType;

	gemm_batch_nonuniform_wsquery_core(&requested_ws);
	ara_trsm_batch_wsquery<T>(requested_ws, num_ops, 0);
	
	if(!external_only)
	{
		// Align to sizeof(GramType) bytes
		requested_ws.d_data_bytes = aligned_bytes<GramType>(requested_ws.d_data_bytes);
		
		requested_ws.d_data_bytes += ( 
			num_ops * bs * bs + // G
			num_ops * bs +      // diag_R
			num_ops             // max_diag
		) * sizeof(GramType);	
		
		// Align to sizeof(int) bytes
		requested_ws.d_data_bytes = aligned_bytes<int>(requested_ws.d_data_bytes);
		
		requested_ws.d_data_bytes += (
			num_ops + // block_ranks
			num_ops + // op_samples
			num_ops   // small_vectors
		) * sizeof(int);
		
		// Pointers
        requested_ws.d_data_bytes = aligned_bytes<T*>(requested_ws.d_data_bytes);
		requested_ws.d_ptrs_bytes += num_ops * sizeof(T*); // Y_batch
				
		// Align to sizeof(GramType*) bytes
		requested_ws.d_data_bytes = aligned_bytes<GramType*>(requested_ws.d_data_bytes);
		requested_ws.d_ptrs_bytes += num_ops * sizeof(GramType*); // G_batch
	}
}

///////////////////////////////////////////////////////////////////////////
// Samplers
///////////////////////////////////////////////////////////////////////////
template<class T>
struct DenseSampler 
{
	T **M_batch;
	int *ldm_batch, *rows_batch, *cols_batch;
	kblasHandle_t handle;
	int max_rows, max_cols;
	T one, zero;
    char trans;

	DenseSampler(T **M_batch, int *ldm_batch, int *rows_batch, int *cols_batch, int max_rows, int max_cols, kblasHandle_t handle) 
	{
		this->M_batch = M_batch;
		this->ldm_batch = ldm_batch;
		this->rows_batch = rows_batch;
		this->cols_batch = cols_batch;
		this->max_rows = max_rows;
		this->max_cols = max_cols;
		this->handle = handle;

        this->one = make_one<T>();
        this->zero = make_zero<T>();
        this->trans = AraTransposeVal<T>::val;
	}

	// A = M * B or A = M' * B
	int sample(T** B_batch, int* ldb_batch, int* samples_batch, T** A_batch, int* lda_batch, int max_samples, int num_ops, int transpose) 
	{
		
		if(!transpose)
		{
			check_error( kblas_gemm_batch(
				handle, KBLAS_NoTrans, KBLAS_NoTrans, rows_batch, samples_batch, cols_batch, 
				max_rows, max_samples, max_cols, one, (const T**)M_batch, ldm_batch, 
				(const T**)B_batch, ldb_batch, zero, A_batch, lda_batch, num_ops
			) );
		}
		else
		{
			check_error( kblas_gemm_batch(
				handle, trans, KBLAS_NoTrans, cols_batch, samples_batch, rows_batch, 
				max_cols, max_samples, max_rows, one, (const T**)M_batch, ldm_batch,  
				(const T**)B_batch, ldb_batch, zero, A_batch, lda_batch, num_ops
			) );
		}
		
		return 1;
	}
};

template<class T, typename FuncPtr>
struct FunctionSampler 
{
	FuncPtr func;
	void* data;
	
	FunctionSampler(FuncPtr func, void* data)
	{
		this->func = func;
		this->data = data;
	}
	
	// A = M * B or A = M' * B
	int sample(T** B_batch, int* ldb_batch, int* samples_batch, T** A_batch, int* lda_batch, int max_samples, int num_ops, int transpose) 
	{
		return func(data, B_batch, ldb_batch, samples_batch, A_batch, lda_batch, max_samples, num_ops, transpose);
	}
};

///////////////////////////////////////////////////////////////////////////
template<class T, class Sampler>
int kblas_ara_batch_template(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, Sampler& sampler, 
	T** A_batch, int* lda_batch, T** B_batch, int* ldb_batch, int* ranks_batch, 
	typename kblas_base_type<T>::mag_type tol, int max_rows, int max_cols, 
    int *max_rank_batch, int bs, int r, int *info, kblasRandState_t rand_state, 
    int relative, int num_ops,
    const typename kblas_base_type<T>::mag_type* tol_batch = nullptr
)
{
    if (num_ops == 0) return KBLAS_Success;
    typedef typename AraGramType<T>::type GramType;
	
    const char trans = AraTransposeVal<T>::val;
    const double tolerance_scale = 1; //7.978845608028654; // (10 * sqrt(2 / pi))
	tol *= tolerance_scale;

	int rank = 0;
    T one = make_one<T>(), zero = make_zero<T>(), minus_one = -make_one<T>();
    GramType g_one = make_one<GramType>();

	T** Y_batch;
	GramType* diag_R, *max_diag;
	GramType* G_strided;
	GramType** G_batch_mp;
	int* block_ranks, *op_samples, *small_vectors;
	
	///////////////////////////////////////////////////////////////////////////
	// Workspace allocation
	///////////////////////////////////////////////////////////////////////////
	KBlasWorkspaceState external_ws, total_ws;
	kblas_ara_batch_wsquery<T>(external_ws, bs, num_ops, 1);
	kblas_ara_batch_wsquery<T>(total_ws, bs, num_ops, 0);
	
	KBlasWorkspaceState available_ws = handle->work_space.getAvailable();
	
	if(!total_ws.isSufficient(&available_ws))
		return KBLAS_InsufficientWorkspace;
	
	// Align workspace to sizeof(GramType) bytes
	external_ws.d_data_bytes = aligned_bytes<GramType>(external_ws.d_data_bytes);
	
	G_strided = (GramType*)((KBlasWorkspace::WS_Byte*)handle->work_space.d_data + external_ws.d_data_bytes);
	diag_R    = G_strided + num_ops * bs * bs;
	max_diag  = diag_R + num_ops * bs;
	
	external_ws.d_data_bytes += ( 
			num_ops * bs * bs + // G
			num_ops * bs + // diag_R
			num_ops        // max_diag
	) * sizeof(GramType);

	// Align to sizeof(int) bytes
	external_ws.d_data_bytes = aligned_bytes<int>(external_ws.d_data_bytes);

	block_ranks = (int*)((KBlasWorkspace::WS_Byte*)handle->work_space.d_data + external_ws.d_data_bytes);
	op_samples = block_ranks + num_ops;
	small_vectors = op_samples + num_ops;

	// Align to sizeof(T*) bytes
	external_ws.d_data_bytes = aligned_bytes<T*>(external_ws.d_data_bytes);
	Y_batch = (T**)((KBlasWorkspace::WS_Byte*)handle->work_space.d_ptrs + external_ws.d_ptrs_bytes);	

	external_ws.d_ptrs_bytes += num_ops * sizeof(T*); // Y_batch
	
	// Align to sizeof(GramType*) bytes
	external_ws.d_data_bytes = aligned_bytes<GramType*>(external_ws.d_data_bytes);
	G_batch_mp = (GramType**)((KBlasWorkspace::WS_Byte*)handle->work_space.d_ptrs + external_ws.d_ptrs_bytes);	

	///////////////////////////////////////////////////////////////////////////
	// Initializations
	///////////////////////////////////////////////////////////////////////////
	cudaStream_t kblas_stream = kblasGetStream(handle);
	
	// Initialize operation data
	fillArray(max_diag, num_ops, -g_one, kblas_stream);
	fillArray(small_vectors, num_ops, 0, kblas_stream);
	fillArray(ranks_batch, num_ops, 0, kblas_stream);
	fillArray(info, num_ops, KBLAS_ARA_NoConvergence, kblas_stream);

	// Copy over A to Y so we can advance Y 
	copyGPUArray(A_batch, Y_batch, num_ops, kblas_stream);
	
	// Generate array of pointers from strided data
	generateArrayOfPointers(G_strided, G_batch_mp, bs * bs, 0, num_ops, kblas_stream);

	///////////////////////////////////////////////////////////////////////////
	// Main Loop
	///////////////////////////////////////////////////////////////////////////
	while(true)
	{
		// Set the op samples to 0 if the operation has converged
		int max_samples = kblas_ara_batch_set_samples(
			op_samples, small_vectors, ranks_batch, bs,
			max_rank_batch, r, num_ops, kblas_stream
		);
		
		if(max_samples == 0) break;
		
		// Generate random matrices Omega stored in B
		// Omega = randn(n, samples)
		T** Omega = B_batch;
		check_error( kblas_rand_batch(
			handle, cols_batch, op_samples, Omega, ldb_batch, 
			max_cols, rand_state, num_ops
		) );
		
        // printDenseMatrixGPU(Omega, ldb_batch, cols_batch, op_samples, 0, 16, "Omega");
		
		// Take samples and store them in A
		// Y = M * Omega
		sampler.sample(
			Omega, ldb_batch, op_samples, 
			Y_batch, lda_batch, max_samples, num_ops, 0
		);

		// Set diag(R) = 1
		fillArray(diag_R, num_ops * bs, g_one, kblas_stream);
	    copyGPUArray(op_samples, block_ranks, num_ops, kblas_stream);

		// Block CGS with one reorthogonalization step
		for(int i = 0; i < 2; i++)
		{
			// Project samples
			// Y = Y - Q * (Q' * Y) = Y - Q * Z
			// Store Z = Q' * Y in B
			T** Z_batch = B_batch, **Q_batch = A_batch;
			check_error( kblas_gemm_batch(
				handle, trans, KBLAS_NoTrans, ranks_batch, block_ranks, rows_batch, 
				rank, max_samples, max_rows, one, (const T**)Q_batch, lda_batch,  
				(const T**)Y_batch, lda_batch, zero, Z_batch, ldb_batch, num_ops
			) );
			check_error( kblas_gemm_batch(
				handle, KBLAS_NoTrans, KBLAS_NoTrans, rows_batch, block_ranks, ranks_batch, 
				max_rows, max_samples, rank, minus_one, (const T**)Q_batch, lda_batch,  
				(const T**)Z_batch, ldb_batch, one, Y_batch, lda_batch, num_ops
			) );

            // printDenseMatrixGPU(Y_batch, lda_batch, rows_batch, op_samples, 0, 16, "Y");

			// Pivoted panel orthogonalization using syrk+pstrf+trsm
			// Compute G = A'*A in mixed precision
			T **R_batch = B_batch;
			check_error( kblas_ara_mp_syrk_batch_template(
				handle, rows_batch, block_ranks, max_rows, max_samples, 
				(const T**)Y_batch, lda_batch, G_batch_mp, block_ranks, num_ops
			) );

            // printDenseMatrixGPU(G_batch_mp, op_samples, op_samples, op_samples, 0, 16, "G");

			check_error( kblas_ara_fused_potrf_batch_template( 
				block_ranks, G_batch_mp, block_ranks, R_batch, ldb_batch, diag_R, bs, 
				block_ranks, num_ops, kblas_stream
			) );

            // printDenseMatrixGPU(R_batch, ldb_batch, op_samples, op_samples, 0, 16, "R");

			/*check_error( kblas_gemm_batch(
				handle, KBLAS_Trans, KBLAS_NoTrans, op_samples, op_samples, rows_batch, 
				samples, samples, max_rows, 1, (const Real**)Y_batch, lda_batch, 
				(const Real**)Y_batch, lda_batch, 0, R_batch, ldb_batch, num_ops
			) ); 
			check_error( kblas_ara_fused_potrf_batch( 
				op_samples, R_batch, ldb_batch, R_batch, ldb_batch, diag_R, bs, 
				block_ranks, num_ops, kblas_stream
			) );
			*/
			// printDenseMatrixGPU(G_batch, ldb_batch, block_ranks, block_ranks, 0, 8, "R");
			
			check_error( kblas_ara_trsm_batch_template(
				handle, Y_batch, lda_batch, R_batch, ldb_batch, rows_batch, 
                block_ranks, num_ops, max_rows, bs
			) );

            // printDenseMatrixGPU(Y_batch, lda_batch, rows_batch, op_samples, 0, 16, "Ys");
		}
        
		// Count the number of vectors that have a small magnitude
		// also updates the rank, max diagonal and advances the Y_batch pointers
		check_error( kblas_ara_svec_count_batch_template(
			diag_R, bs, op_samples, ranks_batch, max_diag, Y_batch, lda_batch, 
			tol, r, small_vectors, info, relative, num_ops, kblas_stream, tol_batch
		) );
		
		// Advance the rank
		rank += max_samples;
	}
	
    int max_rank = thrust::reduce(
		thrust::cuda::par.on(kblas_stream),
		ranks_batch, ranks_batch + num_ops, 0,
        thrust::maximum<int>()
	);

	// Finally, B = M' * A
	sampler.sample(
		A_batch, lda_batch, ranks_batch, 
		B_batch, ldb_batch, max_rank, num_ops, 1
	);
    
	// printDenseMatrixGPU(A_batch, lda_batch, rows_batch, ranks_batch, 0, 16, "A");
	// printDenseMatrixGPU(B_batch, ldb_batch, cols_batch, ranks_batch, 0, 16, "B");
	return KBLAS_Success;
}

//////////////////////////////////////////////////////////////
// Wrappers
//////////////////////////////////////////////////////////////
int kblas_sara_batch(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, float** M_batch, int* ldm_batch, 
	float** A_batch, int* lda_batch, float** B_batch, int* ldb_batch, int* ranks_batch, 
	float tol, int max_rows, int max_cols, int *max_rank_batch, int bs, int r, int *info, kblasRandState_t rand_state, 
	int relative, int num_ops
)
{
	DenseSampler<float> dense_sampler(M_batch, ldm_batch, rows_batch, cols_batch, max_rows, max_cols, handle);
	
	return kblas_ara_batch_template<float, DenseSampler<float> >(
		handle, rows_batch, cols_batch, dense_sampler, A_batch, lda_batch, B_batch, ldb_batch, 
		ranks_batch, tol, max_rows, max_cols, max_rank_batch, bs, r, info, rand_state, relative, num_ops
	);
}

int kblas_dara_batch(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, double** M_batch, int* ldm_batch, 
	double** A_batch, int* lda_batch, double** B_batch, int* ldb_batch, int* ranks_batch, 
	double tol, int max_rows, int max_cols, int *max_rank_batch, int bs, int r, int *info, kblasRandState_t rand_state, 
	int relative, int num_ops
)
{
	DenseSampler<double> dense_sampler(M_batch, ldm_batch, rows_batch, cols_batch, max_rows, max_cols, handle);
	
	return kblas_ara_batch_template<double, DenseSampler<double> >(
		handle, rows_batch, cols_batch, dense_sampler, A_batch, lda_batch, B_batch, ldb_batch, 
		ranks_batch, tol, max_rows, max_cols, max_rank_batch, bs, r, info, rand_state, relative, num_ops
	);
}

int kblas_cara_batch(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, cuComplex** M_batch, int* ldm_batch, 
	cuComplex** A_batch, int* lda_batch, cuComplex** B_batch, int* ldb_batch, int* ranks_batch, 
	float tol, int max_rows, int max_cols, int *max_rank_batch, int bs, int r, int *info, kblasRandState_t rand_state, 
	int relative, int num_ops
)
{
	DenseSampler<cuComplex> dense_sampler(M_batch, ldm_batch, rows_batch, cols_batch, max_rows, max_cols, handle);

    // printDenseMatrixGPU(M_batch, ldm_batch, rows_batch, cols_batch, 0, 8, "M");

	return kblas_ara_batch_template<cuComplex, DenseSampler<cuComplex> >(
		handle, rows_batch, cols_batch, dense_sampler, A_batch, lda_batch, B_batch, ldb_batch, 
		ranks_batch, tol, max_rows, max_cols, max_rank_batch, bs, r, info, rand_state, relative, num_ops
	);
}

int kblas_zara_batch(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, cuDoubleComplex** M_batch, int* ldm_batch, 
	cuDoubleComplex** A_batch, int* lda_batch, cuDoubleComplex** B_batch, int* ldb_batch, int* ranks_batch, 
	double tol, int max_rows, int max_cols, int *max_rank_batch, int bs, int r, int *info, kblasRandState_t rand_state, 
	int relative, int num_ops
)
{
	DenseSampler<cuDoubleComplex> dense_sampler(M_batch, ldm_batch, rows_batch, cols_batch, max_rows, max_cols, handle);
	
    // printDenseMatrixGPU(M_batch, ldm_batch, rows_batch, cols_batch, 0, 16, "M");

	return kblas_ara_batch_template<cuDoubleComplex, DenseSampler<cuDoubleComplex> >(
		handle, rows_batch, cols_batch, dense_sampler, A_batch, lda_batch, B_batch, ldb_batch, 
		ranks_batch, tol, max_rows, max_cols, max_rank_batch, bs, r, info, rand_state, relative, num_ops
	);
}

int kblas_sara_batch_tol(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, float** M_batch, int* ldm_batch,
	float** A_batch, int* lda_batch, float** B_batch, int* ldb_batch, int* ranks_batch,
	const float* tol_batch, int max_rows, int max_cols, int *max_rank_batch, int bs, int r, int *info, kblasRandState_t rand_state,
	int relative, int num_ops
)
{
    if (num_ops < 0 || (num_ops > 0 && !tol_batch))
        return KBLAS_Error_WrongInput;
    if (num_ops == 0) return KBLAS_Success;
	DenseSampler<float> dense_sampler(M_batch, ldm_batch, rows_batch, cols_batch, max_rows, max_cols, handle);

	return kblas_ara_batch_template<float, DenseSampler<float> >(
		handle, rows_batch, cols_batch, dense_sampler, A_batch, lda_batch, B_batch, ldb_batch,
		ranks_batch, 0, max_rows, max_cols, max_rank_batch, bs, r, info, rand_state, relative, num_ops, tol_batch
	);
}

int kblas_dara_batch_tol(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, double** M_batch, int* ldm_batch,
	double** A_batch, int* lda_batch, double** B_batch, int* ldb_batch, int* ranks_batch,
	const double* tol_batch, int max_rows, int max_cols, int *max_rank_batch, int bs, int r, int *info, kblasRandState_t rand_state,
	int relative, int num_ops
)
{
    if (num_ops < 0 || (num_ops > 0 && !tol_batch))
        return KBLAS_Error_WrongInput;
    if (num_ops == 0) return KBLAS_Success;
	DenseSampler<double> dense_sampler(M_batch, ldm_batch, rows_batch, cols_batch, max_rows, max_cols, handle);

	return kblas_ara_batch_template<double, DenseSampler<double> >(
		handle, rows_batch, cols_batch, dense_sampler, A_batch, lda_batch, B_batch, ldb_batch,
		ranks_batch, 0, max_rows, max_cols, max_rank_batch, bs, r, info, rand_state, relative, num_ops, tol_batch
	);
}

int kblas_cara_batch_tol(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, cuComplex** M_batch, int* ldm_batch,
	cuComplex** A_batch, int* lda_batch, cuComplex** B_batch, int* ldb_batch, int* ranks_batch,
	const float* tol_batch, int max_rows, int max_cols, int *max_rank_batch, int bs, int r, int *info, kblasRandState_t rand_state,
	int relative, int num_ops
)
{
    if (num_ops < 0 || (num_ops > 0 && !tol_batch))
        return KBLAS_Error_WrongInput;
    if (num_ops == 0) return KBLAS_Success;
	DenseSampler<cuComplex> dense_sampler(M_batch, ldm_batch, rows_batch, cols_batch, max_rows, max_cols, handle);

    // printDenseMatrixGPU(M_batch, ldm_batch, rows_batch, cols_batch, 0, 8, "M");

	return kblas_ara_batch_template<cuComplex, DenseSampler<cuComplex> >(
		handle, rows_batch, cols_batch, dense_sampler, A_batch, lda_batch, B_batch, ldb_batch,
		ranks_batch, 0, max_rows, max_cols, max_rank_batch, bs, r, info, rand_state, relative, num_ops, tol_batch
	);
}

int kblas_zara_batch_tol(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, cuDoubleComplex** M_batch, int* ldm_batch,
	cuDoubleComplex** A_batch, int* lda_batch, cuDoubleComplex** B_batch, int* ldb_batch, int* ranks_batch,
	const double* tol_batch, int max_rows, int max_cols, int *max_rank_batch, int bs, int r, int *info, kblasRandState_t rand_state,
	int relative, int num_ops
)
{
    if (num_ops < 0 || (num_ops > 0 && !tol_batch))
        return KBLAS_Error_WrongInput;
    if (num_ops == 0) return KBLAS_Success;
	DenseSampler<cuDoubleComplex> dense_sampler(M_batch, ldm_batch, rows_batch, cols_batch, max_rows, max_cols, handle);

    // printDenseMatrixGPU(M_batch, ldm_batch, rows_batch, cols_batch, 0, 16, "M");

	return kblas_ara_batch_template<cuDoubleComplex, DenseSampler<cuDoubleComplex> >(
		handle, rows_batch, cols_batch, dense_sampler, A_batch, lda_batch, B_batch, ldb_batch,
		ranks_batch, 0, max_rows, max_cols, max_rank_batch, bs, r, info, rand_state, relative, num_ops, tol_batch
	);
}

int kblas_sara_batch_fn(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, sara_sampler sampler, void* sampler_data, 
	float** A_batch, int* lda_batch, float** B_batch, int* ldb_batch, int* ranks_batch, 
	float tol, int max_rows, int max_cols, int *max_rank_batch, int bs, int r, int *info, kblasRandState_t rand_state, 
	int relative, int num_ops
)
{
	FunctionSampler<float, sara_sampler> func_sampler(sampler, sampler_data);
		
	return kblas_ara_batch_template<float, FunctionSampler<float, sara_sampler> >(
		handle, rows_batch, cols_batch, func_sampler, A_batch, lda_batch, B_batch, ldb_batch, 
		ranks_batch, tol, max_rows, max_cols, max_rank_batch, bs, r, info, rand_state, relative, num_ops
	);
}

int kblas_dara_batch_fn(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, dara_sampler sampler, void* sampler_data, 
	double** A_batch, int* lda_batch, double** B_batch, int* ldb_batch, int* ranks_batch, 
	double tol, int max_rows, int max_cols, int *max_rank_batch, int bs, int r, int *info, kblasRandState_t rand_state, 
	int relative, int num_ops
)
{
	FunctionSampler<double, dara_sampler> func_sampler(sampler, sampler_data);
		
	return kblas_ara_batch_template<double, FunctionSampler<double, dara_sampler> >(
		handle, rows_batch, cols_batch, func_sampler, A_batch, lda_batch, B_batch, ldb_batch, 
		ranks_batch, tol, max_rows, max_cols, max_rank_batch, bs, r, info, rand_state, relative, num_ops
	);
}

int kblas_cara_batch_fn(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, cara_sampler sampler, void* sampler_data, 
	cuComplex** A_batch, int* lda_batch, cuComplex** B_batch, int* ldb_batch, int* ranks_batch, 
	float tol, int max_rows, int max_cols, int *max_rank_batch, int bs, int r, int *info, kblasRandState_t rand_state, 
	int relative, int num_ops
)
{
	FunctionSampler<cuComplex, cara_sampler> func_sampler(sampler, sampler_data);
		
	return kblas_ara_batch_template<cuComplex, FunctionSampler<cuComplex, cara_sampler> >(
		handle, rows_batch, cols_batch, func_sampler, A_batch, lda_batch, B_batch, ldb_batch, 
		ranks_batch, tol, max_rows, max_cols, max_rank_batch, bs, r, info, rand_state, relative, num_ops
	);
}

int kblas_zara_batch_fn(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, zara_sampler sampler, void* sampler_data, 
	cuDoubleComplex** A_batch, int* lda_batch, cuDoubleComplex** B_batch, int* ldb_batch, int* ranks_batch, 
	double tol, int max_rows, int max_cols, int *max_rank_batch, int bs, int r, int *info, kblasRandState_t rand_state, 
	int relative, int num_ops
)
{
	FunctionSampler<cuDoubleComplex, zara_sampler> func_sampler(sampler, sampler_data);
		
	return kblas_ara_batch_template<cuDoubleComplex, FunctionSampler<cuDoubleComplex, zara_sampler> >(
		handle, rows_batch, cols_batch, func_sampler, A_batch, lda_batch, B_batch, ldb_batch, 
		ranks_batch, tol, max_rows, max_cols, max_rank_batch, bs, r, info, rand_state, relative, num_ops
	);
}

//////////////////////////////////////////////////////////////
// Workspace queries
//////////////////////////////////////////////////////////////
void kblasSara_batch_wsquery(kblasHandle_t handle, int bs, int num_ops)
{
	KBlasWorkspaceState ws;
	kblas_ara_batch_wsquery<float>(ws, bs, num_ops, 0);
	handle->work_space.requested_ws_state.pad(&ws);
}

void kblasDara_batch_wsquery(kblasHandle_t handle, int bs, int num_ops)
{
	KBlasWorkspaceState ws;
	kblas_ara_batch_wsquery<double>(ws, bs, num_ops, 0);
	handle->work_space.requested_ws_state.pad(&ws);
}

void kblasCara_batch_wsquery(kblasHandle_t handle, int bs, int num_ops)
{
	KBlasWorkspaceState ws;
	kblas_ara_batch_wsquery<cuComplex>(ws, bs, num_ops, 0);
	handle->work_space.requested_ws_state.pad(&ws);
}

void kblasZara_batch_wsquery(kblasHandle_t handle, int bs, int num_ops)
{
	KBlasWorkspaceState ws;
	kblas_ara_batch_wsquery<cuDoubleComplex>(ws, bs, num_ops, 0);
	handle->work_space.requested_ws_state.pad(&ws);
}

void kblasSara_trsm_batch_wsquery(kblasHandle_t handle, int num_ops)
{
	KBlasWorkspaceState ws;
	ara_trsm_batch_wsquery<float>(ws, num_ops, 0);
	handle->work_space.requested_ws_state.pad(&ws);
}

void kblasDara_trsm_batch_wsquery(kblasHandle_t handle, int num_ops)
{
	KBlasWorkspaceState ws;
	ara_trsm_batch_wsquery<double>(ws, num_ops, 0);
	handle->work_space.requested_ws_state.pad(&ws);
}

void kblasCara_trsm_batch_wsquery(kblasHandle_t handle, int num_ops)
{
	KBlasWorkspaceState ws;
	ara_trsm_batch_wsquery<cuComplex>(ws, num_ops, 0);
	handle->work_space.requested_ws_state.pad(&ws);
}

void kblasZara_trsm_batch_wsquery(kblasHandle_t handle, int num_ops)
{
	KBlasWorkspaceState ws;
	ara_trsm_batch_wsquery<cuDoubleComplex>(ws, num_ops, 0);
	handle->work_space.requested_ws_state.pad(&ws);
}

//////////////////////////////////////////////////////////////
// utility wrappers
//////////////////////////////////////////////////////////////
int kblas_sara_svec_count_batch(
	double* diag_R, int bs, int* op_samples, int* ranks_batch, double* max_diag, float** Y_batch, 
	int* ldy_batch, float tol, int r, int* small_vectors, int *info, int relative, int num_ops, cudaStream_t stream
)
{
	return kblas_ara_svec_count_batch_template(
		diag_R, bs, op_samples, ranks_batch, max_diag, Y_batch, ldy_batch, 
		tol, r, small_vectors, info, relative, num_ops, stream
	);
}

int kblas_dara_svec_count_batch(
	double* diag_R, int bs, int* op_samples, int* ranks_batch, double* max_diag, double** Y_batch, 
	int* ldy_batch, double tol, int r, int* small_vectors, int *info, int relative, int num_ops, cudaStream_t stream
)
{
	return kblas_ara_svec_count_batch_template(
		diag_R, bs, op_samples, ranks_batch, max_diag, Y_batch, ldy_batch, 
		tol, r, small_vectors, info, relative, num_ops, stream
	);	
}

int kblas_sara_trsm_batch(
	kblasHandle_t handle, float** B_batch, int* ldb_batch, float** A_batch, int* lda_batch, 
	int* rows_batch, int* cols_batch, int num_ops, int max_rows, int bs
)
{
	return kblas_ara_trsm_batch_template(
		handle, B_batch, ldb_batch, A_batch, lda_batch, 
		rows_batch, cols_batch, num_ops, max_rows, bs
	);
}

int kblas_dara_trsm_batch(
	kblasHandle_t handle, double** B_batch, int* ldb_batch, double** A_batch, int* lda_batch, 
	int* rows_batch, int* cols_batch, int num_ops, int max_rows, int bs
)
{
	return kblas_ara_trsm_batch_template(
		handle, B_batch, ldb_batch, A_batch, lda_batch, 
		rows_batch, cols_batch, num_ops, max_rows, bs
	);
}

int kblas_sara_fused_potrf_batch( 
	int* op_samples, double** A_batch, int* lda_batch, float** R_batch, int* ldr_batch, 
	double* diag_R, int bs, int* block_ranks, int num_ops, cudaStream_t stream
)
{
	return kblas_ara_fused_potrf_batch_template(
		op_samples, A_batch, lda_batch, R_batch, ldr_batch, 
		diag_R, bs, block_ranks, num_ops, stream
	);
}

int kblas_dara_fused_potrf_batch( 
	int* op_samples, double** A_batch, int* lda_batch, double** R_batch, int* ldr_batch, 
	double* diag_R, int bs, int* block_ranks, int num_ops, cudaStream_t stream
)
{
	return kblas_ara_fused_potrf_batch_template(
		op_samples, A_batch, lda_batch, R_batch, ldr_batch, 
		diag_R, bs, block_ranks, num_ops, stream
	);	
}

int kblas_sara_mp_syrk_batch(
	kblasHandle_t handle, int* m, int* n, int max_m, int max_n, 
	const float** A, int* lda, double** B, int* ldb, int batchCount 
)
{
	return kblas_ara_mp_syrk_batch_template(
		handle, m, n, max_m, max_n, A, lda, B, ldb, batchCount 
	);
}

int kblas_dara_mp_syrk_batch(
	kblasHandle_t handle, int* m, int* n, int max_m, int max_n, 
	const double** A, int* lda, double** B, int* ldb, int batchCount 
)
{
	return kblas_ara_mp_syrk_batch_template(
		handle, m, n, max_m, max_n, A, lda, B, ldb, batchCount 
	);
}
