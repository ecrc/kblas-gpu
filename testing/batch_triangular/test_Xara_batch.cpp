#include <stdio.h>
#include <math.h>
#include <string.h>
#include <algorithm>

#include <cublas_v2.h>
#include "kblas.h"

#include "testing_helper.h"
#include "batch_rand.h"
#include "batch_pstrf.h"
#include "batch_block_copy.h"
#include "batch_ara.h"

#ifdef PREC_d
typedef double Real;
#define get_max_error					dget_max_error 
#define generateArrayOfPointers			generateDArrayOfPointers
#define generateArrayOfPointersHost		generateDArrayOfPointersHost
#define generate_randomMatricesArray	generateDrandomMatricesArray
#define ara_sampler						dara_sampler
#define generate_singular_values		generateDsingular_values
#define ARA_TOLERANCE					1e-10
#define cublasXcopy                             cublasDcopy
#define kblasXgemm_batch_strided                kblasDgemm_batch_strided
#else
typedef float Real;
#define get_max_error					sget_max_error
#define generateArrayOfPointers			generateSArrayOfPointers
#define generateArrayOfPointersHost		generateSArrayOfPointersHost
#define generate_randomMatricesArray	generateSrandomMatricesArray
#define ara_sampler						sara_sampler
#define generate_singular_values		generateSsingular_values
#define ARA_TOLERANCE					1e-10
#define cublasXcopy                             cublasScopy
#define kblasXgemm_batch_strided                kblasSgemm_batch_strided
#endif

#define FLOPS_ARA(m, n, k, r)  (2 * (m) * (n) * ((r) + (k)) + (m) * (k) * (3 + 4 * (r) + 4 * (k)))

typedef Real*		RealArray;
typedef int*		IntArray;
typedef Real**		RealPtrArray;

#define SYNC_TIMERS(timers, run_time) \
	max_time = -1;	\
	for(int g = 0; g < num_gpus; g++) \
	{ \
		cudaSetDevice(opts.devices[g]); \
		double gpu_time = gpuTimerToc(timers[g]); \
		if(max_time < gpu_time)	\
			max_time = gpu_time; \
	} \
	run_time[i] = max_time;

#define COPY_DATA_UP(d_A, h_A, el_size, el_type)  \
	for(int g = 0; g < num_gpus; g++) \
	{	\
		cudaSetDevice(opts.devices[g]); \
		int entries = batchCount_gpu * (el_size); \
		check_error( cudaMemcpy((d_A)[g], (h_A) + entries * g, entries * sizeof(el_type), cudaMemcpyHostToDevice) ); \
	}	\
	syncGPUs(&opts);

#define COPY_DATA_DOWN(h_A, d_A, el_size, el_type) \
	for(int g = 0; g < num_gpus; g++)	\
	{	\
		cudaSetDevice(opts.devices[g]);	\
		int entries = batchCount_gpu * (el_size); \
		check_error( cudaMemcpy((h_A) + entries * g, (d_A)[g], entries * sizeof(el_type), cudaMemcpyDeviceToHost ) );	\
	}	\
	syncGPUs(&opts);
	
void syncGPUs(kblas_opts* opts)
{
	for(int g = 0; g < opts->ngpu; g++)
	{
		cudaSetDevice(opts->devices[g]);
		cudaDeviceSynchronize();
	}
}

void printResults(RealPtrArray A_batch, int *lda_batch, int* rows_batch, int* cols_batch, int batchCount, char label)
{
	for(int i = 0; i < batchCount; i++)
	{
		printf("%c = [\n", label);
		printMatrix(rows_batch[i], cols_batch[i], A_batch[i], lda_batch[i], stdout);
		printf("];\n");
	}
}

template<class T>
void printGpuArray(T* dev_data, int num_elements, const char* format)
{
	if(num_elements == 0) return;
	
	T* host_data;
	TESTING_MALLOC_CPU(host_data, T, num_elements);
	
	check_error( cudaMemcpy(host_data, dev_data, num_elements * sizeof(T), cudaMemcpyDeviceToHost ) );
	
	printf("================\n");
	for(int i = 0; i < num_elements; i++)
		printf(format, host_data[i]);
	printf("\n================\n");
	
	TESTING_FREE_CPU(host_data);
}

void cblas_gemm(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const int m, const int n, const int k, const float alpha, const float *a, const int lda, const float *b, const int ldb, const float beta, float *c, const int ldc)
{ cblas_sgemm(CblasColMajor, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }

void cblas_gemm(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const int m, const int n, const int k, const double alpha, const double *a, const int lda, const double *b, const int ldb, const double beta, double *c, const int ldc)
{ cblas_dgemm(CblasColMajor, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }

void testResults(
	RealPtrArray M_batch, int *ldm_batch, RealPtrArray A_batch, int *lda_batch, 
	RealPtrArray B_batch, int *ldb_batch, int* rows_batch, int* cols_batch, 
	int* ranks_batch, int* sval_ranks, int batchCount,
	double& avgError, double& avgRankDiff, double& sdevRankDiff
)
{
	double* rank_diff;
	TESTING_MALLOC_CPU(rank_diff, double, batchCount);
	
	Real total_diff = 0;
	for(int i = 0; i < batchCount; i++)
	{
		Real* M = M_batch[i], *A = A_batch[i], *B = B_batch[i];
		int ldm = ldm_batch[i], lda = lda_batch[i], ldb = ldb_batch[i];
		int rows = rows_batch[i], cols = cols_batch[i], rank = ranks_batch[i];
		
		Real* temp_M;
		TESTING_MALLOC_CPU(temp_M, Real, rows * cols);
		
		Real m_normf = 0;
		for(int r = 0; r < rows; r++)
		{			
			for(int c = 0; c < cols; c++) 
			{
				temp_M[r + c * rows] = M[r + c * ldm];
				m_normf += temp_M[r + c * rows] * temp_M[r + c * rows];
			}
		}
		
		cblas_gemm(
			CblasNoTrans, CblasTrans, rows, cols, rank, 
			1, A, lda, B, ldb, -1, temp_M, rows
		);
		
		Real diff_normf = 0;
		for(int r = 0; r < rows; r++) 
			for(int c = 0; c < cols; c++) 
				diff_normf += temp_M[r + c * rows] * temp_M[r + c * rows];
		
		TESTING_FREE_CPU(temp_M);
		total_diff += sqrt(diff_normf / m_normf);
		rank_diff[i] = abs(rank - sval_ranks[i]);
		//if(sqrt(diff_normf / m_normf) < 1e-4)
		// 	printf("Operation %d had diff %e and rank %d (%d real)\n", i, sqrt(diff_normf / m_normf), rank, sval_ranks[i]);
		
	}
	
	// printf("Ranks = [\n");
	// for(int i = 0; i < batchCount; i++)
	//	printf("%d %d;\n", ranks_batch[i], sval_ranks[i]);
	// printf("];\n");
	
	avgError = total_diff / batchCount;
	avg_and_stdev(rank_diff, batchCount, &avgRankDiff, &sdevRankDiff, 0);
	
	TESTING_FREE_CPU(rank_diff);
}

double get_ara_batch_gflops(int* rows_batch, int* cols_batch, int* ranks_batch, int r, int batchCount)
{
	double total_gflops = 0;
	
	for(int i = 0; i < batchCount; i++)
	{
		double op_flops = FLOPS_ARA(rows_batch[i], cols_batch[i], ranks_batch[i], r);
		total_gflops += op_flops * 1e-9;
	}
	
	return total_gflops;
}

void set_sval_ranks(Real** svals_ptrs, int* cols_batch, int batchCount, int* sval_ranks, Real tol)
{
	for(int i = 0; i < batchCount; i++)
	{
		Real* svals = svals_ptrs[i];
		int cols = cols_batch[i];
		Real rel_tol = svals[0] * tol;
		int rank = 0;

		while(svals[rank] >= rel_tol && rank < cols)
			rank++;
		sval_ranks[i] = rank;
	}
}

struct LRSampler 
{
	LRSampler(
		kblasHandle_t handle,  int *rows_batch, int *cols_batch, int *ranks_batch, 
		int max_rows, int max_cols, int max_rank,
		Real** A_batch, int *lda_batch, Real** B_batch, int* ldb_batch, 
		Real** workspace_batch, int* ldws_batch
	)
	{
		this->handle = handle;
		
		this->rows_batch = rows_batch; 
		this->cols_batch = cols_batch; 
		this->ranks_batch = ranks_batch;  
		
		this->max_rank = max_rank;
		this->max_rows = max_rows;
		this->max_cols = max_cols;
		
		this->A_batch = A_batch; 
		this->lda_batch = lda_batch;
		this->B_batch = B_batch; 
		this->ldb_batch = ldb_batch; 
		this->workspace_batch = workspace_batch;
		this->ldws_batch = ldws_batch;
	}
	
	LRSampler() {}
	
	kblasHandle_t handle;
	Real** A_batch, **B_batch, **workspace_batch;
	int *lda_batch, *ldb_batch, *ldws_batch;
	int *rows_batch, *cols_batch, *ranks_batch;
	int max_rank, max_rows, max_cols;
};

int low_rank_sample(void* data, Real** input_batch, int* ldinput_batch, int* samples_batch, Real** output_batch, int* ldoutput_batch, int max_samples, int num_ops, int transpose)
{
	LRSampler* sampler = (LRSampler*)data;

	if(!transpose)
	{
		// W = B^T * Input
		check_kblas_error( kblas_gemm_batch(
			sampler->handle, KBLAS_Trans, KBLAS_NoTrans, sampler->ranks_batch, samples_batch, sampler->cols_batch, 
			sampler->max_rank, max_samples, sampler->max_cols, 1, (const Real**)sampler->B_batch, sampler->ldb_batch,  
			(const Real**)input_batch, ldinput_batch, 0, sampler->workspace_batch, sampler->ldws_batch, num_ops
		) );
		
		// Output = A * W
		check_kblas_error( kblas_gemm_batch(
			sampler->handle, KBLAS_NoTrans, KBLAS_NoTrans, sampler->rows_batch, samples_batch, sampler->ranks_batch, 
			sampler->max_rows, max_samples, sampler->max_rank, 1, (const Real**)sampler->A_batch, sampler->lda_batch,  
			(const Real**)sampler->workspace_batch, sampler->ldws_batch, 0, output_batch, ldoutput_batch, num_ops
		) );
	}
	else
	{
		// W = A^T * Input
		check_kblas_error( kblas_gemm_batch(
			sampler->handle, KBLAS_Trans, KBLAS_NoTrans, sampler->ranks_batch, samples_batch, sampler->rows_batch, 
			sampler->max_rank, max_samples, sampler->max_rows, 1, (const Real**)sampler->A_batch, sampler->lda_batch,  
			(const Real**)input_batch, ldinput_batch, 0, sampler->workspace_batch, sampler->ldws_batch, num_ops
		) );
		// Output = B * W
		check_kblas_error( kblas_gemm_batch(
			sampler->handle, KBLAS_NoTrans, KBLAS_NoTrans, sampler->cols_batch, samples_batch, sampler->ranks_batch, 
			sampler->max_cols, max_samples, sampler->max_rank, 1, (const Real**)sampler->B_batch, sampler->ldb_batch,  
			(const Real**)sampler->workspace_batch, sampler->ldws_batch, 0, output_batch, ldoutput_batch, num_ops
		) );
	}
	
	return 1;
	
}

int main(int argc, char** argv)
{
	kblas_opts opts;
	parse_opts(argc, argv, &opts);
	int num_gpus = opts.ngpu;
	int max_rows, max_cols, max_rank;
	int batchCount, batchCount_gpu;
	int nruns = opts.nruns;
	int warmup = opts.warmup;
	int decay_rank;
	
	double max_time;
	
	// Host stuff
	RealArray h_M, h_A, h_B, h_svals;
	RealPtrArray h_M_ptrs, h_A_ptrs, h_B_ptrs, h_svals_ptrs;
	IntArray h_rows_batch, h_cols_batch, h_ranks, h_sval_ranks; 
	IntArray h_ldm_batch, h_lda_batch, h_ldb_batch;
	
	Real tol = ARA_TOLERANCE, tol2 = tol * 100;
	const int BLOCK_SIZE = 32, ARA_R = 10;
	const Real ARA_DECAY_MIN = 0.3, ARA_DECAY_MAX = 0.31;
	const Real ARA_DECAY_INCREMENT = 0.025;
	
	const int ARA_MIN_RANK = 16, ARA_MAX_RANK =128;
	const int ARA_RANK_INCREMENT = 4;
	
	// GPU stuff
	RealArray d_M[num_gpus], d_A[num_gpus], d_B[num_gpus], d_U[num_gpus], d_V[num_gpus], d_M_Acpy[num_gpus];
	RealArray d_A2[num_gpus], d_B2[num_gpus], d_workspace[num_gpus], d_svals[num_gpus];
	RealPtrArray d_M_ptrs[num_gpus], d_A_ptrs[num_gpus], d_B_ptrs[num_gpus];
	RealPtrArray d_A2_ptrs[num_gpus], d_B2_ptrs[num_gpus], d_workspace_ptrs[num_gpus];
	
	IntArray d_rows_batch[num_gpus], d_cols_batch[num_gpus], d_ranks[num_gpus], d_ranks2[num_gpus];
	IntArray d_ldm_batch[num_gpus], d_lda_batch[num_gpus], d_ldb_batch[num_gpus], d_ldws_batch[num_gpus];
	
	GPU_Timer_t kblas_timers[num_gpus];
	kblasHandle_t kblas_handles[num_gpus];
	kblasRandState_t rand_state[num_gpus];
	LRSampler low_rank_sampler[num_gpus];
	
	double kblas_ara_time[nruns], kblas_svd_time[nruns];
	double avgError, avgRankDiff, sdevRankDiff;
	double avg_kblas_ara_time, avg_kblas_svd_time;
	double std_dev_kblas_ara_time, std_dev_kblas_svd_time;
	double ara_gflops;
	
	magma_init();
	
	for(int g = 0; g < num_gpus; g++)
	{
		cudaSetDevice(opts.devices[g]);
		
		#ifdef PREC_d
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		#endif

		// Init kblas handles and timers
		kblasCreate(&kblas_handles[g]);
		kblas_timers[g] = newGPU_Timer(kblasGetStream(kblas_handles[g]));
		kblasInitRandState(kblas_handles[g], &rand_state[g], 16384*2, 0);
		kblasEnableMagma(kblas_handles[g]);
    }

	printf("%-15s%-10s%-10s%-15s%-15s%-15s%-15s%-15s%-15s%-15s\n", "batchCount", "N", "K", "rank", "svdTime", "araTime", "ARA_GFlops", "avgError", "avgRankDiff", "sdevRankDiff");
	printf("=====================================================================================================================================\n");

	Real decay = ARA_DECAY_MIN;
	int rank = ARA_MIN_RANK;
	int sgen_mode = 2;
	
	while(decay < ARA_DECAY_MAX)
	//while(rank < ARA_MAX_RANK)
	{
		for(int itest = 0; itest < opts.ntest; ++itest)
		{
			max_rows = opts.msize[itest];
			max_cols = opts.nsize[itest];
			max_rank = max_cols;
			
			if(max_rows < ARA_MAX_RANK || max_cols < ARA_MAX_RANK)
			{
				printf("Testing cannot be used on matrices smaller than %d x %d.\n", ARA_MAX_RANK, ARA_MAX_RANK);
				continue;
			}
			
			for(int iter = 0; iter < opts.niter; ++iter)
			{
				for(int btest = 0; btest < opts.btest; ++btest)
				{
					batchCount = opts.batchCount;
					if(opts.btest > 1)
						batchCount = opts.batch[btest];

					batchCount_gpu = batchCount / num_gpus;
					
					////////////////////////////////////////////////////////////////////////
					// Set up the data
					////////////////////////////////////////////////////////////////////////
					TESTING_MALLOC_CPU(h_M, Real, batchCount * max_rows * max_cols);
					TESTING_MALLOC_CPU(h_A, Real, batchCount * max_rows * max_rank);
					TESTING_MALLOC_CPU(h_B, Real, batchCount * max_cols * max_rank);
					TESTING_MALLOC_CPU(h_svals, Real, batchCount * max_cols);
					
					TESTING_MALLOC_CPU(h_rows_batch, int, batchCount);
					TESTING_MALLOC_CPU(h_cols_batch, int, batchCount);
					TESTING_MALLOC_CPU(h_ranks,      int, batchCount);
					TESTING_MALLOC_CPU(h_ldm_batch,  int, batchCount);
					TESTING_MALLOC_CPU(h_lda_batch,  int, batchCount);
					TESTING_MALLOC_CPU(h_ldb_batch,  int, batchCount);
					TESTING_MALLOC_CPU(h_sval_ranks, int, batchCount);
					
					TESTING_MALLOC_CPU(h_M_ptrs,  Real*, batchCount);
					TESTING_MALLOC_CPU(h_A_ptrs,  Real*, batchCount);
					TESTING_MALLOC_CPU(h_B_ptrs,  Real*, batchCount);
					TESTING_MALLOC_CPU(h_svals_ptrs,  Real*, batchCount);
					
					generateRandDimensions(max_rows, max_rows, h_rows_batch, 1234, batchCount);
					generateRandDimensions(max_cols, max_cols, h_cols_batch, 5678, batchCount);
					fillIntArray(h_ldm_batch, max_rows, batchCount);
					fillIntArray(h_lda_batch, max_rows, batchCount);
					fillIntArray(h_ldb_batch, max_cols, batchCount);
					
					generateArrayOfPointersHost(h_M, h_M_ptrs, max_rows * max_cols, batchCount);
					generateArrayOfPointersHost(h_A, h_A_ptrs, max_rows * max_rank, batchCount);
					generateArrayOfPointersHost(h_B, h_B_ptrs, max_cols * max_rank, batchCount);
					generateArrayOfPointersHost(h_svals, h_svals_ptrs, max_cols, batchCount);
					
					generate_singular_values(h_svals_ptrs, rank, tol, 1, 12345, batchCount);

					generate_randomMatricesArray(
						h_M_ptrs, h_ldm_batch, h_svals_ptrs, h_rows_batch, h_cols_batch,
						sgen_mode, 0, decay, decay, 0, batchCount, 16
					);
					
					// printResults(h_M_ptrs, h_ldm_batch, h_rows_batch, h_cols_batch, 1, 'M');
					
					for(int g = 0; g < num_gpus; g++)
					{
						cudaSetDevice(opts.devices[g]);
						
						TESTING_MALLOC_DEV(d_M[g], Real, batchCount_gpu * max_rows * max_cols);
                                                TESTING_MALLOC_DEV(d_M_Acpy[g], Real, batchCount_gpu * max_rows * max_cols);
                                                TESTING_MALLOC_DEV(d_U[g], Real, batchCount_gpu * max_rows * max_cols);
                                                TESTING_MALLOC_DEV(d_V[g], Real, batchCount_gpu * max_rows * max_cols);
						TESTING_MALLOC_DEV(d_A[g], Real, batchCount_gpu * max_rows * max_rank);
						TESTING_MALLOC_DEV(d_B[g], Real, batchCount_gpu * max_cols * max_rank);
						TESTING_MALLOC_DEV(d_A2[g], Real, batchCount_gpu * max_rows * max_rank);
						TESTING_MALLOC_DEV(d_B2[g], Real, batchCount_gpu * max_cols * max_rank);
						TESTING_MALLOC_DEV(d_workspace[g], Real, batchCount_gpu * max_rank * max_rank);
						TESTING_MALLOC_DEV(d_svals[g], Real, batchCount_gpu * max_cols);
						
						TESTING_MALLOC_DEV(d_rows_batch[g], int, batchCount_gpu);
						TESTING_MALLOC_DEV(d_cols_batch[g], int, batchCount_gpu);
						TESTING_MALLOC_DEV(d_ranks[g],      int, batchCount_gpu);
						TESTING_MALLOC_DEV(d_ranks2[g],     int, batchCount_gpu);
						TESTING_MALLOC_DEV(d_ldm_batch[g],  int, batchCount_gpu);
						TESTING_MALLOC_DEV(d_lda_batch[g],  int, batchCount_gpu);
						TESTING_MALLOC_DEV(d_ldb_batch[g],  int, batchCount_gpu);
						TESTING_MALLOC_DEV(d_ldws_batch[g], int, batchCount_gpu);
						
						TESTING_MALLOC_DEV(d_M_ptrs[g], Real*, batchCount_gpu);
						TESTING_MALLOC_DEV(d_A_ptrs[g], Real*, batchCount_gpu);
						TESTING_MALLOC_DEV(d_B_ptrs[g], Real*, batchCount_gpu);
						TESTING_MALLOC_DEV(d_A2_ptrs[g], Real*, batchCount_gpu);
						TESTING_MALLOC_DEV(d_B2_ptrs[g], Real*, batchCount_gpu);
						TESTING_MALLOC_DEV(d_workspace_ptrs[g], Real*, batchCount_gpu);
						
						generateArrayOfPointers(d_M[g], d_M_ptrs[g], max_rows * max_cols, batchCount_gpu, 0);
						generateArrayOfPointers(d_A[g], d_A_ptrs[g], max_rows * max_rank, batchCount_gpu, 0);
						generateArrayOfPointers(d_B[g], d_B_ptrs[g], max_cols * max_rank, batchCount_gpu, 0);
						generateArrayOfPointers(d_A2[g], d_A2_ptrs[g], max_rows * max_rank, batchCount_gpu, 0);
						generateArrayOfPointers(d_B2[g], d_B2_ptrs[g], max_cols * max_rank, batchCount_gpu, 0);
						generateArrayOfPointers(d_workspace[g], d_workspace_ptrs[g], max_rank * max_rank, batchCount_gpu, 0);
						
						fillGPUIntArray(d_ldws_batch[g], max_rank, batchCount_gpu, 0);
						
						kblas_gemm_batch_strided_wsquery(kblas_handles[g], batchCount_gpu);
						kblas_gesvj_batch_wsquery<Real>(kblas_handles[g], max_rows, max_cols, batchCount_gpu);
						kblas_ara_batch_wsquery<Real>(kblas_handles[g], BLOCK_SIZE, batchCount_gpu);
						kblas_rsvd_batch_wsquery<Real>(kblas_handles[g], max_rows, max_cols, 64, batchCount_gpu);

						check_kblas_error( kblasAllocateWorkspace(kblas_handles[g]) );
						
						low_rank_sampler[g] = LRSampler(
							kblas_handles[g], d_rows_batch[g], d_cols_batch[g], d_ranks[g], 
							max_rows, max_cols, max_rank,
							d_A_ptrs[g], d_lda_batch[g], d_B_ptrs[g], d_ldb_batch[g], 
							d_workspace_ptrs[g], d_ldws_batch[g]
						);
					}

					ara_gflops = 0; 
						
					for(int i = 0; i < nruns; i++)
					{	
						COPY_DATA_UP(d_M, h_M, max_rows * max_cols, Real);
						COPY_DATA_UP(d_rows_batch, h_rows_batch, 1, int);
						COPY_DATA_UP(d_cols_batch, h_cols_batch, 1, int);
						COPY_DATA_UP(d_ldm_batch,  h_ldm_batch,  1, int);
						COPY_DATA_UP(d_lda_batch,  h_lda_batch,  1, int);
						COPY_DATA_UP(d_ldb_batch,  h_ldb_batch,  1, int);
						
						for(int g = 0; g < num_gpus; g++)
						{
							cudaSetDevice(opts.devices[g]);
							
							// kblasInitRandState(kblas_handles[g], &rand_state[g], 16384*2, 0);
							
							gpuTimerTic(kblas_timers[g]);
							
							check_kblas_error( kblas_ara_batch(
								kblas_handles[g], d_rows_batch[g], d_cols_batch[g], d_M_ptrs[g], d_ldm_batch[g], 
								d_A_ptrs[g], d_lda_batch[g], d_B_ptrs[g], d_ldb_batch[g], d_ranks[g], 
								tol, max_rows, max_cols, max_rank, BLOCK_SIZE, ARA_R, rand_state[g], 0, batchCount_gpu
							) );
						
							gpuTimerRecordEnd(kblas_timers[g]);
							
							// kblasDestroyRandState(rand_state[g]);
						}
						SYNC_TIMERS(kblas_timers, kblas_ara_time);
					
						COPY_DATA_DOWN(h_A, d_A, max_rows * max_rank, Real);
						COPY_DATA_DOWN(h_B, d_B, max_cols * max_rank, Real);
						COPY_DATA_DOWN(h_ranks, d_ranks, 1, int);
						
						// printResults(h_A_ptrs, h_lda_batch, h_rows_batch, h_ranks, batchCount, 'A');
						// printResults(h_B_ptrs, h_ldb_batch, h_cols_batch, h_ranks, batchCount, 'B');
						
						set_sval_ranks(h_svals_ptrs, h_cols_batch, batchCount, h_sval_ranks, tol);
						decay_rank = h_sval_ranks[0];
						
						testResults(
							h_M_ptrs, h_ldm_batch, h_A_ptrs, h_lda_batch, h_B_ptrs, h_ldb_batch, 
							h_rows_batch, h_cols_batch, h_ranks, h_sval_ranks, batchCount,
							avgError, avgRankDiff, sdevRankDiff
						);
						
						ara_gflops += get_ara_batch_gflops(h_rows_batch, h_cols_batch, h_ranks, ARA_R, batchCount) / kblas_ara_time[i];
						
						for(int g = 0; g < num_gpus; g++)
						{
							cudaSetDevice(opts.devices[g]);
							gpuTimerTic(kblas_timers[g]);
							
							/*
							check_kblas_error( 
								kblas_gesvj_batch(
									kblas_handles[g], max_rows, max_cols, d_M[g], max_rows, max_rows * max_cols, 
									d_svals[g], max_cols, batchCount_gpu
							) );
							*/
                                                cublasXcopy(kblasGetCublasHandle(kblas_handles[g]), max_rows * max_cols, d_M[g], 1, d_M_Acpy[g], 1);
							check_kblas_error( kblas_rsvd_batch( 
								kblas_handles[g], max_rows, max_cols, ARA_MAX_RANK, 
								d_M[g], max_rows, max_rows * max_cols, 
								d_svals[g], max_cols, rand_state[g], batchCount_gpu
							) );
                                                cublasXcopy(kblasGetCublasHandle(kblas_handles[g]), max_rows*ARA_MAX_RANK, d_M[g], 1, d_U[g], 1);
                                                kblasXgemm_batch_strided(kblas_handles[g], KBLAS_Trans, KBLAS_NoTrans, max_cols, ARA_MAX_RANK,
                                                max_rows, 1.0, d_M_Acpy[g], max_rows, max_cols, d_U[g], max_rows, ARA_MAX_RANK, 0.0, d_V[g], max_cols,
                                                ARA_MAX_RANK, batchCount_gpu);

							gpuTimerRecordEnd(kblas_timers[g]);
						}
						SYNC_TIMERS(kblas_timers, kblas_svd_time);
						
						// Low rank compression
						/*for(int g = 0; g < num_gpus; g++)
						{
							cudaSetDevice(opts.devices[g]);
							
							check_kblas_error( kblas_ara_batch_fn(
								kblas_handles[g], d_rows_batch[g], d_cols_batch[g], low_rank_sample, (void*)&low_rank_sampler[g], 
								d_A2_ptrs[g], d_lda_batch[g], d_B2_ptrs[g], d_ldb_batch[g], d_ranks2[g], 
								tol2, max_rows, max_cols, max_rank, BLOCK_SIZE, ARA_R, rand_state[g], batchCount_gpu
							) );
						}
						
						COPY_DATA_DOWN(h_A, d_A2, max_rows * max_rank, Real);
						COPY_DATA_DOWN(h_B, d_B2, max_cols * max_rank, Real);
						COPY_DATA_DOWN(h_ranks, d_ranks2, 1, int);
						
						// printResults(h_A_ptrs, h_lda_batch, h_rows_batch, h_ranks, batchCount, 'A');
						// printResults(h_B_ptrs, h_ldb_batch, h_cols_batch, h_ranks, batchCount, 'B');
						
						set_sval_ranks(h_svals_ptrs, h_cols_batch, batchCount, h_sval_ranks, tol2);
						
						avg_error = testResults(
							h_M_ptrs, h_ldm_batch, h_A_ptrs, h_lda_batch, h_B_ptrs, h_ldb_batch, 
							h_rows_batch, h_cols_batch, h_ranks, h_sval_ranks, batchCount
						);
						printf("Average error = %e\n", avg_error);
						*/
					}
					
					ara_gflops /= nruns;
					
					avg_and_stdev(&kblas_ara_time[0], nruns, &avg_kblas_ara_time, &std_dev_kblas_ara_time, warmup);
					avg_and_stdev(&kblas_svd_time[0], nruns, &avg_kblas_svd_time, &std_dev_kblas_svd_time, warmup);
					
					printf(
							"%-15d%-10d%-10d%-15d%-15.5f%-15.5f%-15.2f%-15e%-15.3f%-15.3f\n", 
							batchCount, max_rows, max_cols, decay_rank, avg_kblas_svd_time, avg_kblas_ara_time, 
							ara_gflops, avgError, avgRankDiff, sdevRankDiff
					);
					// Free the data
					TESTING_FREE_CPU(h_M); 
					TESTING_FREE_CPU(h_A); 
					TESTING_FREE_CPU(h_B);
					TESTING_FREE_CPU(h_svals);
					TESTING_FREE_CPU(h_rows_batch);
					TESTING_FREE_CPU(h_cols_batch);
					TESTING_FREE_CPU(h_ranks);
					TESTING_FREE_CPU(h_ldm_batch);
					TESTING_FREE_CPU(h_lda_batch);
					TESTING_FREE_CPU(h_ldb_batch);
					TESTING_FREE_CPU(h_M_ptrs);
					TESTING_FREE_CPU(h_A_ptrs);
					TESTING_FREE_CPU(h_B_ptrs);
					TESTING_FREE_CPU(h_svals_ptrs);
					TESTING_FREE_CPU(h_sval_ranks);
					
					for(int g = 0; g < num_gpus; g++)
					{
						cudaSetDevice(opts.devices[g]);
						TESTING_FREE_DEV(d_M[g]);
                                                TESTING_FREE_DEV(d_U[g]);
                                                TESTING_FREE_DEV(d_V[g]);
                                                TESTING_FREE_DEV(d_M_Acpy[g]); 
						TESTING_FREE_DEV(d_A[g]); 
						TESTING_FREE_DEV(d_B[g]); 
						TESTING_FREE_DEV(d_A2[g]); 
						TESTING_FREE_DEV(d_B2[g]); 
						TESTING_FREE_DEV(d_workspace[g]);
						TESTING_FREE_DEV(d_svals[g]); 
						TESTING_FREE_DEV(d_rows_batch[g]);
						TESTING_FREE_DEV(d_cols_batch[g]);
						TESTING_FREE_DEV(d_ranks[g]); 
						TESTING_FREE_DEV(d_ranks2[g]); 
						TESTING_FREE_DEV(d_ldm_batch[g]);
						TESTING_FREE_DEV(d_lda_batch[g]);
						TESTING_FREE_DEV(d_ldb_batch[g]);
						TESTING_FREE_DEV(d_ldws_batch[g]);
						TESTING_FREE_DEV(d_M_ptrs[g]); 
						TESTING_FREE_DEV(d_A_ptrs[g]); 
						TESTING_FREE_DEV(d_B_ptrs[g]); 
						TESTING_FREE_DEV(d_A2_ptrs[g]); 
						TESTING_FREE_DEV(d_B2_ptrs[g]); 
						TESTING_FREE_DEV(d_workspace_ptrs[g]); 
					}
				}
			}
		}

		decay += ARA_DECAY_INCREMENT;
		rank  += ARA_RANK_INCREMENT;
		
		if(rank > ARA_MAX_RANK) rank = ARA_MAX_RANK;
		if(decay > ARA_DECAY_MAX) decay = ARA_DECAY_MAX;
	}
	
    return 0;
}
