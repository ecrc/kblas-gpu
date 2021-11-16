#include <stdio.h>
#include <math.h>
#include <string.h>
#include <algorithm>

#include "testing_helper.h"

#ifdef PREC_d
typedef double Real;
#define kblas_geqp2_batch_strided		kblasDgeqp2_batch_strided
#define kblas_orgqr_batch_strided		kblasDorgqr_batch_strided
#define generateXrandomMatrices			generateDrandomMatrices
#define GEQP2_TOL						1e-4
#define GEQP2_DECAY						0.4
#else
typedef float Real;
#define kblas_geqp2_batch_strided		kblasSgeqp2_batch_strided
#define kblas_orgqr_batch_strided		kblasSorgqr_batch_strided
#define generateXrandomMatrices			generateSrandomMatrices
#define GEQP2_TOL						1e-3
#define GEQP2_DECAY						0.3
#endif

#define FLOPS_ARA(m, n, k, r)  (2 * (m) * (n) * ((r) + (k)) + (m) * (k) * (3 + 4 * (r) + 4 * (k)))

typedef Real*		RealArray;
typedef int*		IntArray;

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

void cblas_gemm(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const int m, const int n, const int k, const float alpha, const float *a, const int lda, const float *b, const int ldb, const float beta, float *c, const int ldc)
{ cblas_sgemm(CblasColMajor, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }

void cblas_gemm(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const int m, const int n, const int k, const double alpha, const double *a, const int lda, const double *b, const int ldb, const double beta, double *c, const int ldc)
{ cblas_dgemm(CblasColMajor, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }

void testResults(
	Real* A_strided, int stride_a, Real* Ap_strided, int stride_ap, 
	Real* Q_strided, int stride_q, Real* T_strided, int stride_t, 
	int* piv_strided, int stride_piv, int rows, int cols, int max_rank, 
	int* exact_ranks, int* detected_ranks, int batchCount, 
	double& avgError, double& avgRankDiff, double& sdevRankDiff
)
{
	double* rank_diff;
	TESTING_MALLOC_CPU(rank_diff, double, batchCount);
	
	Real total_diff = 0;
	for(int i = 0; i < batchCount; i++)
	{
		Real* A = A_strided + i * stride_a;
		Real* Ap = Ap_strided + i * stride_ap;
		Real* Q = Q_strided + i * stride_q;
		Real* T = T_strided + i * stride_t;
		int* piv = piv_strided + i * stride_piv;
		int rank = detected_ranks[i];
		
		// Apply the column pivots to A and compute the frobenius norm of A 
		Real Anorm_f = 0;
		for(int c = 0; c < cols; c++)
		{
			int cp = piv[c];
			for(int r = 0; r < rows; r++)
			{
				Ap[r + c * rows] = A[r + cp * rows];
				Anorm_f += A[r + cp * rows] * A[r + cp * rows];
			}
		}
		
		// Find T = Q' * A * P
		cblas_gemm(
			CblasTrans, CblasNoTrans, rank, cols, rows, 
			1, Q, rows, Ap, rows, 0, T, rank
		);
		
		// Find ||A * P - Q * T||
		cblas_gemm(
			CblasNoTrans, CblasNoTrans, rows, cols, rank, 
			1, Q, rows, T, rank, -1, Ap, rows
		);
		
		Real diff_normf = 0;
		for(int r = 0; r < rows; r++) 
			for(int c = 0; c < cols; c++) 
				diff_normf += Ap[r + c * rows] * Ap[r + c * rows];

		total_diff += sqrt(diff_normf / Anorm_f);
		rank_diff[i] = abs(rank - exact_ranks[i]);
		// if(sqrt(diff_normf / Anorm_f) < 1e-4)
		// printf("Operation %d had diff %e and rank %d (%d real)\n", i, sqrt(diff_normf / Anorm_f), rank, exact_ranks[i]);
		
	}
	
	avgError = total_diff / batchCount;
	avg_and_stdev(rank_diff, batchCount, &avgRankDiff, &sdevRankDiff, 0);
	
	TESTING_FREE_CPU(rank_diff);
}

void set_sval_ranks(Real* svals_strided, int stride_s, int cols, int batchCount, int* ranks, Real tol)
{
	for(int i = 0; i < batchCount; i++)
	{
		Real* svals = svals_strided + i * stride_s;
		int rank = 0;
		while(svals[rank] >= tol && rank < cols)
			rank++;
		ranks[i] = rank;
	}
}

int main(int argc, char** argv)
{
	kblas_opts opts;
	int num_gpus, warmup, nruns, num_omp_threads;
	int batchCount, batchCount_gpu;
	int rows, cols, max_rank;
	double max_time, tol;

	// Host arrays
	RealArray host_A, host_S, host_A_pivoted, host_Q, host_T;
	IntArray host_piv, host_ranks, gpu_ranks;
	
	parse_opts(argc, argv, &opts);
	num_gpus = opts.ngpu;
	warmup = opts.warmup;
	nruns = opts.nruns;
	num_omp_threads = opts.omp_numthreads;

	// Device arrays
	RealArray d_A[num_gpus], d_tau[num_gpus];
	IntArray  d_ranks[num_gpus], d_piv[num_gpus];
	
	// Handles and timers
	kblasHandle_t kblas_handles[num_gpus];
	GPU_Timer_t kblas_timers[num_gpus];
	
	double kblas_time[nruns];
	
	int cuda_version;
	check_error( cudaRuntimeGetVersion(&cuda_version) );

	tol = GEQP2_TOL;
	
	for(int g = 0; g < num_gpus; g++)
	{
		cudaSetDevice(opts.devices[g]);

		#ifdef DOUBLE_PRECISION
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		#endif

		// Init kblas handles and timers
		kblasCreate(&kblas_handles[g]);
		kblas_timers[g] = newGPU_Timer(kblasGetStream(kblas_handles[g]));
    }

	printf("%-15s%-10s%-10s%-15s%-15s%-15s%-15s\n", "batchCount", "N", "K", "kblasGEQP(s)", "kblasErr", "avgRankDiff", "sdevRankDiff");
	printf("============================================================================================================================\n");
	for(int itest = 0; itest < opts.ntest; ++itest)
	{
		for(int iter = 0; iter < opts.niter; ++iter)
		{
			for(int btest = 0; btest < opts.btest; ++btest)
			{
				batchCount_gpu = opts.batchCount;
				if(opts.btest > 1)
					batchCount_gpu = opts.batch[btest];

				batchCount = batchCount_gpu * num_gpus;
				
				rows = opts.msize[itest];
				cols = opts.nsize[itest];

				////////////////////////////////////////////////////////////////////////
				// Set up the data
				////////////////////////////////////////////////////////////////////////
				TESTING_MALLOC_CPU(host_A, Real, batchCount * rows * cols);
				TESTING_MALLOC_CPU(host_Q, Real, batchCount * rows * cols);
				TESTING_MALLOC_CPU(host_A_pivoted, Real, batchCount * rows * cols);
				TESTING_MALLOC_CPU(host_T, Real, batchCount * cols * cols);
				TESTING_MALLOC_CPU(host_S, Real, batchCount * cols);
				TESTING_MALLOC_CPU(host_piv, int, batchCount * cols);
				TESTING_MALLOC_CPU(host_ranks, int, batchCount);
				TESTING_MALLOC_CPU(gpu_ranks, int, batchCount);
				
				int rand_seed = itest + iter * opts.ntest + btest * opts.ntest * opts.niter;
	
				generateXrandomMatrices(
					host_A, rows * cols, host_S, cols, rows, cols,
					0, GEQP2_DECAY, rand_seed, batchCount, num_omp_threads
				);
				
				set_sval_ranks(host_S, cols, cols, batchCount, host_ranks, tol);
				
				// Allocate the device data
				for(int g = 0; g < num_gpus; g++)
				{
					cudaSetDevice(opts.devices[g]);
					TESTING_MALLOC_DEV(d_A[g], Real, batchCount_gpu * rows * cols);
					TESTING_MALLOC_DEV(d_tau[g], Real, batchCount_gpu * cols);
					TESTING_MALLOC_DEV(d_ranks[g], int, batchCount_gpu);
					TESTING_MALLOC_DEV(d_piv[g], int, batchCount_gpu * cols);
				}

				////////////////////////////////////////////////////////////////////////
				// Time the runs
				////////////////////////////////////////////////////////////////////////
				for(int i = 0; i < nruns; i++)
				{
					////////////////////////////////////////////////////////////////////////
					// KBLAS
					////////////////////////////////////////////////////////////////////////
					// Reset the GPU data and sync
					COPY_DATA_UP(d_A, host_A, rows * cols, Real);
					
					// Clear out tau
					for(int g = 0; g < num_gpus; g++)
						check_error(cudaMemset(d_tau[g], 0, batchCount_gpu * cols));
					
					// Launch all kernels - we only time geqp2 here since we only want to keep track 
					// of its performance and not orgqr
					for(int g = 0; g < num_gpus; g++)
					{
						cudaSetDevice(opts.devices[g]);
						
						gpuTimerTic(kblas_timers[g]);

						check_kblas_error( kblas_geqp2_batch_strided(
							kblas_handles[g], rows, cols, d_A[g], rows, rows * cols, 
							d_tau[g], cols, d_piv[g], cols, d_ranks[g], tol, batchCount_gpu
						) );
						
						gpuTimerRecordEnd(kblas_timers[g]);
					}
					// The time all gpus finish at is the max of all the individual timers
					SYNC_TIMERS(kblas_timers, kblas_time);
					
					// Expand the reflectors into the orthogonal factor so we can verify the error 
					// Determine the max rank, since orgqr currently does not have a non-uniform version
					for(int g = 0; g < num_gpus; g++)
					{
						cudaStream_t stream = kblasGetStream(kblas_handles[g]);
						int gpu_max_rank = getMaxElement(d_ranks[g], batchCount_gpu, stream);
						if(gpu_max_rank > max_rank) 
							max_rank = gpu_max_rank;
					}
					
					for(int g = 0; g < num_gpus; g++)
					{
						check_kblas_error( kblas_orgqr_batch_strided(
							kblas_handles[g], rows, max_rank, d_A[g], rows, rows * cols,
							d_tau[g], cols, batchCount_gpu
						) );
					}

					// Copy the data down from all the GPUs
					COPY_DATA_DOWN(host_Q, d_A, rows * cols, Real);
					COPY_DATA_DOWN(host_piv, d_piv, cols, int);
					COPY_DATA_DOWN(gpu_ranks, d_ranks, 1, int);
				}
				
				// Test the errors 
				double avgRankDiff, sdevRankDiff;
				double kblas_err;
				
				testResults(
					host_A, rows * cols, host_A_pivoted, rows * cols, 
					host_Q, rows * cols, host_T, cols * cols, 
					host_piv, cols, rows, cols, max_rank, 
					host_ranks, gpu_ranks, batchCount, 
					kblas_err, avgRankDiff, sdevRankDiff
				);
				
				double avg_kblas_time, std_dev_kblas_time;
				avg_and_stdev(&kblas_time[0], nruns, &avg_kblas_time, &std_dev_kblas_time, warmup);

				printf(
					"%-15d%-10d%-10d%-15.5f%-15.3e%-15.3f%-15.3f\n",
					batchCount, rows, cols, avg_kblas_time, kblas_err, avgRankDiff, sdevRankDiff
				);

				////////////////////////////////////////////////////////////////////////
				// Free up the data
				////////////////////////////////////////////////////////////////////////
				TESTING_FREE_CPU(host_A); 
				TESTING_FREE_CPU(host_Q);
				TESTING_FREE_CPU(host_A_pivoted); 
				TESTING_FREE_CPU(host_T); 
				TESTING_FREE_CPU(host_S);
				TESTING_FREE_CPU(host_piv);
				TESTING_FREE_CPU(host_ranks);
				TESTING_FREE_CPU(gpu_ranks);
				
				for(int g = 0; g < num_gpus; g++)
				{
					cudaSetDevice(opts.devices[g]);
					
					TESTING_FREE_DEV(d_A[g]);
					TESTING_FREE_DEV(d_tau[g]);
					TESTING_FREE_DEV(d_ranks[g]);
					TESTING_FREE_DEV(d_piv[g]);
				}
			}
		}
	}

    return 0;
}
