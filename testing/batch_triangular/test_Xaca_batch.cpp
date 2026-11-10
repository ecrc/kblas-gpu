/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/batch_triangular/test_Xgesvj_batch.cpp

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Wajih Halim Boukaram
 * @date 2018-11-14
 **/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <algorithm>
//#include <batch_svd.h>

#include "testing_helper.h"
#include "batch_rand.h"

#define CUSOLVER_STREAMS 	10
#define RSVD_RANK			128

#ifdef PREC_d
typedef double Real;
#define generateXrandom					generateDrandom
#define LAPACKE_Xgesvj					LAPACKE_dgesvj
#define generateXrandomMatrices			generateDrandomMatrices
#define kblasXgesvj_batch				kblasDgesvj_batch_strided
#define kblasXrsvd_batch				kblasDrsvd_batch_strided
#define kblasXrsvd_batch_wsquery		kblasDrsvd_batch_wsquery
#define kblasXgesvj_batch_wsquery		kblasDgesvj_batch_wsquery
#define kblasXaca_batch_wsquery               kblasDgesvj_batch_wsquery
#define kblasXgemm_batch_strided                kblasDgemm_batch_strided
#define cublasXcopy                             cublasDcopy
#define kblasXacaf_batch                             kblas_dacaf_batch
#define LAPACKE_Xlange_work                           LAPACKE_dlange_work
#define cublasXgemm                                   cublasDgemm
#else
typedef float Real;
#define generateXrandom					generateSrandom
#define LAPACKE_Xgesvj					LAPACKE_sgesvj
#define generateXrandomMatrices			generateSrandomMatrices
#define kblasXgesvj_batch				kblasSgesvj_batch_strided
#define kblasXrsvd_batch				kblasSrsvd_batch_strided
#define kblasXrsvd_batch_wsquery		kblasSrsvd_batch_wsquery
#define kblasXgesvj_batch_wsquery		kblasSgesvj_batch_wsquery
#define kblasXaca_batch_wsquery               kblasSgesvj_batch_wsquery
#define kblasXgemm_batch_strided                kblasSgemm_batch_strided
#define cublasXcopy                             cublasScopy
#define Xacaf_batch                             Sacaf_batch
#define LAPACKE_Xlange_work                           LAPACKE_slange_work
#define kblasXacaf_batch                             kblas_sacaf_batch
#define cublasXgemm                                   cublasSgemm
#endif

typedef Real*	RealArray;
typedef int*	IntArray;
int kblas_acaf_gpu( kblasHandle_t handle,
                    int m, int n,
                    Real* A, int lda,
                    Real* U, int ldu,
                    Real* V, int ldv,
                    Real* S,
                    double maxacc, int maxrk,
                    double* acc, Real* rk);

#define COPY_DATA_UP() \
	for(int g = 0; g < num_gpus; g++) \
	{	\
		cudaSetDevice(opts.devices[g]); \
		int entries = batchCount_gpu * rows * cols; \
		check_error( cudaMemcpy(d_A[g], host_A_original + entries * g, entries * sizeof(Real), cudaMemcpyHostToDevice) ); \
	}	\
	syncGPUs(&opts);

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

#define COPY_DATA_DOWN() \
	for(int g = 0; g < num_gpus; g++)	\
	{	\
		cudaSetDevice(opts.devices[g]);	\
		int entries = batchCount_gpu * cols; \
		check_error( cudaMemcpy(host_A, d_A_cpy[g], rows * cols * sizeof(Real), cudaMemcpyDeviceToHost ) );	\
	}	\
	syncGPUs(&opts);

double batch_cpu_svd(Real* A, Real* S, int rows, int cols, int num_ops, int num_threads)
{
	double total_time = gettime();
	int stride_A = rows * cols;
	int stride_S = cols;

	#pragma omp parallel for num_threads(num_threads)
	for(int i = 0; i < num_ops; i++)
	{
		Real* a_op = A + i * stride_A;
		Real* s_op = S + i * stride_S;

        Real stat[6] = {1, 0, 0, 0, 0, 0};
        int info = LAPACKE_Xgesvj(CblasColMajor, 'G', 'U', 'N', rows, cols, a_op, rows, s_op, cols, NULL, cols, &stat[0]);
        if( info > 0 )
			printf( "The algorithm computing SVD failed to converge for operation %d.\n", i);
	}

	total_time = gettime() - total_time;
	return total_time;
}

Real testSValues(Real* S1, Real* S2, int stride_s, int rank, int num_ops)
{
	Real avg_diff = 0;
	for(int i = 0; i < num_ops; i++)
	{
		Real diff = 0, norm2 = 0;
		Real* s1_op = S1 + i * stride_s;
		Real* s2_op = S2 + i * stride_s;
		for(int j = 0; j < rank; j++)
		{
			diff += (s1_op[j] - s2_op[j]) * (s1_op[j] - s2_op[j]);
			norm2 += s2_op[j] * s2_op[j];
		}
		avg_diff += sqrt(diff / norm2);
	}
	return avg_diff / num_ops;
}

void syncGPUs(kblas_opts* opts)
{
	for(int g = 0; g < opts->ngpu; g++)
	{
		cudaSetDevice(opts->devices[g]);
		cudaDeviceSynchronize();
	}
}

int main(int argc, char** argv)
{
	kblas_opts opts;
	int num_gpus, warmup, nruns, num_omp_threads;
	int batchCount, batchCount_gpu;
	int rows, cols;
	double max_time;
	int work_size = 0;
        double acc=1e-8;

	// Host arrays
	RealArray host_A, host_S, host_A_original, host_S_exact, gpu_results;

	parse_opts(argc, argv, &opts);
	num_gpus = opts.ngpu;
	warmup = opts.warmup;
	nruns = opts.nruns;
	num_omp_threads = opts.omp_numthreads;
	
		// Device arrays
		RealArray d_A[num_gpus], d_S[num_gpus], d_U[num_gpus], d_V[num_gpus], d_A_cpy[num_gpus], d_work2[num_gpus];
		IntArray  d_info[num_gpus];
		double *d_work1[num_gpus];   
	
		// Handles and timers
		kblasHandle_t kblas_handles[num_gpus];
		GPU_Timer_t kblas_timers[num_gpus];
		kblasRandState_t rand_state[num_gpus];
		
		double cpu_time[nruns], kblas_time[nruns], cusolver_time[nruns];
	
		int cuda_version;
		check_error( cudaRuntimeGetVersion(&cuda_version) );
	
		for(int g = 0; g < num_gpus; g++)
		{
			cudaSetDevice(opts.devices[g]);
	
			#ifdef DOUBLE_PRECISION
			cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
			#endif
	
			// Init kblas handles and timers
			kblasCreate(&kblas_handles[g]);
			kblas_timers[g] = newGPU_Timer(kblasGetStream(kblas_handles[g]));
			kblasInitRandState(kblas_handles[g], &rand_state[g], 16384*2, 0);
	    }
	
		printf("%-15s%-10s%-10s%-15s%-15s%-15s%-15s%-15s%-15s\n", "batchCount", "N", "K", "kblasGESVJ(s)", "cpuGESVJ(s)", "CUSOLVER(s)", "kblasErr", "CPUerr", "cusovlerErr");
		printf("============================================================================================================================\n");
		for(int itest = 0; itest < opts.ntest; ++itest)
		{
			for(int iter = 0; iter < opts.niter; ++iter)
			{
				for(int btest = 0; btest < opts.btest; ++btest)
				{
					batchCount = opts.batchCount;
					if(opts.btest > 1)
						batchCount = opts.batch[btest];
	
					batchCount_gpu = batchCount / num_gpus;
	
					rows = opts.msize[itest];
					cols = opts.nsize[itest];
	
					////////////////////////////////////////////////////////////////////////
					// Set up the data
					////////////////////////////////////////////////////////////////////////
					TESTING_MALLOC_CPU(host_A, Real, batchCount * rows * cols);
					TESTING_MALLOC_CPU(host_S, Real, batchCount * cols);
					TESTING_MALLOC_CPU(host_A_original, Real, batchCount * rows * cols);
					TESTING_MALLOC_CPU(host_S_exact, Real, batchCount * cols);
					TESTING_MALLOC_CPU(gpu_results, Real, batchCount * cols);
	
					int rand_seed = itest + iter * opts.ntest + btest * opts.ntest * opts.niter;
					int rank = std::min(cols, RSVD_RANK);
				generateXrandomMatrices(
					host_A_original, rows * cols, host_S_exact, cols, rows, cols,
					0, 0.4, rand_seed, batchCount, num_omp_threads
				);
				// Allocate the device data
				for(int g = 0; g < num_gpus; g++)
				{
					cudaSetDevice(opts.devices[g]);
					TESTING_MALLOC_DEV(d_A[g], Real, batchCount_gpu * rows * cols);
					TESTING_MALLOC_DEV(d_S[g], Real, batchCount_gpu * cols);
                                        TESTING_MALLOC_DEV(d_A_cpy[g], Real, batchCount_gpu * rows * cols);
                                        TESTING_MALLOC_DEV(d_U[g], Real, batchCount_gpu * rows * RSVD_RANK); 
                                        TESTING_MALLOC_DEV(d_V[g], Real, batchCount_gpu * cols * RSVD_RANK); 
                                        TESTING_MALLOC_DEV(d_work1[g], double, batchCount_gpu * rows * RSVD_RANK);
                                        TESTING_MALLOC_DEV(d_work2[g], IntArray, batchCount_gpu);


					// KBlas workspace
					kblasXaca_batch_wsquery(kblas_handles[g], rows, cols, batchCount_gpu);

				}

				////////////////////////////////////////////////////////////////////////
				// Time the runs
				////////////////////////////////////////////////////////////////////////
				// Keep track of the times for each run
				double cpu_err = 0, kblas_err = 0, cusolver_err = 0;
				for(int i = 0; i < nruns; i++)
				{
					if(opts.check)
					{
						memcpy(host_A, host_A_original, sizeof(Real) * batchCount * rows * cols);
						cpu_time[i] = batch_cpu_svd(host_A, host_S, rows, cols, batchCount, num_omp_threads);
						cpu_err += testSValues(host_S, host_S_exact, cols, cols, batchCount);
					}

					////////////////////////////////////////////////////////////////////////
					// KBLAS
					////////////////////////////////////////////////////////////////////////
					// Reset the GPU data and sync
					COPY_DATA_UP();
					// Launch all kernels
					for(int g = 0; g < num_gpus; g++)
					{
                                                cudaSetDevice(opts.devices[g]);
                                                cublasXcopy(kblasGetCublasHandle(kblas_handles[g]), rows*cols, d_A[g], 1, d_A_cpy[g], 1);

						gpuTimerTic(kblas_timers[g]);
                                                kblas_acaf_gpu(kblas_handles[g], rows, cols, d_A[g], rows, 
                                                               d_U[g], rows, d_V[g], cols, d_S[g], acc, 
                                                               RSVD_RANK, d_work1[g], d_work2[g]);
						gpuTimerRecordEnd(kblas_timers[g]);

                                                kblasXgemm_batch_strided(kblas_handles[g], KBLAS_NoTrans, KBLAS_Trans, rows, cols,
                                                RSVD_RANK, 1.0, d_U[g], rows, rows*RSVD_RANK, d_V[g], cols, cols*RSVD_RANK, -1.0, d_A_cpy[g], rows,
                                                rows*cols, 1);
					}
					// The time all gpus finish at is the max of all the individual timers
					SYNC_TIMERS(kblas_timers, kblas_time);

					// Copy the data down from all the GPUs and compare with the CPU results
					COPY_DATA_DOWN();


                                        //check_error( cudaMemcpy(host_A, d_A_cpy[g], rows * cols * sizeof(Real), cudaMemcpyDeviceToHost ) );

                                        Real res = 0.0e0;
	                                Real *work_dlange;
	                                work_dlange=(Real*)malloc(rows*sizeof(Real));
	                                res = LAPACKE_Xlange_work( LAPACK_COL_MAJOR, 'i', rows, cols, host_A, rows, work_dlange);
	                                free(work_dlange);
					kblas_err += res; 

				}

				double avg_kblas_time, std_dev_kblas_time;
				double avg_cusolver_time=0.0, std_dev_cusolver_time=1.0;
				double avg_cpu_time, std_dev_cpu_time;

				avg_and_stdev(&cpu_time[0], nruns, &avg_cpu_time, &std_dev_cpu_time, warmup);
				avg_and_stdev(&kblas_time[0], nruns, &avg_kblas_time, &std_dev_kblas_time, warmup);
				avg_and_stdev(&cusolver_time[0], nruns, &avg_cusolver_time, &std_dev_cusolver_time, warmup);

				printf(
					"%-15d%-10d%-10d%-15.5f%-15.5f%-15.5f%-15.3e%-15.3e%-15.3e\n",
					batchCount, rows, cols, avg_kblas_time, avg_cpu_time, avg_cusolver_time,
					kblas_err / nruns, cpu_err / nruns, cusolver_err / nruns
				);

				////////////////////////////////////////////////////////////////////////
				// Free up the data
				////////////////////////////////////////////////////////////////////////
				TESTING_FREE_CPU(host_A); TESTING_FREE_CPU(host_S);
				TESTING_FREE_CPU(host_A_original); TESTING_FREE_CPU(host_S_exact);
				TESTING_FREE_CPU(gpu_results);

				for(int g = 0; g < num_gpus; g++)
				{
					cudaSetDevice(opts.devices[g]);
					TESTING_FREE_DEV(d_A[g]); TESTING_FREE_DEV(d_S[g]);
                                        TESTING_FREE_DEV(d_A_cpy[g]);
                                        TESTING_FREE_DEV(d_U[g]);
                                        TESTING_FREE_DEV(d_V[g]);
                                        TESTING_FREE_DEV(d_work1[g]);
                                        TESTING_FREE_DEV(d_work2[g]);
					if(opts.cuda)
					{
						TESTING_FREE_DEV(d_info[g]);
					}
				}
			}
		}
	}

    return 0;
}
