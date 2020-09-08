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

#include "testing_helper.h"
#include "batch_rand.h"

// The number of streams on each GPU for cusolver
#define CUSOLVER_STREAMS 	10
#define RSVD_RANK			64
//#define USE_RSVD


#ifdef PREC_d
typedef double Real;
#define generateXrandom					generateDrandom
#define LAPACKE_Xgesvj					LAPACKE_dgesvj
#define generateXrandomMatrices			generateDrandomMatrices
#define kblasXgesvj_batch				kblasDgesvj_batch_strided
#define kblasXrsvd_batch				kblasDrsvd_batch_strided
#define kblasXrsvd_batch_wsquery		kblasDrsvd_batch_wsquery
#define kblasXgesvj_batch_wsquery		kblasDgesvj_batch_wsquery
#define cusolverDnXgesvd_bufferSize		cusolverDnDgesvd_bufferSize
#define cusolverDnXgesvd				cusolverDnDgesvd
#else
typedef float Real;
#define generateXrandom					generateSrandom
#define LAPACKE_Xgesvj					LAPACKE_sgesvj
#define generateXrandomMatrices			generateSrandomMatrices
#define kblasXgesvj_batch				kblasSgesvj_batch_strided
#define kblasXrsvd_batch				kblasSrsvd_batch_strided
#define kblasXrsvd_batch_wsquery		kblasSrsvd_batch_wsquery
#define kblasXgesvj_batch_wsquery		kblasSgesvj_batch_wsquery
#define cusolverDnXgesvd_bufferSize		cusolverDnSgesvd_bufferSize
#define cusolverDnXgesvd				cusolverDnSgesvd
#endif

typedef Real*	RealArray;
typedef int*	IntArray;

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
		check_error( cudaMemcpy(gpu_results + entries * g, d_S[g], entries * sizeof(Real), cudaMemcpyDeviceToHost ) );	\
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

	// Host arrays
	RealArray host_A, host_S, host_A_original, host_S_exact, gpu_results;

	parse_opts(argc, argv, &opts);
	num_gpus = opts.ngpu;
	warmup = opts.warmup;
	nruns = opts.nruns;
	num_omp_threads = opts.omp_numthreads;

	// Device arrays
	RealArray d_A[num_gpus], d_S[num_gpus];
	IntArray  d_info[num_gpus];
	RealArray cusolver_ws[num_gpus];

	// Handles and timers
	cusolverDnHandle_t cusolver_handles[num_gpus][CUSOLVER_STREAMS];
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

		cudaStream_t cuda_streams[CUSOLVER_STREAMS];
		for(int i = 0 ; i < CUSOLVER_STREAMS; i++)
		{
			check_error( cudaStreamCreate(&cuda_streams[i]) );
			check_cusolver_error( cusolverDnCreate(&cusolver_handles[g][i]) );
			check_cusolver_error( cusolverDnSetStream(cusolver_handles[g][i], cuda_streams[i]) );
		}

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
				#ifdef USE_RSVD
				int rank = std::min(cols, RSVD_RANK);
				generateXrandomMatrices(
					host_A_original, rows * cols, host_S_exact, cols, rows, cols,
					0, 0.4, rand_seed, batchCount, num_omp_threads
				);
				#else
				generateXrandomMatrices(
					host_A_original, rows * cols, host_S_exact, cols, rows, cols,
					1e7, 0.4, rand_seed, batchCount, num_omp_threads
				);
				#endif
				// Allocate the device data
				for(int g = 0; g < num_gpus; g++)
				{
					cudaSetDevice(opts.devices[g]);
					TESTING_MALLOC_DEV(d_A[g], Real, batchCount_gpu * rows * cols);
					TESTING_MALLOC_DEV(d_S[g], Real, batchCount_gpu * cols);

					// KBlas workspace
					#ifdef USE_RSVD
					kblasXrsvd_batch_wsquery(kblas_handles[g], rows, cols, rank, batchCount_gpu);
					#else
					kblasXgesvj_batch_wsquery(kblas_handles[g], rows, cols, batchCount_gpu);
					#endif
					kblasAllocateWorkspace(kblas_handles[g]);

					// Cusolver workspace
					if(opts.cuda)
					{
						TESTING_MALLOC_DEV(d_info[g], int, CUSOLVER_STREAMS);

						work_size = 0;
						check_cusolver_error(cusolverDnXgesvd_bufferSize(cusolver_handles[g][0], rows, cols, &work_size));
						TESTING_MALLOC_DEV(cusolver_ws[g], Real, CUSOLVER_STREAMS * work_size);
					}
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
						gpuTimerTic(kblas_timers[g]);

						#ifdef USE_RSVD
						kblasXrsvd_batch(kblas_handles[g], rows, cols, rank, d_A[g], rows, rows * cols, d_S[g], cols, rand_state[g], batchCount_gpu);
						#else
						kblasXgesvj_batch(kblas_handles[g], rows, cols, d_A[g], rows, rows * cols, d_S[g], cols, batchCount_gpu);
						#endif
						gpuTimerRecordEnd(kblas_timers[g]);
					}
					// The time all gpus finish at is the max of all the individual timers
					SYNC_TIMERS(kblas_timers, kblas_time);

					// Copy the data down from all the GPUs and compare with the CPU results
					COPY_DATA_DOWN();
					#ifdef USE_RSVD
					kblas_err += testSValues(gpu_results, host_S_exact, cols, rank, batchCount);
					#else
					kblas_err += testSValues(gpu_results, host_S_exact, cols, cols, batchCount);
					#endif

					////////////////////////////////////////////////////////////////////////
					// CUSPARSE
					////////////////////////////////////////////////////////////////////////
					// Reset the GPU data and sync
					if(opts.cuda && cuda_version >= 8000)
					{
						COPY_DATA_UP();

						cusolver_time[i] = gettime();
						// Launch all kernels
						for(int g = 0; g < num_gpus; g++)
						{
							cudaSetDevice(opts.devices[g]);

							int ops_done = 0;
							while(ops_done < batchCount_gpu)
							{
								int ops_to_do = std::min(CUSOLVER_STREAMS, batchCount_gpu - ops_done);

								// #pragma omp parallel for num_threads(ops_to_do)
								for(int i = 0; i < ops_to_do; i++)
								{
									int op_index = ops_done + i;
									Real* matrix = d_A[g] + rows * cols * op_index;
									Real* S_m = d_S[g] + cols * op_index;

									Real* work_m = cusolver_ws[g] + work_size * i;
									int* dev_info_m = d_info[g] + i;

									check_cusolver_error(
										cusolverDnXgesvd(
											cusolver_handles[g][i], 'O', 'N', rows, cols,
											matrix, rows, S_m, NULL, rows,
											NULL, cols, work_m, work_size, NULL,
											dev_info_m
										)
									);
								}
								ops_done += CUSOLVER_STREAMS;
							}
						}
						syncGPUs(&opts);
						cusolver_time[i] = gettime() - cusolver_time[i];

						// Copy the data down from all the GPUs and compare with the CPU results
						COPY_DATA_DOWN();
						cusolver_err += testSValues(gpu_results, host_S_exact, cols, cols, batchCount);
					}
				}

				double avg_kblas_time, std_dev_kblas_time;
				double avg_cusolver_time, std_dev_cusolver_time;
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
					if(opts.cuda)
					{
						TESTING_FREE_DEV(d_info[g]);
						TESTING_FREE_DEV(cusolver_ws[g]);
					}
				}
			}
		}
	}

    return 0;
}
