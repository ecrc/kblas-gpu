/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/batch_triangular/test_Xgeqrf_batch.cpp

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Wajih Halim Boukaram
 * @date 2020-12-10
 **/

#include <stdio.h>
#include <string.h>

#include "testing_helper.h"

#ifdef PREC_d
typedef double Real;
#define generateArrayOfPointers 		generateDArrayOfPointers
#define generateArrayOfPointersHost		generateDArrayOfPointersHost 
#define kblasXtsqrf_vbatch				kblasDtsqrf_vbatch
#define LAPACKE_Xgeqrf					LAPACKE_dgeqrf
#else
typedef float Real;
#define generateArrayOfPointers 		generateSArrayOfPointers
#define generateArrayOfPointersHost		generateSArrayOfPointersHost 
#define kblasXtsqrf_vbatch				kblasStsqrf_vbatch
#define LAPACKE_Xgeqrf					LAPACKE_sgeqrf
#endif

typedef Real*	RealArray;
typedef Real**	RealPtrArray;
typedef int*	IntArray;

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

double batch_qr_cpu(Real** m_batch, int* ld_batch, Real** tau_batch, int* rows_batch, int* cols_batch, int num_ops, int omp_threads)
{
	double total_time = gettime();
	
	#pragma omp parallel for num_threads(omp_threads)
    for(int op = 0; op < num_ops; op++)
    {
		int rows = rows_batch[op], cols = cols_batch[op];
		int ld   = ld_batch[op];
		
		int rank = (rows < cols ? rows : cols);
		
        Real* m = m_batch[op];
        Real* tau = tau_batch[op];

		LAPACKE_Xgeqrf(LAPACK_COL_MAJOR, rows, cols, m, ld, tau);
    }

	total_time = gettime() - total_time;
	return total_time;
}

Real compare_results_R(Real** m1_batch, Real** m2_batch, int* ld_batch, int* rows_batch, int* cols_batch, int num_ops)
{
	Real err = 0;
    for(int op = 0; op < num_ops; op++)
    {
        Real* m1_op = m1_batch[op];
        Real* m2_op = m2_batch[op];
		int rows = rows_batch[op];
		int cols = cols_batch[op];
		int ld   = ld_batch[op];
		
		// printMatrix(rows, cols, m1_op, ld, stdout);
		// printMatrix(rows, cols, m2_op, ld, stdout);
		
        Real err_op = 0, norm_f_op = 0;
	    for(int i = 0; i < rows; i++)
        {
            for(int j = i; j < cols; j++)
            {
                Real diff_entry = fabs(fabs(m1_op[i + j * ld]) - fabs(m2_op[i + j * ld]));
                err_op += diff_entry * diff_entry;
                norm_f_op += m1_op[i + j * ld] * m1_op[i + j * ld];
	        }
        }
        err += sqrt(err_op / norm_f_op);
    }
	return  err / num_ops;
}

double getOpCount(int* rows_batch, int* cols_batch, int num_ops)
{
	double hh_ops = 0;
	for(int op = 0; op < num_ops; op++)
    {
		int rows = rows_batch[op], cols = cols_batch[op];
		hh_ops += (double)(2.0 * rows * cols * cols - (2.0 / 3.0) * cols * cols * cols) * 1e-9;
	}
	return hh_ops;
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
	int num_gpus, warmup, nruns, num_omp_threads, info;
	int batchCount, batchCount_gpu;
	int max_rows, max_cols;

	double max_time, hh_ops;
	double avg_cpu_time, sdev_cpu_time;
	double avg_kblas_time, sdev_kblas_time;

	// Host arrays
	RealArray    m, m_original, tau, gpu_results;
	IntArray     rows_batch, cols_batch, ld_batch;
	RealPtrArray m_ptrs, tau_ptrs, gpu_results_ptrs;
	
	parse_opts(argc, argv, &opts);
	num_gpus = opts.ngpu;
	warmup = opts.warmup;
	nruns = opts.nruns;
	num_omp_threads = opts.omp_numthreads;

	// Device arrays
	RealArray d_m[num_gpus], d_tau[num_gpus];
	IntArray d_rows_batch[num_gpus], d_cols_batch[num_gpus], d_ld_batch[num_gpus];
	RealPtrArray d_m_ptrs[num_gpus], d_tau_ptrs[num_gpus];
	IntArray d_info[num_gpus];

	// Handles and timers
    #ifdef USE_MAGMA
	magma_init();
	magma_queue_t queues[num_gpus];
	#endif
	kblasHandle_t kblas_handles[num_gpus];
	GPU_Timer_t kblas_timers[num_gpus];

	double cpu_time[nruns], kblas_time[nruns];
	
	for(int g = 0; g < num_gpus; g++)
	{
		cudaSetDevice(opts.devices[g]);

		#ifdef PREC_d
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		#endif

		 #ifdef USE_MAGMA
		// Init magma queue
		magma_queue_create(opts.devices[g], &queues[g]);
		#endif

		// Init kblas handle and timer
		kblasCreate(&kblas_handles[g]);
		kblas_timers[g] = newGPU_Timer(kblasGetStream(kblas_handles[g]));
    }

	printf("%-15s%-10s%-10s%-15s%-15s\n", "batchCount", "N", "K", "kblasQERF", "kblasErr");
	printf("=================================================================\n");
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

				max_rows = opts.msize[itest];
				max_cols = opts.nsize[itest];
			
				////////////////////////////////////////////////////////////////////////
				// Set up the data
				////////////////////////////////////////////////////////////////////////
				TESTING_MALLOC_CPU(m, Real, batchCount * max_rows * max_cols);
				TESTING_MALLOC_CPU(m_original, Real, batchCount * max_rows * max_cols);
				TESTING_MALLOC_CPU(tau, Real, batchCount * max_cols);
				
				TESTING_MALLOC_CPU(m_ptrs, Real*, batchCount);
				TESTING_MALLOC_CPU(tau_ptrs, Real*, batchCount);
				TESTING_MALLOC_CPU(gpu_results_ptrs, Real*, batchCount);
				
				TESTING_MALLOC_CPU(rows_batch, int, batchCount);
				TESTING_MALLOC_CPU(cols_batch, int, batchCount);
				TESTING_MALLOC_CPU(ld_batch,   int, batchCount);
				TESTING_MALLOC_CPU(gpu_results, Real, batchCount * max_rows * max_cols);
			
				generateArrayOfPointersHost(m, m_ptrs, max_rows * max_cols, batchCount);
				generateArrayOfPointersHost(gpu_results, gpu_results_ptrs, max_rows * max_cols, batchCount);
				generateArrayOfPointersHost(tau, tau_ptrs, max_cols, batchCount);
				
				generateRandDimensions(32, max_rows, rows_batch, 1234, batchCount);
				generateRandDimensions(32, max_cols, cols_batch, 5678, batchCount);
				fillIntArray(ld_batch, max_rows, batchCount);
				
				hh_ops = getOpCount(rows_batch, cols_batch, batchCount);
				
				for(int op = 0; op < batchCount; op++)
				{
					Real* m_op = m_ptrs[op];
					int rows = rows_batch[op];
					int cols = cols_batch[op];
					int ld   = ld_batch[op];
					
					for(int i = 0; i < rows; i++)
						for(int j = 0; j < cols; j++)
							m_op[i + j * ld] = (Real)rand() / RAND_MAX;
				}
				
				memcpy(m_original, m, sizeof(Real) * batchCount * max_rows * max_cols);
				
				// Allocate the data
				for(int g = 0; g < num_gpus; g++)
				{
					cudaSetDevice(opts.devices[g]);

					TESTING_MALLOC_DEV(d_m[g], Real, batchCount_gpu * max_rows * max_cols);
					TESTING_MALLOC_DEV(d_tau[g], Real, batchCount_gpu * max_cols);

					// Generate array of pointers for the cublas and magma routines
					TESTING_MALLOC_DEV(d_m_ptrs[g], Real*, batchCount_gpu);
					TESTING_MALLOC_DEV(d_tau_ptrs[g], Real*, batchCount_gpu);
					TESTING_MALLOC_DEV(d_rows_batch[g], int, batchCount_gpu);
					TESTING_MALLOC_DEV(d_cols_batch[g], int, batchCount_gpu);
					TESTING_MALLOC_DEV(d_ld_batch[g],   int, batchCount_gpu);

					generateArrayOfPointers(d_m[g], d_m_ptrs[g], max_rows * max_cols, batchCount_gpu, 0);
					generateArrayOfPointers(d_tau[g], d_tau_ptrs[g], max_cols, batchCount_gpu, 0);
				}

				////////////////////////////////////////////////////////////////////////
				// Time the runs
				////////////////////////////////////////////////////////////////////////
				// Keep track of the times for each run
				double kblas_err = 0, magma_err = 0, cublas_err = 0;

				for(int i = 0; i < nruns; i++)
				{
					memcpy(m, m_original, sizeof(Real) * batchCount * max_rows * max_cols);
					cpu_time[i] = batch_qr_cpu(m_ptrs, ld_batch, tau_ptrs, rows_batch, cols_batch, batchCount, num_omp_threads);

					////////////////////////////////////////////////////////////////////////
					// KBLAS
					////////////////////////////////////////////////////////////////////////
					// Reset the GPU data and sync
					COPY_DATA_UP(d_m, m_original, max_rows * max_cols, Real);
					COPY_DATA_UP(d_rows_batch, rows_batch, 1, int);
					COPY_DATA_UP(d_cols_batch, cols_batch, 1, int);
					COPY_DATA_UP(d_ld_batch,   ld_batch,   1, int);
					
					// Launch all kernels
					for(int g = 0; g < num_gpus; g++)
					{
						cudaSetDevice(opts.devices[g]);
						gpuTimerTic(kblas_timers[g]);

						check_kblas_error( kblasXtsqrf_vbatch(
							kblas_handles[g], d_rows_batch[g], d_cols_batch[g], max_rows, max_cols, 
							d_m_ptrs[g], d_ld_batch[g], d_tau_ptrs[g], batchCount_gpu
						) );
						
						gpuTimerRecordEnd(kblas_timers[g]);
					}
					// The time all gpus finish at is the max of all the individual timers
					SYNC_TIMERS(kblas_timers, kblas_time);

					// Copy the data down from all the GPUs and compare with the CPU results
					COPY_DATA_DOWN(gpu_results, d_m, max_rows * max_cols, Real);
					kblas_err += compare_results_R(gpu_results_ptrs, m_ptrs, ld_batch, rows_batch, cols_batch, batchCount);
				}

				avg_and_stdev(cpu_time, nruns, &avg_cpu_time, &sdev_cpu_time, warmup);
				avg_and_stdev(kblas_time, nruns, &avg_kblas_time, &sdev_kblas_time, warmup);

				printf(
					"%-15d%-10d%-10d%-15.3f%-15.3e\n",
					batchCount, max_rows, max_cols, hh_ops / avg_kblas_time, kblas_err / nruns
				);

				////////////////////////////////////////////////////////////////////////
				// Free up the data
				////////////////////////////////////////////////////////////////////////
				TESTING_FREE_CPU(m); 
				TESTING_FREE_CPU(m_original);
				TESTING_FREE_CPU(tau); 
				TESTING_FREE_CPU(gpu_results);
				TESTING_FREE_CPU(rows_batch); 
				TESTING_FREE_CPU(cols_batch);
				TESTING_FREE_CPU(ld_batch);
				
				// Allocate the data
				for(int g = 0; g < num_gpus; g++)
				{
					cudaSetDevice(opts.devices[g]);

					TESTING_FREE_DEV(d_m[g]); 
					TESTING_FREE_DEV(d_tau[g]);
					TESTING_FREE_DEV(d_m_ptrs[g]); 
					TESTING_FREE_DEV(d_tau_ptrs[g]);
					TESTING_FREE_DEV(d_rows_batch[g]);
					TESTING_FREE_DEV(d_cols_batch[g]);
					TESTING_FREE_DEV(d_ld_batch[g]);
				}
			}
		}
	}

    return 0;
}
