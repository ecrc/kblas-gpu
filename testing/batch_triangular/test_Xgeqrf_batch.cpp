#include <stdio.h>
#include <string.h>

#include <kblas.h>
#include <magma_v2.h>
#include <mkl.h>

#include "testing_helper.h"

#ifdef PREC_d
typedef double Real;
#define generateArrayOfPointers generateDArrayOfPointers
#define cublasXgeqrf_batched 	cublasDgeqrfBatched
#define magmaXgeqrf_batched		magma_dgeqrf_batched
#define kblasXgeqrf_batched		kblasDgeqrf_batch_strided
#define kblasXtsqrf_batched		kblasDtsqrf_batch_strided
#define LAPACKE_Xgeqrf			LAPACKE_dgeqrf
#else
typedef float Real;
#define generateArrayOfPointers generateSArrayOfPointers
#define cublasXgeqrf_batched 	cublasSgeqrfBatched
#define magmaXgeqrf_batched		magma_sgeqrf_batched
#define kblasXgeqrf_batched		kblasSgeqrf_batch_strided
#define kblasXtsqrf_batched		kblasStsqrf_batch_strided
#define LAPACKE_Xgeqrf			LAPACKE_sgeqrf
#endif

typedef Real*	RealArray;
typedef Real**	RealPtrArray;
typedef int*	IntArray;

#define COPY_DATA_UP() \
	for(g = 0; g < num_gpus; g++) \
	{	\
		cudaSetDevice(opts.devices[g]); \
		int entries = batchCount_gpu * rows * cols; \
		check_error( cudaMemcpy(d_m[g], m_original + entries * g, entries * sizeof(Real), cudaMemcpyHostToDevice) ); \
	}	\
	syncGPUs(&opts); 

#define SYNC_TIMERS(timers, run_time) \
	max_time = -1;	\
	for(g = 0; g < num_gpus; g++) \
	{ \
		cudaSetDevice(opts.devices[g]); \
		double gpu_time = gpuTimerToc(timers[g]); \
		if(max_time < gpu_time)	\
			max_time = gpu_time; \
	} \
	run_time[i] = max_time;

#define COPY_DATA_DOWN() \
	for(g = 0; g < num_gpus; g++)	\
	{	\
		cudaSetDevice(opts.devices[g]);	\
		int entries = batchCount_gpu * rows * cols; \
		check_error( cudaMemcpy(gpu_results + entries * g, d_m[g], entries * sizeof(Real), cudaMemcpyDeviceToHost ) );	\
	}	\
	syncGPUs(&opts);

double batch_qr_cpu(Real* m, Real* tau, int rows, int cols, int num_ops, int num_threads)
{
	double total_time = gettime();
	
	#pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i < num_ops; i++)
    {
        Real* m_op = m + i * rows * cols;
        Real* tau_op = tau + i * cols;
		LAPACKE_Xgeqrf(LAPACK_COL_MAJOR, rows, cols, m_op, rows, tau_op);
    }
	
	total_time = gettime() - total_time;
	return total_time;
}

Real compare_results_R(Real* m1, Real* m2, int rows, int cols, int num_ops)
{
	Real err = 0;
    for(int op = 0; op < num_ops; op++)
    {
        Real* m1_op = m1 + op * rows * cols;
        Real* m2_op = m2 + op * rows * cols;
        Real err_op = 0, norm_f_op = 0;
	    for(int i = 0; i < cols; i++)
        {
            for(int j = i; j < cols; j++)
            {
                Real diff_entry = abs(abs(m1_op[i + j * rows]) - abs(m2_op[i + j * rows]));
                err_op += diff_entry * diff_entry;
                norm_f_op += m1_op[i + j * rows] * m1_op[i + j * rows];
	        }
        }
        err += sqrt(err_op / norm_f_op);
    }
	return  err / num_ops;
}

void syncGPUs(kblas_opts* opts)
{
	for(int g = 0; g < opts->ngpu; g++)
	{
		cudaSetDevice(opts->devices[g]);
		cudaDeviceSynchronize();
	}
}

void avg_and_stdev(double* values, int num_vals, double& avg, double& std_dev, int warmup)
{
	if(num_vals == 0) return;

	int start = 0;
	if(warmup == 1 && num_vals != 1)
		start = 1;
	
	avg = 0;
	for(int i = start; i < num_vals; i++) 
		avg += values[i];
	avg /= num_vals;
	
	std_dev = 0;
	for(int i = 0; i < num_vals; i++)
		std_dev += (values[i] - avg) * (values[i] - avg);
	std_dev = sqrt(std_dev / num_vals);
}

int main(int argc, char** argv)
{
	kblas_opts opts;
	int num_gpus, warmup, nruns, num_omp_threads, info;
	int g, itest, iter, btest, batchCount, batchCount_gpu;
	int rows, cols, i;

	double max_time, hh_ops, kblas_err, magma_err, cublas_err;
	double avg_cpu_time, sdev_cpu_time;
	double avg_kblas_time, sdev_kblas_time;
	double avg_magma_time, sdev_magma_time;
	double avg_cublas_time, sdev_cublas_time;
	
	// Host arrays
	RealArray m, m_original, tau, gpu_results;

	parse_opts(argc, argv, &opts);
	num_gpus = opts.ngpu;
	warmup = opts.warmup;
	nruns = opts.nruns;
	num_omp_threads = opts.omp_numthreads;
	
	// Device arrays
	RealArray d_m[num_gpus], d_tau[num_gpus];
	RealPtrArray d_m_ptrs[num_gpus], d_tau_ptrs[num_gpus];
	IntArray d_info[num_gpus];
	
	// Handles and timers
    magma_queue_t queues[num_gpus];
	cublasHandle_t cublas_handles[num_gpus];
	kblasHandle_t kblas_handles[num_gpus];
	GPU_Timer_t magma_timers[num_gpus], cublas_timers[num_gpus], kblas_timers[num_gpus];
	
	double cpu_time[nruns], kblas_time[nruns], magma_time[nruns], cublas_time[nruns];
	magma_init();
		
	for(int g = 0; g < num_gpus; g++)
	{
		cudaSetDevice(opts.devices[g]);
		
		#ifdef DOUBLE_PRECISION
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		#endif
		
		// Init magma queue and timer
		magma_queue_create(opts.devices[g], &queues[g]);
		magma_timers[g] = newGPU_Timer(magma_queue_get_cuda_stream(queues[g]));
		
		// Init cublas handle and timer
		check_cublas_error( cublasCreate(&cublas_handles[g]) );
		cudaStream_t cublas_stream;
		check_cublas_error( cublasGetStream(cublas_handles[g], &cublas_stream) );
		cublas_timers[g] = newGPU_Timer(cublas_stream);
		
		// Init kblas handle and timer
		kblasCreate(&kblas_handles[g]);
		kblas_timers[g] = newGPU_Timer(kblasGetStream(kblas_handles[g]));
    }
    
	printf("%-15s%-10s%-10s%-15s%-15s%-15s%-15s%-15s%-15s\n", "batchCount", "N", "K", "kblasQERF", "magmaQERF", "cublasQERF", "kblasErr", "magmaErr", "cublasErr");
	printf("============================================================================================================================\n");
	for(itest = 0; itest < opts.ntest; ++itest) 
	{
		for(iter = 0; iter < opts.niter; ++iter) 
		{
			for(btest = 0; btest < opts.btest; ++btest) 
			{
				batchCount = opts.batchCount;
				if(opts.btest > 1)
					batchCount = opts.batch[btest];
				
				batchCount_gpu = batchCount / num_gpus;
				
				rows = opts.msize[itest];
				cols = opts.nsize[itest];
				
				hh_ops = (double)(2.0 * rows * cols * cols - (2.0 / 3.0) * cols * cols * cols) * batchCount * 1e-9;
				
				////////////////////////////////////////////////////////////////////////
				// Set up the data
				////////////////////////////////////////////////////////////////////////
				TESTING_MALLOC_CPU(m, Real, batchCount * rows * cols);
				TESTING_MALLOC_CPU(m_original, Real, batchCount * rows * cols);
				TESTING_MALLOC_CPU(tau, Real, batchCount * cols);
				TESTING_MALLOC_CPU(gpu_results, Real, batchCount * rows * cols);
				for(i = 0; i < batchCount * rows * cols; i++)
					m[i] = m_original[i] = (Real)rand() / RAND_MAX;

				// Allocate the data
				for(g = 0; g < num_gpus; g++)
				{
					cudaSetDevice(opts.devices[g]);
					
					TESTING_MALLOC_DEV(d_m[g], Real, batchCount_gpu * rows * cols);
					TESTING_MALLOC_DEV(d_tau[g], Real, batchCount_gpu * cols);
					
					// Generate array of pointers for the cublas and magma routines
					TESTING_MALLOC_DEV(d_m_ptrs[g], Real*, batchCount_gpu);
					TESTING_MALLOC_DEV(d_tau_ptrs[g], Real*, batchCount_gpu);
					TESTING_MALLOC_DEV(d_info[g], int, batchCount_gpu);
					
					generateArrayOfPointers(d_m[g], d_m_ptrs[g], rows * cols, batchCount_gpu, 0);
					generateArrayOfPointers(d_tau[g], d_tau_ptrs[g], cols, batchCount_gpu, 0);
				}
				
				////////////////////////////////////////////////////////////////////////
				// Time the runs
				////////////////////////////////////////////////////////////////////////
				// Keep track of the times for each run
				kblas_err = magma_err = cublas_err = 0;

				for(i = 0; i < nruns; i++)
				{
					memcpy(m, m_original, sizeof(Real) * batchCount * rows * cols);
					cpu_time[i] = batch_qr_cpu(m, tau, rows, cols, batchCount, num_omp_threads);
					
					////////////////////////////////////////////////////////////////////////
					// KBLAS
					////////////////////////////////////////////////////////////////////////
					// Reset the GPU data and sync
					COPY_DATA_UP();
					// Launch all kernels
					for(g = 0; g < num_gpus; g++)
					{
						cudaSetDevice(opts.devices[g]);
						gpuTimerTic(kblas_timers[g]);
						// kblasXgeqrf_batched(kblas_handles[g], rows, cols, d_m[g], rows, rows * cols, d_tau[g], cols, batchCount_gpu);
						kblasXtsqrf_batched(kblas_handles[g], rows, cols, d_m[g], rows, rows * cols, d_tau[g], cols, batchCount_gpu);
						gpuTimerRecordEnd(kblas_timers[g]);
					}
					// The time all gpus finish at is the max of all the individual timers
					SYNC_TIMERS(kblas_timers, kblas_time);
					
					// Copy the data down from all the GPUs and compare with the CPU results
					COPY_DATA_DOWN();
					kblas_err += compare_results_R(gpu_results, m, rows, cols, batchCount);
					
					////////////////////////////////////////////////////////////////////////
					// MAGMA
					////////////////////////////////////////////////////////////////////////
					// Reset the GPU data and sync
					COPY_DATA_UP();
					// Launch all kernels
					for(g = 0; g < num_gpus; g++)
					{
						cudaSetDevice(opts.devices[g]);
						gpuTimerTic(magma_timers[g]);
						magmaXgeqrf_batched(
							rows, cols, d_m_ptrs[g], rows, d_tau_ptrs[g], d_info[g], 
							batchCount_gpu, queues[g]
						);
						gpuTimerRecordEnd(magma_timers[g]);
					}
					// The time all gpus finish at is the max of all the individual timers
					SYNC_TIMERS(magma_timers, magma_time);
					
					// Copy the data down from all the GPUs and compare with the CPU results
					COPY_DATA_DOWN();
					magma_err += compare_results_R(gpu_results, m, rows, cols, batchCount);
					
					////////////////////////////////////////////////////////////////////////
					// CUBLAS
					////////////////////////////////////////////////////////////////////////
					// Reset the GPU data and sync
					COPY_DATA_UP();
					// Launch all kernels
					for(g = 0; g < num_gpus; g++)
					{
						cudaSetDevice(opts.devices[g]);
						gpuTimerTic(cublas_timers[g]);
						check_cublas_error(
								cublasXgeqrf_batched(
									cublas_handles[g], rows, cols, d_m_ptrs[g], rows,
									d_tau_ptrs[g], &info, batchCount_gpu
							)
						);
						gpuTimerRecordEnd(cublas_timers[g]);
					}
					// The time all gpus finish at is the max of all the individual timers
					SYNC_TIMERS(cublas_timers, cublas_time);
					
					// Copy the data down from all the GPUs and compare with the CPU results
					COPY_DATA_DOWN();
					cublas_err += compare_results_R(gpu_results, m, rows, cols, batchCount);
				}

				avg_and_stdev(cpu_time, nruns, avg_cpu_time, sdev_cpu_time, warmup);
				avg_and_stdev(kblas_time, nruns, avg_kblas_time, sdev_kblas_time, warmup);
				avg_and_stdev(magma_time, nruns, avg_magma_time, sdev_magma_time, warmup);
				avg_and_stdev(cublas_time, nruns, avg_cublas_time, sdev_cublas_time, warmup);
				
				printf(
					"%-15d%-10d%-10d%-15.3f%-15.3f%-15.3f%-15.3e%-15.3e%-15.3e\n", 
					batchCount, rows, cols, hh_ops / avg_kblas_time, hh_ops / avg_magma_time, hh_ops / avg_cublas_time, 
					kblas_err / nruns, magma_err / nruns, cublas_err / nruns
				);
				
				////////////////////////////////////////////////////////////////////////
				// Free up the data
				////////////////////////////////////////////////////////////////////////
				TESTING_FREE_CPU(m); TESTING_FREE_CPU(m_original); 
				TESTING_FREE_CPU(tau); TESTING_FREE_CPU(gpu_results);

				// Allocate the data
				for(g = 0; g < num_gpus; g++)
				{
					cudaSetDevice(opts.devices[g]);
					
					TESTING_FREE_DEV(d_m[g]); TESTING_FREE_DEV(d_tau[g]);
					TESTING_FREE_DEV(d_m_ptrs[g]); TESTING_FREE_DEV(d_tau_ptrs[g]);
					TESTING_FREE_DEV(d_info[g]);
				}
			}
		}
	}

    return 0;
}