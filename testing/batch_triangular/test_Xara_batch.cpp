#include <stdio.h>
#include <math.h>
#include <string.h>
#include <algorithm>

#include <cublas_v2.h>
#include <curand.h>
#include "kblas.h"

#include "testing_helper.h"
#include "batch_rand.h"

#ifdef PREC_d
typedef double Real;
#define avg_and_stdev	avg_and_stdev 
#else
typedef float Real;
#define avg_and_stdev	avg_and_stdevf
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
		int entries = batchCount_gpu * rows * cols; \
		check_error( cudaMemcpy(gpu_results + entries * g, d_A[g], entries * sizeof(Real), cudaMemcpyDeviceToHost ) );	\
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

template<class T>
inline int generateRandomMatrices(T* d_m, int rows, int cols, unsigned int seed, int num_ops, curandGenerator_t gen);

template<>
inline int generateRandomMatrices(float* d_m, int rows, int cols, unsigned int seed, int num_ops, curandGenerator_t gen)
{
    curandGenerateNormal(gen, d_m, num_ops * rows * cols, 0, 1);
	check_error( cudaGetLastError() );
	return 0;
}

template<>
inline int generateRandomMatrices(double* d_m, int rows, int cols, unsigned int seed, int num_ops, curandGenerator_t gen)
{
    curandGenerateNormalDouble(gen, d_m, num_ops * rows * cols, 0, 1);
	check_error( cudaGetLastError() );
	return 0;
}


void printResults(RealArray A_batch, int ld, int rows, int cols, int batchCount)
{
	for(int i = 0; i < batchCount; i++)
		printMatrix(rows, cols, A_batch + rows * cols * i, ld, stdout);
}

void avg_and_stdevf(float* values, int num_vals, float* avg, float* std_dev, int warmup)
{
	if(num_vals == 0) return;

	int start = 0;
	if(warmup == 1 && num_vals != 1)
		start = 1;

	*avg = 0;
	for(int i = start; i < num_vals; i++)
		*avg += values[i];
	*avg /= num_vals;

	*std_dev = 0;
	for(int i = 0; i < num_vals; i++)
		*std_dev += (values[i] - *avg) * (values[i] - *avg);
	*std_dev = sqrt(*std_dev / num_vals);
}

int main(int argc, char** argv)
{
	kblas_opts opts;
	parse_opts(argc, argv, &opts);
	int num_gpus = opts.ngpu;
	int rows, cols;
	int batchCount, batchCount_gpu;
	
	// Host stuff
	RealArray host_A, gpu_results;
	
	// GPU stuff
	RealArray d_A[num_gpus];
	
	GPU_Timer_t kblas_timers[num_gpus];
	kblasHandle_t kblas_handles[num_gpus];
	kblasRandState_t rand_state[num_gpus];
	
	curandGenerator_t gen[num_gpus];
	
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
		
		curandCreateGenerator(&gen[g], CURAND_RNG_PSEUDO_DEFAULT);
		curandSetStream(gen[g], 0);
		curandSetPseudoRandomGeneratorSeed(gen[g], 0);
    }
	
	for(int itest = 0; itest < opts.ntest; ++itest)
	{
		rows = opts.msize[itest];
		cols = opts.nsize[itest];
		
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
				TESTING_MALLOC_CPU(host_A, Real, batchCount * rows * cols);
				TESTING_MALLOC_CPU(gpu_results, Real, batchCount * rows * cols);
				
				for(int g = 0; g < num_gpus; g++)
				{
					cudaSetDevice(opts.devices[g]);
					TESTING_MALLOC_DEV(d_A[g], Real, batchCount_gpu * rows * cols);
					kblas_rand_batch(kblas_handles[g], rows, cols, d_A[g], rows, rows * cols, rand_state[g], batchCount_gpu);
				}
				
				// Copy the results 
				COPY_DATA_DOWN();
				
				// printResults(gpu_results, rows, rows, cols, batchCount);
				if(opts.check)
				{
					std::sort(gpu_results, gpu_results + rows * cols * batchCount);
					int duplicates = 0, zeros = 0;
					for(int i = 0; i < rows * cols * batchCount - 1; i++)
						if(gpu_results[i] == gpu_results[i+1])
							duplicates++;
						else if(gpu_results[i] == 0)
							zeros++;
						
					Real avg, stdev;
					avg_and_stdev(gpu_results, rows * cols * batchCount, &avg, &stdev, 0);
					printf("%d numbers generated (%d duplicates %d zeros) with %e mean and %e variance\n", rows * cols * batchCount, duplicates, zeros, avg, stdev * stdev);
				}
				
				for(int g = 0; g < num_gpus; g++)
					generateRandomMatrices(d_A[g], rows, cols, g, batchCount_gpu, gen[g]);
				
				// Free the data
				TESTING_FREE_CPU(host_A);
				TESTING_FREE_CPU(gpu_results);
				
				for(int g = 0; g < num_gpus; g++)
				{
					cudaSetDevice(opts.devices[g]);
					TESTING_FREE_DEV(d_A[g]); 
				}
			}
		}
	}

    return 0;
}
