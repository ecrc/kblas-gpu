/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_l2/test_dsymv_mgpu.c

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ahmad Abdelfattah
 * @date 2018-11-14
 **/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas.h>
#include "kblas.h"
#include "testing_utils.h"

#define FMULS_SYMV(n) ((n) * (n) + 2. * (n))
#define FADDS_SYMV(n) ((n) * (n)           )

#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 6. * FMULS_SYMV(n) + 2. * FADDS_SYMV(n))
#else
#define FLOPS(n) (      FMULS_SYMV(n) +      FADDS_SYMV(n))
#endif

#define EVENT_TIMING

int main(int argc, char** argv)
{
	if(argc < 7)
	{
		printf("USAGE: %s <ngpus> <upper'u'-or-lower'l'> <start-dim> <stop-dim> <step-dim> <offset>\n", argv[0]);
		printf("==> <ngpus>: Number of GPUs to use \n");
		printf("==> <upper'u'-or-lower'l'>: Access either the upper or lower triangular part of the matrix \n");
		printf("==> <start-dim> <stop-dim> <step-dim>: test for dimensions in the range start-dim : stop-dim with step step-dim \n");
		printf("==> <offset>: skip the [0:offset-1] rows and the [0:offset-1] columns \n");
		exit(-1);
	}

	int ngpus = atoi(argv[1]);
	char uplo = *argv[2];
	int istart = atoi(argv[3]);
	int istop = atoi(argv[4]);
	int istep = atoi(argv[5]);
	int offset = atoi(argv[6]);

	int i, j, m, k, r, dim;
	const int nruns = NRUNS;

	int gpus_avail;
	hipGetDeviceCount(&gpus_avail);
	if(ngpus > gpus_avail){printf("Error: Can't run on %d gpus, only %d gpus are available \n", ngpus, gpus_avail); exit(-1);}

	int ngpus_local = ngpus;	// for now (without mpi)

	hipStream_t streams[MAX_NGPUS][MAX_STREAMS];

	hipblasHandle_t cublas_handle;
	cublasAtomicsMode_t mode = CUBLAS_ATOMICS_ALLOWED;
	hipblasCreate(&cublas_handle);
	cublasSetAtomicsMode(cublas_handle, mode);

	// create streams
	for(k = 0; k < ngpus; k++)
	{
		hipSetDevice(k);
		hipStreamCreate(&streams[k][0]);
	}

	int N = istop;
    int M = N;
    int LDA = M;
    int incx = 1;
	int incy = 1;

	hipblasFillMode_t uplo_;
	if(uplo == 'L' || uplo == 'l')
		uplo_ = HIPBLAS_FILL_MODE_LOWER;
	else if (uplo == 'U' || uplo == 'u')
		uplo_ = HIPBLAS_FILL_MODE_UPPER;

	double alpha = 2.3, beta = -1.0;

	hipError_t err;
	hipEvent_t start, stop;

	hipSetDevice(0);
	hipEventCreate(&start);
	hipEventCreate(&stop);

	#ifdef EVENT_TIMING
	hipEvent_t _start[MAX_NGPUS], _stop[MAX_NGPUS];
	for(k = 0; k < ngpus; k++)
	{
		hipSetDevice(k);
		hipEventCreate(&_start[k]);
		hipEventCreate(&_stop[k]);
	}
	#endif

    // point to host memory
    double* A = NULL;
    double* x = NULL;
    double* y = NULL;
    double* ycuda = NULL;
    double* ykblas[MAX_NGPUS] = {NULL};

    if(uplo == 'L' || uplo == 'l')printf("Lower triangular test .. \n");
	else if (uplo == 'U' || uplo == 'u') printf("Upper triangular test .. \n");
	else { printf("upper/lower triangular configuration is not properly specified\n"); exit(-1);}

	// cpu alloc / init
	{
		int vecsize_x = N * abs(incx);
		int vecsize_y = N * abs(incy);
		// alloc memory cpu
		printf("Allocating matrices on cpu .. \n");
    	A = (double*)malloc(N*LDA*sizeof(double));
    	x = (double*)malloc(vecsize_x*sizeof(double));
    	y = (double*)malloc(vecsize_x*sizeof(double));
    	ycuda = (double*)malloc(vecsize_y*sizeof(double));
    	for(k = 0; k < ngpus; k++)
    		ykblas[k] = (double*)malloc(vecsize_y*sizeof(double));

    	printf("Initializing on cpu .. \n");
    	// Initialize matrix and vector on cpu
    	for(i = 0; i < M; i++)
    		for(j = 0; j < N; j++)
    			A[j*LDA+i] = kblas_drand();

    	for(i = 0; i < M; i++)
      		for(j = i; j < M; j++)
				A[i * N + j] = A[j * M + i];

    	for(i = 0; i < vecsize_x; i++)
      		x[i] = kblas_drand();

    	for(i = 0; i < vecsize_y; i++)
    		y[i] = kblas_drand();
	}

	int alloc = 1, alloc_mgpu = 1;

	printf("------------------- Testing DSYMV ------------------------\n");
	printf("  Matrix       KBLAS-1 GPU     KBLAS-%-2d GPU(s)    Max.  \n", ngpus);
	printf(" Dimension     (Gflop/s)         (Gflop/s)         Error  \n");
	printf("-----------   -------------   ----------------   ----------\n");

	int test_single = 1;
	for(dim = istart; dim <= istop; dim += istep)
	{
		float elapsedTime;
		float flops = FLOPS( (float)(dim-offset) ) / 1e6;
		float single_gpu_perf = 0.0;
		float mgpu_perf = 0.0;
		double error;
		int m = dim;
		int n = m;
		int lda_ = ((m+31)/32)*32;

		printf("%-11d   ", m);


		// single gpu test
		if(test_single)
		{
			double* dA_single = NULL;
    		double* dx_single = NULL;
    		double* dy_single = NULL;

    		int vecsize_x = m * abs(incx);
			int vecsize_y = m * abs(incy);

			alloc = 1;
    		// alloc A, x, y on gpus
    		// for cublas test
    		hipError_t e1, e2, e3;
    		hipSetDevice(0);
    		e1 = hipMalloc((void**)&dA_single, n*lda_*sizeof(double));
    		e2 = hipMalloc((void**)&dx_single, vecsize_x*sizeof(double));
    		e3 = hipMalloc((void**)&dy_single, vecsize_y*sizeof(double));

    		if((e1 != hipSuccess) || (e2 != hipSuccess) || (e3 != hipSuccess) )
    		{
    			if(dA_single)hipFree(dA_single);
    			if(dx_single)hipFree(dx_single);
    			if(dy_single)hipFree(dy_single);
    			alloc = 0;
    			printf("-----           ");
    		}

    		if(alloc == 1)
    		{
    			// offload A, y to gpus
    			hipSetDevice(0);
    			hipblasSetMatrix(m, m, sizeof(double), A, LDA, dA_single, lda_);
    			hipMemcpy(dx_single, x, vecsize_x*sizeof(double), hipMemcpyHostToDevice);

    			elapsedTime = 0.0;
      			for(r = 0; r < nruns; r++)
      			{
    				hipMemcpy(dy_single, y, vecsize_y*sizeof(double), hipMemcpyHostToDevice);

    				// handle offset
    				double* da = dA_single + (offset * lda_) + offset;
    				double* dx = dx_single + (offset * incx);
    				double* dy = dy_single + (offset * incy);
    				int m_ = m-offset;

    				hipSetDevice(0);
      				hipEventRecord(start, 0);
      				//hipblasStatus_t s = cublasDsymv(cublas_handle, uplo_, m, &alpha, dA_single, lda_, dx_single, incx, &beta, dy_single, incy);
      				kblas_dsymv(uplo, m_, alpha, da, lda_, dx, incx, beta, dy, incy);
      				hipEventRecord(stop, 0);
      				hipEventSynchronize(stop);
      				float time = 0;
      				hipEventElapsedTime(&time, start, stop);
      				elapsedTime += time;
      			}
      			elapsedTime /= nruns;
      			single_gpu_perf = flops / elapsedTime;

				hipMemcpy(ycuda, dy_single, vecsize_y * sizeof(double), hipMemcpyDeviceToHost);

				printf("%-13.2f   ", single_gpu_perf);

				if(dA_single)hipFree(dA_single);
				if(dx_single)hipFree(dx_single);
				if(dy_single)hipFree(dy_single);
			}
		} // end of 1 gpu test

		// mgpu test
		{
    		// point to device memory
    		double* dA[MAX_NGPUS] = {NULL};
    		double* dx[MAX_NGPUS] = {NULL};
    		double* dy[MAX_NGPUS] = {NULL};

    		int vecsize_x = m * abs(incx);
			int vecsize_y = m * abs(incy);

			alloc_mgpu = 1;
    		// for kblas test
    		// alloc
    		int nb = get_dsymv_mgpu_bs(uplo);
    		kblas_dmalloc_mgpu_1D(m, m, dA, ngpus, lda_, nb);

    		for(k = 0; k < ngpus; k++)
    		{
    			// check allocation
    			if(dA[k] == NULL) {alloc_mgpu = 0; break;}
    		}

    		hipError_t e1, e2;
    		for(k = 0; k < ngpus; k++)
    		{
    			hipSetDevice(k);
    			e1 = hipMalloc((void**)&dx[k], vecsize_x*sizeof(double));
    			e2 = hipMalloc((void**)&dy[k], vecsize_y*sizeof(double));
    			if( (e1 != hipSuccess) || (e2 != hipSuccess) )
    			{
    				alloc_mgpu = 0; break;
    			}
    			// init y
    			hipMemset(dy[k], 0, vecsize_y*sizeof(double));
    		}

    		if(alloc_mgpu == 1)
    		{
    			// offload (broadcast) x to all gpus
    			for(k = 0; k < ngpus; k++)
    			{
    				hipSetDevice(k);
    				hipMemcpy(dx[k], x, vecsize_x*sizeof(double), hipMemcpyHostToDevice);
				}

				// offload A and y on gpus
    			kblas_dsetmatrix_mgpu_1D(m, m, A, LDA, dA, lda_, ngpus, nb);

				float time = 0;
				float _time = 0;
				elapsedTime = 0;
				for(r = 0; r < nruns; r++)
				{
					// offload y among gpus
					kblas_dsetvector_mgpu_1D(m, y, dy, ngpus, nb);

    				// ---- kblas
      				hipSetDevice(0);
      				hipEventRecord(start, 0);

      				#ifdef EVENT_TIMING
      				for(k = 0; k < ngpus; k++){hipSetDevice(k); hipEventRecord(_start[k], streams[k][0]);}
      				#endif

      				kblas_dsymv_mgpu_async(uplo, m, alpha, dA, lda_, dx, incx, beta, dy, incy, ngpus, offset, streams);

      				#ifdef EVENT_TIMING
      				for(k = 0; k < ngpus; k++){hipSetDevice(k); hipEventRecord(_stop[k], streams[k][0]);}
      				#endif

      				// sync
					for(k = 0; k < ngpus; k++)
					{
						hipSetDevice(k);
						hipStreamSynchronize(streams[k][0]);
					}

      				hipSetDevice(0);
      				hipEventRecord(stop, 0);
      				hipEventSynchronize(stop);
      				time = 0;
      				hipEventElapsedTime(&time, start, stop);

      				// event timing
      				#ifdef EVENT_TIMING
      				float gpu_time = 0, max_gpu_time = 0;
      				for(k = 0; k < ngpus; k++)
					{
						hipSetDevice(k);
						hipEventElapsedTime(&gpu_time, _start[k], _stop[k]);
						if(gpu_time > max_gpu_time) max_gpu_time = gpu_time;
					}
      				_time = max_gpu_time;
      				#endif
      				// end of event timing

      				#ifndef EVENT_TIMING
      				elapsedTime += time;
      				#else
      				elapsedTime += _time;
      				#endif
      			}

      			for(k = 0; k < ngpus; k++)
      			{
      				hipSetDevice(k);
      				hipMemcpy(ykblas[k], dy[k], vecsize_y * sizeof(double), hipMemcpyDeviceToHost);
      			}

      			//reduce the result
      			for(k = 0; k < vecsize_y; k++)
      				for(j = 1; j < ngpus_local; j++)
      					ykblas[0][k] += ykblas[j][k];

      			elapsedTime /= nruns;
      			mgpu_perf = flops / elapsedTime;
      			printf("%-16.2f   ", mgpu_perf);
      		}
      		else printf("-----           ");

      		for(k = 0; k < ngpus_local; k++)
			{
				hipSetDevice(k);
				if(dA[k])hipFree(dA[k]);
				if(dx[k])hipFree(dx[k]);
				if(dy[k])hipFree(dy[k]);
			}
    	} // end of mgpu test


    	// testing correctness
    	if(test_single)
    	{
    		if(alloc == 1 && alloc_mgpu == 1)
    		{
    			// testing error -- specify ref. vector and result vector
      			double* yref = ycuda;
      			double* yres = ykblas[0];

      			error = dget_max_error(yref, yres, m, incy);
      			//print
    			printf("%-10e;\n", error);
    		}
    		else
    			printf ("N/A       \n");
    	}

    	if(alloc == 0 && alloc_mgpu == 0)break;
	} // end of for loop

	// finalize
	hipSetDevice(0);
	hipEventDestroy(start);
	hipEventDestroy(stop);
	#ifdef EVENT_TIMING
	for(k = 0; k < ngpus; k++)
	{
		hipSetDevice(k);
		hipEventDestroy(_start[k]);
		hipEventDestroy(_stop[k]);
	}
	#endif

	// destroy streams
	for(k = 0; k < ngpus; k++)
	{
		hipSetDevice(k);
		hipStreamDestroy(streams[k][0]);
	}

    if(A)free(A);
    if(x)free(x);
    if(ycuda)free(ycuda);
    for(k = 0; k < ngpus_local; k++)if(ykblas[k])free(ykblas[k]);
	hipblasDestroy(cublas_handle);
    return EXIT_SUCCESS;
}

