/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_l2/test_zgemv_mgpu.c

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

#define FMULS_GEMV(n) ((n) * (n) + 2. * (n))
#define FADDS_GEMV(n) ((n) * (n)           )

#define PRECISION_z

#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 6. * FMULS_GEMV(n) + 2. * FADDS_GEMV(n))
#else
#define FLOPS(n) (      FMULS_GEMV(n) +      FADDS_GEMV(n))
#endif

#define EVENT_TIMING

int main(int argc, char** argv)
{
	if(argc < 7)
	{
		printf("USAGE: %s <ngpus> <no-trans'n' or trans't' or conj-trans'c'> <start-dim> <stop-dim> <step-dim> <offset>\n", argv[0]);
		printf("==> <ngpus>: Number of GPUs to use \n");
		printf("==> <no-trans'n' or trans't' or conj-trans'c'>: Process the matrix in non-transposed,transposed, or conjugate transposed configuration \n");
		printf("==> <start-dim> <stop-dim> <step-dim>: test for dimensions in the range start-dim : stop-dim with step step-dim \n");
		printf("==> <offset>: skip the [0:offset-1] rows and the [0:offset-1] columns \n");
		exit(-1);
	}

	int ngpus = atoi(argv[1]);
	char trans = *argv[2];
	int istart = atoi(argv[3]);
	int istop = atoi(argv[4]);
	int istep = atoi(argv[5]);
	int offset = atoi(argv[6]);

	int i, j, m, k, r, dim;
	const int nruns = NRUNS;
	int gpus_avail;
	hipGetDeviceCount(&gpus_avail);
	if(ngpus > gpus_avail){printf("Error: Can't run on %d gpus, only %d gpus are available \n", ngpus, gpus_avail); exit(-1);}

	int ngpus_local = ngpus;	// for now (no mpi)

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

	int M = istop;
    int N = M;
    int LDA = M;
    int incx = 1;
	int incy = 1;

	hipblasOperation_t trans_;
	int alloc = 1, alloc_mgpu = 1;
	if(trans == 'N' || trans == 'n')
		trans_ = HIPBLAS_OP_N;
	else if (trans == 'T' || trans == 't')
		trans_ = HIPBLAS_OP_T;
	else if (trans == 'C' || trans == 'c')
		trans_ = HIPBLAS_OP_C;

	hipDoubleComplex alpha = make_hipDoubleComplex(2.3, 0);
	hipDoubleComplex beta  = make_hipDoubleComplex(-1.0, 0);

	hipError_t err;
	hipEvent_t start, stop;

	hipSetDevice(0);
	hipEventCreate(&start);
	hipEventCreate(&stop);

    // point to host memory
    hipDoubleComplex* A = NULL;
    hipDoubleComplex* x = NULL;
    hipDoubleComplex* y = NULL;
    hipDoubleComplex* ysingle = NULL;
    hipDoubleComplex* ykblas[MAX_NGPUS] = {NULL};

    if(trans == 'N' || trans == 'n')printf("Non-transposed test .. \n");
	else if (trans == 'T' || trans == 't') printf("Transposed case test .. \n");
	else if (trans == 'C' || trans == 'c') printf("Conjugate transposed test .. \n");
	else { printf("transpose/non-transpose configuration is not properly specified\n"); exit(-1);}

	// cpu alloc / init
	{
		int vecsize_x = N * abs(incx);
		int vecsize_y = M * abs(incy);
		// alloc memory cpu
		printf("Allocating matrices on cpu .. \n");
    	A = (hipDoubleComplex*)malloc(N*LDA*sizeof(hipDoubleComplex));
    	x = (hipDoubleComplex*)malloc(vecsize_x*sizeof(hipDoubleComplex));
    	y = (hipDoubleComplex*)malloc(vecsize_y*sizeof(hipDoubleComplex));
    	ysingle = (hipDoubleComplex*)malloc(vecsize_y*sizeof(hipDoubleComplex));
    	for(k = 0; k < ngpus; k++)
    		ykblas[k] = (hipDoubleComplex*)malloc(vecsize_y*sizeof(hipDoubleComplex));

    	printf("Initializing on cpu .. \n");
    	// Initialize matrix and vector on cpu
    	for(i = 0; i < M; i++)
    		for(j = 0; j < N; j++)
     				A[j*LDA+i] = kblas_zrand();

    	for(i = 0; i < vecsize_x; i++)
      		x[i] = kblas_zrand();

    	for(i = 0; i < vecsize_y; i++)
    		y[i] = kblas_zrand();
	}

	printf("------------------- Testing ZGEMV ------------------------\n");
	printf("  Matrix       KBLAS-1 GPU     KBLAS %-2d GPU(s)    Max.  \n", ngpus);
	printf(" Dimension     (Gflop/s)         (Gflop/s)         Error  \n");
	printf("-----------   -------------   ----------------   ----------\n");

	for(dim = istart; dim <= istop; dim += istep)
	{
		float elapsedTime;
		float flops = FLOPS( (float)(dim-offset) ) / 1e6;
		float single_gpu_perf;
		float mgpu_perf;
		double error;
		int m = dim;
		int n = m;
		int lda_ = ((m+31)/32)*32;

		printf("%-11d   ", m);

		// single gpu test
		{
			hipDoubleComplex* dA_single = NULL;
    		hipDoubleComplex* dx_single = NULL;
    		hipDoubleComplex* dy_single = NULL;

    		int vecsize_x = n * abs(incx);
			int vecsize_y = m * abs(incy);

			alloc = 1;
    		// alloc A, x, y on gpus
    		// for cublas test
    		hipError_t e1, e2, e3;
    		hipSetDevice(0);
    		e1 = hipMalloc((void**)&dA_single, n*lda_*sizeof(hipDoubleComplex));
    		e2 = hipMalloc((void**)&dx_single, vecsize_x*sizeof(hipDoubleComplex));
    		e3 = hipMalloc((void**)&dy_single, vecsize_y*sizeof(hipDoubleComplex));

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
    			hipblasSetMatrix(m, n, sizeof(hipDoubleComplex), A, LDA, dA_single, lda_);

    			float time = 0;
    			elapsedTime = 0.0;
    			for(r = 0; r < nruns; r++)
    			{
    				hipMemcpy(dy_single, y, vecsize_y*sizeof(hipDoubleComplex), hipMemcpyHostToDevice);
    				hipMemcpy(dx_single, x, vecsize_x*sizeof(hipDoubleComplex), hipMemcpyHostToDevice);

    				// handle offset
    				hipDoubleComplex* da = dA_single + (offset * lda_) + offset;
    				hipDoubleComplex* dx = dx_single + (offset * incx);
    				hipDoubleComplex* dy = dy_single + (offset * incy);
    				int m_ = m-offset;
    				int n_ = n-offset;

    				hipSetDevice(0);
      				hipEventRecord(start, 0);
      				//hipblasStatus_t s = hipblasZgemv(cublas_handle, trans_, m, n, &alpha, dA_single, lda_, dx_single, incx, &beta, dy_single, incy);
      				kblas_zgemv(trans, m_, n_, alpha, da, lda_, dx, incx, beta, dy, incy);
      				hipEventRecord(stop, 0);
      				hipEventSynchronize(stop);
      				time = 0.0;
      				hipEventElapsedTime(&time, start, stop);
      				elapsedTime += time;
      			}
      			elapsedTime /= nruns;
      			single_gpu_perf = flops / elapsedTime;

				hipMemcpy(ysingle, dy_single, vecsize_y * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost);

				printf("%-13.2f   ", single_gpu_perf);

				if(dA_single)hipFree(dA_single);
				if(dx_single)hipFree(dx_single);
				if(dy_single)hipFree(dy_single);
			}
		} // end of 1 gpu test

		// mgpu test
		{
			alloc_mgpu = 1;
    		// point to device memory
    		hipDoubleComplex* dA[MAX_NGPUS] = {NULL};
    		hipDoubleComplex* dx[MAX_NGPUS] = {NULL};
    		hipDoubleComplex* dy[MAX_NGPUS] = {NULL};

    		int vecsize_x = n * abs(incx);
			int vecsize_y = m * abs(incy);

    		// for kblas test
    		// alloc
    		int nb = get_zgemv_mgpu_bs(trans);
    		kblas_zmalloc_mgpu_1D(m, n, dA, ngpus, lda_, nb);

    		for(k = 0; k < ngpus; k++)
    		{
    			// check allocation
    			if(dA[k] == NULL) {alloc_mgpu = 0; break;}
    		}

    		hipError_t e1, e2;
    		for(k = 0; k < ngpus; k++)
    		{
    			hipSetDevice(k);
    			e1 = hipMalloc((void**)&dx[k], vecsize_x*sizeof(hipDoubleComplex));
    			e2 = hipMalloc((void**)&dy[k], vecsize_y*sizeof(hipDoubleComplex));
    			if( (e1 != hipSuccess) || (e2 != hipSuccess) )
    			{
    				alloc_mgpu = 0; break;
    			}
    			// init y
    			hipMemset(dy[k], 0, vecsize_y*sizeof(hipDoubleComplex));
    		}

    		if(alloc_mgpu == 1)
    		{
    			// offload (broadcast) x to all gpus
    			for(k = 0; k < ngpus; k++)
    			{
    				hipSetDevice(k);
    				hipError_t e = hipMemcpy(dx[k], x, vecsize_x*sizeof(hipDoubleComplex), hipMemcpyHostToDevice);
					//printf("gpu %d: hipMemcpy x - %s \n", k, hipGetErrorString(e));
				}

				// offload A and y on gpus
				kblas_zsetmatrix_mgpu_1D(m, n, A, LDA, dA, lda_, ngpus, nb);

				float time = 0;
				elapsedTime = 0.0;
				for(r = 0; r < nruns; r++)
				{
					// offload (broadcast) y to all gpus
    			    for(k = 0; k < ngpus; k++)
    			    {
    			    	hipSetDevice(k);
    				    hipError_t e = hipMemcpy(dy[k], y, vecsize_y*sizeof(hipDoubleComplex), hipMemcpyHostToDevice);
					}

    				// ---- kblas
      				hipSetDevice(0);
      				hipEventRecord(start, 0);
      				kblas_zgemv_mgpu_async(trans, m, n, alpha, dA, lda_, dx, incx, beta, dy, incy, ngpus, offset, offset, streams);
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
      				elapsedTime += time;
      			}

      			for(k = 0; k < ngpus; k++)
      			{
      				hipSetDevice(k);
      				hipMemcpy(ykblas[k], dy[k], vecsize_y * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost);
      			}
      			elapsedTime /= nruns;
      			mgpu_perf = flops / elapsedTime;
      			//reduce the result
      			//for(k = 0; k < vecsize_y; k++)
      			//	ykblas[0][k] = ykblas[1][k];
      			for(k = 0; k < vecsize_y; k++)
      				for(j = 1; j < ngpus_local; j++)
      				{
      					ykblas[0][k].x += ykblas[j][k].x;
      					ykblas[0][k].y += ykblas[j][k].y;
      				}

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
    	{
    		if(alloc == 1 && alloc_mgpu == 1)
    		{
    			// testing error -- specify ref. vector and result vector
      			hipDoubleComplex* yref = ysingle + (offset * incy);
      			hipDoubleComplex* yres = ykblas[0] + (offset * incy);
      			error = zget_max_error(yref, yres, m-offset, incy);
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

	// destroy streams
	for(k = 0; k < ngpus; k++)
	{
		hipSetDevice(k);
		hipStreamDestroy(streams[k][0]);
	}

    if(A)free(A);
    if(x)free(x);
    if(ysingle)free(ysingle);
    for(k = 0; k < ngpus_local; k++)if(ykblas[k])free(ykblas[k]);
	hipblasDestroy(cublas_handle);
    return EXIT_SUCCESS;
}

