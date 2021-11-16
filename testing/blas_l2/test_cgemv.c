/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_l2/test_cgemv.c

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

#define PRECISION_c

#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 6. * FMULS_GEMV(n) + 2. * FADDS_GEMV(n))
#else
#define FLOPS(n) (      FMULS_GEMV(n) +      FADDS_GEMV(n))
#endif

int main(int argc, char** argv)
{
	if(argc < 6)
	{
		printf("USAGE: %s <device-id> <no-trans'n' or trans't' or conj-trans'c'> <start-dim> <stop-dim> <step-dim>\n", argv[0]);
		printf("==> <device-id>: GPU device id to use \n");
		printf("==> <no-trans'n' or trans't' or conj-trans'c'>: Process the matrix in non-transposed,transposed, or conjugate transposed configuration \n");
		printf("==> <start-dim> <stop-dim> <step-dim>: test for dimensions in the range start-dim : stop-dim with step step-dim \n");
		exit(-1);
	}

	int dev = atoi(argv[1]);
	char trans = *argv[2];
	int istart = atoi(argv[3]);
	int istop = atoi(argv[4]);
	int istep = atoi(argv[5]);

	const int nruns = NRUNS;

	hipError_t ed = hipSetDevice(dev);
	if(ed != hipSuccess){printf("Error setting device : %s \n", hipGetErrorString(ed) ); exit(-1);}

	hipblasHandle_t cublas_handle;
	cublasAtomicsMode_t mode = CUBLAS_ATOMICS_ALLOWED;
	hipblasCreate(&cublas_handle);
	cublasSetAtomicsMode(cublas_handle, mode);

	struct hipDeviceProp_t deviceProp;
	hipGetDeviceProperties(&deviceProp, dev);

	int dim, r;

    int M = istop;
    int N = M;
    int LDA = M;
    int LDA_ = ((M+31)/32)*32;

	int incx = 1;
	int incy = 1;
	int vecsize_x = N * abs(incx);
	int vecsize_y = M * abs(incy);

	hipblasOperation_t trans_;
	if(trans == 'N' || trans == 'n')
		trans_ = HIPBLAS_OP_N;
	else if (trans == 'T' || trans == 't')
		trans_ = HIPBLAS_OP_T;
	else if (trans == 'C' || trans == 'c')
		trans_ = HIPBLAS_OP_C;

	hipFloatComplex alpha = make_hipFloatComplex(1.2, -0.6);
    hipFloatComplex beta = make_hipFloatComplex(-0.3 , 0.1);

	hipError_t err;
	hipEvent_t start, stop;

	hipEventCreate(&start);
	hipEventCreate(&stop);

    // point to host memory
    hipFloatComplex* A = NULL;
    hipFloatComplex* x = NULL;
    hipFloatComplex* ycuda = NULL;
    hipFloatComplex* ykblas = NULL;

    // point to device memory
    hipFloatComplex* dA = NULL;
    hipFloatComplex* dx = NULL;
    hipFloatComplex* dy = NULL;

    if(trans == 'N' || trans == 'n')printf("non-transposed test .. \n");
	else if (trans == 'T' || trans == 't') printf("transposed test .. \n");
	else if (trans == 'C' || trans == 'c') printf("Conjugate transposed test .. \n");
	else { printf("transpose configuration is not properly specified\n"); exit(-1);}
	printf("Allocating Matrices\n");

    A = (hipFloatComplex*)malloc(N*LDA*sizeof(hipFloatComplex));
    x = (hipFloatComplex*)malloc(vecsize_x*sizeof(hipFloatComplex));
    ycuda = (hipFloatComplex*)malloc(vecsize_y*sizeof(hipFloatComplex));
    ykblas = (hipFloatComplex*)malloc(vecsize_y*sizeof(hipFloatComplex));

    err = hipMalloc((void**)&dA, N*LDA_*sizeof(hipFloatComplex));
    if(err != hipSuccess){printf("ERROR: %s \n", hipGetErrorString(err)); exit(1);}
    err = hipMalloc((void**)&dx, vecsize_x*sizeof(hipFloatComplex));
    if(err != hipSuccess){printf("ERROR: %s \n", hipGetErrorString(err)); exit(1);}
    err = hipMalloc((void**)&dy, vecsize_y*sizeof(hipFloatComplex));
	if(err != hipSuccess){printf("ERROR: %s \n", hipGetErrorString(err)); exit(1);}

    // Initialize matrix and vector
    printf("Initializing on cpu .. \n");
    int i, j, m;
    for(i = 0; i < M ; i++)
    	for(j = 0; j < N; j++)
    	    A[j*LDA+i] = kblas_crand();

    for(i = 0; i < vecsize_x; i++)
      x[i] = kblas_crand();

    hipMemcpy(dA, A, M*N*sizeof(hipFloatComplex), hipMemcpyHostToDevice);
    hipMemcpy(dx, x, vecsize_x*sizeof(hipFloatComplex), hipMemcpyHostToDevice);


	printf("------------------- Testing CGEMV ----------------\n");
    printf("  Matrix        CUBLAS       KBLAS          Max.  \n");
    printf(" Dimension     (Gflop/s)   (Gflop/s)       Error  \n");
    printf("-----------   ----------   ----------   ----------\n");

    for(m = istart; m <= istop; m += istep)
    {
    	float elapsedTime;

      	int lda = ((m+31)/32)*32;
      	float flops = FLOPS( (float)m ) / 1e6;

		hipblasSetMatrix(m, m, sizeof(hipFloatComplex), A, LDA, dA, lda);

		for(i = 0; i < vecsize_y; i++)
    	{
      		ycuda[i] = kblas_crand();
      		ykblas[i].x = ycuda[i].x;
      		ykblas[i].y = ycuda[i].y;
    	}

      	// --- cuda test
      	elapsedTime = 0.0;
      	for(r = 0; r < nruns; r++)
      	{
      		hipMemcpy(dy, ycuda, vecsize_y * sizeof(hipFloatComplex), hipMemcpyHostToDevice);

      		hipEventRecord(start, 0);

      		hipblasCgemv(cublas_handle, trans_, m, m, &alpha, dA, lda, dx, incx, &beta, dy, incy);

      		hipEventRecord(stop, 0);
      		hipEventSynchronize(stop);

      		float time = 0;
      		hipEventElapsedTime(&time, start, stop);
      		elapsedTime += time;
      	}
      	elapsedTime /= nruns;
      	float cuda_perf = flops / elapsedTime;

      	hipMemcpy(ycuda, dy, vecsize_y * sizeof(hipFloatComplex), hipMemcpyDeviceToHost);
      	// end of cuda test

      	// ---- kblas
      	elapsedTime = 0.0;
      	for(r = 0; r < nruns; r++)
      	{
      		hipMemcpy(dy, ykblas, vecsize_y * sizeof(hipFloatComplex), hipMemcpyHostToDevice);

      		hipEventRecord(start, 0);

      		kblas_cgemv( trans, m, m, alpha, dA, lda, dx, incx, beta, dy, incy);

      		hipEventRecord(stop, 0);
      		hipEventSynchronize(stop);

      		float time = 0.0;
      		hipEventElapsedTime(&time, start, stop);
      		elapsedTime += time;
      	}
      	elapsedTime /= nruns;
      	float kblas_perf = flops / elapsedTime;

      	hipMemcpy(ykblas, dy, vecsize_y * sizeof(hipFloatComplex), hipMemcpyDeviceToHost);

      	// testing error -- specify ref. vector and result vector
      	hipFloatComplex* yref = ycuda;
      	hipFloatComplex* yres = ykblas;

      	float error = cget_max_error(yref, yres, m, incy);

      	//for(i = 0; i < m; i++) printf("[%d]:   %-8.2f   %-8.2f\n", i, ycuda[i], ykblas[i]);

      	//printf("-----------   ----------   ----------   ----------   ----------   ----------\n");
    	printf("%-11d   %-10.2f   %-10.2f   %-10e;\n", m, cuda_perf, kblas_perf, error);

    }

	hipEventDestroy(start);
	hipEventDestroy(stop);

    if(dA)hipFree(dA);
    if(dx)hipFree(dx);
    if(dy)hipFree(dy);

    if(A)free(A);
    if(x)free(x);
    if(ycuda)free(ycuda);
	if(ykblas)free(ykblas);

	hipblasDestroy(cublas_handle);
    return EXIT_SUCCESS;
}

