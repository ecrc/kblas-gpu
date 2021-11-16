/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_l2/test_zhemv.c

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

#define PRECISION_z

#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 6. * FMULS_SYMV(n) + 2. * FADDS_SYMV(n))
#else
#define FLOPS(n) (      FMULS_SYMV(n) +      FADDS_SYMV(n))
#endif

int main(int argc, char** argv)
{
	if(argc < 6)
	{
		printf("USAGE: %s <device-id> <upper'u'-or-lower'l'> <start-dim> <stop-dim> <step-dim>\n", argv[0]);
		printf("==> <device-id>: GPU device id to use \n");
		printf("==> <upper'u'-or-lower'l'>: Access either upper or lower triangular part of the matrix \n");
		printf("==> <start-dim> <stop-dim> <step-dim>: test for dimensions in the range start-dim : stop-dim with step step-dim \n");
		exit(-1);
	}

	int dev = atoi(argv[1]);
	char uplo = *argv[2];
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

    int M = istop;
    int N = M;
    int LDA = M;
    int LDA_ = ((M+31)/32)*32;

	int incx = 1;
	int incy = 1;
	int vecsize_x = N * abs(incx);
	int vecsize_y = M * abs(incy);

	hipblasFillMode_t uplo_;
	if(uplo == 'L' || uplo == 'l')
		uplo_ = HIPBLAS_FILL_MODE_LOWER;
	else if (uplo == 'U' || uplo == 'u')
		uplo_ = HIPBLAS_FILL_MODE_UPPER;

	hipError_t err;
	hipEvent_t start, stop;

	hipEventCreate(&start);
	hipEventCreate(&stop);

    // point to host memory
    hipDoubleComplex* A = NULL;
    hipDoubleComplex* x = NULL;
    hipDoubleComplex* ycuda = NULL;
    hipDoubleComplex* ykblas = NULL;

    // point to device memory
    hipDoubleComplex* dA = NULL;
    hipDoubleComplex* dx = NULL;
    hipDoubleComplex* dy = NULL;

    if(uplo == 'L' || uplo == 'l')printf("Lower triangular test .. \n");
	else if (uplo == 'U' || uplo == 'u') printf("Upper triangular test .. \n");
	else { printf("upper/lower triangular configuration is not properly specified\n"); exit(-1);}
	printf("Allocating Matrices\n");
    A = (hipDoubleComplex*)malloc(N*LDA*sizeof(hipDoubleComplex));
    x = (hipDoubleComplex*)malloc(vecsize_x*sizeof(hipDoubleComplex));
    ycuda = (hipDoubleComplex*)malloc(vecsize_y*sizeof(hipDoubleComplex));
    ykblas = (hipDoubleComplex*)malloc(vecsize_y*sizeof(hipDoubleComplex));

    err = hipMalloc((void**)&dA, LDA_*N*sizeof(hipDoubleComplex));
    if(err != hipSuccess){printf("ERROR: %s \n", hipGetErrorString(err)); exit(1);}
    err = hipMalloc((void**)&dx, vecsize_x*sizeof(hipDoubleComplex));
    if(err != hipSuccess){printf("ERROR: %s \n", hipGetErrorString(err)); exit(1);}
    err = hipMalloc((void**)&dy, vecsize_y*sizeof(hipDoubleComplex));
	if(err != hipSuccess){printf("ERROR: %s \n", hipGetErrorString(err)); exit(1);}

    // Initialize matrix and vector
    printf("Initializing on cpu .. \n");
    int i, j, m;
    for(i = 0; i < M; i++)
    		for(j = 0; j < N; j++)
     				A[j*LDA+i] = kblas_zrand();

    // make 'A' Hermitian
    {
        int i, j;
        for(i=0; i<M; i++) {
            //A[i*LDA+i] = make_hipDoubleComplex( A[i*LDA+i].x, 0.0 );
            for(j=0; j<i; j++)
                A[i*LDA+j] = hipConj(A[j*LDA+i]);
        }
    }

    // init vector 'x'
    for(i = 0; i < vecsize_x; i++)
      x[i] = kblas_zrand();

    hipMemcpy(dx, x, vecsize_x*sizeof(hipDoubleComplex), hipMemcpyHostToDevice);

	printf("------------------- Testing ZHEMV ----------------\n");
    printf("  Matrix        CUBLAS       KBLAS          Max.  \n");
    printf(" Dimension     (Gflop/s)   (Gflop/s)       Error  \n");
    printf("-----------   ----------   ----------   ----------\n");

    // init alpha and beta
    hipDoubleComplex alpha = make_hipDoubleComplex(1.2, -0.6);
    hipDoubleComplex beta = make_hipDoubleComplex(2.5, 0.1);

    int r;
    for(m = istart; m <= istop; m += istep)
    {
    	float elapsedTime;

      	int lda = ((m+31)/32)*32;
      	float flops = FLOPS( (float)m ) / 1e6;

		hipblasSetMatrix(m, m, sizeof(hipDoubleComplex), A, LDA, dA, lda);

		for(i = 0; i < vecsize_y; i++)
    	{
      		ycuda[i] = kblas_zrand();
      		ykblas[i].x = ycuda[i].x;
      		ykblas[i].y = ycuda[i].y;
    	}

      	// --- cuda test
      	elapsedTime = 0;
      	for(r = 0; r < nruns; r++)
      	{
      		hipMemcpy(dy, ycuda, vecsize_y * sizeof(hipDoubleComplex), hipMemcpyHostToDevice);

      		hipEventRecord(start, 0);
      		cublasZhemv(cublas_handle, uplo_, m, &alpha, dA, lda, dx, incx, &beta, dy, incy);
      		hipEventRecord(stop, 0);
      		hipEventSynchronize(stop);

      		float time = 0.0;
      		hipEventElapsedTime(&time, start, stop);
      		elapsedTime += time;
      	}
      	elapsedTime /= nruns;
      	float cuda_perf = flops / elapsedTime;

      	hipMemcpy(ycuda, dy, vecsize_y * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost);
      	// end of cuda test

      	// ---- kblas
      	elapsedTime = 0;
      	for(r = 0; r < nruns; r++)
      	{
      		hipMemcpy(dy, ykblas, vecsize_y * sizeof(hipDoubleComplex), hipMemcpyHostToDevice);

      		hipEventRecord(start, 0);
      		kblas_zhemv(uplo, m, alpha, dA, lda, dx, incx, beta, dy, incy);
      		hipEventRecord(stop, 0);
      		hipEventSynchronize(stop);

      		float time = 0.0;
      		hipEventElapsedTime(&time, start, stop);
      		elapsedTime += time;
      	}
      	elapsedTime /= nruns;
      	float kblas_perf = flops / elapsedTime;

      	hipMemcpy(ykblas, dy, vecsize_y * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost);

      	// testing error -- specify ref. vector and result vector
      	hipDoubleComplex* yref = ycuda;
      	hipDoubleComplex* yres = ykblas;

      	double error = zget_max_error(yref, yres, m, incy);

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

