/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_l2/test_zgemv2.c

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
#include <kblas.h>
#include "testing_utils.h"

#define FMULS_GEMV(m_, n_) ((m_) * (n_) + 2. * (m_))
#define FADDS_GEMV(m_, n_) ((m_) * (n_)           )

#define PRECISION_c

#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(m, n) ( 6. * FMULS_GEMV(m, n) + 2. * FADDS_GEMV(m, n))
#else
#define FLOPS(m, n) (      FMULS_GEMV(m, n) +      FADDS_GEMV(m, n))
#endif

int main(int argc, char** argv)
{
	if(argc < 7)
	{
		printf("USAGE: %s <device-id> <no-trans'n' or trans't' or conj-trans'c'> <start-dim> <stop-dim> <step-dim> <ratio>\n", argv[0]);
		printf("==> <device-id>: GPU device id to use \n");
		printf("==> <no-trans'n' or trans't' or conj-trans'c'>: Process the matrix in non-transposed,transposed, or conjugate transposed configuration \n");
		printf("==> <start-dim> <stop-dim> <step-dim>: test for dimensions (#rows) in the range start-dim : stop-dim with step step-dim \n");
		printf("==> <ratio>: Integer > 0, the ratio between rows and cols, ratio = (rows/cols)\n");
		exit(-1);
	}

	int dev = atoi(argv[1]);
	char trans = *argv[2];
	int istart = atoi(argv[3]);
	int istop = atoi(argv[4]);
	int istep = atoi(argv[5]);
	int ratio = atoi(argv[6]);

	const int nruns = NRUNS;

	hipError_t ed = hipSetDevice(dev);
	if(ed != hipSuccess){printf("Error setting device : %s \n", hipGetErrorString(ed) ); exit(-1);}

	hipblasHandle_t cublas_handle;
	cublasAtomicsMode_t mode = CUBLAS_ATOMICS_ALLOWED;
	hipblasCreate(&cublas_handle);
	cublasSetAtomicsMode(cublas_handle, mode);

	struct hipDeviceProp_t deviceProp;
	hipGetDeviceProperties(&deviceProp, dev);

	if(ratio == 0){printf("Ratio must not be zero\n"); exit(1);}

    int M = istop;
    int N = M/ratio;
    int LDA = M;
    int LDA_ = ((M+31)/32)*32;

	int incx = 1;
	int incy = 1;

	int vecsize_x;
	int vecsize_y;

	if(trans == 'n' || trans == 'N')
	{
		vecsize_x = N * abs(incx);
		vecsize_y = M * abs(incy);
	}
	else
	{
		vecsize_x = M * abs(incx);
		vecsize_y = N * abs(incy);
	}

	hipblasOperation_t trans_;
	if(trans == 'N' || trans == 'n')
		trans_ = HIPBLAS_OP_N;
	else if (trans == 'T' || trans == 't')
		trans_ = HIPBLAS_OP_T;
	else if (trans == 'C' || trans == 'c')
		trans_ = HIPBLAS_OP_C;

	hipDoubleComplex alpha = kblas_zrand();
	hipDoubleComplex beta  = kblas_zrand();

	hipError_t err;
	hipEvent_t start, stop;

	hipEventCreate(&start);
	hipEventCreate(&stop);

    // point to host memory
    hipDoubleComplex* A = NULL;
    hipDoubleComplex* x = NULL;
    hipDoubleComplex* ycuda = NULL;
    hipDoubleComplex* ykblas = NULL;
    hipDoubleComplex* ykblas2 = NULL;

    // point to device memory
    hipDoubleComplex* dA = NULL;
    hipDoubleComplex* dx = NULL;
    hipDoubleComplex* dy = NULL;

    if(trans == 'N' || trans == 'n')printf("non-transposed test .. \n");
	else if (trans == 'T' || trans == 't') printf("transposed test .. \n");
	else if (trans == 'C' || trans == 'c') printf("Conjugate transposed test .. \n");
	else { printf("transpose configuration is not properly specified\n"); exit(-1);}
	printf("Allocating Matrices\n");
    A = (hipDoubleComplex*)malloc(N*LDA*sizeof(hipDoubleComplex));
    x = (hipDoubleComplex*)malloc(vecsize_x*sizeof(hipDoubleComplex));
    ycuda   = (hipDoubleComplex*)malloc(vecsize_y*sizeof(hipDoubleComplex));
    ykblas  = (hipDoubleComplex*)malloc(vecsize_y*sizeof(hipDoubleComplex));
    ykblas2 = (hipDoubleComplex*)malloc(vecsize_y*sizeof(hipDoubleComplex));

    err = hipMalloc((void**)&dA, N*LDA_*sizeof(hipDoubleComplex));
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

    for(i = 0; i < vecsize_x; i++)
      x[i] = kblas_zrand();

    //hipMemcpy(dA, A, M*N*sizeof(hipDoubleComplex), hipMemcpyHostToDevice);
    hipMemcpy(dx, x, vecsize_x*sizeof(hipDoubleComplex), hipMemcpyHostToDevice);


    printf("--------------------------------------- Testing ZGEMV --------------------------------------\n");
    printf("  Matrix      Matrix       CUBLAS      KBLAS-GEMV   KBLAS-GEMV2   KBLAS-GEMV     KBLAS-GEMV2\n");
    printf("   Rows       Columns     (Gflop/s)    (Gflop/s)    (Gflop/s)     MAX Error       MAX Error \n");
    printf("----------- -----------   ----------   ----------   -----------   ------------   -----------\n");

    int r;
    for(m = istart; m <= istop; m += istep)
    {
    	float elapsedTime;

      	int lda = ((m+31)/32)*32;
      	int n = (m/ratio);
      	if(n == 0) n = 1;

      	float flops = FLOPS( (float)m, (float)n ) / 1e6;

		hipblasSetMatrix(m, n, sizeof(hipDoubleComplex), A, LDA, dA, lda);

		for(i = 0; i < vecsize_y; i++)
    	{
      		ycuda[i] = kblas_zrand();
      		ykblas[i] = ycuda[i];
      		ykblas2[i] = ycuda[i];
    	}

      	// --- cuda test
      	elapsedTime  = 0;
      	for(r = 0; r < nruns; r++)
      	{
      		hipMemcpy(dy, ycuda, vecsize_y * sizeof(hipDoubleComplex), hipMemcpyHostToDevice);

      		hipEventRecord(start, 0);
      		hipblasZgemv(cublas_handle, trans_, m, n, &alpha, dA, lda, dx, incx, &beta, dy, incy);
      		hipEventRecord(stop, 0);
      		hipEventSynchronize(stop);
      		float time = 0;
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
      		kblas_zgemv( trans, m, n, alpha, dA, lda, dx, incx, beta, dy, incy);
      		hipEventRecord(stop, 0);
      		hipEventSynchronize(stop);

      		float time = 0;
      		hipEventElapsedTime(&time, start, stop);
      		elapsedTime += time;
      	}
      	elapsedTime /= nruns;
      	float kblas_perf = flops / elapsedTime;


      	hipMemcpy(ykblas, dy, vecsize_y * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost);

      	// ---- kblas-2
      	elapsedTime = 0;
      	for(r = 0; r < nruns; r++)
      	{
      		hipMemcpy(dy, ykblas2, vecsize_y * sizeof(hipDoubleComplex), hipMemcpyHostToDevice);

      		hipEventRecord(start, 0);
      		kblas_zgemv2(trans, m, n, alpha, dA, lda, dx, incx, beta, dy, incy);
      		hipEventRecord(stop, 0);
      		hipEventSynchronize(stop);

      		float time = 0;
      		hipEventElapsedTime(&time, start, stop);
      		elapsedTime += time;
      	}
      	elapsedTime /= nruns;
      	float kblas2_perf = flops / elapsedTime;


      	hipMemcpy(ykblas2, dy, vecsize_y * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost);

      	// testing error -- specify ref. vector and result vector
      	hipDoubleComplex* yref;
      	hipDoubleComplex* yres;

      	// error1
      	yref = ycuda;
		yres = ykblas;
		double error;
      	if(trans == 'n' || trans == 'N')error = zget_max_error(yref, yres, m, incy);
      	else error = zget_max_error(yref, yres, n, incy);
		// error 2
		yref = ycuda;
		yres = ykblas2;
		double error2;
      	if(trans == 'n' || trans == 'N')error2 = zget_max_error(yref, yres, m, incy);
      	else error2 = zget_max_error(yref, yres, n, incy);

      	printf("%-11d %-11d   %-10.2f   %-10.2f   %-11.2f   %-10e   %-11e\n", m, n, cuda_perf, kblas_perf, kblas2_perf, error, error2);
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
	if(ykblas2)free(ykblas2);

	hipblasDestroy(cublas_handle);
    return EXIT_SUCCESS;
}

