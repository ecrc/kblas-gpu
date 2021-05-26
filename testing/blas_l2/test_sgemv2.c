/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_l2/test_sgemv2.c

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ahmad Abdelfattah
 * @date 2020-12-10
 **/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <kblas.h>
#include "testing_utils.h"

#define FMULS_GEMV(m_, n_) ((m_) * (n_) + 2. * (m_))
#define FADDS_GEMV(m_, n_) ((m_) * (n_)           )

#define PRECISION_s

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

	cudaError_t ed = cudaSetDevice(dev);
	if(ed != cudaSuccess){printf("Error setting device : %s \n", cudaGetErrorString(ed) ); exit(-1);}

	cublasHandle_t cublas_handle;
	cublasAtomicsMode_t mode = CUBLAS_ATOMICS_ALLOWED;
	cublasCreate(&cublas_handle);
	cublasSetAtomicsMode(cublas_handle, mode);

	struct cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

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

	cublasOperation_t trans_;
	if(trans == 'N' || trans == 'n')
		trans_ = CUBLAS_OP_N;
	else if (trans == 'T' || trans == 't')
		trans_ = CUBLAS_OP_T;
	else if (trans == 'C' || trans == 'c')
		trans_ = CUBLAS_OP_C;

	float alpha = kblas_srand();
	float beta  = kblas_srand();

	cudaError_t err;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    // point to host memory
    float* A = NULL;
    float* x = NULL;
    float* ycuda = NULL;
    float* ykblas = NULL;
    float* ykblas2 = NULL;

    // point to device memory
    float* dA = NULL;
    float* dx = NULL;
    float* dy = NULL;

    if(trans == 'N' || trans == 'n')printf("non-transposed test .. \n");
	else if (trans == 'T' || trans == 't') printf("transposed test .. \n");
	else if (trans == 'C' || trans == 'c') printf("Conjugate transposed test .. \n");
	else { printf("transpose configuration is not properly specified\n"); exit(-1);}
	printf("Allocating Matrices\n");
    A = (float*)malloc(N*LDA*sizeof(float));
    x = (float*)malloc(vecsize_x*sizeof(float));
    ycuda   = (float*)malloc(vecsize_y*sizeof(float));
    ykblas  = (float*)malloc(vecsize_y*sizeof(float));
    ykblas2 = (float*)malloc(vecsize_y*sizeof(float));

    err = cudaMalloc((void**)&dA, N*LDA_*sizeof(float));
    if(err != cudaSuccess){printf("ERROR: %s \n", cudaGetErrorString(err)); exit(1);}
    err = cudaMalloc((void**)&dx, vecsize_x*sizeof(float));
    if(err != cudaSuccess){printf("ERROR: %s \n", cudaGetErrorString(err)); exit(1);}
    err = cudaMalloc((void**)&dy, vecsize_y*sizeof(float));
	if(err != cudaSuccess){printf("ERROR: %s \n", cudaGetErrorString(err)); exit(1);}

    // Initialize matrix and vector
    printf("Initializing on cpu .. \n");
    int i, j, m;
    for(i = 0; i < M; i++)
    		for(j = 0; j < N; j++)
    			A[j*LDA+i] = kblas_srand();

    for(i = 0; i < vecsize_x; i++)
      x[i] = kblas_srand();

    //cudaMemcpy(dA, A, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, vecsize_x*sizeof(float), cudaMemcpyHostToDevice);


    printf("--------------------------------------- Testing SGEMV --------------------------------------\n");
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

		cublasSetMatrix(m, n, sizeof(float), A, LDA, dA, lda);

		for(i = 0; i < vecsize_y; i++)
    	{
      		ycuda[i] = kblas_srand();
      		ykblas[i] = ycuda[i];
      		ykblas2[i] = ycuda[i];
    	}

      	// --- cuda test
      	elapsedTime  = 0;
      	for(r = 0; r < nruns; r++)
      	{
      		cudaMemcpy(dy, ycuda, vecsize_y * sizeof(float), cudaMemcpyHostToDevice);

      		cudaEventRecord(start, 0);
      		cublasSgemv(cublas_handle, trans_, m, n, &alpha, dA, lda, dx, incx, &beta, dy, incy);
      		cudaEventRecord(stop, 0);
      		cudaEventSynchronize(stop);
      		float time = 0;
      		cudaEventElapsedTime(&time, start, stop);
      		elapsedTime += time;
      	}
      	elapsedTime /= nruns;
      	float cuda_perf = flops / elapsedTime;

      	cudaMemcpy(ycuda, dy, vecsize_y * sizeof(float), cudaMemcpyDeviceToHost);
      	// end of cuda test

      	// ---- kblas
      	elapsedTime = 0;
      	for(r = 0; r < nruns; r++)
      	{
      		cudaMemcpy(dy, ykblas, vecsize_y * sizeof(float), cudaMemcpyHostToDevice);

      		cudaEventRecord(start, 0);
      		kblas_sgemv( trans, m, n, alpha, dA, lda, dx, incx, beta, dy, incy);
      		cudaEventRecord(stop, 0);
      		cudaEventSynchronize(stop);

      		float time = 0;
      		cudaEventElapsedTime(&time, start, stop);
      		elapsedTime += time;
      	}
      	elapsedTime /= nruns;
      	float kblas_perf = flops / elapsedTime;


      	cudaMemcpy(ykblas, dy, vecsize_y * sizeof(float), cudaMemcpyDeviceToHost);

      	// ---- kblas-2
      	elapsedTime = 0;
      	for(r = 0; r < nruns; r++)
      	{
      		cudaMemcpy(dy, ykblas2, vecsize_y * sizeof(float), cudaMemcpyHostToDevice);

      		cudaEventRecord(start, 0);
      		kblas_sgemv2(trans, m, n, alpha, dA, lda, dx, incx, beta, dy, incy);
      		cudaEventRecord(stop, 0);
      		cudaEventSynchronize(stop);

      		float time = 0;
      		cudaEventElapsedTime(&time, start, stop);
      		elapsedTime += time;
      	}
      	elapsedTime /= nruns;
      	float kblas2_perf = flops / elapsedTime;


      	cudaMemcpy(ykblas2, dy, vecsize_y * sizeof(float), cudaMemcpyDeviceToHost);

      	// testing error -- specify ref. vector and result vector
      	float* yref;
      	float* yres;

      	// error1
      	yref = ycuda;
		yres = ykblas;
		float error;
      	if(trans == 'n' || trans == 'N')error = sget_max_error(yref, yres, m, incy);
      	else error = sget_max_error(yref, yres, n, incy);
		// error 2
		yref = ycuda;
		yres = ykblas2;
		float error2;
      	if(trans == 'n' || trans == 'N')error2 = sget_max_error(yref, yres, m, incy);
      	else error2 = sget_max_error(yref, yres, n, incy);

      	printf("%-11d %-11d   %-10.2f   %-10.2f   %-11.2f   %-10e   %-11e\n", m, n, cuda_perf, kblas_perf, kblas2_perf, error, error2);
    }

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    if(dA)cudaFree(dA);
    if(dx)cudaFree(dx);
    if(dy)cudaFree(dy);

    if(A)free(A);
    if(x)free(x);
    if(ycuda)free(ycuda);
	if(ykblas)free(ykblas);
	if(ykblas2)free(ykblas2);

	cublasDestroy(cublas_handle);
    return EXIT_SUCCESS;
}

