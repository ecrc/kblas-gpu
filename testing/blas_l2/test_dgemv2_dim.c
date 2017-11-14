/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_l2/test_dgemv2_dim.c

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 2.0.0
 * @author Ahmad Abdelfattah
 * @date 2017-11-13
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

#define PRECISION_d

#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(m, n) ( 6. * FMULS_GEMV(m, n) + 2. * FADDS_GEMV(m, n))
#else
#define FLOPS(m, n) (      FMULS_GEMV(m, n) +      FADDS_GEMV(m, n))
#endif

int main(int argc, char** argv)
{
	if(argc < 6)
	{
		printf("USAGE: %s <device-id> <no-trans'n' or trans't' or conj-trans'c'> <M> <N> <print-header>\n", argv[0]);
		printf("==> <device-id>: GPU device id to use \n");
		printf("==> <no-trans'n' or trans't' or conj-trans'c'>: Process the matrix in non-transposed,transposed, or conjugate transposed configuration \n");
		printf("==> <M> <N> dimensions of the matrix \n");
		printf("==> <print-header>: if not zero, header information will be printed\n");
		exit(-1);
	}

	int dev = atoi(argv[1]);
	char trans = *argv[2];
	int input_m = atoi(argv[3]);
	int input_n = atoi(argv[4]);
	int print_header = atoi(argv[5]);

	const int nruns = NRUNS;

	cudaError_t ed = cudaSetDevice(dev);
	if(ed != cudaSuccess){printf("Error setting device : %s \n", cudaGetErrorString(ed) ); exit(-1);}

	cublasHandle_t cublas_handle;
	cublasAtomicsMode_t mode = CUBLAS_ATOMICS_ALLOWED;
	cublasCreate(&cublas_handle);
	cublasSetAtomicsMode(cublas_handle, mode);

	struct cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

    int M = input_m;
    int N = input_n;
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

	double alpha = kblas_drand();
	double beta  = kblas_drand();

	cudaError_t err;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    // point to host memory
    double* A = NULL;
    double* x = NULL;
    double* ycuda = NULL;
    double* ykblas = NULL;
    double* ykblas2 = NULL;

    // point to device memory
    double* dA = NULL;
    double* dx = NULL;
    double* dy = NULL;

    if(print_header != 0)
    {
    	if(trans == 'N' || trans == 'n')printf("non-transposed test .. \n");
		else if (trans == 'T' || trans == 't') printf("transposed test .. \n");
		else if (trans == 'C' || trans == 'c') printf("Conjugate transposed test .. \n");
		else { printf("transpose configuration is not properly specified\n"); exit(-1);}
		printf("Allocating Matrices\n");
    }
    A = (double*)malloc(N*LDA*sizeof(double));
    x = (double*)malloc(vecsize_x*sizeof(double));
    ycuda   = (double*)malloc(vecsize_y*sizeof(double));
    ykblas  = (double*)malloc(vecsize_y*sizeof(double));
    ykblas2 = (double*)malloc(vecsize_y*sizeof(double));

    err = cudaMalloc((void**)&dA, N*LDA_*sizeof(double));
    if(err != cudaSuccess){printf("ERROR: %s \n", cudaGetErrorString(err)); exit(1);}
    err = cudaMalloc((void**)&dx, vecsize_x*sizeof(double));
    if(err != cudaSuccess){printf("ERROR: %s \n", cudaGetErrorString(err)); exit(1);}
    err = cudaMalloc((void**)&dy, vecsize_y*sizeof(double));
	if(err != cudaSuccess){printf("ERROR: %s \n", cudaGetErrorString(err)); exit(1);}

    // Initialize matrix and vector
    if(print_header != 0){printf("Initializing on cpu .. \n");}

    int i, j, m;
    for(i = 0; i < M; i++)
    		for(j = 0; j < N; j++)
    			A[j*LDA+i] = kblas_drand();

    for(i = 0; i < vecsize_x; i++)
      x[i] = kblas_drand();

    //cudaMemcpy(dA, A, M*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, vecsize_x*sizeof(double), cudaMemcpyHostToDevice);

	if(print_header != 0)
	{
    	printf("--------------------------------------- Testing DGEMV --------------------------------------\n");
    	printf("  Matrix      Matrix       CUBLAS      KBLAS-GEMV   KBLAS-GEMV2   KBLAS-GEMV     KBLAS-GEMV2\n");
    	printf("   Rows       Columns     (Gflop/s)    (Gflop/s)    (Gflop/s)     MAX Error       MAX Error \n");
    	printf("----------- -----------   ----------   ----------   -----------   ------------   -----------\n");
    }

    int r;

    // remove the for loop
    {
    	float elapsedTime;

    	int m = M;
      	int n = N;
      	int lda = ((m+31)/32)*32;
      	if(n == 0) n = 1;

      	float flops = FLOPS( (float)m, (float)n ) / 1e6;

		cublasSetMatrix(m, n, sizeof(double), A, LDA, dA, lda);

		for(i = 0; i < vecsize_y; i++)
    	{
      		ycuda[i] = kblas_drand();
      		ykblas[i] = ycuda[i];
      		ykblas2[i] = ycuda[i];
    	}

      	// --- cuda test
      	elapsedTime  = 0;
      	for(r = 0; r < nruns; r++)
      	{
      		cudaMemcpy(dy, ycuda, vecsize_y * sizeof(double), cudaMemcpyHostToDevice);

      		cudaEventRecord(start, 0);
      		cublasDgemv(cublas_handle, trans_, m, n, &alpha, dA, lda, dx, incx, &beta, dy, incy);
      		cudaEventRecord(stop, 0);
      		cudaEventSynchronize(stop);
      		float time = 0;
      		cudaEventElapsedTime(&time, start, stop);
      		elapsedTime += time;
      	}
      	elapsedTime /= nruns;
      	float cuda_perf = flops / elapsedTime;

      	cudaMemcpy(ycuda, dy, vecsize_y * sizeof(double), cudaMemcpyDeviceToHost);
      	// end of cuda test

      	// ---- kblas
      	elapsedTime = 0;
      	for(r = 0; r < nruns; r++)
      	{
      		cudaMemcpy(dy, ykblas, vecsize_y * sizeof(double), cudaMemcpyHostToDevice);

      		cudaEventRecord(start, 0);
      		kblas_dgemv( trans, m, n, alpha, dA, lda, dx, incx, beta, dy, incy);
      		cudaEventRecord(stop, 0);
      		cudaEventSynchronize(stop);

      		float time = 0;
      		cudaEventElapsedTime(&time, start, stop);
      		elapsedTime += time;
      	}
      	elapsedTime /= nruns;
      	float kblas_perf = flops / elapsedTime;


      	cudaMemcpy(ykblas, dy, vecsize_y * sizeof(double), cudaMemcpyDeviceToHost);

      	// ---- kblas-2
      	elapsedTime = 0;
      	for(r = 0; r < nruns; r++)
      	{
      		cudaMemcpy(dy, ykblas2, vecsize_y * sizeof(double), cudaMemcpyHostToDevice);

      		cudaEventRecord(start, 0);
      		kblas_dgemv2(trans, m, n, alpha, dA, lda, dx, incx, beta, dy, incy);
      		cudaEventRecord(stop, 0);
      		cudaEventSynchronize(stop);

      		float time = 0;
      		cudaEventElapsedTime(&time, start, stop);
      		elapsedTime += time;
      	}
      	elapsedTime /= nruns;
      	float kblas2_perf = flops / elapsedTime;


      	cudaMemcpy(ykblas2, dy, vecsize_y * sizeof(double), cudaMemcpyDeviceToHost);

      	// testing error -- specify ref. vector and result vector
      	double* yref;
      	double* yres;

      	// error1
      	yref = ycuda;
		yres = ykblas;
		double error;
      	if(trans == 'n' || trans == 'N')error = dget_max_error(yref, yres, m, incy);
      	else error = dget_max_error(yref, yres, n, incy);
		// error 2
		yref = ycuda;
		yres = ykblas2;
		double error2;
      	if(trans == 'n' || trans == 'N')error2 = dget_max_error(yref, yres, m, incy);
      	else error2 = dget_max_error(yref, yres, n, incy);

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

