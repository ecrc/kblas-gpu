/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_l2/test_zgemv2_offset.c

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

int main(int argc, char** argv)
{
	if(argc < 7)
	{
		printf("USAGE: %s <device-id> <no-trans'n' or trans't' or conj-trans'c'> <matrix-dim> <start-offset> <stop-offset> <step-offset>\n", argv[0]);
		printf("==> <device-id>: GPU device id to use \n");
		printf("==> <no-trans'n' or trans't' or conj-trans'c'>: Process the matrix in non-transposed,transposed, or conjugate transposed configuration \n");
		printf("==> <matrix-dim>: The dimension of the matrix\n");
		printf("==> <start-offset> <stop-offset> <step-offset>: Offset range. For every <offset> in the offset range, test is performed on a submatrix whose dimension is <matrix-dim>-<offset>\n");
		exit(-1);
	}

	int dev = atoi(argv[1]);
	char trans = *argv[2];
	int dim = atoi(argv[3]);
	int istart = atoi(argv[4]);
	int istop = atoi(argv[5]);
	int istep = atoi(argv[6]);

	const int nruns = NRUNS;

	hipError_t ed = hipSetDevice(dev);
	if(ed != hipSuccess){printf("Error setting device : %s \n", hipGetErrorString(ed) ); exit(-1);}

	hipblasHandle_t cublas_handle;
	cublasAtomicsMode_t mode = CUBLAS_ATOMICS_ALLOWED;
	hipblasCreate(&cublas_handle);
	cublasSetAtomicsMode(cublas_handle, mode);

	struct hipDeviceProp_t deviceProp;
	hipGetDeviceProperties(&deviceProp, dev);

	if(istop >= dim){printf("Error: maximum offset value causes zero or negative submatrix dimension\n"); exit(-1);}

    int M = dim;
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

	hipDoubleComplex alpha = kblas_zrand();
	hipDoubleComplex beta = kblas_zrand();

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

    if(trans == 'N' || trans == 'n')printf("non-transposed test .. \n");
	else if (trans == 'T' || trans == 't') printf("transposed test .. \n");
	else if (trans == 'C' || trans == 'c') printf("Conjugate transposed test .. \n");
	else { printf("transpose configuration is not properly specified\n"); exit(-1);}
	printf("Allocating Matrices\n");
    A = (hipDoubleComplex*)malloc(N*LDA*sizeof(hipDoubleComplex));
    x = (hipDoubleComplex*)malloc(vecsize_x*sizeof(hipDoubleComplex));
    ycuda = (hipDoubleComplex*)malloc(vecsize_y*sizeof(hipDoubleComplex));
    ykblas = (hipDoubleComplex*)malloc(vecsize_y*sizeof(hipDoubleComplex));

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

    hipblasSetMatrix(dim, dim, sizeof(hipDoubleComplex), A, LDA, dA, LDA_);
	hipMemcpy(dx, x, vecsize_x*sizeof(hipDoubleComplex), hipMemcpyHostToDevice);

	printf("------------------- Testing ZGEMV ----------------\n");
    printf("  Matrix        CUBLAS       KBLAS          Max.  \n");
    printf(" Dimension     (Gflop/s)   (Gflop/s)       Error  \n");
    printf("-----------   ----------   ----------   ----------\n");

    int r;
    for(m = istart; m <= istop; m += istep)
    {
    	float elapsedTime;
    	int offset = m;
    	int dim_ = dim-offset;
      	float flops = FLOPS( (float)dim_ ) / 1e6;

		for(i = 0; i < vecsize_y; i++)
    	{
    		// init y for now until beta is supported in gemv2
      		ycuda[i] = kblas_zrand();
      		ykblas[i] = ycuda[i];
    	}

        // handle the offset
        hipDoubleComplex* dA_ = dA + offset * LDA_ + offset;
        hipDoubleComplex* dx_ = dx + offset * incx;
        hipDoubleComplex* dy_ = dy + offset * incy;
        int vecsize_y_ = vecsize_y - offset;
        int vecsize_x_ = vecsize_x - offset;

      	// --- cuda test
      	elapsedTime = 0;
      	for(r = 0; r < nruns; r++)
      	{
      		hipMemcpy(dy_, ycuda, vecsize_y_ * sizeof(hipDoubleComplex), hipMemcpyHostToDevice);

      		hipEventRecord(start, 0);
      		hipblasZgemv(cublas_handle, trans_, dim_, dim_, &alpha, dA_, LDA_, dx_, incx, &beta, dy_, incy);
      		hipEventRecord(stop, 0);
      		hipEventSynchronize(stop);
      		float time  = 0;
      		hipEventElapsedTime(&time, start, stop);
      		elapsedTime += time;
      	}
      	elapsedTime /= nruns;
      	float cuda_perf = flops / elapsedTime;

      	hipMemcpy(ycuda, dy_, vecsize_y_ * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost);
      	// end of cuda test

      	// ---- kblas
      	elapsedTime = 0;
      	for(r = 0; r < nruns; r++)
      	{
      		hipMemcpy(dy_, ykblas, vecsize_y_ * sizeof(hipDoubleComplex), hipMemcpyHostToDevice);

      		hipEventRecord(start, 0);
      		kblas_zgemv2_offset( trans, dim, dim, alpha, dA, LDA_, dx, incx, beta, dy, incy, offset, offset);
      		hipEventRecord(stop, 0);
      		hipEventSynchronize(stop);

      		float time  = 0;
      		hipEventElapsedTime(&time, start, stop);
      		elapsedTime += time;
      	}
      	elapsedTime /= nruns;
      	float kblas_perf = flops / elapsedTime;

      	hipMemcpy(ykblas, dy_, vecsize_y_ * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost);

      	// testing error -- specify ref. vector and result vector
      	hipDoubleComplex* yref = ycuda;
      	hipDoubleComplex* yres = ykblas;

      	double error = zget_max_error(yref, yres, dim_, incy);

      	//for(i = 0; i < m; i++) printf("[%d]:   %-8.2f   %-8.2f\n", i, ycuda[i], ykblas[i]);

      	//printf("-----------   ----------   ----------   ----------   ----------   ----------\n");
    	printf("%-11d   %-10.2f   %-10.2f   %-10e;\n", dim_, cuda_perf, kblas_perf, error);

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

