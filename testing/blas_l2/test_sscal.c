/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_l2/test_sscal.c

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

#define FMULS_SCAL(n) (n)
#define FADDS_SCAL(n) (0)

#define PRECISION_s

#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 6. * FMULS_SCAL(n) + 2. * FADDS_SCAL(n))
#else
#define FLOPS(n) (      FMULS_SCAL(n) +      FADDS_SCAL(n))
#endif

float get_max_error(int n, float* ref, int inc_ref, float *res, int inc_res)
{
	int i, j;
	float max_err = -1.0;
	float err = -1.0;
	for(i = 0; i < n; i++)
	{
		float rf = ref[i * inc_ref];
		float rs = res[i * inc_res];
		err = fabs(rs - rf);
		if(rf != 0.0)err /= fabs(rf);
		if(err > max_err)max_err = err;
	}
	return max_err;
}

int main(int argc, char** argv)
{
	if(argc < 5)
	{
		printf("USAGE: %s <device-id> <start-dim> <stop-dim> <step-dim>\n", argv[0]);
		printf("==> <device-id>: GPU device id to use \n");
		printf("==> <start-dim> <stop-dim> <step-dim>: test for dimensions in the range start-dim : stop-dim with step step-dim \n");
		exit(-1);
	}

	int dev = atoi(argv[1]);
	int istart = atoi(argv[2]);
	int istop = atoi(argv[3]);
	int istep = atoi(argv[4]);

	const int nruns = NRUNS;

	hipError_t ed = hipSetDevice(dev);
	if(ed != hipSuccess){printf("Error setting device : %s \n", hipGetErrorString(ed) ); exit(-1);}

	struct hipDeviceProp_t deviceProp;
	hipGetDeviceProperties(&deviceProp, dev);

    int N = istop;
    int incx = 1;

	int vecsize = N * abs(incx);

	hipError_t err;
	hipEvent_t start, stop;

	hipEventCreate(&start);
	hipEventCreate(&stop);

    // point to host memory
    float* x = NULL;
    float* xcublas = NULL;
    float* xkblas = NULL;

    // point to device memory
    float* dx = NULL;

	printf("Allocating vectors\n");
    x 		= (float*)malloc(vecsize*sizeof(float));
    xcublas = (float*)malloc(vecsize*sizeof(float));
    xkblas 	= (float*)malloc(vecsize*sizeof(float));

    hipMalloc((void**)&dx, vecsize*sizeof(float));

    // Initialize vectors
    printf("Initializing\n");
    int i, j, m;
    for(i = 0; i < N; i++)
      x[i] = kblas_srand();

    float alpha =  kblas_srand();

    printf("--------------- Testing SSCAL ----------------\n");
    printf("  Matrix       CUBLAS      KBLAS        Max.  \n");
    printf(" Dimension    (Gflop/s)   (Gflop/s)    Error  \n");
    printf("-----------   ---------   ---------   --------\n");

    int r;
    for(m = istart; m <= istop; m += istep)
    {
    	float elapsedTime;

      	float flops = FLOPS( (float)m ) / 1e6;

		// cublas test
		elapsedTime  = 0;
		for(r = 0; r < nruns; r++)
		{
			hipMemcpy(dx, x, m * sizeof(float), hipMemcpyHostToDevice);
			hipEventRecord(start, 0);
      		hipblasSscal(m, alpha, dx, incx);
      		hipEventRecord(stop, 0);
      		hipEventSynchronize(stop);
      		float time  = 0;
      		hipEventElapsedTime(&time, start, stop);
      		elapsedTime += time;
      	}
      	elapsedTime /= nruns;
      	float cublas_perf = flops / elapsedTime;

      	hipMemcpy(xcublas, dx, vecsize * sizeof(float), hipMemcpyDeviceToHost);
      	// end of cuda test

      	// ---- kblas
      	elapsedTime = 0;
      	for(r = 0; r < nruns; r++)
      	{
      		hipMemcpy(dx, x, m * sizeof(float), hipMemcpyHostToDevice);
      		hipEventRecord(start, 0);
      		kblas_sscal(m, alpha, dx, incx);
      		hipEventRecord(stop, 0);
      		hipEventSynchronize(stop);

      		float time = 0.0;
      		hipEventElapsedTime(&time, start, stop);
      		elapsedTime += time;
      	}
      	elapsedTime /= nruns;
      	float kblas_perf = flops / elapsedTime;

      	hipMemcpy(xkblas, dx, vecsize * sizeof(float), hipMemcpyDeviceToHost);

      	// testing error -- specify ref. vector and result vector
      	float* yref = xcublas;
      	float* yres = xkblas;

      	float error = get_max_error(m, yref, incx, yres, incx);

      	printf("%-11d   %-9.2f   %-9.2f   %-10e;\n", m, cublas_perf, kblas_perf, error);
    }

	hipEventDestroy(start);
	hipEventDestroy(stop);

    if(dx)hipFree(dx);

    if(x)free(x);
    if(xcublas)free(xcublas);
	if(xkblas)free(xkblas);

	return EXIT_SUCCESS;
}

