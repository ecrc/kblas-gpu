#include "hip/hip_runtime.h"
/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/sscal.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ahmad Abdelfattah
 * @date 2018-11-14
 **/

#include <stdio.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas.h>
#include "scal_core.cuh"

#ifdef __HIP_PLATFORM_HCC__
#define SHFL_DOWN(val, offset) __shfl_down(val, offset)
#else
#define SHFL_DOWN(val, offset) __shfl_down_sync(0xffffffff, val, offset)
#endif

#ifdef __HIP_PLATFORM_HCC__
#define SHFL(val, offset) __shfl(val, offset)
#else
#define SHFL(val, offset) __shfl_sync(0xffffffff, val, offset)
#endif

#define sscal_nbx		(128)

int kblas_sscal_driver(int n, float alpha, float *x, int incx, hipStream_t stream)
{
	int gridx = n / sscal_nbx + (n % sscal_nbx != 0);

	dim3 dimBlock(sscal_nbx, 1);
	dim3 dimGrid(gridx, 1);

	hipLaunchKernelGGL(HIP_KERNEL_NAME(scal<float>), dim3(dimGrid), dim3(dimBlock), 0, stream, n, alpha, x, incx);

	return 0;
}


extern "C"
int kblas_sscal(int n, float alpha, float *x, int incx)
{
	return kblas_sscal_driver(n, alpha, x, incx, 0);
}

extern "C"
int kblas_sscal_async(int n, float alpha, float *x, int incx, hipStream_t stream)
{
	return kblas_sscal_driver(n, alpha, x, incx, stream);
}
