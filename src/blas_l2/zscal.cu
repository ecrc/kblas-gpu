#include "hip/hip_runtime.h"
/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/zscal.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ahmad Abdelfattah
 * @date 2018-11-14
 **/

#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas.h>
#include "scal_core.cuh"

#define zscal_nbx		(128)

int kblas_zscal_driver(int n, hipDoubleComplex alpha, hipDoubleComplex *x, int incx, hipStream_t stream)
{
	int gridx = n / zscal_nbx + (n % zscal_nbx != 0);

	dim3 dimBlock(zscal_nbx, 1);
	dim3 dimGrid(gridx, 1);

	hipLaunchKernelGGL(HIP_KERNEL_NAME(scal<hipDoubleComplex>), dim3(dimGrid), dim3(dimBlock), 0, stream, n, alpha, x, incx);

	return 0;
}


extern "C"
int kblas_zscal(int n, hipDoubleComplex alpha, hipDoubleComplex *x, int incx)
{
	return kblas_zscal_driver(n, alpha, x, incx, 0);
}

extern "C"
int kblas_zscal_async(int n, hipDoubleComplex alpha, hipDoubleComplex *x, int incx, hipStream_t stream)
{
	return kblas_zscal_driver(n, alpha, x, incx, stream);
}
