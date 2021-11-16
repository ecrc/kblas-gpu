#include "hip/hip_runtime.h"
/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/cscal.cu

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

#define cscal_nbx		(128)

int kblas_cscal_driver(int n, hipFloatComplex alpha, hipFloatComplex *x, int incx, hipStream_t stream)
{
	int gridx = n / cscal_nbx + (n % cscal_nbx != 0);

	dim3 dimBlock(cscal_nbx, 1);
	dim3 dimGrid(gridx, 1);

	hipLaunchKernelGGL(HIP_KERNEL_NAME(scal<hipFloatComplex>), dim3(dimGrid), dim3(dimBlock), 0, stream, n, alpha, x, incx);

	return 0;
}


extern "C"
int kblas_cscal(int n, hipFloatComplex alpha, hipFloatComplex *x, int incx)
{
	return kblas_cscal_driver(n, alpha, x, incx, 0);
}

extern "C"
int kblas_cscal_async(int n, hipFloatComplex alpha, hipFloatComplex *x, int incx, hipStream_t stream)
{
	return kblas_cscal_driver(n, alpha, x, incx, stream);
}
