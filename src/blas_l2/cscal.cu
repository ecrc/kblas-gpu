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
 * @version 2.0.0
 * @author Ahmad Abdelfattah
 * @date 2017-11-13
 **/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "scal_core.cuh"

#define cscal_nbx		(128)

int kblas_cscal_driver(int n, cuFloatComplex alpha, cuFloatComplex *x, int incx, cudaStream_t stream)
{
	int gridx = n / cscal_nbx + (n % cscal_nbx != 0);

	dim3 dimBlock(cscal_nbx, 1);
	dim3 dimGrid(gridx, 1);

	scal<cuFloatComplex><<<dimGrid, dimBlock, 0, stream>>>(n, alpha, x, incx);

	return 0;
}


extern "C"
int kblas_cscal(int n, cuFloatComplex alpha, cuFloatComplex *x, int incx)
{
	return kblas_cscal_driver(n, alpha, x, incx, 0);
}

extern "C"
int kblas_cscal_async(int n, cuFloatComplex alpha, cuFloatComplex *x, int incx, cudaStream_t stream)
{
	return kblas_cscal_driver(n, alpha, x, incx, stream);
}
