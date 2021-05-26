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
 * @version 4.0.0
 * @author Ahmad Abdelfattah
 * @date 2020-12-10
 **/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "scal_core.cuh"

#define sscal_nbx		(128)

int kblas_sscal_driver(int n, float alpha, float *x, int incx, cudaStream_t stream)
{
	int gridx = n / sscal_nbx + (n % sscal_nbx != 0);

	dim3 dimBlock(sscal_nbx, 1);
	dim3 dimGrid(gridx, 1);

	scal<float><<<dimGrid, dimBlock, 0, stream>>>(n, alpha, x, incx);

	return 0;
}


extern "C"
int kblas_sscal(int n, float alpha, float *x, int incx)
{
	return kblas_sscal_driver(n, alpha, x, incx, 0);
}

extern "C"
int kblas_sscal_async(int n, float alpha, float *x, int incx, cudaStream_t stream)
{
	return kblas_sscal_driver(n, alpha, x, incx, stream);
}
