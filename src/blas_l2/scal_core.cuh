#include "hip/hip_runtime.h"
/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/scal_core.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ahmad Abdelfattah
 * @date 2018-11-14
 **/

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas.h>
#include "kblas_operators.h"
#include <stdio.h>

template <class T>
__global__ void
scal(int n, T alpha, T *x, int incx)
{
	const int tx = threadIdx.x;
	const int bx = blockIdx.x;

	const int gtx = bx * blockDim.x + tx;

	if(gtx < n) x[gtx * incx] *= alpha;
}
//--------------------------------------------------------------------------------------------------------//
