#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "operators.h"
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
