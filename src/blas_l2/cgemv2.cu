#include "hip/hip_runtime.h"
/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/cgemv2.cu

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
#include "gemv2_core.cuh"

#if(TARGET_SM >= 30)

#define cgemvn_nb               (32)
#define cgemvn_ntcol            (4)
#define cgemvn_ept              (2)
#define cgemvn_width    (cgemvn_ntcol*cgemvn_ept)
#define cgemvn_by               (16)

#define cgemvt_nb               (32)
#define cgemvt_ntcol            (2)
#define cgemvt_ept              (4)
#define cgemvt_width    (cgemvt_ntcol*cgemvt_ept)
#define cgemvt_by               (8)

#else

#define cgemvn_nb               (64)
#define cgemvn_ntcol    		(8)
#define cgemvn_ept              (2)
#define cgemvn_width    (cgemvn_ntcol*cgemvn_ept)
#define cgemvn_by               (1)

#define cgemvt_nb               (64)
#define cgemvt_ntcol    		(8)
#define cgemvt_ept              (2)
#define cgemvt_width    (cgemvt_ntcol*cgemvt_ept)
#define cgemvt_by               (1)
#endif


extern "C"
int kblas_cscal_async(int n, hipFloatComplex alpha, hipFloatComplex *x, int incx, hipStream_t stream);


int kblas_cgemv2_driver(	char trans, int rows, int cols,
						hipFloatComplex alpha, hipFloatComplex *dA, int lda,
						hipFloatComplex *dX, int incx,
						hipFloatComplex  beta, hipFloatComplex *dY, int incy,
						hipStream_t stream)
{
	if(trans == 'n' || trans == 'N')
	{
		// scaling with beta
		kblas_cscal_async(rows, beta, dY, incy, stream);

		int mod_r = rows % cgemvn_nb;
		int mod_c = cols % cgemvn_width;

		int blocks = rows/cgemvn_nb;
		if(mod_r != 0) blocks += 1;

		const int thread_x = cgemvn_nb;
		const int thread_y = cgemvn_ntcol;
		const int ept = cgemvn_ept;

		int threshold = mod_c / ept;
		int ept_ = mod_c % ept;
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks, cgemvn_by);
		switch(ept_)
		{
			case 0: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn<hipFloatComplex, cgemvn_nb, cgemvn_ntcol, ept, cgemvn_width, 0>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 1: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn<hipFloatComplex, cgemvn_nb, cgemvn_ntcol, ept, cgemvn_width, 1>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 2: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn<hipFloatComplex, cgemvn_nb, cgemvn_ntcol, ept, cgemvn_width, 2>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 3: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn<hipFloatComplex, cgemvn_nb, cgemvn_ntcol, ept, cgemvn_width, 3>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 4: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn<hipFloatComplex, cgemvn_nb, cgemvn_ntcol, ept, cgemvn_width, 4>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 5: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn<hipFloatComplex, cgemvn_nb, cgemvn_ntcol, ept, cgemvn_width, 5>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 6: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn<hipFloatComplex, cgemvn_nb, cgemvn_ntcol, ept, cgemvn_width, 6>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 7: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn<hipFloatComplex, cgemvn_nb, cgemvn_ntcol, ept, cgemvn_width, 7>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 8: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn<hipFloatComplex, cgemvn_nb, cgemvn_ntcol, ept, cgemvn_width, 8>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			default: printf("irregular part %d is not supported, please extend the case statement of cgemv\n", ept_); exit(1);
		}
	}	// end of non-transpose case
	else if(trans == 't' || trans == 'T' || trans == 'c' || trans == 'C')
	{
		// scaling with beta
		kblas_cscal_async(cols, beta, dY, incy, stream);

		int mod_r = rows % cgemvt_nb;
		int mod_c = cols % cgemvt_width;

		int blocks = cols/cgemvt_width;
		if(mod_c != 0) blocks += 1;

		const int thread_x = cgemvt_nb;
		const int thread_y = cgemvt_ntcol;
		const int ept = cgemvt_ept;

		int threshold = mod_c / ept;
		int ept_ = mod_c % ept;

		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks, cgemvt_by);

		int conj;
		if(trans == 'c' || trans == 'C')conj = 1;
		else conj = 0;
		//printf("modr = %d, modc = %d, threshold = %d, ept_ = %d \n", mod_r, mod_c, threshold, ept_);
		switch(ept_)
		{
			case 0: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt<hipFloatComplex, cgemvt_nb, cgemvt_ntcol, ept, cgemvt_width, 0>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 1: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt<hipFloatComplex, cgemvt_nb, cgemvt_ntcol, ept, cgemvt_width, 1>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 2: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt<hipFloatComplex, cgemvt_nb, cgemvt_ntcol, ept, cgemvt_width, 2>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 3: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt<hipFloatComplex, cgemvt_nb, cgemvt_ntcol, ept, cgemvt_width, 3>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 4: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt<hipFloatComplex, cgemvt_nb, cgemvt_ntcol, ept, cgemvt_width, 4>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 5: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt<hipFloatComplex, cgemvt_nb, cgemvt_ntcol, ept, cgemvt_width, 5>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 6: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt<hipFloatComplex, cgemvt_nb, cgemvt_ntcol, ept, cgemvt_width, 6>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 7: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt<hipFloatComplex, cgemvt_nb, cgemvt_ntcol, ept, cgemvt_width, 7>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 8: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt<hipFloatComplex, cgemvt_nb, cgemvt_ntcol, ept, cgemvt_width, 8>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			default: printf("irregular part %d is not supported, please extend the case statement of cgemv\n", ept_); exit(1);
		}
	}
	else
	{
		printf("CGEMV error: Unrecognized transpose mode %c \n", trans);
		return -1;
	}

	return 0;
}

extern "C"
int kblas_cgemv2(char trans, int rows, int cols,
				hipFloatComplex alpha, hipFloatComplex *dA, int lda,
				hipFloatComplex *dX, int incx,
				hipFloatComplex  beta, hipFloatComplex *dY, int incy)
{
	return kblas_cgemv2_driver(	trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, 0);
}

extern "C"
int kblas_cgemv2_async(	char trans, int rows, int cols,
						hipFloatComplex alpha, hipFloatComplex *dA, int lda,
						hipFloatComplex *dX, int incx,
						hipFloatComplex  beta, hipFloatComplex *dY, int incy,
						hipStream_t stream)
{
	return kblas_cgemv2_driver(	trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, stream);
}
