#include "hip/hip_runtime.h"
/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/zgemv2.cu

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

#define zgemvn_nb       (32)
#define zgemvn_ntcol    (4)
#define zgemvn_ept      (2)
#define zgemvn_width    (zgemvn_ntcol*zgemvn_ept)
#define zgemvn_by       (4)

#define zgemvt_nb       (32)
#define zgemvt_ntcol    (4)
#define zgemvt_ept      (2)
#define zgemvt_width    (zgemvt_ntcol*zgemvt_ept)
#define zgemvt_by       (4)

#else

#define zgemvn_nb               (64)
#define zgemvn_ntcol    		(8)
#define zgemvn_ept              (2)
#define zgemvn_width    (zgemvn_ntcol*zgemvn_ept)
#define zgemvn_by               (1)

#define zgemvt_nb               (64)
#define zgemvt_ntcol    		(8)
#define zgemvt_ept              (2)
#define zgemvt_width    (zgemvt_ntcol*zgemvt_ept)
#define zgemvt_by               (1)
#endif


extern "C"
int kblas_zscal_async(int n, hipDoubleComplex alpha, hipDoubleComplex *x, int incx, hipStream_t stream);


int kblas_zgemv2_driver(	char trans, int rows, int cols,
						hipDoubleComplex alpha, hipDoubleComplex *dA, int lda,
						hipDoubleComplex *dX, int incx,
						hipDoubleComplex  beta, hipDoubleComplex *dY, int incy,
						hipStream_t stream)
{
	if(trans == 'n' || trans == 'N')
	{
		// scaling with beta
		kblas_zscal_async(rows, beta, dY, incy, stream);

		int mod_r = rows % zgemvn_nb;
		int mod_c = cols % zgemvn_width;

		int blocks = rows/zgemvn_nb;
		if(mod_r != 0) blocks += 1;

		const int thread_x = zgemvn_nb;
		const int thread_y = zgemvn_ntcol;
		const int ept = zgemvn_ept;

		int threshold = mod_c / ept;
		int ept_ = mod_c % ept;
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks, zgemvn_by);
		switch(ept_)
		{
			case 0: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn<hipDoubleComplex, zgemvn_nb, zgemvn_ntcol, ept, zgemvn_width, 0>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 1: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn<hipDoubleComplex, zgemvn_nb, zgemvn_ntcol, ept, zgemvn_width, 1>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 2: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn<hipDoubleComplex, zgemvn_nb, zgemvn_ntcol, ept, zgemvn_width, 2>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 3: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn<hipDoubleComplex, zgemvn_nb, zgemvn_ntcol, ept, zgemvn_width, 3>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 4: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn<hipDoubleComplex, zgemvn_nb, zgemvn_ntcol, ept, zgemvn_width, 4>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 5: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn<hipDoubleComplex, zgemvn_nb, zgemvn_ntcol, ept, zgemvn_width, 5>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 6: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn<hipDoubleComplex, zgemvn_nb, zgemvn_ntcol, ept, zgemvn_width, 6>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 7: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn<hipDoubleComplex, zgemvn_nb, zgemvn_ntcol, ept, zgemvn_width, 7>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 8: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn<hipDoubleComplex, zgemvn_nb, zgemvn_ntcol, ept, zgemvn_width, 8>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			default: printf("irregular part %d is not supported, please extend the case statement of zgemv\n", ept_); exit(1);
		}
	}	// end of non-transpose case
	else if(trans == 't' || trans == 'T' || trans == 'c' || trans == 'C')
	{
		// scaling with beta
		kblas_zscal_async(cols, beta, dY, incy, stream);

		int mod_r = rows % zgemvt_nb;
		int mod_c = cols % zgemvt_width;

		int blocks = cols/zgemvt_width;
		if(mod_c != 0) blocks += 1;

		const int thread_x = zgemvt_nb;
		const int thread_y = zgemvt_ntcol;
		const int ept = zgemvt_ept;

		int threshold = mod_c / ept;
		int ept_ = mod_c % ept;

		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks, zgemvt_by);

		int conj;
		if(trans == 'c' || trans == 'C')conj = 1;
		else conj = 0;
		//printf("modr = %d, modc = %d, threshold = %d, ept_ = %d \n", mod_r, mod_c, threshold, ept_);
		switch(ept_)
		{
			case 0: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt<hipDoubleComplex, zgemvt_nb, zgemvt_ntcol, ept, zgemvt_width, 0>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 1: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt<hipDoubleComplex, zgemvt_nb, zgemvt_ntcol, ept, zgemvt_width, 1>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 2: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt<hipDoubleComplex, zgemvt_nb, zgemvt_ntcol, ept, zgemvt_width, 2>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 3: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt<hipDoubleComplex, zgemvt_nb, zgemvt_ntcol, ept, zgemvt_width, 3>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 4: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt<hipDoubleComplex, zgemvt_nb, zgemvt_ntcol, ept, zgemvt_width, 4>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 5: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt<hipDoubleComplex, zgemvt_nb, zgemvt_ntcol, ept, zgemvt_width, 5>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 6: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt<hipDoubleComplex, zgemvt_nb, zgemvt_ntcol, ept, zgemvt_width, 6>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 7: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt<hipDoubleComplex, zgemvt_nb, zgemvt_ntcol, ept, zgemvt_width, 7>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 8: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt<hipDoubleComplex, zgemvt_nb, zgemvt_ntcol, ept, zgemvt_width, 8>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			default: printf("irregular part %d is not supported, please extend the case statement of zgemv\n", ept_); exit(1);
		}
	}
	else
	{
		printf("ZGEMV error: Unrecognized transpose mode %c \n", trans);
		return -1;
	}

	return 0;
}

extern "C"
int kblas_zgemv2(char trans, int rows, int cols,
				hipDoubleComplex alpha, hipDoubleComplex *dA, int lda,
				hipDoubleComplex *dX, int incx,
				hipDoubleComplex  beta, hipDoubleComplex *dY, int incy)
{
	return kblas_zgemv2_driver(	trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, 0);
}

extern "C"
int kblas_zgemv2_async(	char trans, int rows, int cols,
						hipDoubleComplex alpha, hipDoubleComplex *dA, int lda,
						hipDoubleComplex *dX, int incx,
						hipDoubleComplex  beta, hipDoubleComplex *dY, int incy,
						hipStream_t stream)
{
	return kblas_zgemv2_driver(	trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, stream);
}
