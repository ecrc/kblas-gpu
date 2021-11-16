#include "hip/hip_runtime.h"
/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/sgemv2_offset.cu

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
#include "gemv2_offset_core.cuh"

#if(TARGET_SM >= 30)

#define sgemvn_offset_nb               	(64)
#define sgemvn_offset_ntcol    			(4)
#define sgemvn_offset_ept              	(8)
#define sgemvn_offset_width    			(sgemvn_offset_ntcol*sgemvn_offset_ept)
#define sgemvn_offset_by               	(8)

#define sgemvt_offset_nb               	(32)
#define sgemvt_offset_ntcol            	(4)
#define sgemvt_offset_ept              	(8)
#define sgemvt_offset_width    			(sgemvt_offset_ntcol*sgemvt_offset_ept)
#define sgemvt_offset_by               	(8)

#else

#define sgemvn_offset_nb               	(64)
#define sgemvn_offset_ntcol    			(8)
#define sgemvn_offset_ept              	(2)
#define sgemvn_offset_width    			(sgemvn_offset_ntcol*sgemvn_offset_ept)
#define sgemvn_offset_by				(1)

#define sgemvt_offset_nb               	(64)
#define sgemvt_offset_ntcol    			(8)
#define sgemvt_offset_ept              	(2)
#define sgemvt_offset_width    			(sgemvt_offset_ntcol*sgemvt_offset_ept)
#define sgemvt_offset_by               	(1)
#endif


extern "C"
int kblas_sscal_async(int n, float alpha, float *x, int incx, hipStream_t stream);


int kblas_sgemv2_offset_driver(char trans, int rows, int cols,
						float alpha, float *dA, int lda,
						float *dX, int incx,
						float  beta, float *dY, int incy,
						int offset_r, int offset_c,
						hipStream_t stream)
{
	if(trans == 'n' || trans == 'N')
	{
		// offset necessary calculations
		int offset_r_ = offset_r % sgemvn_offset_nb;
		int offset_c_ = offset_c % sgemvn_offset_width;
		int rows_ = rows - (offset_r - offset_r_);
		int cols_ = cols - (offset_c - offset_c_);

		// Advance pointers
		dA += (offset_c - offset_c_) * lda + (offset_r - offset_r_);
		dX += (offset_c - offset_c_) * incx;
		dY += (offset_r - offset_r_) * incy;

		// scaling with beta
		kblas_sscal_async(rows_, beta, dY, incy, stream);

		int mod_r = rows_ % sgemvn_offset_nb;
		int mod_c = cols_ % sgemvn_offset_width;

		int blocks = rows_/sgemvn_offset_nb;
		if(mod_r != 0) blocks += 1;

		const int thread_x = sgemvn_offset_nb;
		const int thread_y = sgemvn_offset_ntcol;
		const int ept = sgemvn_offset_ept;

		int threshold = mod_c / ept;
		int ept_ = mod_c % ept;
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks, sgemvn_offset_by);
		//printf("rows_ = %d - cols_ = %d - mod_r = %d - mod_c = %d - offset_r_ = %d - offset_c_ = %d \n", rows_, cols_, mod_r, mod_c, offset_r_, offset_c_);
		switch(ept_)
		{
			case 0: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_offset<float, sgemvn_offset_nb, sgemvn_offset_ntcol, ept, sgemvn_offset_width, 0>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 1: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_offset<float, sgemvn_offset_nb, sgemvn_offset_ntcol, ept, sgemvn_offset_width, 1>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 2: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_offset<float, sgemvn_offset_nb, sgemvn_offset_ntcol, ept, sgemvn_offset_width, 2>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 3: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_offset<float, sgemvn_offset_nb, sgemvn_offset_ntcol, ept, sgemvn_offset_width, 3>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 4: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_offset<float, sgemvn_offset_nb, sgemvn_offset_ntcol, ept, sgemvn_offset_width, 4>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 5: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_offset<float, sgemvn_offset_nb, sgemvn_offset_ntcol, ept, sgemvn_offset_width, 5>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 6: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_offset<float, sgemvn_offset_nb, sgemvn_offset_ntcol, ept, sgemvn_offset_width, 6>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 7: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_offset<float, sgemvn_offset_nb, sgemvn_offset_ntcol, ept, sgemvn_offset_width, 7>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 8: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_offset<float, sgemvn_offset_nb, sgemvn_offset_ntcol, ept, sgemvn_offset_width, 8>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			default: printf("irregular part %d is not supported, please extend the case statement of sgemv\n", ept_); exit(1);
		}
	}	// end of non-transpose case
	else if(trans == 't' || trans == 'T' || trans == 'c' || trans == 'C')
	{
		// offset necessary calculations
		int offset_r_ = offset_r % sgemvt_offset_nb;
		int offset_c_ = offset_c % sgemvt_offset_width;
		int rows_ = rows - (offset_r - offset_r_);
		int cols_ = cols - (offset_c - offset_c_);

		// Advance pointers
		dA += (offset_c - offset_c_) * lda + (offset_r - offset_r_);
		dX += (offset_r - offset_r_) * incx;
		dY += (offset_c - offset_c_) * incy;

		// scaling with beta
		kblas_sscal_async(cols_, beta, dY, incy, stream);

		int mod_r = rows_ % sgemvt_offset_nb;
		int mod_c = cols_ % sgemvt_offset_width;

		int blocks = cols_/sgemvt_offset_width;
		if(mod_c != 0) blocks += 1;

		const int thread_x = sgemvt_offset_nb;
		const int thread_y = sgemvt_offset_ntcol;
		const int ept = sgemvt_offset_ept;

		int threshold = mod_c / ept;
		int ept_ = mod_c % ept;

		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks, sgemvt_offset_by);

		int conj;
		if(trans == 'c' || trans == 'C')conj = 1;
		else conj = 0;
		//printf("modr = %d, modc = %d, threshold = %d, ept_ = %d \n", mod_r, mod_c, threshold, ept_);
		//printf("rows_ = %d - cols_ = %d - mod_r = %d - mod_c = %d - offset_r_ = %d - offset_c_ = %d \n", rows_, cols_, mod_r, mod_c, offset_r_, offset_c_);
		switch(ept_)
		{
			case 0: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_offset<float, sgemvt_offset_nb, sgemvt_offset_ntcol, ept, sgemvt_offset_width, 0>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 1: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_offset<float, sgemvt_offset_nb, sgemvt_offset_ntcol, ept, sgemvt_offset_width, 1>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 2: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_offset<float, sgemvt_offset_nb, sgemvt_offset_ntcol, ept, sgemvt_offset_width, 2>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 3: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_offset<float, sgemvt_offset_nb, sgemvt_offset_ntcol, ept, sgemvt_offset_width, 3>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 4: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_offset<float, sgemvt_offset_nb, sgemvt_offset_ntcol, ept, sgemvt_offset_width, 4>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 5: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_offset<float, sgemvt_offset_nb, sgemvt_offset_ntcol, ept, sgemvt_offset_width, 5>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 6: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_offset<float, sgemvt_offset_nb, sgemvt_offset_ntcol, ept, sgemvt_offset_width, 6>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 7: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_offset<float, sgemvt_offset_nb, sgemvt_offset_ntcol, ept, sgemvt_offset_width, 7>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 8: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_offset<float, sgemvt_offset_nb, sgemvt_offset_ntcol, ept, sgemvt_offset_width, 8>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			default: printf("irregular part %d is not supported, please extend the case statement of sgemv\n", ept_); exit(1);
		}
	}
	else
	{
		printf("SGEMV error: Unrecognized transpose mode %c \n", trans);
		return -1;
	}

	return 0;
}

extern "C"
int kblas_sgemv2_offset(char trans, int rows, int cols,
				float alpha, float *dA, int lda,
				float *dX, int incx,
				float  beta, float *dY, int incy,
				int offset_r, int offset_c)
{
	return kblas_sgemv2_offset_driver(trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, offset_r, offset_c, 0);
}

extern "C"
int kblas_sgemv2_offset_async(	char trans, int rows, int cols,
						float alpha, float *dA, int lda,
						float *dX, int incx,
						float  beta, float *dY, int incy,
						int offset_r, int offset_c,
						hipStream_t stream)
{
	return kblas_sgemv2_offset_driver(	trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, offset_r, offset_c, stream);
}
