/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/dgemv2_offset.cu

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
#include "gemv2_offset_core.cuh"

#if(SM >= 30)

#define dgemvn_offset_nb               	(32)
#define dgemvn_offset_ntcol    			(4)
#define dgemvn_offset_ept              	(4)
#define dgemvn_offset_width    			(dgemvn_offset_ntcol*dgemvn_offset_ept)
#define dgemvn_offset_by               	(8)

#define dgemvt_offset_nb               	(32)
#define dgemvt_offset_ntcol            	(4)
#define dgemvt_offset_ept              	(2)
#define dgemvt_offset_width    			(dgemvt_offset_ntcol*dgemvt_offset_ept)
#define dgemvt_offset_by               	(4)

#else

#define dgemvn_offset_nb               	(64)
#define dgemvn_offset_ntcol    			(8)
#define dgemvn_offset_ept              	(2)
#define dgemvn_offset_width    			(dgemvn_offset_ntcol*dgemvn_offset_ept)
#define dgemvn_offset_by				(1)

#define dgemvt_offset_nb               	(64)
#define dgemvt_offset_ntcol    			(8)
#define dgemvt_offset_ept              	(2)
#define dgemvt_offset_width    			(dgemvt_offset_ntcol*dgemvt_offset_ept)
#define dgemvt_offset_by               	(1)
#endif


extern "C"
int kblas_dscal_async(int n, double alpha, double *x, int incx, cudaStream_t stream);


int kblas_dgemv2_offset_driver(char trans, int rows, int cols,
						double alpha, double *dA, int lda,
						double *dX, int incx,
						double  beta, double *dY, int incy,
						int offset_r, int offset_c,
						cudaStream_t stream)
{
	if(trans == 'n' || trans == 'N')
	{
		// offset necessary calculations
		int offset_r_ = offset_r % dgemvn_offset_nb;
		int offset_c_ = offset_c % dgemvn_offset_width;
		int rows_ = rows - (offset_r - offset_r_);
		int cols_ = cols - (offset_c - offset_c_);

		// Advance pointers
		dA += (offset_c - offset_c_) * lda + (offset_r - offset_r_);
		dX += (offset_c - offset_c_) * incx;
		dY += (offset_r - offset_r_) * incy;

		// scaling with beta
		kblas_dscal_async(rows_, beta, dY, incy, stream);

		int mod_r = rows_ % dgemvn_offset_nb;
		int mod_c = cols_ % dgemvn_offset_width;

		int blocks = rows_/dgemvn_offset_nb;
		if(mod_r != 0) blocks += 1;

		const int thread_x = dgemvn_offset_nb;
		const int thread_y = dgemvn_offset_ntcol;
		const int ept = dgemvn_offset_ept;

		int threshold = mod_c / ept;
		int ept_ = mod_c % ept;
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks, dgemvn_offset_by);
		//printf("rows_ = %d - cols_ = %d - mod_r = %d - mod_c = %d - offset_r_ = %d - offset_c_ = %d \n", rows_, cols_, mod_r, mod_c, offset_r_, offset_c_);
		switch(ept_)
		{
			case 0: gemvn_offset<double, dgemvn_offset_nb, dgemvn_offset_ntcol, ept, dgemvn_offset_width, 0><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 1: gemvn_offset<double, dgemvn_offset_nb, dgemvn_offset_ntcol, ept, dgemvn_offset_width, 1><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 2: gemvn_offset<double, dgemvn_offset_nb, dgemvn_offset_ntcol, ept, dgemvn_offset_width, 2><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 3: gemvn_offset<double, dgemvn_offset_nb, dgemvn_offset_ntcol, ept, dgemvn_offset_width, 3><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 4: gemvn_offset<double, dgemvn_offset_nb, dgemvn_offset_ntcol, ept, dgemvn_offset_width, 4><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 5: gemvn_offset<double, dgemvn_offset_nb, dgemvn_offset_ntcol, ept, dgemvn_offset_width, 5><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 6: gemvn_offset<double, dgemvn_offset_nb, dgemvn_offset_ntcol, ept, dgemvn_offset_width, 6><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 7: gemvn_offset<double, dgemvn_offset_nb, dgemvn_offset_ntcol, ept, dgemvn_offset_width, 7><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 8: gemvn_offset<double, dgemvn_offset_nb, dgemvn_offset_ntcol, ept, dgemvn_offset_width, 8><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			default: printf("irregular part %d is not supported, please extend the case statement of dgemv\n", ept_); exit(1);
		}
	}	// end of non-transpose case
	else if(trans == 't' || trans == 'T' || trans == 'c' || trans == 'C')
	{
		// offset necessary calculations
		int offset_r_ = offset_r % dgemvt_offset_nb;
		int offset_c_ = offset_c % dgemvt_offset_width;
		int rows_ = rows - (offset_r - offset_r_);
		int cols_ = cols - (offset_c - offset_c_);

		// Advance pointers
		dA += (offset_c - offset_c_) * lda + (offset_r - offset_r_);
		dX += (offset_r - offset_r_) * incx;
		dY += (offset_c - offset_c_) * incy;

		// scaling with beta
		kblas_dscal_async(cols_, beta, dY, incy, stream);

		int mod_r = rows_ % dgemvt_offset_nb;
		int mod_c = cols_ % dgemvt_offset_width;

		int blocks = cols_/dgemvt_offset_width;
		if(mod_c != 0) blocks += 1;

		const int thread_x = dgemvt_offset_nb;
		const int thread_y = dgemvt_offset_ntcol;
		const int ept = dgemvt_offset_ept;

		int threshold = mod_c / ept;
		int ept_ = mod_c % ept;

		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks, dgemvt_offset_by);

		int conj;
		if(trans == 'c' || trans == 'C')conj = 1;
		else conj = 0;
		//printf("modr = %d, modc = %d, threshold = %d, ept_ = %d \n", mod_r, mod_c, threshold, ept_);
		//printf("rows_ = %d - cols_ = %d - mod_r = %d - mod_c = %d - offset_r_ = %d - offset_c_ = %d \n", rows_, cols_, mod_r, mod_c, offset_r_, offset_c_);
		switch(ept_)
		{
			case 0: gemvt_offset<double, dgemvt_offset_nb, dgemvt_offset_ntcol, ept, dgemvt_offset_width, 0><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 1: gemvt_offset<double, dgemvt_offset_nb, dgemvt_offset_ntcol, ept, dgemvt_offset_width, 1><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 2: gemvt_offset<double, dgemvt_offset_nb, dgemvt_offset_ntcol, ept, dgemvt_offset_width, 2><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 3: gemvt_offset<double, dgemvt_offset_nb, dgemvt_offset_ntcol, ept, dgemvt_offset_width, 3><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 4: gemvt_offset<double, dgemvt_offset_nb, dgemvt_offset_ntcol, ept, dgemvt_offset_width, 4><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 5: gemvt_offset<double, dgemvt_offset_nb, dgemvt_offset_ntcol, ept, dgemvt_offset_width, 5><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 6: gemvt_offset<double, dgemvt_offset_nb, dgemvt_offset_ntcol, ept, dgemvt_offset_width, 6><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 7: gemvt_offset<double, dgemvt_offset_nb, dgemvt_offset_ntcol, ept, dgemvt_offset_width, 7><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 8: gemvt_offset<double, dgemvt_offset_nb, dgemvt_offset_ntcol, ept, dgemvt_offset_width, 8><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			default: printf("irregular part %d is not supported, please extend the case statement of dgemv\n", ept_); exit(1);
		}
	}
	else
	{
		printf("DGEMV error: Unrecognized transpose mode %c \n", trans);
		return -1;
	}

	return 0;
}

extern "C"
int kblas_dgemv2_offset(char trans, int rows, int cols,
				double alpha, double *dA, int lda,
				double *dX, int incx,
				double  beta, double *dY, int incy,
				int offset_r, int offset_c)
{
	return kblas_dgemv2_offset_driver(trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, offset_r, offset_c, 0);
}

extern "C"
int kblas_dgemv2_offset_async(	char trans, int rows, int cols,
						double alpha, double *dA, int lda,
						double *dX, int incx,
						double  beta, double *dY, int incy,
						int offset_r, int offset_c,
						cudaStream_t stream)
{
	return kblas_dgemv2_offset_driver(	trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, offset_r, offset_c, stream);
}
