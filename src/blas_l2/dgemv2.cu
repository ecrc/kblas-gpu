/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/dgemv2.cu

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
#include "gemv2_core.cuh"

#if(SM >= 30)

#define dgemvn_nb               (32)
#define dgemvn_ntcol    		(4)
#define dgemvn_ept              (4)
#define dgemvn_width    (dgemvn_ntcol*dgemvn_ept)
#define dgemvn_by               (4)

#define dgemvt_nb               (32)
#define dgemvt_ntcol            (4)
#define dgemvt_ept              (2)
#define dgemvt_width    (dgemvt_ntcol*dgemvt_ept)
#define dgemvt_by               (4)

#else

#define dgemvn_nb               (64)
#define dgemvn_ntcol    		(8)
#define dgemvn_ept              (2)
#define dgemvn_width    (dgemvn_ntcol*dgemvn_ept)
#define dgemvn_by               (1)

#define dgemvt_nb               (64)
#define dgemvt_ntcol    		(8)
#define dgemvt_ept              (2)
#define dgemvt_width    (dgemvt_ntcol*dgemvt_ept)
#define dgemvt_by               (1)
#endif


extern "C"
int kblas_dscal_async(int n, double alpha, double *x, int incx, cudaStream_t stream);


int kblas_dgemv2_driver(char trans, int rows, int cols,
						double alpha, double *dA, int lda,
						double *dX, int incx,
						double  beta, double *dY, int incy,
						cudaStream_t stream)
{
	if(trans == 'n' || trans == 'N')
	{
		// scaling with beta
		kblas_dscal_async(rows, beta, dY, incy, stream);

		int mod_r = rows % dgemvn_nb;
		int mod_c = cols % dgemvn_width;

		int blocks = rows/dgemvn_nb;
		if(mod_r != 0) blocks += 1;

		const int thread_x = dgemvn_nb;
		const int thread_y = dgemvn_ntcol;
		const int ept = dgemvn_ept;

		int threshold = mod_c / ept;
		int ept_ = mod_c % ept;
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks, dgemvn_by);
		switch(ept_)
		{
			case 0: gemvn<double, dgemvn_nb, dgemvn_ntcol, ept, dgemvn_width, 0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 1: gemvn<double, dgemvn_nb, dgemvn_ntcol, ept, dgemvn_width, 1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 2: gemvn<double, dgemvn_nb, dgemvn_ntcol, ept, dgemvn_width, 2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 3: gemvn<double, dgemvn_nb, dgemvn_ntcol, ept, dgemvn_width, 3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 4: gemvn<double, dgemvn_nb, dgemvn_ntcol, ept, dgemvn_width, 4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 5: gemvn<double, dgemvn_nb, dgemvn_ntcol, ept, dgemvn_width, 5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 6: gemvn<double, dgemvn_nb, dgemvn_ntcol, ept, dgemvn_width, 6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 7: gemvn<double, dgemvn_nb, dgemvn_ntcol, ept, dgemvn_width, 7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 8: gemvn<double, dgemvn_nb, dgemvn_ntcol, ept, dgemvn_width, 8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			default: printf("irregular part %d is not supported, please extend the case statement of dgemv\n", ept_); exit(1);
		}
	}	// end of non-transpose case
	else if(trans == 't' || trans == 'T' || trans == 'c' || trans == 'C')
	{
		// scaling with beta
		kblas_dscal_async(cols, beta, dY, incy, stream);

		int mod_r = rows % dgemvt_nb;
		int mod_c = cols % dgemvt_width;

		int blocks = cols/dgemvt_width;
		if(mod_c != 0) blocks += 1;

		const int thread_x = dgemvt_nb;
		const int thread_y = dgemvt_ntcol;
		const int ept = dgemvt_ept;

		int threshold = mod_c / ept;
		int ept_ = mod_c % ept;

		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks, dgemvt_by);

		int conj;
		if(trans == 'c' || trans == 'C')conj = 1;
		else conj = 0;
		//printf("modr = %d, modc = %d, threshold = %d, ept_ = %d \n", mod_r, mod_c, threshold, ept_);
		switch(ept_)
		{
			case 0: gemvt<double, dgemvt_nb, dgemvt_ntcol, ept, dgemvt_width, 0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 1: gemvt<double, dgemvt_nb, dgemvt_ntcol, ept, dgemvt_width, 1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 2: gemvt<double, dgemvt_nb, dgemvt_ntcol, ept, dgemvt_width, 2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 3: gemvt<double, dgemvt_nb, dgemvt_ntcol, ept, dgemvt_width, 3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 4: gemvt<double, dgemvt_nb, dgemvt_ntcol, ept, dgemvt_width, 4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 5: gemvt<double, dgemvt_nb, dgemvt_ntcol, ept, dgemvt_width, 5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 6: gemvt<double, dgemvt_nb, dgemvt_ntcol, ept, dgemvt_width, 6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 7: gemvt<double, dgemvt_nb, dgemvt_ntcol, ept, dgemvt_width, 7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 8: gemvt<double, dgemvt_nb, dgemvt_ntcol, ept, dgemvt_width, 8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
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
int kblas_dgemv2(char trans, int rows, int cols,
				double alpha, double *dA, int lda,
				double *dX, int incx,
				double  beta, double *dY, int incy)
{
	return kblas_dgemv2_driver(	trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, 0);
}

extern "C"
int kblas_dgemv2_async(	char trans, int rows, int cols,
						double alpha, double *dA, int lda,
						double *dX, int incx,
						double  beta, double *dY, int incy,
						cudaStream_t stream)
{
	return kblas_dgemv2_driver(	trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, stream);
}
