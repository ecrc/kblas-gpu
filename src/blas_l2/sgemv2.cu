/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/sgemv2.cu

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
#include "gemv2_core.cuh"

#if(TARGET_SM >= 30)

#define sgemvn_nb               (32)
#define sgemvn_ntcol            (4)
#define sgemvn_ept              (4)
#define sgemvn_width    (sgemvn_ntcol*sgemvn_ept)
#define sgemvn_by               (8)

#define sgemvt_nb               (32)
#define sgemvt_ntcol            (4)
#define sgemvt_ept              (8)
#define sgemvt_width    (sgemvt_ntcol*sgemvt_ept)
#define sgemvt_by               (4)

#else

#define sgemvn_nb               (64)
#define sgemvn_ntcol    		(8)
#define sgemvn_ept              (2)
#define sgemvn_width    (sgemvn_ntcol*sgemvn_ept)
#define sgemvn_by               (1)

#define sgemvt_nb               (64)
#define sgemvt_ntcol    		(8)
#define sgemvt_ept              (2)
#define sgemvt_width    (sgemvt_ntcol*sgemvt_ept)
#define sgemvt_by               (1)
#endif


extern "C"
int kblas_sscal_async(int n, float alpha, float *x, int incx, cudaStream_t stream);


int kblas_sgemv2_driver(	char trans, int rows, int cols,
						float alpha, float *dA, int lda,
						float *dX, int incx,
						float  beta, float *dY, int incy,
						cudaStream_t stream)
{
	if(trans == 'n' || trans == 'N')
	{
		// scaling with beta
		kblas_sscal_async(rows, beta, dY, incy, stream);

		int mod_r = rows % sgemvn_nb;
		int mod_c = cols % sgemvn_width;

		int blocks = rows/sgemvn_nb;
		if(mod_r != 0) blocks += 1;

		const int thread_x = sgemvn_nb;
		const int thread_y = sgemvn_ntcol;
		const int ept = sgemvn_ept;

		int threshold = mod_c / ept;
		int ept_ = mod_c % ept;
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks, sgemvn_by);
		switch(ept_)
		{
			case 0: gemvn<float, sgemvn_nb, sgemvn_ntcol, ept, sgemvn_width, 0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 1: gemvn<float, sgemvn_nb, sgemvn_ntcol, ept, sgemvn_width, 1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 2: gemvn<float, sgemvn_nb, sgemvn_ntcol, ept, sgemvn_width, 2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 3: gemvn<float, sgemvn_nb, sgemvn_ntcol, ept, sgemvn_width, 3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 4: gemvn<float, sgemvn_nb, sgemvn_ntcol, ept, sgemvn_width, 4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 5: gemvn<float, sgemvn_nb, sgemvn_ntcol, ept, sgemvn_width, 5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 6: gemvn<float, sgemvn_nb, sgemvn_ntcol, ept, sgemvn_width, 6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 7: gemvn<float, sgemvn_nb, sgemvn_ntcol, ept, sgemvn_width, 7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 8: gemvn<float, sgemvn_nb, sgemvn_ntcol, ept, sgemvn_width, 8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			default: printf("irregular part %d is not supported, please extend the case statement of sgemv\n", ept_); exit(1);
		}
	}	// end of non-transpose case
	else if(trans == 't' || trans == 'T' || trans == 'c' || trans == 'C')
	{
		// scaling with beta
		kblas_sscal_async(cols, beta, dY, incy, stream);

		int mod_r = rows % sgemvt_nb;
		int mod_c = cols % sgemvt_width;

		int blocks = cols/sgemvt_width;
		if(mod_c != 0) blocks += 1;

		const int thread_x = sgemvt_nb;
		const int thread_y = sgemvt_ntcol;
		const int ept = sgemvt_ept;

		int threshold = mod_c / ept;
		int ept_ = mod_c % ept;

		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks, sgemvt_by);

		int conj;
		if(trans == 'c' || trans == 'C')conj = 1;
		else conj = 0;
		//printf("modr = %d, modc = %d, threshold = %d, ept_ = %d \n", mod_r, mod_c, threshold, ept_);
		switch(ept_)
		{
			case 0: gemvt<float, sgemvt_nb, sgemvt_ntcol, ept, sgemvt_width, 0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 1: gemvt<float, sgemvt_nb, sgemvt_ntcol, ept, sgemvt_width, 1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 2: gemvt<float, sgemvt_nb, sgemvt_ntcol, ept, sgemvt_width, 2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 3: gemvt<float, sgemvt_nb, sgemvt_ntcol, ept, sgemvt_width, 3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 4: gemvt<float, sgemvt_nb, sgemvt_ntcol, ept, sgemvt_width, 4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 5: gemvt<float, sgemvt_nb, sgemvt_ntcol, ept, sgemvt_width, 5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 6: gemvt<float, sgemvt_nb, sgemvt_ntcol, ept, sgemvt_width, 6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 7: gemvt<float, sgemvt_nb, sgemvt_ntcol, ept, sgemvt_width, 7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 8: gemvt<float, sgemvt_nb, sgemvt_ntcol, ept, sgemvt_width, 8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
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
int kblas_sgemv2(char trans, int rows, int cols,
				float alpha, float *dA, int lda,
				float *dX, int incx,
				float  beta, float *dY, int incy)
{
	return kblas_sgemv2_driver(	trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, 0);
}

extern "C"
int kblas_sgemv2_async(	char trans, int rows, int cols,
						float alpha, float *dA, int lda,
						float *dX, int incx,
						float  beta, float *dY, int incy,
						cudaStream_t stream)
{
	return kblas_sgemv2_driver(	trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, stream);
}
