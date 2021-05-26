/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/sgemv.cu

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
#include "gemv_core.cuh"

#if(TARGET_SM >= 30)

#define sgemvn_bs		(64)
#define sgemvn_ty		(8)
#define sgemvn_by		(8)

#define sgemvt_bs		(64)
#define sgemvt_ty		(4)
#define sgemvt_by		(4)

#else

#define sgemvn_bs		(32)
#define sgemvn_ty		(8)
#define sgemvn_by		(1)

#define sgemvt_bs		(32)
#define sgemvt_ty		(8)
#define sgemvt_by		(1)

#endif

extern "C"
int kblas_sscal_async(int n, float alpha, float *x, int incx, cudaStream_t stream);

int kblas_sgemv_driver( char trans, int rows, int cols,
						float alpha, float *dA, int lda,
						float *dX, int incx,
						float  beta, float *dY, int incy, cudaStream_t stream)
{
	if(trans == 'n' || trans == 'N')
	{
		// scaling with beta
		kblas_sscal_async(rows, beta, dY, incy, stream);

		int mod_r = rows % sgemvn_bs;
		int mod_c = cols % sgemvn_bs;

		if(mod_r == 0)
		{
			if(mod_c == 0)
			{
				// special case
				int blocks = rows/sgemvn_bs;
				const int thread_x = sgemvn_bs;
				const int thread_y = sgemvn_ty;
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, sgemvn_by);
				const int elements_per_thread = thread_x/(2*thread_y);
				gemvn_special<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy);
				//gemvn_special_<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy);
			}
			else
			{
				// generic case for columns only
				int blocks = rows/sgemvn_bs;
				blocks += 1;	// dummy thread block
				const int thread_x = sgemvn_bs;
				const int thread_y = sgemvn_ty;
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, sgemvn_by);
				const int elements_per_thread = thread_x/(2*thread_y);
				const int irregular_cols = mod_c % elements_per_thread;
				switch(irregular_cols)
				{
					/**
					 * The kernel for irregular dimensions has an extra template parameter.
				 	 * This parameter must be among the values listed in the switch-case statement below.
				 	 * The possible values are in the range 0 - (elements_per_thread-1)
				 	 * Make sure these values are updated whenever you change the configuration parameters.
					**/
					case  0: gemvn_generic<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread,  0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  1: gemvn_generic<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread,  1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  2: gemvn_generic<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread,  2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  3: gemvn_generic<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread,  3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  4: gemvn_generic<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread,  4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  5: gemvn_generic<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread,  5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  6: gemvn_generic<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread,  6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  7: gemvn_generic<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread,  7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  8: gemvn_generic<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread,  8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					default: printf("SGEMV-N error: improper template parameter. Please read the inline documentation for this function. \n"); return -1;

				}
			}
		}
		else	// mod_r != 0
		{
			if(mod_c == 0)
			{
				// generic case for columns only
				int blocks = (rows/sgemvn_bs) + (mod_r != 0);
				const int thread_x = sgemvn_bs;
				const int thread_y = sgemvn_ty;
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, sgemvn_by);
				const int elements_per_thread = thread_x/(2*thread_y);
				gemvn_generic<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread, 0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c);
			}
			else
			{
				// generic case for rows and cols
				int blocks = (rows/sgemvn_bs) + (mod_r != 0);
				const int thread_x = sgemvn_bs;
				const int thread_y = sgemvn_ty;
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, sgemvn_by);
				const int elements_per_thread = thread_x/(2*thread_y);
				const int irregular_cols = mod_c % elements_per_thread;
				switch(irregular_cols)
				{
					/**
					 * The kernel for irregular dimensions has an extra template parameter.
				 	 * This parameter must be among the values listed in the switch-case statement below.
				 	 * The possible values are in the range 0 - (elements_per_thread-1)
				 	 * Make sure these values are updated whenever you change the configuration parameters.
					**/
					case  0: gemvn_generic<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread,  0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  1: gemvn_generic<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread,  1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  2: gemvn_generic<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread,  2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  3: gemvn_generic<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread,  3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  4: gemvn_generic<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread,  4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  5: gemvn_generic<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread,  5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  6: gemvn_generic<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread,  6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  7: gemvn_generic<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread,  7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  8: gemvn_generic<float, sgemvn_bs, sgemvn_bs, sgemvn_ty, elements_per_thread,  8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					default: printf("SGEMV-N error: improper template parameter. Please read the inline documentation for this function. \n"); return -1;

				}
			}
		}
	}	// end of non-transpose case
	else if (trans == 't' || trans == 'T' || trans == 'c' || trans == 'C')
	{
		int conj;
		if(trans == 'c' || trans == 'C') conj = 1;
		else conj = 0;
		// scaling with beta
		kblas_sscal_async(cols, beta, dY, incy, stream);

		int mod_r = rows % sgemvt_bs;
		int mod_c = cols % sgemvt_bs;

		if(mod_c == 0)
		{
			if(mod_r == 0)
			{
				// special case
				int blocks = cols/sgemvt_bs;
				const int thread_x = sgemvt_bs;
				const int thread_y = sgemvt_ty;
				const int elements_per_thread = thread_x/(2*thread_y);
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, sgemvt_by);
				gemvt_special<float, sgemvt_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, conj);
			}
			else
			{
				// mod_r != 0
				int blocks = cols/sgemvt_bs;
				blocks += 1;	// dummy thread block
				const int thread_x = sgemvt_bs;
				const int thread_y = sgemvt_ty;
				const int elements_per_thread = thread_x/(2*thread_y);
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, sgemvt_by);
				gemvt_generic<float, sgemvt_bs, thread_x, thread_y, elements_per_thread, 0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj);
			}
		}
		else	// mod_c != 0
		{
			int blocks = cols/sgemvt_bs + (mod_c != 0);
			const int thread_x = sgemvt_bs;
			const int thread_y = sgemvt_ty;
			const int elements_per_thread = thread_x/(2*thread_y);
			const int irregular_cols = mod_c % elements_per_thread;
			dim3 dimBlock(thread_x, thread_y);
			dim3 dimGrid(blocks, sgemvt_by);

			switch(irregular_cols)
			{
				/**
				  * The kernel for irregular dimensions has an extra template parameter.
				  * This parameter must be among the values listed in the switch-case statement below.
				  * The possible values are in the range 0 - (elements_per_thread-1)
				  * Make sure these values are updated whenever you change the configuration parameters.
				**/
				case  0: gemvt_generic<float, sgemvt_bs, thread_x, thread_y, elements_per_thread,  0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  1: gemvt_generic<float, sgemvt_bs, thread_x, thread_y, elements_per_thread,  1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  2: gemvt_generic<float, sgemvt_bs, thread_x, thread_y, elements_per_thread,  2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  3: gemvt_generic<float, sgemvt_bs, thread_x, thread_y, elements_per_thread,  3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  4: gemvt_generic<float, sgemvt_bs, thread_x, thread_y, elements_per_thread,  4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  5: gemvt_generic<float, sgemvt_bs, thread_x, thread_y, elements_per_thread,  5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  6: gemvt_generic<float, sgemvt_bs, thread_x, thread_y, elements_per_thread,  6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  7: gemvt_generic<float, sgemvt_bs, thread_x, thread_y, elements_per_thread,  7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  8: gemvt_generic<float, sgemvt_bs, thread_x, thread_y, elements_per_thread,  8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				default: printf("SGEMV-T error: improper template parameter. Please read the inline documentation for this function. \n"); return -1;
			}
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
int kblas_sgemv(char trans, int rows, int cols,
				float alpha, float *dA, int lda,
				float *dX, int incx,
				float  beta, float *dY, int incy)
{
	return kblas_sgemv_driver( trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, 0);
}

extern "C"
int kblas_sgemv_async( 	char trans, int rows, int cols,
						float alpha, float *dA, int lda,
						float *dX, int incx,
						float  beta, float *dY, int incy,
						cudaStream_t stream)
{
	return kblas_sgemv_driver( trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, stream);
}
