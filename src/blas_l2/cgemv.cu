/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/cgemv.cu

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
#include "gemv_core.cuh"

#if (SM >= 30)

#define cgemvn_bs		(32)
#define cgemvn_ty		(16)
#define cgemvn_by		(2)

#define cgemvt_bs		(32)
#define cgemvt_ty		(16)
#define cgemvt_by		(2)

#else

#define cgemvn_bs		(32)
#define cgemvn_ty		(4)
#define cgemvn_by		(2)

#define cgemvt_bs		(32)
#define cgemvt_ty		(4)
#define cgemvt_by		(2)

#endif

extern "C"
int kblas_cscal_async(int n, cuFloatComplex alpha, cuFloatComplex *x, int incx, cudaStream_t stream);

int kblas_cgemv_driver( char trans, int rows, int cols,
						cuFloatComplex alpha, cuFloatComplex *dA, int lda,
						cuFloatComplex *dX, int incx,
						cuFloatComplex  beta, cuFloatComplex *dY, int incy,
						cudaStream_t stream)
{
	if(trans == 'n' || trans == 'N')
	{
		// scaling with beta
		kblas_cscal_async(rows, beta, dY, incy, stream);

		int mod_r = rows % cgemvn_bs;
		int mod_c = cols % cgemvn_bs;

		if(mod_r == 0)
		{
			if(mod_c == 0)
			{
				// special case
				int blocks = rows/cgemvn_bs;
				const int thread_x = cgemvn_bs;
				const int thread_y = cgemvn_ty;
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, cgemvn_by);
				const int elements_per_thread = thread_x/(2*thread_y);
				gemvn_special<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy);
			}
			else
			{
				// generic case for columns only
				int blocks = rows/cgemvn_bs;
				blocks += 1;	// dummy thread block
				const int thread_x = cgemvn_bs;
				const int thread_y = cgemvn_ty;
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, cgemvn_by);
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
					case  0: gemvn_generic<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread,  0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  1: gemvn_generic<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread,  1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  2: gemvn_generic<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread,  2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  3: gemvn_generic<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread,  3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  4: gemvn_generic<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread,  4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  5: gemvn_generic<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread,  5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  6: gemvn_generic<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread,  6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  7: gemvn_generic<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread,  7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  8: gemvn_generic<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread,  8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					default: printf("CGEMV-N error: improper template parameter. Please read the inline documentation for this function. \n"); return -1;

				}
			}
		}
		else	// mod_r != 0
		{
			if(mod_c == 0)
			{
				// generic case for columns only
				int blocks = (rows/cgemvn_bs) + (mod_r != 0);
				const int thread_x = cgemvn_bs;
				const int thread_y = cgemvn_ty;
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, cgemvn_by);
				const int elements_per_thread = thread_x/(2*thread_y);
				gemvn_generic<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread, 0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c);
			}
			else
			{
				// generic case for rows and cols
				int blocks = (rows/cgemvn_bs) + (mod_r != 0);
				const int thread_x = cgemvn_bs;
				const int thread_y = cgemvn_ty;
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, cgemvn_by);
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
					case  0: gemvn_generic<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread,  0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  1: gemvn_generic<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread,  1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  2: gemvn_generic<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread,  2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  3: gemvn_generic<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread,  3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  4: gemvn_generic<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread,  4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  5: gemvn_generic<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread,  5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  6: gemvn_generic<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread,  6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  7: gemvn_generic<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread,  7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  8: gemvn_generic<cuFloatComplex, cgemvn_bs, cgemvn_bs, cgemvn_ty, elements_per_thread,  8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					default: printf("CGEMV-N error: improper template parameter. Please read the inline documentation for this function. \n"); return -1;

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
		kblas_cscal_async(cols, beta, dY, incy, stream);

		int mod_r = rows % cgemvt_bs;
		int mod_c = cols % cgemvt_bs;

		if(mod_c == 0)
		{
			if(mod_r == 0)
			{
				// special case
				int blocks = cols/cgemvt_bs;
				const int thread_x = cgemvt_bs;
				const int thread_y = cgemvt_ty;
				const int elements_per_thread = thread_x/(2*thread_y);
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, cgemvt_by);
				gemvt_special<cuFloatComplex, cgemvt_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, conj);
			}
			else
			{
				// mod_r != 0
				int blocks = cols/cgemvt_bs;
				blocks += 1;	// dummy thread block
				const int thread_x = cgemvt_bs;
				const int thread_y = cgemvt_ty;
				const int elements_per_thread = thread_x/(2*thread_y);
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, cgemvt_by);
				gemvt_generic<cuFloatComplex, cgemvt_bs, thread_x, thread_y, elements_per_thread, 0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj);
			}
		}
		else	// mod_c != 0
		{
			int blocks = cols/cgemvt_bs + (mod_c != 0);
			const int thread_x = cgemvt_bs;
			const int thread_y = cgemvt_ty;
			const int elements_per_thread = thread_x/(2*thread_y);
			const int irregular_cols = mod_c % elements_per_thread;
			dim3 dimBlock(thread_x, thread_y);
			dim3 dimGrid(blocks, cgemvt_by);

			switch(irregular_cols)
			{
				/**
				 * The kernel for irregular dimensions has an extra template parameter.
				 * This parameter must be among the values listed in the switch-case statement below.
				 * The possible values are in the range 0 - (elements_per_thread-1)
				 * Make sure these values are updated whenever you change the configuration parameters.
				**/
				case  0: gemvt_generic<cuFloatComplex, cgemvt_bs, thread_x, thread_y, elements_per_thread,  0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  1: gemvt_generic<cuFloatComplex, cgemvt_bs, thread_x, thread_y, elements_per_thread,  1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  2: gemvt_generic<cuFloatComplex, cgemvt_bs, thread_x, thread_y, elements_per_thread,  2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  3: gemvt_generic<cuFloatComplex, cgemvt_bs, thread_x, thread_y, elements_per_thread,  3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  4: gemvt_generic<cuFloatComplex, cgemvt_bs, thread_x, thread_y, elements_per_thread,  4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  5: gemvt_generic<cuFloatComplex, cgemvt_bs, thread_x, thread_y, elements_per_thread,  5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  6: gemvt_generic<cuFloatComplex, cgemvt_bs, thread_x, thread_y, elements_per_thread,  6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  7: gemvt_generic<cuFloatComplex, cgemvt_bs, thread_x, thread_y, elements_per_thread,  7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  8: gemvt_generic<cuFloatComplex, cgemvt_bs, thread_x, thread_y, elements_per_thread,  8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				default: printf("CGEMV-T error: improper template parameter. Please read the inline documentation for this function. \n"); return -1;
			}
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
int kblas_cgemv( char trans, int rows, int cols,
				cuFloatComplex alpha, cuFloatComplex *dA, int lda,
				cuFloatComplex *dX, int incx,
				cuFloatComplex  beta, cuFloatComplex *dY, int incy)
{
	return kblas_cgemv_driver( trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, 0);
}

extern "C"
int kblas_cgemv_async( 	char trans, int rows, int cols,
						cuFloatComplex alpha, cuFloatComplex *dA, int lda,
						cuFloatComplex *dX, int incx,
						cuFloatComplex  beta, cuFloatComplex *dY, int incy,
						cudaStream_t stream)
{
	return kblas_cgemv_driver( trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, stream);
}
