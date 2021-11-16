#include "hip/hip_runtime.h"
/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/dgemv.cu

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
#include "gemv_core.cuh"

#if(TARGET_SM >= 30)

#define dgemvn_bs		(64)
#define dgemvn_ty		(4)
#define dgemvn_by		(4)

#define dgemvt_bs		(64)
#define dgemvt_ty		(4)
#define dgemvt_by		(2)

#else

#define dgemvn_bs		(64)
#define dgemvn_ty		(8)
#define dgemvn_by		(2)

#define dgemvt_bs		(64)
#define dgemvt_ty		(8)
#define dgemvt_by		(2)

#endif

extern "C"
int kblas_dscal_async(int n, double alpha, double *x, int incx, hipStream_t stream);


int kblas_dgemv_driver(	char trans, int rows, int cols,
						double alpha, double *dA, int lda,
						double *dX, int incx,
						double  beta, double *dY, int incy,
						hipStream_t stream)
{
	if(trans == 'n' || trans == 'N')
	{
		// scaling with beta
		kblas_dscal_async(rows, beta, dY, incy, stream);

		int mod_r = rows % dgemvn_bs;
		int mod_c = cols % dgemvn_bs;

		if(mod_r == 0)
		{
			if(mod_c == 0)
			{
				// special case
				int blocks = rows/dgemvn_bs;
				const int thread_x = dgemvn_bs;
				const int thread_y = dgemvn_ty;
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, dgemvn_by);
				const int elements_per_thread = thread_x/(2*thread_y);
				hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_special<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy);
			}
			else
			{
				// generic case for columns only
				int blocks = rows/dgemvn_bs;
				blocks += 1;	// dummy thread block
				const int thread_x = dgemvn_bs;
				const int thread_y = dgemvn_ty;
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, dgemvn_by);
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
					case  0: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  0>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  1: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  1>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  2: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  2>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  3: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  3>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  4: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  4>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  5: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  5>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  6: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  6>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  7: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  7>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  8: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  8>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					default: printf("DGEMV-N error: improper template parameter. Please read the inline documentation for this function. \n"); return -1;

				}
			}
		}
		else	// mod_r != 0
		{
			if(mod_c == 0)
			{
				// generic case for columns only
				int blocks = (rows/dgemvn_bs) + (mod_r != 0);
				const int thread_x = dgemvn_bs;
				const int thread_y = dgemvn_ty;
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, dgemvn_by);
				const int elements_per_thread = thread_x/(2*thread_y);
				hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread, 0>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c);
			}
			else
			{
				// generic case for rows and cols
				int blocks = (rows/dgemvn_bs) + (mod_r != 0);
				const int thread_x = dgemvn_bs;
				const int thread_y = dgemvn_ty;
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, dgemvn_by);
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
					case  0: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  0>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  1: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  1>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  2: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  2>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  3: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  3>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  4: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  4>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  5: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  5>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  6: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  6>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  7: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  7>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  8: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  8>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					default: printf("DGEMV-N error: improper template parameter. Please read the inline documentation for this function. \n"); return -1;

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
		kblas_dscal_async(cols, beta, dY, incy, stream);

		int mod_r = rows % dgemvt_bs;
		int mod_c = cols % dgemvt_bs;

		if(mod_c == 0)
		{
			if(mod_r == 0)
			{
				// special case
				int blocks = cols/dgemvt_bs;
				const int thread_x = dgemvt_bs;
				const int thread_y = dgemvt_ty;
				const int elements_per_thread = thread_x/(2*thread_y);
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, dgemvt_by);
				hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_special<double, dgemvt_bs, thread_x, thread_y, elements_per_thread>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, conj);
			}
			else
			{
				// mod_r != 0
				int blocks = cols/dgemvt_bs;
				blocks += 1;	// dummy thread block
				const int thread_x = dgemvt_bs;
				const int thread_y = dgemvt_ty;
				const int elements_per_thread = thread_x/(2*thread_y);
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, dgemvt_by);
				hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread, 0>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj);
			}
		}
		else	// mod_c != 0
		{
			int blocks = cols/dgemvt_bs + (mod_c != 0);
			const int thread_x = dgemvt_bs;
			const int thread_y = dgemvt_ty;
			const int elements_per_thread = thread_x/(2*thread_y);
			const int irregular_cols = mod_c % elements_per_thread;
			dim3 dimBlock(thread_x, thread_y);
			dim3 dimGrid(blocks, dgemvt_by);

			switch(irregular_cols)
			{
				/**
				  * The kernel for irregular dimensions has an extra template parameter.
				  * This parameter must be among the values listed in the switch-case statement below.
				  * The possible values are in the range 0 - (elements_per_thread-1)
				  * Make sure these values are updated whenever you change the configuration parameters.
				**/
				case  0: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread,  0>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  1: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread,  1>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  2: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread,  2>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  3: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread,  3>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  4: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread,  4>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  5: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread,  5>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  6: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread,  6>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  7: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread,  7>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  8: hipLaunchKernelGGL(HIP_KERNEL_NAME(gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread,  8>), dim3(dimGrid), dim3(dimBlock), 0, stream, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				default: printf("DGEMV-T error: improper template parameter. Please read the inline documentation for this function. \n"); return -1;
			}
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
int kblas_dgemv(char trans, int rows, int cols,
				double alpha, double *dA, int lda,
				double *dX, int incx,
				double  beta, double *dY, int incy)
{
	return kblas_dgemv_driver(	trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, 0);
}

extern "C"
int kblas_dgemv_async(	char trans, int rows, int cols,
						double alpha, double *dA, int lda,
						double *dX, int incx,
						double  beta, double *dY, int incy,
						hipStream_t stream)
{
	return kblas_dgemv_driver(	trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, stream);
}
