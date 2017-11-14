/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/dgemv_mgpu.cu

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
#include "gemv_mgpu_core.cuh"
#include "gemv_mgpu_offset_core.cuh"
#include "defs.h"

#if(SM >= 30)

#define dgemvn_mgpu_bs		(64)
#define dgemvn_mgpu_ty		(4)
//#define dgemvn_mgpu_by		(2)

#define dgemvt_mgpu_bs		(64)
#define dgemvt_mgpu_ty		(4)
//#define dgemvt_mgpu_by		(2)

#else

#define dgemvn_mgpu_bs		(64)
#define dgemvn_mgpu_ty		(8)
//#define dgemvn_mgpu_by		(1)

#define dgemvt_mgpu_bs		(64)
#define dgemvt_mgpu_ty		(8)
//#define dgemvt_mgpu_by		(1)

#endif

extern "C"
int kblas_dscal_async(int n, double alpha, double *x, int incx, cudaStream_t stream);

extern "C"
int kblas_dgemv_mgpu_driver( char trans, int rows, int cols,
						double alpha, double *dA, int lda,
						double *dX, int incx,
						double  beta, double *dY, int incy, int gpu_gid, int ngpus,
						cudaStream_t stream = 0)
{
    const double d_zero = 0.0;

	if(trans == 'n' || trans == 'N')
	{
		//******** config parameters
		const int thread_x = dgemvn_mgpu_bs;
		const int thread_y = dgemvn_mgpu_ty;
		const int elements_per_thread = thread_x/(2*thread_y);
		int grid_y_n = 1 * ngpus;
		//**************************

		// scaling with beta
		//if(gpu_gid == 0)cublasDscal(rows, beta, dY, incy);
		if(gpu_gid == 0)kblas_dscal_async(rows, beta, dY, incy, stream);
		else kblas_dscal_async(rows, d_zero, dY, incy, stream);

		int cols_ = dgemvn_mgpu_bs * ( (cols/dgemvn_mgpu_bs)/ngpus );
		if(gpu_gid < (cols/dgemvn_mgpu_bs)%ngpus) cols_ += dgemvn_mgpu_bs;
		if(gpu_gid == (cols/dgemvn_mgpu_bs)%ngpus) cols_ += cols%dgemvn_mgpu_bs;

		int mod_r = rows % dgemvn_mgpu_bs;
		int mod_c = cols_ % dgemvn_mgpu_bs;

		if(mod_r == 0)
		{
			if(mod_c == 0)
			{
				// special case
				int blocks = rows/dgemvn_mgpu_bs;
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, grid_y_n);
				if(blocks == 0) return 0;
				gemvn_mgpu_special<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, gpu_gid, ngpus);
			}
			else
			{
				// generic case for columns only
				const int irregular_cols = mod_c % elements_per_thread;

				int blocks = rows/dgemvn_mgpu_bs;
				blocks += 1;	// dummy thread block
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, grid_y_n);
				if(blocks == 0) return 0;
				switch(irregular_cols)
				{
					/**
					 * The kernel for irregular dimensions has an extra template parameter.
				 	 * This parameter must be among the values listed in the switch-case statement below.
				 	 * The possible values are in the range 0 - (elements_per_thread-1)
				 	 * Make sure these values are updated whenever you change the configuration parameters.
					**/
					case  0: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case  1: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case  2: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case  3: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case  4: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case  5: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case  6: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case  7: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case  8: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case  9: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  9><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case 10: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread, 10><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case 11: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread, 11><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case 12: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread, 12><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case 13: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread, 13><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case 14: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread, 14><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case 15: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread, 15><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					default: printf("DGEMV-N error: improper template parameter. Please read the inline documentation for this function. \n"); return -1;

				}
			}
		}
		else	// mod_r != 0
		{
			if(mod_c == 0)
			{
				// generic case for columns only
				int blocks = (rows/dgemvn_mgpu_bs) + (mod_r != 0);
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, grid_y_n);
				if(blocks == 0) return 0;
				gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread, 0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus);
			}
			else
			{
				// generic case for rows and cols
				const int irregular_cols = mod_c % elements_per_thread;

				int blocks = (rows/dgemvn_mgpu_bs) + (mod_r != 0);

				//printf("gpu_gid = %d, cols_ = %d \n", gpu_gid, cols_);
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, grid_y_n);
				if(blocks == 0) return 0;
				switch(irregular_cols)
				{
					/**
					 * The kernel for irregular dimensions has an extra template parameter.
				 	 * This parameter must be among the values listed in the switch-case statement below.
				 	 * The possible values are in the range 0 - (elements_per_thread-1)
				 	 * Make sure these values are updated whenever you change the configuration parameters.
					**/
					case  0: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case  1: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case  2: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case  3: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case  4: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case  5: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case  6: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case  7: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case  8: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case  9: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  9><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case 10: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread, 10><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case 11: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread, 11><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case 12: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread, 12><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case 13: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread, 13><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case 14: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread, 14><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
					case 15: gemvn_mgpu_generic<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread, 15><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus); break;
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
		//************ config parameters
		const int thread_x = dgemvt_mgpu_bs;
		const int thread_y = dgemvt_mgpu_ty;
		const int elements_per_thread = thread_x/(2*thread_y);
		int grid_y_t = 1 * ngpus;
		//******************************

		// scaling with beta
		//if(gpu_gid == 0)cublasDscal(cols, beta, dY, incy);
		if(gpu_gid == 0)kblas_dscal_async(cols, beta, dY, incy, stream);
		else kblas_dscal_async(cols, d_zero, dY, incy, stream);

		int cols_ = dgemvt_mgpu_bs * ( (cols/dgemvt_mgpu_bs)/ngpus );
		if(gpu_gid < (cols/dgemvt_mgpu_bs)%ngpus) cols_ += dgemvt_mgpu_bs;
		if(gpu_gid == (cols/dgemvt_mgpu_bs)%ngpus) cols_ += cols%dgemvt_mgpu_bs;

		int mod_r = rows % dgemvt_mgpu_bs;
		int mod_c = cols_ % dgemvt_mgpu_bs;

		if(mod_c == 0)
		{
			if(mod_r == 0)
			{
				// special case
				int blocks = cols_/dgemvt_mgpu_bs;
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, grid_y_t);
				if(blocks == 0) return 0;
				gemvt_mgpu_special<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, gpu_gid, ngpus, conj);
			}
			else
			{
				// mod_r != 0
				int blocks = cols_/dgemvt_mgpu_bs;
				blocks += 1;	// dummy thread block
				dim3 dimBlock(thread_x, thread_y);
				dim3 dimGrid(blocks, grid_y_t);
				if(blocks == 0) return 0;
				gemvt_mgpu_generic<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread, 0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus, conj);
			}
		}
		else	// mod_c != 0
		{
			const int irregular_cols = mod_c % elements_per_thread;

			int blocks = cols_/dgemvt_mgpu_bs + (mod_c != 0);
			dim3 dimBlock(thread_x, thread_y);
			dim3 dimGrid(blocks, grid_y_t);
			if(blocks == 0) return 0;

			switch(irregular_cols)
			{
				/**
				  * The kernel for irregular dimensions has an extra template parameter.
				  * This parameter must be among the values listed in the switch-case statement below.
				  * The possible values are in the range 0 - (elements_per_thread-1)
				  * Make sure these values are updated whenever you change the configuration parameters.
				**/
				case  0: gemvt_mgpu_generic<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus, conj); break;
				case  1: gemvt_mgpu_generic<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus, conj); break;
				case  2: gemvt_mgpu_generic<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus, conj); break;
				case  3: gemvt_mgpu_generic<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus, conj); break;
				case  4: gemvt_mgpu_generic<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus, conj); break;
				case  5: gemvt_mgpu_generic<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus, conj); break;
				case  6: gemvt_mgpu_generic<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus, conj); break;
				case  7: gemvt_mgpu_generic<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus, conj); break;
				case  8: gemvt_mgpu_generic<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus, conj); break;
				case  9: gemvt_mgpu_generic<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  9><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus, conj); break;
				case 10: gemvt_mgpu_generic<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread, 10><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus, conj); break;
				case 11: gemvt_mgpu_generic<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread, 11><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus, conj); break;
				case 12: gemvt_mgpu_generic<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread, 12><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus, conj); break;
				case 13: gemvt_mgpu_generic<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread, 13><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus, conj); break;
				case 14: gemvt_mgpu_generic<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread, 14><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus, conj); break;
				case 15: gemvt_mgpu_generic<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread, 15><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, gpu_gid, ngpus, conj); break;
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
/*************************************************************************************/
extern "C"
int kblas_dgemv_mgpu_driver_offset( char trans, int rows, int cols,
						double alpha, double *dA, int lda,
						double *dX, int incx,
						double  beta, double *dY, int incy, int gpu_gid, int ngpus,
						int offset_r, int offset_c,
						cudaStream_t stream = 0)
{
    const double d_zero = 0.0;
	if(trans == 'n' || trans == 'N')
	{

		//**** Config parameters
		const int thread_x = dgemvn_mgpu_bs;
		const int thread_y = dgemvn_mgpu_ty;
		const int elements_per_thread = thread_x/(2*thread_y);
		const int grid_y_n = 2 * ngpus;
		//*************************

		/** offset necessary calculation **/
		int offset_r_ = offset_r % dgemvn_mgpu_bs;
		int offset_c_ = offset_c % dgemvn_mgpu_bs;
		int total_blocks_skipped_r = offset_r / dgemvn_mgpu_bs;
		int total_blocks_skipped_c = offset_c / dgemvn_mgpu_bs;
		int my_skipped_blocks_r = total_blocks_skipped_r;
		int my_skipped_blocks_c = total_blocks_skipped_c/ngpus;
		if(gpu_gid < (total_blocks_skipped_c%ngpus)) my_skipped_blocks_c += 1;
		int ref_gpu = total_blocks_skipped_c%ngpus;
		int new_gpu_gid = (gpu_gid - ref_gpu + ngpus) % ngpus;
		// Advance pointers accordingly
		dA += my_skipped_blocks_c * dgemvn_mgpu_bs * lda;
		dA += total_blocks_skipped_r * dgemvn_mgpu_bs;
		dX += total_blocks_skipped_c * dgemvn_mgpu_bs * incx;
		dY += my_skipped_blocks_r * dgemvn_mgpu_bs * incy;
		rows -= total_blocks_skipped_r * dgemvn_mgpu_bs;
		cols -= total_blocks_skipped_c * dgemvn_mgpu_bs;
		/** end offset necessary calculation **/

		int nstripes = (cols/dgemvn_mgpu_bs) + ((cols%dgemvn_mgpu_bs) != 0);

		// scaling with beta
		if(gpu_gid == 0)kblas_dscal_async(rows-offset_r_, beta, dY+(offset_r_*incy), incy, stream);
		else kblas_dscal_async(rows-offset_r_, d_zero, dY+(offset_r_*incy), incy, stream);

		int cols_ = dgemvn_mgpu_bs * ( (cols/dgemvn_mgpu_bs)/ngpus );
		if(new_gpu_gid < (cols/dgemvn_mgpu_bs)%ngpus) cols_ += dgemvn_mgpu_bs;
		if(new_gpu_gid == (cols/dgemvn_mgpu_bs)%ngpus) cols_ += cols%dgemvn_mgpu_bs;

		int mod_r = rows % dgemvn_mgpu_bs;
		int mod_c = cols_ % dgemvn_mgpu_bs;

		if(mod_r == 0 && mod_c == 0)
		{
			// special case
			int blocks = rows/dgemvn_mgpu_bs;
			dim3 dimBlock(thread_x, thread_y);
			dim3 dimGrid(blocks, grid_y_n);
			if(blocks == 0) return 0;
			gemvn_mgpu_special_offset<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread>
									<<<dimGrid, dimBlock, 0, stream>>>
									(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_);
		}
		else
		{
			// generic case for columns only
			const int irregular_cols = mod_c % elements_per_thread;

			int blocks = (rows/dgemvn_mgpu_bs) + (mod_r != 0);
			if(mod_r == 0)blocks += 1;	// dummy thread block, will return immediately if mod_r == 0

			dim3 dimBlock(thread_x, thread_y);
			dim3 dimGrid(blocks, grid_y_n);
			if(blocks == 0) return 0;

			switch(irregular_cols)
			{
				/**
				 * The kernel for irregular dimensions has an extra template parameter.
				 * This parameter must be among the values listed in the switch-case statement below.
				 * The possible values are in the range 0 - (elements_per_thread-1)
				 * Make sure these values are updated whenever you change the configuration parameters.
				**/
				case  0: gemvn_mgpu_generic_offset<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_); break;
				case  1: gemvn_mgpu_generic_offset<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_); break;
				case  2: gemvn_mgpu_generic_offset<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_); break;
				case  3: gemvn_mgpu_generic_offset<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_); break;
				case  4: gemvn_mgpu_generic_offset<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_); break;
				case  5: gemvn_mgpu_generic_offset<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_); break;
				case  6: gemvn_mgpu_generic_offset<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_); break;
				case  7: gemvn_mgpu_generic_offset<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_); break;
				case  8: gemvn_mgpu_generic_offset<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_); break;
				case  9: gemvn_mgpu_generic_offset<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread,  9><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_); break;
				case 10: gemvn_mgpu_generic_offset<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread, 10><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_); break;
				case 11: gemvn_mgpu_generic_offset<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread, 11><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_); break;
				case 12: gemvn_mgpu_generic_offset<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread, 12><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_); break;
				case 13: gemvn_mgpu_generic_offset<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread, 13><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_); break;
				case 14: gemvn_mgpu_generic_offset<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread, 14><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_); break;
				case 15: gemvn_mgpu_generic_offset<double, dgemvn_mgpu_bs, dgemvn_mgpu_bs, dgemvn_mgpu_ty, elements_per_thread, 15><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_); break;
				default: printf("DGEMV-N error: improper template parameter. Please read the inline documentation for this function. \n"); return -1;
			}
		}
	}	// end of non-transpose case
	else if (trans == 't' || trans == 'T' || trans == 'c' || trans == 'C')
	{
		int conj;
		if(trans == 'c' || trans == 'C') conj = 1;
		else conj = 0;
		//**** Config parameters
		const int thread_x = dgemvt_mgpu_bs;
		const int thread_y = dgemvt_mgpu_ty;
		const int elements_per_thread = thread_x/(2*thread_y);
		const int grid_y_t = 2 * ngpus;
		//*************************

		/** offset necessary calculation **/
		int offset_r_ = offset_r % dgemvt_mgpu_bs;
		int offset_c_ = offset_c % dgemvt_mgpu_bs;
		int total_blocks_skipped_r = offset_r / dgemvt_mgpu_bs;
		int total_blocks_skipped_c = offset_c / dgemvt_mgpu_bs;
		int my_skipped_blocks_r = total_blocks_skipped_r;
		int my_skipped_blocks_c = total_blocks_skipped_c/ngpus;
		if(gpu_gid < (total_blocks_skipped_c%ngpus)) my_skipped_blocks_c += 1;
		int ref_gpu = total_blocks_skipped_c%ngpus;
		int new_gpu_gid = (gpu_gid - ref_gpu + ngpus) % ngpus;

		//if(new_gpu_gid != 3){return 0;}
		// Advance pointers accordingly
		dA += my_skipped_blocks_c * dgemvt_mgpu_bs * lda;
		dA += my_skipped_blocks_r * dgemvt_mgpu_bs;
		dX += total_blocks_skipped_r * dgemvt_mgpu_bs * incx;
		dY += total_blocks_skipped_c * dgemvt_mgpu_bs * incy;
		rows -= total_blocks_skipped_r * dgemvt_mgpu_bs;
		cols -= total_blocks_skipped_c * dgemvt_mgpu_bs;
		/** end offset necessary calculation **/

		int nstripes = (cols/dgemvt_mgpu_bs) + ((cols%dgemvt_mgpu_bs) != 0);

		// scaling with beta
		//if(gpu_gid == 0)cublasDscal(cols-offset_, beta, dY+(offset_*incy), incy);
		if(gpu_gid == 0)kblas_dscal_async(cols-offset_c_, beta, dY+(offset_c_*incy), incy, stream);
		else kblas_dscal_async(cols-offset_r_, d_zero, dY+(offset_r_*incy), incy, stream);

		int cols_ = dgemvt_mgpu_bs * ( (cols/dgemvt_mgpu_bs)/ngpus );
		if(new_gpu_gid < (cols/dgemvt_mgpu_bs)%ngpus) cols_ += dgemvt_mgpu_bs;
		if(new_gpu_gid == (cols/dgemvt_mgpu_bs)%ngpus) cols_ += cols%dgemvt_mgpu_bs;

		int mod_r = rows % dgemvt_mgpu_bs;
		int mod_c = cols_ % dgemvt_mgpu_bs;

		if(mod_r == 0 && mod_c == 0)
		{
			int blocks = cols_/dgemvt_mgpu_bs;
			dim3 dimBlock(thread_x, thread_y);
			dim3 dimGrid(blocks, grid_y_t);
			if(blocks == 0) return 0;
			gemvt_mgpu_special_offset<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_, conj);

		}
		else
		{
			const int irregular_cols = mod_c % elements_per_thread;
			int blocks = cols_/dgemvt_mgpu_bs + (mod_c != 0);
			int gpu_last = (nstripes+ngpus-1)%ngpus;
			if(mod_c == 0 && new_gpu_gid == gpu_last) blocks += 1; // dummy thread block, will return if mod_c == 0
			dim3 dimBlock(thread_x, thread_y);
			dim3 dimGrid(blocks, grid_y_t);
			if(blocks == 0) return 0;

			switch(irregular_cols)
			{
				/**
				  * The kernel for irregular dimensions has an extra template parameter.
				  * This parameter must be among the values listed in the switch-case statement below.
				  * The possible values are in the range 0 - (elements_per_thread-1)
				  * Make sure these values are updated whenever you change the configuration parameters.
				**/
				case  0: gemvt_mgpu_generic_offset<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_, conj); break;
				case  1: gemvt_mgpu_generic_offset<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_, conj); break;
				case  2: gemvt_mgpu_generic_offset<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_, conj); break;
				case  3: gemvt_mgpu_generic_offset<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_, conj); break;
				case  4: gemvt_mgpu_generic_offset<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_, conj); break;
				case  5: gemvt_mgpu_generic_offset<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_, conj); break;
				case  6: gemvt_mgpu_generic_offset<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_, conj); break;
				case  7: gemvt_mgpu_generic_offset<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_, conj); break;
				case  8: gemvt_mgpu_generic_offset<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_, conj); break;
				case  9: gemvt_mgpu_generic_offset<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread,  9><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_, conj); break;
				case 10: gemvt_mgpu_generic_offset<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread, 10><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_, conj); break;
				case 11: gemvt_mgpu_generic_offset<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread, 11><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_, conj); break;
				case 12: gemvt_mgpu_generic_offset<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread, 12><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_, conj); break;
				case 13: gemvt_mgpu_generic_offset<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread, 13><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_, conj); break;
				case 14: gemvt_mgpu_generic_offset<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread, 14><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_, conj); break;
				case 15: gemvt_mgpu_generic_offset<double, dgemvt_mgpu_bs, thread_x, thread_y, elements_per_thread, 15><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, new_gpu_gid, ngpus, nstripes, offset_r_, offset_c_, conj); break;
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
/***********************************************************************************/
extern "C"
int kblas_dgemv_mgpu( char trans, int rows, int cols,
						double alpha, double **dA, int lda,
						double **dX, int incx,
						double  beta, double **dY, int incy,
						int ngpus,
						int offset_r, int offset_c)
{
    const int ngpus_local = ngpus;
	if(offset_r == 0 && offset_c == 0)
	{
		for(int i = 0; i < ngpus_local; i++)
		{
			cudaSetDevice(gpu_lid[i]);
			kblas_dgemv_mgpu_driver(trans, rows, cols, alpha, dA[i], lda, dX[i], incx, beta, dY[i], incy, gpu_gid[i], ngpus);
		}
	}
	else
	{
		for(int i = 0; i < ngpus_local; i++)
		{
			cudaSetDevice(gpu_lid[i]);
			kblas_dgemv_mgpu_driver_offset(trans, rows, cols, alpha, dA[i], lda, dX[i], incx, beta, dY[i], incy, gpu_gid[i], ngpus, offset_r, offset_c);
		}
	}

	// wait for gpus to finish
	for(int i = 0; i < ngpus_local; i++)
	{
		cudaSetDevice(gpu_lid[i]);
		cudaDeviceSynchronize();
	}
	return 0;
}

/*************************************************************************************/
extern "C"
int kblas_dgemv_mgpu_async( char trans, int rows, int cols,
							double alpha, double **dA, int lda,
							double **dX, int incx,
							double  beta, double **dY, int incy,
							int ngpus,
							int offset_r, int offset_c,
							cudaStream_t stream[MAX_NGPUS][MAX_STREAMS])
{
    const int ngpus_local = ngpus;
	if(offset_r == 0 && offset_c == 0)
	{
		for(int i = 0; i < ngpus_local; i++)
		{
			cudaSetDevice(gpu_lid[i]);
			kblas_dgemv_mgpu_driver(trans, rows, cols, alpha, dA[i], lda, dX[i], incx, beta, dY[i], incy, gpu_gid[i], ngpus, stream[i][0]);
		}
	}
	else
	{
		for(int i = 0; i < ngpus_local; i++)
		{
			cudaSetDevice(gpu_lid[i]);
			kblas_dgemv_mgpu_driver_offset(trans, rows, cols, alpha, dA[i], lda, dX[i], incx, beta, dY[i], incy, gpu_gid[i], ngpus, offset_r, offset_c);
		}
	}

	return 0;
}
/*************************************************************************************/

extern "C"
int get_dgemv_mgpu_bs(char trans)
{
    if(trans == 'n' || trans == 'N')
        return dgemvn_mgpu_bs;
    else if (trans == 't' || trans == 'T' || trans == 'c' || trans == 'C')
        return dgemvt_mgpu_bs;
    else
        {printf("Error ..  input %c is not supported for gemv \n", trans); return -1;}
}
