/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/zgemv_offset.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ahmad Abdelfattah
 * @date 2018-11-14
 **/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "gemv_offset_core.cuh"

#if(TARGET_SM >= 30)

#define zgemvn_offset_bs		(16)
#define zgemvn_offset_ty		(8)
#define zgemvn_offset_by		(2)

#define zgemvt_offset_bs		(32)
#define zgemvt_offset_ty		(8)
#define zgemvt_offset_by		(2)

#else

#define zgemvn_offset_bs		(32)
#define zgemvn_offset_ty		(8)
#define zgemvn_offset_by		(2)

#define zgemvt_offset_bs		(32)
#define zgemvt_offset_ty		(8)
#define zgemvt_offset_by		(2)
#endif


extern "C"
int kblas_zscal_async(int n, cuDoubleComplex alpha, cuDoubleComplex *x, int incx, cudaStream_t stream);

int kblas_zgemv_offset_driver( char trans, int rows, int cols,
						        cuDoubleComplex alpha, cuDoubleComplex *dA, int lda,
						        cuDoubleComplex *dX, int incx,
						        cuDoubleComplex  beta, cuDoubleComplex *dY, int incy,
						        int offset_r, int offset_c,
						        cudaStream_t stream = 0)
{
	if(trans == 'n' || trans == 'N')
	{

		//**** Config parameters
		const int thread_x = zgemvn_offset_bs;
		const int thread_y = zgemvn_offset_ty;
		const int elements_per_thread = thread_x/(2*thread_y);
		const int grid_y_n = zgemvn_offset_by;
		//*************************

		/** offset necessary calculation **/
		int offset_r_ = offset_r % zgemvn_offset_bs;
		int offset_c_ = offset_c % zgemvn_offset_bs;
		int total_blocks_skipped_r = offset_r / zgemvn_offset_bs;
		int total_blocks_skipped_c = offset_c / zgemvn_offset_bs;
		int my_skipped_blocks_r = total_blocks_skipped_r;
		int my_skipped_blocks_c = total_blocks_skipped_c/ngpus;
		if(gpu_gid < (total_blocks_skipped_c%ngpus)) my_skipped_blocks_c += 1;
		int ref_gpu = total_blocks_skipped_c%ngpus;
		int new_gpu_gid = (gpu_gid - ref_gpu + ngpus) % ngpus;
		// Advance pointers accordingly
		dA += my_skipped_blocks_c * zgemvn_offset_bs * lda;
		dA += my_skipped_blocks_r * zgemvn_offset_bs;
		dX += my_skipped_blocks_c * zgemvn_offset_bs * incx;
		dY += my_skipped_blocks_r * zgemvn_offset_bs * incy;
		rows -= my_skipped_blocks_r * zgemvn_offset_bs;
		cols -= my_skipped_blocks_c * zgemvn_offset_bs;
		/** end offset necessary calculation **/

		int nstripes = (cols/zgemvn_offset_bs) + ((cols%zgemvn_offset_bs) != 0);

		// scaling with beta
		//if(gpu_gid == 0)cublasSscal(rows-offset_, beta, dY+(offset_*incy), incy);
		if(gpu_gid == 0)kblas_zscal_async(rows-offset_r_, beta, dY+(offset_r_*incy), incy, stream);

		int cols_ = zgemvn_offset_bs * ( (cols/zgemvn_offset_bs)/ngpus );
		if(new_gpu_gid < (cols/zgemvn_offset_bs)%ngpus) cols_ += zgemvn_offset_bs;
		if(new_gpu_gid == (cols/zgemvn_offset_bs)%ngpus) cols_ += cols%zgemvn_offset_bs;

		int mod_r = rows % zgemvn_offset_bs;
		int mod_c = cols_ % zgemvn_offset_bs;

		if(mod_r == 0 && mod_c == 0)
		{
			// special case
			int blocks = rows/zgemvn_offset_bs;
			dim3 dimBlock(thread_x, thread_y);
			dim3 dimGrid(blocks, grid_y_n);
			if(blocks == 0) return 0;
			gemvn_special_offset<cuDoubleComplex, zgemvn_offset_bs, zgemvn_offset_bs, zgemvn_offset_ty, elements_per_thread>
									<<<dimGrid, dimBlock, 0, stream>>>
									(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, nstripes, offset_r_, offset_c_);
		}
		else
		{
			// generic case for columns only
			const int irregular_cols = mod_c % elements_per_thread;

			int blocks = (rows/zgemvn_offset_bs) + (mod_r != 0);
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
				case  0: gemvn_generic_offset<cuDoubleComplex, zgemvn_offset_bs, zgemvn_offset_bs, zgemvn_offset_ty, elements_per_thread,  0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case  1: gemvn_generic_offset<cuDoubleComplex, zgemvn_offset_bs, zgemvn_offset_bs, zgemvn_offset_ty, elements_per_thread,  1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case  2: gemvn_generic_offset<cuDoubleComplex, zgemvn_offset_bs, zgemvn_offset_bs, zgemvn_offset_ty, elements_per_thread,  2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case  3: gemvn_generic_offset<cuDoubleComplex, zgemvn_offset_bs, zgemvn_offset_bs, zgemvn_offset_ty, elements_per_thread,  3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case  4: gemvn_generic_offset<cuDoubleComplex, zgemvn_offset_bs, zgemvn_offset_bs, zgemvn_offset_ty, elements_per_thread,  4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case  5: gemvn_generic_offset<cuDoubleComplex, zgemvn_offset_bs, zgemvn_offset_bs, zgemvn_offset_ty, elements_per_thread,  5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case  6: gemvn_generic_offset<cuDoubleComplex, zgemvn_offset_bs, zgemvn_offset_bs, zgemvn_offset_ty, elements_per_thread,  6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case  7: gemvn_generic_offset<cuDoubleComplex, zgemvn_offset_bs, zgemvn_offset_bs, zgemvn_offset_ty, elements_per_thread,  7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case  8: gemvn_generic_offset<cuDoubleComplex, zgemvn_offset_bs, zgemvn_offset_bs, zgemvn_offset_ty, elements_per_thread,  8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case  9: gemvn_generic_offset<cuDoubleComplex, zgemvn_offset_bs, zgemvn_offset_bs, zgemvn_offset_ty, elements_per_thread,  9><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case 10: gemvn_generic_offset<cuDoubleComplex, zgemvn_offset_bs, zgemvn_offset_bs, zgemvn_offset_ty, elements_per_thread, 10><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case 11: gemvn_generic_offset<cuDoubleComplex, zgemvn_offset_bs, zgemvn_offset_bs, zgemvn_offset_ty, elements_per_thread, 11><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case 12: gemvn_generic_offset<cuDoubleComplex, zgemvn_offset_bs, zgemvn_offset_bs, zgemvn_offset_ty, elements_per_thread, 12><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case 13: gemvn_generic_offset<cuDoubleComplex, zgemvn_offset_bs, zgemvn_offset_bs, zgemvn_offset_ty, elements_per_thread, 13><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case 14: gemvn_generic_offset<cuDoubleComplex, zgemvn_offset_bs, zgemvn_offset_bs, zgemvn_offset_ty, elements_per_thread, 14><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case 15: gemvn_generic_offset<cuDoubleComplex, zgemvn_offset_bs, zgemvn_offset_bs, zgemvn_offset_ty, elements_per_thread, 15><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				default: printf("ZGEMV-N error: improper template parameter. Please read the inline documentation for this function. \n"); return -1;
			}
		}
	}	// end of non-transpose case
	else if (trans == 't' || trans == 'T' || trans == 'c' || trans == 'C')
	{
		int conj;
		if(trans == 'c' || trans == 'C') conj = 1;
		else conj = 0;
		//**** Config parameters
		const int thread_x = zgemvt_offset_bs;
		const int thread_y = zgemvt_offset_ty;
		const int elements_per_thread = thread_x/(2*thread_y);
		const int grid_y_t = zgemvt_offset_by;
		//*************************

		/** offset necessary calculation **/
		int offset_r_ = offset_r % zgemvt_offset_bs;
		int offset_c_ = offset_c % zgemvt_offset_bs;
		int total_blocks_skipped_r = offset_r / zgemvt_offset_bs;
		int total_blocks_skipped_c = offset_c / zgemvt_offset_bs;
		int my_skipped_blocks_r = total_blocks_skipped_r;
		int my_skipped_blocks_c = total_blocks_skipped_c/ngpus;
		if(gpu_gid < (total_blocks_skipped_c%ngpus)) my_skipped_blocks_c += 1;
		int ref_gpu = total_blocks_skipped_c%ngpus;
		int new_gpu_gid = (gpu_gid - ref_gpu + ngpus) % ngpus;

		//if(new_gpu_gid != 3){return 0;}
		// Advance pointers accordingly
		dA += my_skipped_blocks_c * zgemvt_offset_bs * lda;
		dA += my_skipped_blocks_r * zgemvt_offset_bs;
		dX += my_skipped_blocks_r * zgemvt_offset_bs * incx;
		dY += my_skipped_blocks_c * zgemvt_offset_bs * incy;
		rows -= my_skipped_blocks_r * zgemvt_offset_bs;
		cols -= my_skipped_blocks_c * zgemvt_offset_bs;
		/** end offset necessary calculation **/

		int nstripes = (cols/zgemvt_offset_bs) + ((cols%zgemvt_offset_bs) != 0);

		// scaling with beta
		//if(gpu_gid == 0)cublasSscal(cols-offset_, beta, dY+(offset_*incy), incy);
		if(gpu_gid == 0)kblas_zscal_async(cols-offset_c_, beta, dY+(offset_c_*incy), incy, stream);

		int cols_ = zgemvt_offset_bs * ( (cols/zgemvt_offset_bs)/ngpus );
		if(new_gpu_gid < (cols/zgemvt_offset_bs)%ngpus) cols_ += zgemvt_offset_bs;
		if(new_gpu_gid == (cols/zgemvt_offset_bs)%ngpus) cols_ += cols%zgemvt_offset_bs;

		int mod_r = rows % zgemvt_offset_bs;
		int mod_c = cols_ % zgemvt_offset_bs;

		if(mod_r == 0 && mod_c == 0)
		{
			int blocks = cols_/zgemvt_offset_bs;
			dim3 dimBlock(thread_x, thread_y);
			dim3 dimGrid(blocks, grid_y_t);
			if(blocks == 0) return 0;
			gemvt_special_offset<cuDoubleComplex, zgemvt_offset_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, nstripes, offset_r_, offset_c_, conj);

		}
		else
		{
			const int irregular_cols = mod_c % elements_per_thread;
			int blocks = cols_/zgemvt_offset_bs + (mod_c != 0);
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
				case  0: gemvt_generic_offset<cuDoubleComplex, zgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case  1: gemvt_generic_offset<cuDoubleComplex, zgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case  2: gemvt_generic_offset<cuDoubleComplex, zgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case  3: gemvt_generic_offset<cuDoubleComplex, zgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case  4: gemvt_generic_offset<cuDoubleComplex, zgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case  5: gemvt_generic_offset<cuDoubleComplex, zgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case  6: gemvt_generic_offset<cuDoubleComplex, zgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case  7: gemvt_generic_offset<cuDoubleComplex, zgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case  8: gemvt_generic_offset<cuDoubleComplex, zgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case  9: gemvt_generic_offset<cuDoubleComplex, zgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  9><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case 10: gemvt_generic_offset<cuDoubleComplex, zgemvt_offset_bs, thread_x, thread_y, elements_per_thread, 10><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case 11: gemvt_generic_offset<cuDoubleComplex, zgemvt_offset_bs, thread_x, thread_y, elements_per_thread, 11><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case 12: gemvt_generic_offset<cuDoubleComplex, zgemvt_offset_bs, thread_x, thread_y, elements_per_thread, 12><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case 13: gemvt_generic_offset<cuDoubleComplex, zgemvt_offset_bs, thread_x, thread_y, elements_per_thread, 13><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case 14: gemvt_generic_offset<cuDoubleComplex, zgemvt_offset_bs, thread_x, thread_y, elements_per_thread, 14><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case 15: gemvt_generic_offset<cuDoubleComplex, zgemvt_offset_bs, thread_x, thread_y, elements_per_thread, 15><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				default: printf("ZGEMV-T error: improper template parameter. Please read the inline documentation for this function. \n"); return -1;
			}
		}
	}
	else
	{
		printf("ZGEMV error: Unrecognized transpose mode %c \n", trans);
		return -1;
	}

	return 0;
}
/***********************************************************************************/
extern "C"
int kblas_zgemv_offset( char trans, int rows, int cols,
						cuDoubleComplex alpha, cuDoubleComplex *dA, int lda,
						cuDoubleComplex *dX, int incx,
						cuDoubleComplex beta, cuDoubleComplex *dY, int incy,
						int offset_r, int offset_c)
{
	return kblas_zgemv_offset_driver( trans, rows, cols,
	                                alpha, dA, lda,
	                                dX, incx,
	                                beta, dY, incy,
	                                offset_r, offset_c);
}

/*************************************************************************************/
extern "C"
int kblas_zgemv_offset_async( char trans, int rows, int cols,
							cuDoubleComplex alpha, cuDoubleComplex *dA, int lda,
							cuDoubleComplex *dX, int incx,
							cuDoubleComplex  beta, cuDoubleComplex *dY, int incy,
							int offset_r, int offset_c,
							cudaStream_t stream)
{
	return kblas_zgemv_offset_driver(  trans, rows, cols,
	                            alpha, dA, lda,
	                            dX, incx,
	                            beta, dY, incy, offset_r, offset_c,
	                            stream);
}
/*************************************************************************************/
