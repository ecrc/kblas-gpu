/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/cgemv_offset.cu

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

#define cgemvn_offset_bs		(32)
#define cgemvn_offset_ty		(8)
#define cgemvn_offset_by		(4)

#define cgemvt_offset_bs		(32)
#define cgemvt_offset_ty		(4)
#define cgemvt_offset_by		(4)

#else

#define cgemvn_offset_bs		(32)
#define cgemvn_offset_ty		(4)
#define cgemvn_offset_by		(2)

#define cgemvt_offset_bs		(32)
#define cgemvt_offset_ty		(4)
#define cgemvt_offset_by		(2)
#endif


extern "C"
int kblas_cscal_async(int n, cuFloatComplex alpha, cuFloatComplex *x, int incx, cudaStream_t stream);

int kblas_cgemv_offset_driver( char trans, int rows, int cols,
						        cuFloatComplex alpha, cuFloatComplex *dA, int lda,
						        cuFloatComplex *dX, int incx,
						        cuFloatComplex  beta, cuFloatComplex *dY, int incy,
						        int offset_r, int offset_c,
						        cudaStream_t stream = 0)
{
	if(trans == 'n' || trans == 'N')
	{

		//**** Config parameters
		const int thread_x = cgemvn_offset_bs;
		const int thread_y = cgemvn_offset_ty;
		const int elements_per_thread = thread_x/(2*thread_y);
		const int grid_y_n = cgemvn_offset_by;
		//*************************

		/** offset necessary calculation **/
		int offset_r_ = offset_r % cgemvn_offset_bs;
		int offset_c_ = offset_c % cgemvn_offset_bs;
		int total_blocks_skipped_r = offset_r / cgemvn_offset_bs;
		int total_blocks_skipped_c = offset_c / cgemvn_offset_bs;
		int my_skipped_blocks_r = total_blocks_skipped_r;
		int my_skipped_blocks_c = total_blocks_skipped_c/ngpus;
		if(gpu_gid < (total_blocks_skipped_c%ngpus)) my_skipped_blocks_c += 1;
		int ref_gpu = total_blocks_skipped_c%ngpus;
		int new_gpu_gid = (gpu_gid - ref_gpu + ngpus) % ngpus;
		// Advance pointers accordingly
		dA += my_skipped_blocks_c * cgemvn_offset_bs * lda;
		dA += my_skipped_blocks_r * cgemvn_offset_bs;
		dX += my_skipped_blocks_c * cgemvn_offset_bs * incx;
		dY += my_skipped_blocks_r * cgemvn_offset_bs * incy;
		rows -= my_skipped_blocks_r * cgemvn_offset_bs;
		cols -= my_skipped_blocks_c * cgemvn_offset_bs;
		/** end offset necessary calculation **/

		int nstripes = (cols/cgemvn_offset_bs) + ((cols%cgemvn_offset_bs) != 0);

		// scaling with beta
		//if(gpu_gid == 0)cublasSscal(rows-offset_, beta, dY+(offset_*incy), incy);
		if(gpu_gid == 0)kblas_cscal_async(rows-offset_r_, beta, dY+(offset_r_*incy), incy, stream);

		int cols_ = cgemvn_offset_bs * ( (cols/cgemvn_offset_bs)/ngpus );
		if(new_gpu_gid < (cols/cgemvn_offset_bs)%ngpus) cols_ += cgemvn_offset_bs;
		if(new_gpu_gid == (cols/cgemvn_offset_bs)%ngpus) cols_ += cols%cgemvn_offset_bs;

		int mod_r = rows % cgemvn_offset_bs;
		int mod_c = cols_ % cgemvn_offset_bs;

		if(mod_r == 0 && mod_c == 0)
		{
			// special case
			int blocks = rows/cgemvn_offset_bs;
			dim3 dimBlock(thread_x, thread_y);
			dim3 dimGrid(blocks, grid_y_n);
			if(blocks == 0) return 0;
			gemvn_special_offset<cuFloatComplex, cgemvn_offset_bs, cgemvn_offset_bs, cgemvn_offset_ty, elements_per_thread>
									<<<dimGrid, dimBlock, 0, stream>>>
									(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, nstripes, offset_r_, offset_c_);
		}
		else
		{
			// generic case for columns only
			const int irregular_cols = mod_c % elements_per_thread;

			int blocks = (rows/cgemvn_offset_bs) + (mod_r != 0);
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
				case  0: gemvn_generic_offset<cuFloatComplex, cgemvn_offset_bs, cgemvn_offset_bs, cgemvn_offset_ty, elements_per_thread,  0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case  1: gemvn_generic_offset<cuFloatComplex, cgemvn_offset_bs, cgemvn_offset_bs, cgemvn_offset_ty, elements_per_thread,  1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case  2: gemvn_generic_offset<cuFloatComplex, cgemvn_offset_bs, cgemvn_offset_bs, cgemvn_offset_ty, elements_per_thread,  2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case  3: gemvn_generic_offset<cuFloatComplex, cgemvn_offset_bs, cgemvn_offset_bs, cgemvn_offset_ty, elements_per_thread,  3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case  4: gemvn_generic_offset<cuFloatComplex, cgemvn_offset_bs, cgemvn_offset_bs, cgemvn_offset_ty, elements_per_thread,  4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case  5: gemvn_generic_offset<cuFloatComplex, cgemvn_offset_bs, cgemvn_offset_bs, cgemvn_offset_ty, elements_per_thread,  5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case  6: gemvn_generic_offset<cuFloatComplex, cgemvn_offset_bs, cgemvn_offset_bs, cgemvn_offset_ty, elements_per_thread,  6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case  7: gemvn_generic_offset<cuFloatComplex, cgemvn_offset_bs, cgemvn_offset_bs, cgemvn_offset_ty, elements_per_thread,  7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case  8: gemvn_generic_offset<cuFloatComplex, cgemvn_offset_bs, cgemvn_offset_bs, cgemvn_offset_ty, elements_per_thread,  8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case  9: gemvn_generic_offset<cuFloatComplex, cgemvn_offset_bs, cgemvn_offset_bs, cgemvn_offset_ty, elements_per_thread,  9><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case 10: gemvn_generic_offset<cuFloatComplex, cgemvn_offset_bs, cgemvn_offset_bs, cgemvn_offset_ty, elements_per_thread, 10><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case 11: gemvn_generic_offset<cuFloatComplex, cgemvn_offset_bs, cgemvn_offset_bs, cgemvn_offset_ty, elements_per_thread, 11><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case 12: gemvn_generic_offset<cuFloatComplex, cgemvn_offset_bs, cgemvn_offset_bs, cgemvn_offset_ty, elements_per_thread, 12><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case 13: gemvn_generic_offset<cuFloatComplex, cgemvn_offset_bs, cgemvn_offset_bs, cgemvn_offset_ty, elements_per_thread, 13><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case 14: gemvn_generic_offset<cuFloatComplex, cgemvn_offset_bs, cgemvn_offset_bs, cgemvn_offset_ty, elements_per_thread, 14><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				case 15: gemvn_generic_offset<cuFloatComplex, cgemvn_offset_bs, cgemvn_offset_bs, cgemvn_offset_ty, elements_per_thread, 15><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_); break;
				default: printf("CGEMV-N error: improper template parameter. Please read the inline documentation for this function. \n"); return -1;
			}
		}
	}	// end of non-transpose case
	else if (trans == 't' || trans == 'T' || trans == 'c' || trans == 'C')
	{
		int conj;
		if(trans == 'c' || trans == 'C') conj = 1;
		else conj = 0;
		//**** Config parameters
		const int thread_x = cgemvt_offset_bs;
		const int thread_y = cgemvt_offset_ty;
		const int elements_per_thread = thread_x/(2*thread_y);
		const int grid_y_t = cgemvt_offset_by;
		//*************************

		/** offset necessary calculation **/
		int offset_r_ = offset_r % cgemvt_offset_bs;
		int offset_c_ = offset_c % cgemvt_offset_bs;
		int total_blocks_skipped_r = offset_r / cgemvt_offset_bs;
		int total_blocks_skipped_c = offset_c / cgemvt_offset_bs;
		int my_skipped_blocks_r = total_blocks_skipped_r;
		int my_skipped_blocks_c = total_blocks_skipped_c/ngpus;
		if(gpu_gid < (total_blocks_skipped_c%ngpus)) my_skipped_blocks_c += 1;
		int ref_gpu = total_blocks_skipped_c%ngpus;
		int new_gpu_gid = (gpu_gid - ref_gpu + ngpus) % ngpus;

		//if(new_gpu_gid != 3){return 0;}
		// Advance pointers accordingly
		dA += my_skipped_blocks_c * cgemvt_offset_bs * lda;
		dA += my_skipped_blocks_r * cgemvt_offset_bs;
		dX += my_skipped_blocks_r * cgemvt_offset_bs * incx;
		dY += my_skipped_blocks_c * cgemvt_offset_bs * incy;
		rows -= my_skipped_blocks_r * cgemvt_offset_bs;
		cols -= my_skipped_blocks_c * cgemvt_offset_bs;
		/** end offset necessary calculation **/

		int nstripes = (cols/cgemvt_offset_bs) + ((cols%cgemvt_offset_bs) != 0);

		// scaling with beta
		//if(gpu_gid == 0)cublasSscal(cols-offset_, beta, dY+(offset_*incy), incy);
		if(gpu_gid == 0)kblas_cscal_async(cols-offset_c_, beta, dY+(offset_c_*incy), incy, stream);

		int cols_ = cgemvt_offset_bs * ( (cols/cgemvt_offset_bs)/ngpus );
		if(new_gpu_gid < (cols/cgemvt_offset_bs)%ngpus) cols_ += cgemvt_offset_bs;
		if(new_gpu_gid == (cols/cgemvt_offset_bs)%ngpus) cols_ += cols%cgemvt_offset_bs;

		int mod_r = rows % cgemvt_offset_bs;
		int mod_c = cols_ % cgemvt_offset_bs;

		if(mod_r == 0 && mod_c == 0)
		{
			int blocks = cols_/cgemvt_offset_bs;
			dim3 dimBlock(thread_x, thread_y);
			dim3 dimGrid(blocks, grid_y_t);
			if(blocks == 0) return 0;
			gemvt_special_offset<cuFloatComplex, cgemvt_offset_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, nstripes, offset_r_, offset_c_, conj);

		}
		else
		{
			const int irregular_cols = mod_c % elements_per_thread;
			int blocks = cols_/cgemvt_offset_bs + (mod_c != 0);
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
				case  0: gemvt_generic_offset<cuFloatComplex, cgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case  1: gemvt_generic_offset<cuFloatComplex, cgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case  2: gemvt_generic_offset<cuFloatComplex, cgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case  3: gemvt_generic_offset<cuFloatComplex, cgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case  4: gemvt_generic_offset<cuFloatComplex, cgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case  5: gemvt_generic_offset<cuFloatComplex, cgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case  6: gemvt_generic_offset<cuFloatComplex, cgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case  7: gemvt_generic_offset<cuFloatComplex, cgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case  8: gemvt_generic_offset<cuFloatComplex, cgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case  9: gemvt_generic_offset<cuFloatComplex, cgemvt_offset_bs, thread_x, thread_y, elements_per_thread,  9><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case 10: gemvt_generic_offset<cuFloatComplex, cgemvt_offset_bs, thread_x, thread_y, elements_per_thread, 10><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case 11: gemvt_generic_offset<cuFloatComplex, cgemvt_offset_bs, thread_x, thread_y, elements_per_thread, 11><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case 12: gemvt_generic_offset<cuFloatComplex, cgemvt_offset_bs, thread_x, thread_y, elements_per_thread, 12><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case 13: gemvt_generic_offset<cuFloatComplex, cgemvt_offset_bs, thread_x, thread_y, elements_per_thread, 13><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case 14: gemvt_generic_offset<cuFloatComplex, cgemvt_offset_bs, thread_x, thread_y, elements_per_thread, 14><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
				case 15: gemvt_generic_offset<cuFloatComplex, cgemvt_offset_bs, thread_x, thread_y, elements_per_thread, 15><<<dimGrid, dimBlock, 0, stream>>>(rows, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, nstripes, offset_r_, offset_c_, conj); break;
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
/***********************************************************************************/
extern "C"
int kblas_cgemv_offset( char trans, int rows, int cols,
						cuFloatComplex alpha, cuFloatComplex *dA, int lda,
						cuFloatComplex *dX, int incx,
						cuFloatComplex beta, cuFloatComplex *dY, int incy,
						int offset_r, int offset_c)
{
	return kblas_cgemv_offset_driver( trans, rows, cols,
	                                alpha, dA, lda,
	                                dX, incx,
	                                beta, dY, incy,
	                                offset_r, offset_c);
}

/*************************************************************************************/
extern "C"
int kblas_cgemv_offset_async( char trans, int rows, int cols,
							cuFloatComplex alpha, cuFloatComplex *dA, int lda,
							cuFloatComplex *dX, int incx,
							cuFloatComplex  beta, cuFloatComplex *dY, int incy,
							int offset_r, int offset_c,
							cudaStream_t stream)
{
	return kblas_cgemv_offset_driver(  trans, rows, cols,
	                            alpha, dA, lda,
	                            dX, incx,
	                            beta, dY, incy, offset_r, offset_c,
	                            stream);
}
/*************************************************************************************/
