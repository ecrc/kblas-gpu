/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/dsymv_offset.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ahmad Abdelfattah
 * @date 2018-11-14
 **/

#include "syhemv_offset_core.cuh"

#if(TARGET_SM >= 30)

#define dsymv_upper_bs	(32)
#define dsymv_upper_ty	(2)
#define dsymv_upper_by  (4)

#define dsymv_lower_bs	(32)
#define dsymv_lower_ty	(2)
#define dsymv_lower_by  (4)

#else

#define dsymv_upper_bs	(64)
#define dsymv_upper_ty	(8)
#define dsymv_upper_by  (2)

#define dsymv_lower_bs	(64)
#define dsymv_lower_ty	(8)
#define dsymv_lower_by  (2)

#endif

/*************************************************************************************/
int kblas_dsymv_offset_driver(  char uplo, int m,
							    double alpha, double *dA, int lda,
							    double *dX, int incx,
							    double  beta, double *dY, int incy,
						    	int offset,
							    cudaStream_t stream = 0)
{
	// handle the case when incx and/or incy is -ve
	if(incx < 0) dX -= (m-1) * incx;
	if(incy < 0) dY -= (m-1) * incy;

	if(uplo == 'U' || uplo == 'u')
	{
		/** configuration params **/
		const int dsymv_bs = dsymv_upper_bs;
		const int thread_x = dsymv_bs;
		const int thread_y = dsymv_upper_ty;
		const int elements_per_thread = (dsymv_bs/(2*thread_y)) ;
		/** end configuration params **/

		/** offset necessary calculation **/
		int offset_ = offset % dsymv_bs;
		int total_blocks_skipped = offset / dsymv_bs;
		int my_skipped_blocks = total_blocks_skipped/ngpus;
		if(gpu_gid < (total_blocks_skipped%ngpus)) my_skipped_blocks += 1;
		int ref_gpu = total_blocks_skipped%ngpus;
		int new_gpu_gid = (gpu_gid - ref_gpu + ngpus) % ngpus;
		// Advance pointers accordingly
		dA += my_skipped_blocks * dsymv_bs * lda;
		dA += total_blocks_skipped * dsymv_bs;
		dX += total_blocks_skipped * dsymv_bs * incx;
		dY += total_blocks_skipped * dsymv_bs * incy;
		m  -= total_blocks_skipped * dsymv_bs;
		/** end offset necessary calculation **/

		int mod = m % dsymv_bs;
		int nstripes = m / dsymv_bs + (mod != 0);
		int blocks = nstripes/ngpus;
		if(new_gpu_gid < (nstripes%ngpus) ) blocks += 1;
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks,1);
		dim3 dimGrid_(blocks, dsymv_upper_by);

		if(blocks == 0) return 0;

		if(mod == 0)
		{
			syhemvu_special_d_offset<double, dsymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, nstripes, offset_);
			syhemvu_special_nd_offset<double, dsymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, nstripes, offset_);
		}
		else
		{
			syhemvu_generic_d_offset<double, dsymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, ngpus, nstripes, offset_);
			const int irregular_part = mod % elements_per_thread;
			/**
			 * The upper case kernel for irregular dimensions has an extra template parameter.
			 * This parameter must be among the values listed in the switch-case statement below.
			 * The possible values are in the range 0 - (elements_per_thread-1)
			 * Make sure these values are updated whenever you change the configuration parameters.
			 **/
			switch(irregular_part)
			{
				case  0: syhemvu_generic_nd_offset<double, dsymv_bs, thread_x, thread_y, elements_per_thread,  0><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, nstripes, offset_); break;
				case  1: syhemvu_generic_nd_offset<double, dsymv_bs, thread_x, thread_y, elements_per_thread,  1><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, nstripes, offset_); break;
				case  2: syhemvu_generic_nd_offset<double, dsymv_bs, thread_x, thread_y, elements_per_thread,  2><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, nstripes, offset_); break;
				case  3: syhemvu_generic_nd_offset<double, dsymv_bs, thread_x, thread_y, elements_per_thread,  3><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, nstripes, offset_); break;
				case  4: syhemvu_generic_nd_offset<double, dsymv_bs, thread_x, thread_y, elements_per_thread,  4><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, nstripes, offset_); break;
				case  5: syhemvu_generic_nd_offset<double, dsymv_bs, thread_x, thread_y, elements_per_thread,  5><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, nstripes, offset_); break;
				case  6: syhemvu_generic_nd_offset<double, dsymv_bs, thread_x, thread_y, elements_per_thread,  6><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, nstripes, offset_); break;
				case  7: syhemvu_generic_nd_offset<double, dsymv_bs, thread_x, thread_y, elements_per_thread,  7><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, nstripes, offset_); break;
				case  8: syhemvu_generic_nd_offset<double, dsymv_bs, thread_x, thread_y, elements_per_thread,  8><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, nstripes, offset_); break;
				// return error otherwise:
				default: printf("DSYMV-UPPER ERROR: improper template parameter. Please read the inline documentation for this function. \n"); return -1;
			}
		}

	}
	else if(uplo == 'L' || uplo == 'l')
	{
		/** configuration params **/
		const int dsymv_bs = dsymv_lower_bs;
		const int thread_x = dsymv_bs;
		const int thread_y = dsymv_lower_ty;
		const int elements_per_thread = (dsymv_bs/(2*thread_y)) ;
		/** end configuration params **/

		/** offset necessary calculation **/
		int offset_ = offset % dsymv_bs;
		int total_blocks_skipped = offset / dsymv_bs;
		int my_skipped_blocks = total_blocks_skipped/ngpus;
		if(gpu_gid < (total_blocks_skipped%ngpus)) my_skipped_blocks += 1;
		int ref_gpu = total_blocks_skipped%ngpus;
		int new_gpu_gid = (gpu_gid - ref_gpu + ngpus) % ngpus;
		// Advance pointers accordingly
		dA += my_skipped_blocks * dsymv_bs * lda;
		dA += total_blocks_skipped * dsymv_bs;
		dX += total_blocks_skipped * dsymv_bs * incx;
		dY += total_blocks_skipped * dsymv_bs * incy;
		m  -= total_blocks_skipped * dsymv_bs;
		/** end offset necessary calculation **/

		int mod = m % dsymv_bs;
		int nstripes = m / dsymv_bs + (mod != 0);
		int blocks = nstripes/ngpus;
		if(new_gpu_gid < (nstripes%ngpus) ) blocks += 1;
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks,1);
		dim3 dimGrid_(blocks, dsymv_lower_by);

		if(blocks == 0) return 0;

		if(mod == 0)
		{
			syhemvl_special_d_offset<double, dsymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, nstripes, offset_);
			syhemvl_special_nd_offset<double, dsymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, nstripes, offset_);
		}
		else
		{
		  	syhemvl_generic_d_offset<double, dsymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, nstripes, offset_);
			syhemvl_generic_nd_offset<double, dsymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, nstripes, offset_);
		}
	}
	else{printf("Upper/Lower mode %c is not supported \n", uplo); return -1;}
	return 0;
}

/*************************************************************************************/
extern "C"
int kblas_dsymv_offset( char uplo, int m,
						double alpha, double *dA, int lda,
						double *dX, int incx,
						double beta, double *dY, int incy,
						int offset)
{
	return kblas_dsymv_offset_driver(uplo, m, alpha, dA, lda, dX, incx, beta, dY, incy, offset);
}
/*************************************************************************************/
extern "C"
int kblas_dsymv_offset_async( char uplo, int m,
							double alpha, double *dA, int lda,
							double *dX, int incx,
							double  beta, double *dY, int incy,
							int offset,
							cudaStream_t stream)
{
    return kblas_dsymv_offset_driver(uplo, m, alpha, dA, lda, dX, incx, beta, dY, incy, offset, stream);
}
/*************************************************************************************/
