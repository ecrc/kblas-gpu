/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/ssymv_mgpu.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 2.0.0
 * @author Ahmad Abdelfattah
 * @date 2017-11-13
 **/

#include "syhemv_mgpu_core.cuh"
#include "syhemv_mgpu_offset_core.cuh"
#include "defs.h"

#if(SM >= 30)

#define ssymv_upper_bs	(64)
#define ssymv_upper_ty	(4)

#define ssymv_lower_bs	(64)
#define ssymv_lower_ty	(4)

#else

#define ssymv_upper_bs	(32)
#define ssymv_upper_ty	(8)

#define ssymv_lower_bs	(32)
#define ssymv_lower_ty	(4)

#endif

int kblas_ssymv_mgpu_driver( char uplo, int m,
							float alpha, float *dA, int lda,
							float *dX, int incx,
							float  beta, float *dY, int incy,
							int ngpus, int gpu_gid,
							cudaStream_t stream = 0)
{
	// handle the case when incx and/or incy is -ve
	if(incx < 0) dX -= (m-1) * incx;
	if(incy < 0) dY -= (m-1) * incy;

	if(uplo == 'U' || uplo == 'u')
	{
		/** configuration params **/
		/**
		* If you change the configuration parameters,
		* you must revise the case statement of the upper case
		* to make sure it covers all the possible cases
		**/
		const int ssymv_bs = ssymv_upper_bs;
		const int thread_x = ssymv_bs;
		const int thread_y = ssymv_upper_ty;
		const int elements_per_thread = (ssymv_bs/(2*thread_y)) ;
		const int ssymv_upper_by = 2*ngpus;
		/** end configuration params **/
		int mod = m % ssymv_bs;
		int nstripes = m / ssymv_bs + (mod != 0);
		int blocks = nstripes/ngpus;
		if(gpu_gid < (nstripes%ngpus) ) blocks += 1;
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks,1);
		dim3 dimGrid_(blocks, ssymv_upper_by);

		if(blocks == 0) return 0;

		//if (mod == 0) mod = ssymv_bs;
		if(mod == 0)
		{
		  syhemvu_mgpu_special_d<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, gpu_gid, ngpus, nstripes);
		  syhemvu_mgpu_special_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, gpu_gid, ngpus, nstripes);
		}
		else
		{
			syhemvu_mgpu_generic_d<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, gpu_gid, ngpus, nstripes);
			// for the non-diagonal part choose between a templatized irregular part or a variable one
			const int irregular_part = mod % elements_per_thread;
			if(0)
			{}
			else
			{	// Templatized irregular_part

				/**
				 * The upper case kernel for irregular dimensions has an extra template parameter.
				 * This parameter must be among the values listed in the switch-case statement below.
				 * The possible values are in the range 0 - (elements_per_thread-1)
				 * Make sure these values are updated whenever you change the configuration parameters.
				 **/
				switch(irregular_part)
				{
					case  0: syhemvu_mgpu_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  0><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, gpu_gid, ngpus, nstripes); break;
					case  1: syhemvu_mgpu_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  1><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, gpu_gid, ngpus, nstripes); break;
					case  2: syhemvu_mgpu_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  2><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, gpu_gid, ngpus, nstripes); break;
					case  3: syhemvu_mgpu_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  3><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, gpu_gid, ngpus, nstripes); break;
					case  4: syhemvu_mgpu_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  4><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, gpu_gid, ngpus, nstripes); break;
					case  5: syhemvu_mgpu_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  5><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, gpu_gid, ngpus, nstripes); break;
					case  6: syhemvu_mgpu_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  6><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, gpu_gid, ngpus, nstripes); break;
					case  7: syhemvu_mgpu_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  7><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, gpu_gid, ngpus, nstripes); break;
					case  8: syhemvu_mgpu_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  8><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, gpu_gid, ngpus, nstripes); break;
					// return error otherwise:
					default: printf("SSYMV-UPPER ERROR: improper template parameter. Please read the inline documentation for this function. \n"); return -1;
				}
			}
		}
	}
	else if(uplo == 'L' || uplo == 'l')
	{
		/** configuration params **/
		const int ssymv_bs = ssymv_lower_bs;
		const int thread_x = ssymv_bs;
		const int thread_y = ssymv_lower_ty;
		const int elements_per_thread = (ssymv_bs/(2*thread_y)) ;
		const int ssymv_lower_by = 2*ngpus; 	// design rule, feel free to change it
		/** end configuration params **/

		int mod = m % ssymv_bs;
		int nstripes = m / ssymv_bs + (mod != 0);
		int blocks = nstripes/ngpus;
		if(gpu_gid < (nstripes%ngpus) ) blocks += 1;
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks,1);
		dim3 dimGrid_(blocks, ssymv_lower_by);

		if(blocks == 0) return 0;

		if(mod == 0)
		{
			syhemvl_mgpu_special_d<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, gpu_gid, ngpus, nstripes);
			syhemvl_mgpu_special_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, gpu_gid, ngpus, nstripes);
		}
		else
		{
		  	syhemvl_mgpu_generic_d<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, gpu_gid, ngpus, nstripes);
			syhemvl_mgpu_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, gpu_gid, ngpus, nstripes);
		}
	}
	else{printf("Upper/Lower mode %c is not supported \n", uplo); return -1;}
	return 0;
}

/*************************************************************************************/
int kblas_ssymv_mgpu_driver_offset( char uplo, int m,
							float alpha, float *dA, int lda,
							float *dX, int incx,
							float  beta, float *dY, int incy,
							int ngpus, int gpu_gid,
							int offset,
							cudaStream_t stream = 0)
{
	// handle the case when incx and/or incy is -ve
	if(incx < 0) dX -= (m-1) * incx;
	if(incy < 0) dY -= (m-1) * incy;

	if(uplo == 'U' || uplo == 'u')
	{
		/** configuration params **/
		const int ssymv_bs = ssymv_upper_bs;
		const int thread_x = ssymv_bs;
		const int thread_y = ssymv_upper_ty;
		const int elements_per_thread = (ssymv_bs/(2*thread_y)) ;
		const int ssymv_upper_by = 2*ngpus; 	// design rule, feel free to change it
		/** end configuration params **/

		/** offset necessary calculation **/
		int offset_ = offset % ssymv_bs;
		int total_blocks_skipped = offset / ssymv_bs;
		int my_skipped_blocks = total_blocks_skipped/ngpus;
		if(gpu_gid < (total_blocks_skipped%ngpus)) my_skipped_blocks += 1;
		int ref_gpu = total_blocks_skipped%ngpus;
		int new_gpu_gid = (gpu_gid - ref_gpu + ngpus) % ngpus;
		// Advance pointers accordingly
		dA += my_skipped_blocks * ssymv_bs * lda;
		dA += total_blocks_skipped * ssymv_bs;
		dX += total_blocks_skipped * ssymv_bs * incx;
		dY += total_blocks_skipped * ssymv_bs * incy;
		m  -= total_blocks_skipped * ssymv_bs;
		/** end offset necessary calculation **/

		int mod = m % ssymv_bs;
		int nstripes = m / ssymv_bs + (mod != 0);
		int blocks = nstripes/ngpus;
		if(new_gpu_gid < (nstripes%ngpus) ) blocks += 1;
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks,1);
		dim3 dimGrid_(blocks, ssymv_upper_by);

		if(blocks == 0) return 0;

		if(mod == 0)
		{
			syhemvu_mgpu_special_d_offset<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, new_gpu_gid, ngpus, nstripes, offset_);
			syhemvu_mgpu_special_nd_offset<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, new_gpu_gid, ngpus, nstripes, offset_);
		}
		else
		{
			syhemvu_mgpu_generic_d_offset<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, new_gpu_gid, ngpus, nstripes, offset_);
			const int irregular_part = mod % elements_per_thread;
			/**
			 * The upper case kernel for irregular dimensions has an extra template parameter.
			 * This parameter must be among the values listed in the switch-case statement below.
			 * The possible values are in the range 0 - (elements_per_thread-1)
			 * Make sure these values are updated whenever you change the configuration parameters.
			 **/
			switch(irregular_part)
			{
				case  0: syhemvu_mgpu_generic_nd_offset<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  0><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, new_gpu_gid, ngpus, nstripes, offset_); break;
				case  1: syhemvu_mgpu_generic_nd_offset<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  1><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, new_gpu_gid, ngpus, nstripes, offset_); break;
				case  2: syhemvu_mgpu_generic_nd_offset<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  2><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, new_gpu_gid, ngpus, nstripes, offset_); break;
				case  3: syhemvu_mgpu_generic_nd_offset<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  3><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, new_gpu_gid, ngpus, nstripes, offset_); break;
				case  4: syhemvu_mgpu_generic_nd_offset<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  4><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, new_gpu_gid, ngpus, nstripes, offset_); break;
				case  5: syhemvu_mgpu_generic_nd_offset<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  5><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, new_gpu_gid, ngpus, nstripes, offset_); break;
				case  6: syhemvu_mgpu_generic_nd_offset<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  6><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, new_gpu_gid, ngpus, nstripes, offset_); break;
				case  7: syhemvu_mgpu_generic_nd_offset<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  7><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, new_gpu_gid, ngpus, nstripes, offset_); break;
				case  8: syhemvu_mgpu_generic_nd_offset<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  8><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, new_gpu_gid, ngpus, nstripes, offset_); break;
				// return error otherwise:
				default: printf("SSYMV-UPPER ERROR: improper template parameter. Please read the inline documentation for this function. \n"); return -1;
			}
		}

	}
	else if(uplo == 'L' || uplo == 'l')
	{
		/** configuration params **/
		const int ssymv_bs = ssymv_lower_bs;
		const int thread_x = ssymv_bs;
		const int thread_y = ssymv_lower_ty;
		const int elements_per_thread = (ssymv_bs/(2*thread_y)) ;
		const int ssymv_lower_by = 2*ngpus; 	// design rule, feel free to change it
		/** end configuration params **/

		/** offset necessary calculation **/
		int offset_ = offset % ssymv_bs;
		int total_blocks_skipped = offset / ssymv_bs;
		int my_skipped_blocks = total_blocks_skipped/ngpus;
		if(gpu_gid < (total_blocks_skipped%ngpus)) my_skipped_blocks += 1;
		int ref_gpu = total_blocks_skipped%ngpus;
		int new_gpu_gid = (gpu_gid - ref_gpu + ngpus) % ngpus;
		// Advance pointers accordingly
		dA += my_skipped_blocks * ssymv_bs * lda;
		dA += total_blocks_skipped * ssymv_bs;
		dX += total_blocks_skipped * ssymv_bs * incx;
		dY += total_blocks_skipped * ssymv_bs * incy;
		m  -= total_blocks_skipped * ssymv_bs;
		/** end offset necessary calculation **/

		int mod = m % ssymv_bs;
		int nstripes = m / ssymv_bs + (mod != 0);
		int blocks = nstripes/ngpus;
		if(new_gpu_gid < (nstripes%ngpus) ) blocks += 1;
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks,1);
		dim3 dimGrid_(blocks, ssymv_lower_by);

		if(blocks == 0) return 0;

		if(mod == 0)
		{
			syhemvl_mgpu_special_d_offset<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, new_gpu_gid, ngpus, nstripes, offset_);
			syhemvl_mgpu_special_nd_offset<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, new_gpu_gid, ngpus, nstripes, offset_);
		}
		else
		{
		  	syhemvl_mgpu_generic_d_offset<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, new_gpu_gid, ngpus, nstripes, offset_);
			syhemvl_mgpu_generic_nd_offset<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod, new_gpu_gid, ngpus, nstripes, offset_);
		}
	}
	else{printf("Upper/Lower mode %c is not supported \n", uplo); return -1;}
	return 0;
}

/*************************************************************************************/
extern "C"
int kblas_ssymv_mgpu( char uplo, int m,
							float alpha, float **dA, int lda,
							float **dX, int incx,
							float  beta, float **dY, int incy,
							int ngpus,
							int offset)
{
    const int ngpus_local = ngpus;
	if(offset == 0)
	{
		for(int i = 0; i < ngpus_local; i++)
		{
			cudaSetDevice(gpu_lid[i]);
			kblas_ssymv_mgpu_driver(uplo, m, alpha, dA[i], lda, dX[i], incx, beta, dY[i], incy, ngpus, gpu_gid[i]);
		}
	}
	else
	{
		for(int i = 0; i < ngpus_local; i++)
		{
			cudaSetDevice(gpu_lid[i]);
			kblas_ssymv_mgpu_driver_offset(uplo, m, alpha, dA[i], lda, dX[i], incx, beta, dY[i], incy, ngpus, gpu_gid[i], offset);
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
int kblas_ssymv_mgpu_async( char uplo, int m,
							float alpha, float **dA, int lda,
							float **dX, int incx,
							float  beta, float **dY, int incy,
							int ngpus,
							int offset,
							cudaStream_t stream[MAX_NGPUS][MAX_STREAMS])
{
     const int ngpus_local = ngpus;
	if(offset == 0)
	{
		for(int i = 0; i < ngpus_local; i++)
		{
			cudaSetDevice(gpu_lid[i]);
			kblas_ssymv_mgpu_driver(uplo, m, alpha, dA[i], lda, dX[i], incx, beta, dY[i], incy, ngpus, gpu_gid[i], stream[i][0]);
		}
	}
	else
	{
		for(int i = 0; i < ngpus_local; i++)
		{
			cudaSetDevice(gpu_lid[i]);
			kblas_ssymv_mgpu_driver_offset(uplo, m, alpha, dA[i], lda, dX[i], incx, beta, dY[i], incy, ngpus, gpu_gid[i], offset, stream[i][0]);
		}
	}
	return 0;
}
/*************************************************************************************/

extern "C"
int get_ssymv_mgpu_bs(char uplo)
{
    if(uplo == 'l' || uplo == 'L')
        return ssymv_lower_bs;
    else if (uplo == 'u' || uplo == 'U')
        return ssymv_upper_bs;
    else
        {printf("Error ..  input %c is not supported for symv \n", uplo); return -1;}
}

