/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/zhemv.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ahmad Abdelfattah
 * @date 2018-11-14
 **/

#include "syhemv_core.cuh"

#if(TARGET_SM >= 30)

#define zhemv_upper_bs (32)
#define zhemv_upper_ty (4)
#define zhemv_upper_by (2)

#define zhemv_lower_bs (16)
#define zhemv_lower_ty (2)
#define zhemv_lower_by (2)

#else

#define zhemv_upper_bs (32)
#define zhemv_upper_ty (8)
#define zhemv_upper_by (2)

#define zhemv_lower_bs (16)
#define zhemv_lower_ty (4)
#define zhemv_lower_by (2)

#endif



int kblas_zhemv_driver( char uplo,
						int m, cuDoubleComplex alpha, cuDoubleComplex *dA, int lda,
						cuDoubleComplex *dX, int incx,
						cuDoubleComplex  beta, cuDoubleComplex *dY, int incy,
						cudaStream_t stream)
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
		const int zhemv_bs = zhemv_upper_bs;
		const int thread_x = zhemv_bs;
		const int thread_y = zhemv_upper_ty;
		const int elements_per_thread = (zhemv_bs/(2*thread_y)) ;
		/** end configuration params **/

		int mod = m % zhemv_bs;
		int blocks = m / zhemv_bs + (mod != 0);
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks,1);
		dim3 dimGrid_(blocks,zhemv_upper_by);

		if(mod == 0)
		{
		  syhemvu_special_d<cuDoubleComplex, zhemv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy);
		  syhemvu_special_nd<cuDoubleComplex, zhemv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy);
		}
		else
		{
			syhemvu_generic_d<cuDoubleComplex, zhemv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod);
			/**
			* The upper case kernel for irregular dimensions has an extra template parameter.
			* This parameter must be among the values listed in the switch-case statement below.
			* The possible values are in the range 0 - (elements_per_thread-1)
			* Make sure these values are updated whenever you change the configuration parameters (i.e. the block size).
			**/
			const int irregular_part = mod % elements_per_thread;
			switch(irregular_part)
			{
				case  0: syhemvu_generic_nd<cuDoubleComplex, zhemv_bs, thread_x, thread_y, elements_per_thread,  0><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  1: syhemvu_generic_nd<cuDoubleComplex, zhemv_bs, thread_x, thread_y, elements_per_thread,  1><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  2: syhemvu_generic_nd<cuDoubleComplex, zhemv_bs, thread_x, thread_y, elements_per_thread,  2><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  3: syhemvu_generic_nd<cuDoubleComplex, zhemv_bs, thread_x, thread_y, elements_per_thread,  3><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  4: syhemvu_generic_nd<cuDoubleComplex, zhemv_bs, thread_x, thread_y, elements_per_thread,  4><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  5: syhemvu_generic_nd<cuDoubleComplex, zhemv_bs, thread_x, thread_y, elements_per_thread,  5><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  6: syhemvu_generic_nd<cuDoubleComplex, zhemv_bs, thread_x, thread_y, elements_per_thread,  6><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  7: syhemvu_generic_nd<cuDoubleComplex, zhemv_bs, thread_x, thread_y, elements_per_thread,  7><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  8: syhemvu_generic_nd<cuDoubleComplex, zhemv_bs, thread_x, thread_y, elements_per_thread,  8><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				// return error otherwise:
				default: printf("ZHEMV-UPPER ERROR: improper template parameters. Please read the inline documentation for this function. \n"); return -1;
			}
		}
	}
	else if(uplo == 'L' || uplo == 'l')
	{
		/** configuration params **/
		const int zhemv_bs = zhemv_lower_bs;
		const int thread_x = zhemv_bs;
		const int thread_y = zhemv_lower_ty;
		const int elements_per_thread = (zhemv_bs/(2*thread_y)) ;
		/** end configuration params **/

		int mod = m % zhemv_bs;
		int blocks = m / zhemv_bs + (mod != 0);
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks,1);
		dim3 dimGrid_(blocks,zhemv_lower_by);

		if(mod == 0)
		{
			syhemvl_special_d<cuDoubleComplex, zhemv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy);
			syhemvl_special_nd<cuDoubleComplex, zhemv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy);
		}
		else
		{
		  	syhemvl_generic_d<cuDoubleComplex, zhemv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod);
			syhemvl_generic_nd<cuDoubleComplex, zhemv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod);
		}
	}
	else{printf("Upper/Lower mode %c is not supported \n", uplo); return -1;}
	return 0;
}

extern "C"
int kblas_zhemv( char uplo,
			 int m, cuDoubleComplex alpha, cuDoubleComplex *dA, int lda,
			 cuDoubleComplex *dX, int incx,
			 cuDoubleComplex  beta, cuDoubleComplex *dY, int incy)
{
	return kblas_zhemv_driver( uplo, m, alpha, dA, lda, dX, incx, beta, dY, incy, 0);
}

extern "C"
int kblas_zhemv_async( char uplo,
						int m, cuDoubleComplex alpha, cuDoubleComplex *dA, int lda,
						cuDoubleComplex *dX, int incx,
						cuDoubleComplex  beta, cuDoubleComplex *dY, int incy,
						cudaStream_t stream)
{
	return kblas_zhemv_driver( uplo, m, alpha, dA, lda, dX, incx, beta, dY, incy, stream);
}

