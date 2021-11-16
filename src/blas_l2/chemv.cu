#include "hip/hip_runtime.h"
/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/chemv.cu

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

#define chemv_upper_bs	(32)
#define chemv_upper_ty	(2)
#define chemv_upper_by	(2)

#define chemv_lower_bs	(32)
#define chemv_lower_ty	(2)
#define chemv_lower_by	(2)

#else

#define chemv_upper_bs	(64)
#define chemv_upper_ty	(8)
#define chemv_upper_by	(2)

#define chemv_lower_bs	(32)
#define chemv_lower_ty	(4)
#define chemv_lower_by	(2)

#endif


int kblas_chemv_driver( char uplo,
						int m, hipFloatComplex alpha, hipFloatComplex *dA, int lda,
						hipFloatComplex *dX, int incx,
						hipFloatComplex  beta, hipFloatComplex *dY, int incy,
						hipStream_t stream)
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
		const int chemv_bs = chemv_upper_bs;
		const int thread_x = chemv_bs;
		const int thread_y = chemv_upper_ty;
		const int elements_per_thread = (chemv_bs/(2*thread_y)) ;
		/** end configuration params **/

		int mod = m % chemv_bs;
		int blocks = m / chemv_bs + (mod != 0);
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks,1);
		dim3 dimGrid_(blocks, chemv_upper_by);

		if(mod == 0)
		{
		  hipLaunchKernelGGL(HIP_KERNEL_NAME(syhemvu_special_d<hipFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread>), dim3(dimGrid), dim3(dimBlock), 0, stream,  m, alpha, dA, lda, dX, incx, beta, dY, incy);
		  hipLaunchKernelGGL(HIP_KERNEL_NAME(syhemvu_special_nd<hipFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread>), dim3(dimGrid_), dim3(dimBlock), 0, stream,  m, alpha, dA, lda, dX, incx, beta, dY, incy);
		}
		else
		{
			hipLaunchKernelGGL(HIP_KERNEL_NAME(syhemvu_generic_d<hipFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread>), dim3(dimGrid), dim3(dimBlock), 0, stream,  m, alpha, dA, lda, dX, incx, beta, dY, incy, mod);
			/**
			* The upper case kernel for irregular dimensions has an extra template parameter.
			* This parameter must be among the values listed in the switch-case statement below.
			* The possible values are in the range 0 - (elements_per_thread-1)
			* Make sure these values are updated whenever you change the configuration parameters.
			**/
			const int irregular_part = mod % elements_per_thread;
			switch(irregular_part)
			{
				case  0: hipLaunchKernelGGL(HIP_KERNEL_NAME(syhemvu_generic_nd<hipFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread,  0>), dim3(dimGrid_), dim3(dimBlock), 0, stream,  m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  1: hipLaunchKernelGGL(HIP_KERNEL_NAME(syhemvu_generic_nd<hipFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread,  1>), dim3(dimGrid_), dim3(dimBlock), 0, stream,  m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  2: hipLaunchKernelGGL(HIP_KERNEL_NAME(syhemvu_generic_nd<hipFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread,  2>), dim3(dimGrid_), dim3(dimBlock), 0, stream,  m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  3: hipLaunchKernelGGL(HIP_KERNEL_NAME(syhemvu_generic_nd<hipFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread,  3>), dim3(dimGrid_), dim3(dimBlock), 0, stream,  m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  4: hipLaunchKernelGGL(HIP_KERNEL_NAME(syhemvu_generic_nd<hipFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread,  4>), dim3(dimGrid_), dim3(dimBlock), 0, stream,  m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  5: hipLaunchKernelGGL(HIP_KERNEL_NAME(syhemvu_generic_nd<hipFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread,  5>), dim3(dimGrid_), dim3(dimBlock), 0, stream,  m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  6: hipLaunchKernelGGL(HIP_KERNEL_NAME(syhemvu_generic_nd<hipFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread,  6>), dim3(dimGrid_), dim3(dimBlock), 0, stream,  m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  7: hipLaunchKernelGGL(HIP_KERNEL_NAME(syhemvu_generic_nd<hipFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread,  7>), dim3(dimGrid_), dim3(dimBlock), 0, stream,  m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  8: hipLaunchKernelGGL(HIP_KERNEL_NAME(syhemvu_generic_nd<hipFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread,  8>), dim3(dimGrid_), dim3(dimBlock), 0, stream,  m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				// return error otherwise:
				default: printf("CHEMV-UPPER ERROR: improper template parameter. Please read the inline documentation for this function. \n"); return -1;
			}
		}
	}
	else if(uplo == 'L' || uplo == 'l')
	{
		/** configuration params **/
		const int chemv_bs = chemv_lower_bs;
		const int thread_x = chemv_bs;
		const int thread_y = chemv_lower_ty;
		const int elements_per_thread = (chemv_bs/(2*thread_y)) ;
		/** end configuration params **/

		int mod = m % chemv_bs;
		int blocks = m / chemv_bs + (mod != 0);
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks,1);
		dim3 dimGrid_(blocks,chemv_lower_by);

		if(mod == 0)
		{
			hipLaunchKernelGGL(HIP_KERNEL_NAME(syhemvl_special_d<hipFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread>), dim3(dimGrid), dim3(dimBlock), 0, stream,  m, alpha, dA, lda, dX, incx, beta, dY, incy);
			hipLaunchKernelGGL(HIP_KERNEL_NAME(syhemvl_special_nd<hipFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread>), dim3(dimGrid_), dim3(dimBlock), 0, stream,  m, alpha, dA, lda, dX, incx, beta, dY, incy);
		}
		else
		{
		  	hipLaunchKernelGGL(HIP_KERNEL_NAME(syhemvl_generic_d<hipFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread>), dim3(dimGrid), dim3(dimBlock), 0, stream,  m, alpha, dA, lda, dX, incx, beta, dY, incy, mod);
			hipLaunchKernelGGL(HIP_KERNEL_NAME(syhemvl_generic_nd<hipFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread>), dim3(dimGrid_), dim3(dimBlock), 0, stream,  m, alpha, dA, lda, dX, incx, beta, dY, incy, mod);
		}
	}
	else{printf("Upper/Lower mode %c is not supported \n", uplo); return -1;}
	return 0;
}

extern "C"
int kblas_chemv( char uplo,
				int m, hipFloatComplex alpha, hipFloatComplex *dA, int lda,
				hipFloatComplex *dX, int incx,
				hipFloatComplex  beta, hipFloatComplex *dY, int incy)
{
	return kblas_chemv_driver( uplo, m, alpha, dA, lda, dX, incx, beta, dY, incy, 0);
}

extern "C"
int kblas_chemv_async( 	char uplo,
						int m, hipFloatComplex alpha, hipFloatComplex *dA, int lda,
						hipFloatComplex *dX, int incx,
						hipFloatComplex  beta, hipFloatComplex *dY, int incy, hipStream_t stream)
{
	return kblas_chemv_driver( uplo, m, alpha, dA, lda, dX, incx, beta, dY, incy, stream);
}
