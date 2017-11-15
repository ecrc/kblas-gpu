/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/ssymv.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 2.0.0
 * @author Ahmad Abdelfattah
 * @date 2017-11-13
 **/

/***********************************************************************
    DOCUMENTATION
    =============

    kblas_ssymv performs the matrix-vector operation on a single GPU:

       y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n symmetric matrix.

    Arguments
    ==========

    UPLO   - CHARACTER.
             On entry, UPLO specifies whether the upper or lower
             triangular part of the array A is to be referenced as
             follows:

                UPLO = 'U' or 'u'   Only the upper triangular part of A
                                    is to be referenced.

                UPLO = 'L' or 'l'   Only the lower triangular part of A
                                    is to be referenced.

             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the order of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    ALPHA  - SINGLE PRECISION      .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - SINGLE PRECISION array of DIMENSION ( LDA, n ).
             Before entry with  UPLO = 'U' or 'u', the leading n by n
             upper triangular part of the array A must contain the upper
             triangular part of the symmetric matrix and the strictly
             lower triangular part of A is not referenced.
             Before entry with UPLO = 'L' or 'l', the leading n by n
             lower triangular part of the array A must contain the lower
             triangular part of the symmetric matrix and the strictly
             upper triangular part of A is not referenced.
             Note that the imaginary parts of the diagonal elements need
             not be set and are assumed to be zero.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, n ).
             Unchanged on exit.
             It is recommended that lda is multiple of 16. Otherwise
             performance would be deteriorated as the memory accesses
             would not be fully coalescent.

    X      - SINGLE PRECISION array of dimension at least
             ( 1 + ( n - 1 )*abs( INCX ) ).
             Before entry, the incremented array X must contain the n
             element vector x.
             Unchanged on exit.

    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.

    BETA   - SINGLE PRECISION.
             On entry, BETA specifies the scalar beta. When BETA is
             supplied as zero then Y need not be set on input.
             Unchanged on exit.

    Y      - SINGLE PRECISION array of dimension at least
             ( 1 + ( n - 1 )*abs( INCY ) ).
             Before entry, the incremented array Y must contain the n
             element vector y. On exit, Y is overwritten by the updated
             vector y.

    INCY   - INTEGER.
             On entry, INCY specifies the increment for the elements of
             Y. INCY must not be zero.
             Unchanged on exit.

*****************************************************************************/

#include "syhemv_core.cuh"

#if(SM >= 30)

#define ssymv_upper_bs 	(64)
#define ssymv_upper_ty 	(4)
#define ssymv_upper_by	(2)

#define ssymv_lower_bs 	(64)
#define ssymv_lower_ty 	(4)
#define ssymv_lower_by	(2)

#else

#define ssymv_upper_bs 	(32)
#define ssymv_upper_ty 	(8)
#define ssymv_upper_by	(2)

#define ssymv_lower_bs 	(32)
#define ssymv_lower_ty 	(4)
#define ssymv_lower_by	(2)

#endif


int kblas_ssymv_driver( char uplo,
						int m, float alpha, float *dA, int lda,
						float *dX, int incx,
						float  beta, float *dY, int incy,
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
		const int ssymv_bs = ssymv_upper_bs;
		const int thread_x = ssymv_bs;
		const int thread_y = ssymv_upper_ty;
		const int elements_per_thread = (ssymv_bs/(2*thread_y)) ;
		/** end configuration params **/

		int mod = m % ssymv_bs;
		int blocks = m / ssymv_bs + (mod != 0);
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks,1);
		dim3 dimGrid_(blocks, ssymv_upper_by);

		//if (mod == 0) mod = ssymv_bs;
		if(mod == 0)
		{
		  syhemvu_special_d<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy);
		  syhemvu_special_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy);
		}
		else
		{
			syhemvu_generic_d<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod);
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
					case  0: syhemvu_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  0><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
					case  1: syhemvu_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  1><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
					case  2: syhemvu_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  2><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
					case  3: syhemvu_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  3><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
					case  4: syhemvu_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  4><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
					case  5: syhemvu_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  5><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
					case  6: syhemvu_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  6><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
					case  7: syhemvu_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  7><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
					case  8: syhemvu_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread,  8><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
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
		/** end configuration params **/

		int mod = m % ssymv_bs;
		int blocks = m / ssymv_bs + (mod != 0);
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks,1);
		dim3 dimGrid_(blocks,ssymv_lower_by);

		if(mod == 0)
		{
			syhemvl_special_d<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy);
			syhemvl_special_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy);
		}
		else
		{
		  	syhemvl_generic_d<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod);
			syhemvl_generic_nd<float, ssymv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod);
		}
	}
	else{printf("Upper/Lower mode %c is not supported \n", uplo); return -1;}
	return 0;
}


extern "C"
int kblas_ssymv( char uplo,
				 int m, float alpha, float *dA, int lda,
				float *dX, int incx,
				float  beta, float *dY, int incy)
{
	return kblas_ssymv_driver( uplo, m, alpha, dA, lda, dX, incx, beta, dY, incy, 0);
}

extern "C"
int kblas_ssymv_async( char uplo,
						int m, float alpha, float *dA, int lda,
						float *dX, int incx,
						float  beta, float *dY, int incy,
						cudaStream_t stream)
{
	return kblas_ssymv_driver( uplo, m, alpha, dA, lda, dX, incx, beta, dY, incy, stream);
}
