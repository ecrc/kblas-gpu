 /**
 -- (C) Copyright 2013 King Abdullah University of Science and Technology
  Authors:
  Ahmad Abdelfattah (ahmad.ahmad@kaust.edu.sa)
  David Keyes (david.keyes@kaust.edu.sa)
  Hatem Ltaief (hatem.ltaief@kaust.edu.sa)

  Redistribution  and  use  in  source and binary forms, with or without
  modification,  are  permitted  provided  that the following conditions
  are met:

  * Redistributions  of  source  code  must  retain  the above copyright
    notice,  this  list  of  conditions  and  the  following  disclaimer.
  * Redistributions  in  binary  form must reproduce the above copyright
    notice,  this list of conditions and the following disclaimer in the
    documentation  and/or other materials provided with the distribution.
  * Neither  the  name of the King Abdullah University of Science and
    Technology nor the names of its contributors may be used to endorse 
    or promote products derived from this software without specific prior 
    written permission.

  THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
  LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**/

#include "syhemv_core.cuh"

#if(SM >= 30)

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
						int m, cuFloatComplex alpha, cuFloatComplex *dA, int lda, 
						cuFloatComplex *dX, int incx, 
						cuFloatComplex  beta, cuFloatComplex *dY, int incy,
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
		  syhemvu_special_d<cuFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy);
		  syhemvu_special_nd<cuFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy);
		}
		else
		{
			syhemvu_generic_d<cuFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod);
			/**
			* The upper case kernel for irregular dimensions has an extra template parameter.
			* This parameter must be among the values listed in the switch-case statement below.
			* The possible values are in the range 0 - (elements_per_thread-1)
			* Make sure these values are updated whenever you change the configuration parameters.  
			**/
			const int irregular_part = mod % elements_per_thread;
			switch(irregular_part)
			{
				case  0: syhemvu_generic_nd<cuFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread,  0><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  1: syhemvu_generic_nd<cuFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread,  1><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  2: syhemvu_generic_nd<cuFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread,  2><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  3: syhemvu_generic_nd<cuFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread,  3><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  4: syhemvu_generic_nd<cuFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread,  4><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  5: syhemvu_generic_nd<cuFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread,  5><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  6: syhemvu_generic_nd<cuFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread,  6><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  7: syhemvu_generic_nd<cuFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread,  7><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
				case  8: syhemvu_generic_nd<cuFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread,  8><<<dimGrid_, dimBlock, 0, stream>>>( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod); break;
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
			syhemvl_special_d<cuFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy);
			syhemvl_special_nd<cuFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy);
		}
		else
		{
		  	syhemvl_generic_d<cuFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod);
			syhemvl_generic_nd<cuFloatComplex, chemv_bs, thread_x, thread_y, elements_per_thread><<<dimGrid_, dimBlock, 0, stream>>> ( m, alpha, dA, lda, dX, incx, beta, dY, incy, mod);
		}
	}
	else{printf("Upper/Lower mode %c is not supported \n", uplo); return -1;}	
	return 0;
}

extern "C"
int kblas_chemv( char uplo, 
				int m, cuFloatComplex alpha, cuFloatComplex *dA, int lda, 
				cuFloatComplex *dX, int incx, 
				cuFloatComplex  beta, cuFloatComplex *dY, int incy)
{
	return kblas_chemv_driver( uplo, m, alpha, dA, lda, dX, incx, beta, dY, incy, 0);
}

extern "C"
int kblas_chemv_async( 	char uplo, 
						int m, cuFloatComplex alpha, cuFloatComplex *dA, int lda, 
						cuFloatComplex *dX, int incx, 
						cuFloatComplex  beta, cuFloatComplex *dY, int incy, cudaStream_t stream)
{
	return kblas_chemv_driver( uplo, m, alpha, dA, lda, dX, incx, beta, dY, incy, stream);
}
