/*
    -- KBLAS (version 1.0) --
       Ahmad Abdelfattah, Center of Extreme Computing
	   Hatem Ltaief, Supercomputing Laboratory
	   David Keyes, Center of Extreme Computing
	   King Abdullah University of Science and Technology (KAUST)
       June 2013
	   KBLAS is a subset of BLAS routines highly optimized for NVIDIA GPUs 
*/
/**
	-- Center of Extreme Computing and Supercomputing Laboratory
	-- Division of Applied Mathematics and Computational Science
	-- King Abdullah University of Science and Technology
	-- (C) Copyright 2013

	Redistribution  and  use  in  source and binary forms, with or without
	modification,  are  permitted  provided  that the following conditions
	are met:

	*	Redistributions  of  source  code  must  retain  the above copyright
		notice,  this  list  of  conditions  and  the  following  disclaimer.
	* 	Redistributions  in  binary  form must reproduce the above copyright
		notice,  this list of conditions and the following disclaimer in the
		documentation  and/or other materials provided with the distribution.
	* 	Neither  the  name of the University of Tennessee, Knoxville nor the
		names of its contributors may be used to endorse or promote products
		derived from this software without specific prior written permission.

	THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
	''AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
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

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "gemv_core.cuh"

#if(SM >= 30)

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
int kblas_dscal_async(int n, double alpha, double *x, int incx, cudaStream_t stream);

  
int kblas_dgemv_driver(	char trans, int rows, int cols,
						double alpha, double *dA, int lda, 
						double *dX, int incx, 
						double  beta, double *dY, int incy,
						cudaStream_t stream)
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
				gemvn_special<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy);
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
					case  0: gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  1: gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  2: gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  3: gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  4: gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  5: gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  6: gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  7: gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  8: gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
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
				gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread, 0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c);
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
					case  0: gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  1: gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  2: gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  3: gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  4: gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  5: gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  6: gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  7: gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
					case  8: gemvn_generic<double, dgemvn_bs, dgemvn_bs, dgemvn_ty, elements_per_thread,  8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c); break;
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
				gemvt_special<double, dgemvt_bs, thread_x, thread_y, elements_per_thread><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, conj); 
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
				gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread, 0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj);
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
				case  0: gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread,  0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  1: gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread,  1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  2: gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread,  2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  3: gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread,  3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  4: gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread,  4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  5: gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread,  5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  6: gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread,  6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  7: gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread,  7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
				case  8: gemvt_generic<double, dgemvt_bs, thread_x, thread_y, elements_per_thread,  8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, conj); break;
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
						cudaStream_t stream)
{
	return kblas_dgemv_driver(	trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, stream);
}
