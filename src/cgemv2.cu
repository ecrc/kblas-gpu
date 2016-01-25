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
#include "gemv2_core.cuh"

#if(SM >= 30)

#define cgemvn_nb               (32)
#define cgemvn_ntcol            (4)
#define cgemvn_ept              (2)
#define cgemvn_width    (cgemvn_ntcol*cgemvn_ept)
#define cgemvn_by               (16)

#define cgemvt_nb               (32)
#define cgemvt_ntcol            (2)
#define cgemvt_ept              (4)
#define cgemvt_width    (cgemvt_ntcol*cgemvt_ept)
#define cgemvt_by               (8)

#else

#define cgemvn_nb               (64)
#define cgemvn_ntcol    		(8)
#define cgemvn_ept              (2)
#define cgemvn_width    (cgemvn_ntcol*cgemvn_ept)
#define cgemvn_by               (1)

#define cgemvt_nb               (64)
#define cgemvt_ntcol    		(8)
#define cgemvt_ept              (2)
#define cgemvt_width    (cgemvt_ntcol*cgemvt_ept)
#define cgemvt_by               (1)
#endif


extern "C"
int kblas_cscal_async(int n, cuFloatComplex alpha, cuFloatComplex *x, int incx, cudaStream_t stream);

  
int kblas_cgemv2_driver(	char trans, int rows, int cols,
						cuFloatComplex alpha, cuFloatComplex *dA, int lda, 
						cuFloatComplex *dX, int incx, 
						cuFloatComplex  beta, cuFloatComplex *dY, int incy,
						cudaStream_t stream)
{	
	if(trans == 'n' || trans == 'N')
	{
		// scaling with beta
		kblas_cscal_async(rows, beta, dY, incy, stream);
		
		int mod_r = rows % cgemvn_nb;
		int mod_c = cols % cgemvn_width;	
		
		int blocks = rows/cgemvn_nb;
		if(mod_r != 0) blocks += 1;
		
		const int thread_x = cgemvn_nb;
		const int thread_y = cgemvn_ntcol; 
		const int ept = cgemvn_ept;
		
		int threshold = mod_c / ept; 
		int ept_ = mod_c % ept;
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks, cgemvn_by);
		switch(ept_)
		{
			case 0: gemvn<cuFloatComplex, cgemvn_nb, cgemvn_ntcol, ept, cgemvn_width, 0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 1: gemvn<cuFloatComplex, cgemvn_nb, cgemvn_ntcol, ept, cgemvn_width, 1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 2: gemvn<cuFloatComplex, cgemvn_nb, cgemvn_ntcol, ept, cgemvn_width, 2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 3: gemvn<cuFloatComplex, cgemvn_nb, cgemvn_ntcol, ept, cgemvn_width, 3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 4: gemvn<cuFloatComplex, cgemvn_nb, cgemvn_ntcol, ept, cgemvn_width, 4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 5: gemvn<cuFloatComplex, cgemvn_nb, cgemvn_ntcol, ept, cgemvn_width, 5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 6: gemvn<cuFloatComplex, cgemvn_nb, cgemvn_ntcol, ept, cgemvn_width, 6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 7: gemvn<cuFloatComplex, cgemvn_nb, cgemvn_ntcol, ept, cgemvn_width, 7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			case 8: gemvn<cuFloatComplex, cgemvn_nb, cgemvn_ntcol, ept, cgemvn_width, 8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold); break;
			default: printf("irregular part %d is not supported, please extend the case statement of cgemv\n", ept_); exit(1);
		}
	}	// end of non-transpose case
	else if(trans == 't' || trans == 'T' || trans == 'c' || trans == 'C')
	{
		// scaling with beta
		kblas_cscal_async(cols, beta, dY, incy, stream);
		
		int mod_r = rows % cgemvt_nb;
		int mod_c = cols % cgemvt_width;
		
		int blocks = cols/cgemvt_width;
		if(mod_c != 0) blocks += 1;
		
		const int thread_x = cgemvt_nb;
		const int thread_y = cgemvt_ntcol;
		const int ept = cgemvt_ept;
		
		int threshold = mod_c / ept;
		int ept_ = mod_c % ept;
		
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks, cgemvt_by);
		
		int conj;
		if(trans == 'c' || trans == 'C')conj = 1;
		else conj = 0;
		//printf("modr = %d, modc = %d, threshold = %d, ept_ = %d \n", mod_r, mod_c, threshold, ept_);
		switch(ept_)
		{
			case 0: gemvt<cuFloatComplex, cgemvt_nb, cgemvt_ntcol, ept, cgemvt_width, 0><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 1: gemvt<cuFloatComplex, cgemvt_nb, cgemvt_ntcol, ept, cgemvt_width, 1><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 2: gemvt<cuFloatComplex, cgemvt_nb, cgemvt_ntcol, ept, cgemvt_width, 2><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 3: gemvt<cuFloatComplex, cgemvt_nb, cgemvt_ntcol, ept, cgemvt_width, 3><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 4: gemvt<cuFloatComplex, cgemvt_nb, cgemvt_ntcol, ept, cgemvt_width, 4><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 5: gemvt<cuFloatComplex, cgemvt_nb, cgemvt_ntcol, ept, cgemvt_width, 5><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 6: gemvt<cuFloatComplex, cgemvt_nb, cgemvt_ntcol, ept, cgemvt_width, 6><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 7: gemvt<cuFloatComplex, cgemvt_nb, cgemvt_ntcol, ept, cgemvt_width, 7><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			case 8: gemvt<cuFloatComplex, cgemvt_nb, cgemvt_ntcol, ept, cgemvt_width, 8><<<dimGrid, dimBlock, 0, stream>>>(rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj); break;
			default: printf("irregular part %d is not supported, please extend the case statement of cgemv\n", ept_); exit(1);
		}
	}
	else
	{	
		printf("CGEMV error: Unrecognized transpose mode %c \n", trans);
		return -1;
	}
	
	return 0;
}

extern "C"
int kblas_cgemv2(char trans, int rows, int cols,
				cuFloatComplex alpha, cuFloatComplex *dA, int lda, 
				cuFloatComplex *dX, int incx, 
				cuFloatComplex  beta, cuFloatComplex *dY, int incy)
{
	return kblas_cgemv2_driver(	trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, 0);
}

extern "C"
int kblas_cgemv2_async(	char trans, int rows, int cols,
						cuFloatComplex alpha, cuFloatComplex *dA, int lda, 
						cuFloatComplex *dX, int incx, 
						cuFloatComplex  beta, cuFloatComplex *dY, int incy,
						cudaStream_t stream)
{
	return kblas_cgemv2_driver(	trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, stream);
}
