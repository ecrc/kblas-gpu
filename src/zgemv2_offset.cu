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
#include "gemv2_offset_core.cuh"

#if(SM >= 30)

#define zgemvn_offset_nb               	(32)
#define zgemvn_offset_ntcol    			(4)
#define zgemvn_offset_ept              	(2)
#define zgemvn_offset_width    			(zgemvn_offset_ntcol*zgemvn_offset_ept)
#define zgemvn_offset_by               	(4)

#define zgemvt_offset_nb               	(32)
#define zgemvt_offset_ntcol            	(4)
#define zgemvt_offset_ept              	(4)
#define zgemvt_offset_width    			(zgemvt_offset_ntcol*zgemvt_offset_ept)
#define zgemvt_offset_by               	(4)

#else

#define zgemvn_offset_nb               	(64)
#define zgemvn_offset_ntcol    			(8)
#define zgemvn_offset_ept              	(2)
#define zgemvn_offset_width    			(zgemvn_offset_ntcol*zgemvn_offset_ept)
#define zgemvn_offset_by				(1)

#define zgemvt_offset_nb               	(64)
#define zgemvt_offset_ntcol    			(8)
#define zgemvt_offset_ept              	(2)
#define zgemvt_offset_width    			(zgemvt_offset_ntcol*zgemvt_offset_ept)
#define zgemvt_offset_by               	(1)
#endif


extern "C"
int kblas_zscal_async(int n, cuDoubleComplex alpha, cuDoubleComplex *x, int incx, cudaStream_t stream);

  
int kblas_zgemv2_offset_driver(char trans, int rows, int cols,
						cuDoubleComplex alpha, cuDoubleComplex *dA, int lda, 
						cuDoubleComplex *dX, int incx, 
						cuDoubleComplex  beta, cuDoubleComplex *dY, int incy, 
						int offset_r, int offset_c, 
						cudaStream_t stream)
{	
	if(trans == 'n' || trans == 'N')
	{
		// offset necessary calculations
		int offset_r_ = offset_r % zgemvn_offset_nb;
		int offset_c_ = offset_c % zgemvn_offset_width; 
		int rows_ = rows - (offset_r - offset_r_);
		int cols_ = cols - (offset_c - offset_c_);
		
		// Advance pointers
		dA += (offset_c - offset_c_) * lda + (offset_r - offset_r_);
		dX += (offset_c - offset_c_) * incx; 
		dY += (offset_r - offset_r_) * incy;
		
		// scaling with beta
		kblas_zscal_async(rows_, beta, dY, incy, stream);
		
		int mod_r = rows_ % zgemvn_offset_nb;
		int mod_c = cols_ % zgemvn_offset_width;	
		
		int blocks = rows_/zgemvn_offset_nb;
		if(mod_r != 0) blocks += 1;
		
		const int thread_x = zgemvn_offset_nb;
		const int thread_y = zgemvn_offset_ntcol; 
		const int ept = zgemvn_offset_ept;
		
		int threshold = mod_c / ept; 
		int ept_ = mod_c % ept;
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks, zgemvn_offset_by);
		//printf("rows_ = %d - cols_ = %d - mod_r = %d - mod_c = %d - offset_r_ = %d - offset_c_ = %d \n", rows_, cols_, mod_r, mod_c, offset_r_, offset_c_); 
		switch(ept_)
		{
			case 0: gemvn_offset<cuDoubleComplex, zgemvn_offset_nb, zgemvn_offset_ntcol, ept, zgemvn_offset_width, 0><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 1: gemvn_offset<cuDoubleComplex, zgemvn_offset_nb, zgemvn_offset_ntcol, ept, zgemvn_offset_width, 1><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 2: gemvn_offset<cuDoubleComplex, zgemvn_offset_nb, zgemvn_offset_ntcol, ept, zgemvn_offset_width, 2><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 3: gemvn_offset<cuDoubleComplex, zgemvn_offset_nb, zgemvn_offset_ntcol, ept, zgemvn_offset_width, 3><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 4: gemvn_offset<cuDoubleComplex, zgemvn_offset_nb, zgemvn_offset_ntcol, ept, zgemvn_offset_width, 4><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 5: gemvn_offset<cuDoubleComplex, zgemvn_offset_nb, zgemvn_offset_ntcol, ept, zgemvn_offset_width, 5><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 6: gemvn_offset<cuDoubleComplex, zgemvn_offset_nb, zgemvn_offset_ntcol, ept, zgemvn_offset_width, 6><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 7: gemvn_offset<cuDoubleComplex, zgemvn_offset_nb, zgemvn_offset_ntcol, ept, zgemvn_offset_width, 7><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			case 8: gemvn_offset<cuDoubleComplex, zgemvn_offset_nb, zgemvn_offset_ntcol, ept, zgemvn_offset_width, 8><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, offset_r_, offset_c_); break;
			default: printf("irregular part %d is not supported, please extend the case statement of zgemv\n", ept_); exit(1);
		}
	}	// end of non-transpose case
	else if(trans == 't' || trans == 'T' || trans == 'c' || trans == 'C')
	{
		// offset necessary calculations
		int offset_r_ = offset_r % zgemvt_offset_nb;
		int offset_c_ = offset_c % zgemvt_offset_width; 
		int rows_ = rows - (offset_r - offset_r_);
		int cols_ = cols - (offset_c - offset_c_);
		
		// Advance pointers
		dA += (offset_c - offset_c_) * lda + (offset_r - offset_r_);
		dX += (offset_r - offset_r_) * incx; 
		dY += (offset_c - offset_c_) * incy;
		
		// scaling with beta
		kblas_zscal_async(cols_, beta, dY, incy, stream);
		
		int mod_r = rows_ % zgemvt_offset_nb;
		int mod_c = cols_ % zgemvt_offset_width;
		
		int blocks = cols_/zgemvt_offset_width;
		if(mod_c != 0) blocks += 1;
		
		const int thread_x = zgemvt_offset_nb;
		const int thread_y = zgemvt_offset_ntcol;
		const int ept = zgemvt_offset_ept;
		
		int threshold = mod_c / ept;
		int ept_ = mod_c % ept;
		
		dim3 dimBlock(thread_x, thread_y);
		dim3 dimGrid(blocks, zgemvt_offset_by);
		
		int conj;
		if(trans == 'c' || trans == 'C')conj = 1;
		else conj = 0;
		//printf("modr = %d, modc = %d, threshold = %d, ept_ = %d \n", mod_r, mod_c, threshold, ept_);
		//printf("rows_ = %d - cols_ = %d - mod_r = %d - mod_c = %d - offset_r_ = %d - offset_c_ = %d \n", rows_, cols_, mod_r, mod_c, offset_r_, offset_c_); 
		switch(ept_)
		{
			case 0: gemvt_offset<cuDoubleComplex, zgemvt_offset_nb, zgemvt_offset_ntcol, ept, zgemvt_offset_width, 0><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 1: gemvt_offset<cuDoubleComplex, zgemvt_offset_nb, zgemvt_offset_ntcol, ept, zgemvt_offset_width, 1><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 2: gemvt_offset<cuDoubleComplex, zgemvt_offset_nb, zgemvt_offset_ntcol, ept, zgemvt_offset_width, 2><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 3: gemvt_offset<cuDoubleComplex, zgemvt_offset_nb, zgemvt_offset_ntcol, ept, zgemvt_offset_width, 3><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 4: gemvt_offset<cuDoubleComplex, zgemvt_offset_nb, zgemvt_offset_ntcol, ept, zgemvt_offset_width, 4><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 5: gemvt_offset<cuDoubleComplex, zgemvt_offset_nb, zgemvt_offset_ntcol, ept, zgemvt_offset_width, 5><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 6: gemvt_offset<cuDoubleComplex, zgemvt_offset_nb, zgemvt_offset_ntcol, ept, zgemvt_offset_width, 6><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 7: gemvt_offset<cuDoubleComplex, zgemvt_offset_nb, zgemvt_offset_ntcol, ept, zgemvt_offset_width, 7><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			case 8: gemvt_offset<cuDoubleComplex, zgemvt_offset_nb, zgemvt_offset_ntcol, ept, zgemvt_offset_width, 8><<<dimGrid, dimBlock, 0, stream>>>(rows_, cols_, alpha, dA, lda, dX, incx, beta, dY, incy, mod_r, mod_c, threshold, conj, offset_r_, offset_c_); break;
			default: printf("irregular part %d is not supported, please extend the case statement of zgemv\n", ept_); exit(1);
		}
	}
	else
	{	
		printf("ZGEMV error: Unrecognized transpose mode %c \n", trans);
		return -1;
	}
	
	return 0;
}

extern "C"
int kblas_zgemv2_offset(char trans, int rows, int cols,
				cuDoubleComplex alpha, cuDoubleComplex *dA, int lda, 
				cuDoubleComplex *dX, int incx, 
				cuDoubleComplex  beta, cuDoubleComplex *dY, int incy, 
				int offset_r, int offset_c)
{
	return kblas_zgemv2_offset_driver(trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, offset_r, offset_c, 0);
}

extern "C"
int kblas_zgemv2_offset_async(	char trans, int rows, int cols,
						cuDoubleComplex alpha, cuDoubleComplex *dA, int lda, 
						cuDoubleComplex *dX, int incx, 
						cuDoubleComplex  beta, cuDoubleComplex *dY, int incy, 
						int offset_r, int offset_c, 
						cudaStream_t stream)
{
	return kblas_zgemv2_offset_driver(	trans, rows, cols, alpha, dA, lda, dX, incx, beta, dY, incy, offset_r, offset_c, stream);
}
