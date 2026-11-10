/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/gemv2_core.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ahmad Abdelfattah
 * @date 2020-12-10
 **/

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "kblas_operators.h"
#include <stdio.h>

template <class T, int nb, int tcol, int ept, int width, int ept_>
__global__ void
gemvn(int rows, int cols, T alpha, T *A, int lda, T *x, int incx, T  beta, T *y, int incy, int mod_r, int mod_c, int threshold)
{
    const int	tx   = threadIdx.x ;
    const int	ty   = threadIdx.y ;
    const int	blkc = blockIdx.x ;
    const int	by	=	blockIdx.y;

    T res_1_	= make_zero<T>();
    T areg[ept];
	T breg[ept];

	__shared__ T la[nb * tcol];

    if(blkc == gridDim.x-1)
    {
    	if(mod_r > 0){if(tx >= mod_r)return;}
    }

    // number of full blocks to process
    int count = (cols/width)/gridDim.y + (by < (cols/width)%gridDim.y);

    {
    	int start = by * ((cols/width)/gridDim.y) + min(by, (cols/width)%gridDim.y);

    	// Advance 'A'
		A += nb * blkc;
		A += start * width * lda;

    	// Advance 'x'
    	x += start * width * incx;

    	// Advance 'y'
    	y += (blkc * nb) * incy;
    }

    if(by != gridDim.y-1){if(count == 0) return;}
    else {if(count == 0 && mod_c == 0) return;}

    const int j = ty * ept * lda + tx;

	if(count >= 2)
	{
		// read 1st block
		#pragma unroll
    	for(int k = 0; k < ept; k++)
			areg[k] = A[j + k * lda];
		A += width * lda;
	}

	int Vblocks = 0;
	#pragma unroll
    for(Vblocks = 0; Vblocks < (count/2)*2; Vblocks+=2)
    {
		// read 2nd block
		#pragma unroll
		for(int k = 0; k < ept; k++)
	    	breg[k] = A[j + k * lda];
	    A += width * lda;

	    // compute 1st
	    #pragma unroll
		for(int k = 0; k < ept; k++)
			res_1_ += areg[k] * x[(ty * ept + k) * incx];
	    x += width * incx;

		// prefetch 1st block
		if(Vblocks != ((count/2)*2-2) )
		{
			#pragma unroll
			for(int k = 0; k < ept; k++)
	  			areg[k] = A[j + k * lda];
	  		A += width * lda;
		}

		// compute 2nd
		#pragma unroll
		for(int k = 0; k < ept; k++)
	  		res_1_ += breg[k] * x[(ty * ept + k) * incx];
		x += width * incx;
	}

	if(count%2 >= 1)
	{
		//if(ty == 0 && tx == 0)printf("hi \n");
		// read the remaining block
		#pragma unroll
		for(int k = 0; k < ept; k++)
			areg[k] = A[j + k * lda];
		A += width * lda;

		// process remaining block
		#pragma unroll
		for(int k = 0; k < ept; k++)
			res_1_ += areg[k] * x[(ty * ept + k) * incx];
		x += width * incx;
	}

	//if(ty == 0 && tx == 0)printf("(%d, %d): by = %d\n ", tx, ty, by);
	//if(ty == 0 && tx == 0)printf("(%d, %d): mod_c = %d\n", tx, ty, mod_c);

	if(by == gridDim.y-1)
	{
		#pragma unroll
		for(int k = 0; k < ept; k++){breg[k] = make_zero<T>();}

		//if(ty == 0 && tx == 0)printf("mod_c = %d\n", mod_c);
		if(mod_c != 0)
		{
			if(ty < threshold)
			{
				#pragma unroll
				for(int k = 0; k < ept; k++)
					breg[k] = A[j + k * lda];
			}
			else if(ty == threshold)
			{
				#pragma unroll
				for(int k = 0; k < ept_; k++)
					breg[k] = A[j + k * lda];
			}

			// compute
			if(ty < threshold)
			{
				#pragma unroll
				for(int k = 0; k < ept; k++)
					res_1_ += breg[k] * x[(ty * ept + k) * incx];
			}
			else if (ty == threshold)
			{
				#pragma unroll
				for(int k = 0; k < ept_; k++)
					res_1_ += breg[k] * x[(ty * ept + k) * incx];
			}
			//if(ty == 0 && tx == 0)printf("hi 2\n");
			//int l;
			//#pragma unroll
			//for(l = 0; l < (mod_c/tcol); l++)
			//	breg[l] = A[l*tcol*lda + ty*lda + tx];
			//if(ty < (mod_c%tcol) )
			//	breg[l] = A[l*tcol*lda + ty*lda + tx];

			//#pragma unroll
			//for(l = 0; l < (mod_c/tcol); l++)
			//	res_1_ += breg[l] * x[(l*tcol + ty) * incx];
			//if(ty < (mod_c%tcol) )
			//	res_1_ += breg[l] * x[(l*tcol + ty) * incx];
		}
	}

	la[ty * nb + tx] = res_1_;
    __syncthreads();

    if(ty == 0)
    {
		res_1_ = make_zero<T>();
      	#pragma unroll
      	for(int k = 0; k < tcol; k++)
			res_1_ += la[k * nb + tx];
      	// use atomics
      	atomicAdd(&y[tx * incy], (alpha*res_1_));
      	//y[tx] = alpha * res_1_ + res;
    }
}


template <class T, int nb, int tcol, int ept, int width, int ept_>
__global__ void
gemvt(int rows, int cols, T alpha, T *A, int lda, T *x, int incx, T  beta, T *y, int incy, int mod_r, int mod_c, int threshold, int conj)
{
    const int	tx   = threadIdx.x ;
    const int	ty   = threadIdx.y ;
    const int	blkc = blockIdx.x ;
    const int	by	=	blockIdx.y;

    T res[ept]	= {make_zero<T>()};
    T areg[ept];
	T breg[ept];

	__shared__ T la[nb * width];

    if(blkc == gridDim.x-1 && mod_c != 0){if(ty > threshold) return;}

    // number of full blocks to process
    int count = (rows/nb)/gridDim.y + (by < (rows/nb)%gridDim.y);

    {
    	int start = by * ((rows/nb)/gridDim.y) + min(by, (rows/nb)%gridDim.y);

    	// Advance 'A'
		A += blkc * width * lda;
		A += start * nb;

    	// Advance 'x'
    	x += start * nb * incx;

    	// Advance 'y'
    	y += (blkc * width) * incy;
    }

    if(by != gridDim.y-1){if(count == 0) return;}

    const int j = ty * ept * lda + tx;

	const int irregular = ( (mod_c != 0) && (blkc == gridDim.x-1) && (ty == threshold) );

	if(count >= 2)
	{
		//if(blkc == 0 && by == 0 && tx == 0 && ty == 0)printf("hi-1\n");
		// read 1st block
		if(irregular)
		{
			#pragma unroll
    		for(int k = 0; k < ept_; k++)
				areg[k] = A[j + k * lda];
		}
		else
		{
			#pragma unroll
    		for(int k = 0; k < ept; k++)
				areg[k] = A[j + k * lda];
		}
		A += nb;
	}

	int Vblocks = 0;
	#pragma unroll
    for(Vblocks = 0; Vblocks < (count/2)*2; Vblocks+=2)
    {
    	//if(blkc == 0 && by == 0 && tx == 0 && ty == 0)printf("hi-2\n");
		// read 2nd block
		if(irregular)
		{
			#pragma unroll
			for(int k = 0; k < ept_; k++)
	    		breg[k] = A[j + k * lda];
		}
		else
		{
			#pragma unroll
			for(int k = 0; k < ept; k++)
	    		breg[k] = A[j + k * lda];
	    }
	    A += nb;

	    // compute 1st
	    if(irregular)
	    {
	    	#pragma unroll
			for(int k = 0; k < ept_; k++)
				res[k] += conj_if(conj, areg[k]) * x[tx * incx];
	    }
	    else
	    {
	    	#pragma unroll
			for(int k = 0; k < ept; k++)
				res[k] += conj_if(conj, areg[k]) * x[tx * incx];
	    }
	    x += nb * incx;

		// prefetch 1st block
		if(Vblocks != ((count/2)*2-2) )
		{
			if(irregular)
			{
				#pragma unroll
				for(int k = 0; k < ept_; k++)
	  				areg[k] = A[j + k * lda];
			}
			else
			{
				#pragma unroll
				for(int k = 0; k < ept; k++)
	  				areg[k] = A[j + k * lda];
	  		}
	  		A += nb;
		}

		// compute 2nd
		if(irregular)
		{
			#pragma unroll
			for(int k = 0; k < ept_; k++)
	  			res[k] += conj_if(conj, breg[k]) * x[tx * incx];
		}
		else
		{
			#pragma unroll
			for(int k = 0; k < ept; k++)
	  			res[k] += conj_if(conj, breg[k]) * x[tx * incx];
		}
		x += nb * incx;
	}

	if(count%2 >= 1)
	{
		//printf("hi from (%d, %d) \n", tx, ty);

		// read the remaining block
		if(irregular)
		{
			#pragma unroll
			for(int k = 0; k < ept_; k++)
				areg[k] = A[j + k * lda];
		}
		else
		{
			#pragma unroll
			for(int k = 0; k < ept; k++)
				areg[k] = A[j + k * lda];
		}
		A += nb;

		// process remaining block
		if(irregular)
		{
			#pragma unroll
			for(int k = 0; k < ept_; k++)
				res[k] += conj_if(conj, areg[k]) * x[tx * incx];
		}
		else
		{
			#pragma unroll
			for(int k = 0; k < ept; k++)
				res[k] += conj_if(conj, areg[k]) * x[tx * incx];
		}
		x += nb * incx;
	}

	if(by == gridDim.y-1)
	{
		#pragma unroll
		for(int k = 0; k < ept; k++){breg[k] = make_zero<T>();}

		//if(ty == 0 && tx == 0)printf("mod_c = %d\n", mod_c);

		//if(blkc == 0 && by == 0 && tx == 0 && ty == 0)printf("hi-4\n");
		if(tx < mod_r)
		{
			if(irregular)
			{
				#pragma unroll
				for(int k = 0; k < ept_; k++)
					breg[k] = A[j + k * lda];
			}
			else
			{
				#pragma unroll
				for(int k = 0; k < ept; k++)
					breg[k] = A[j + k * lda];
			}

			// compute
			if(irregular)
			{
				#pragma unroll
				for(int k = 0; k < ept_; k++)
					res[k] += conj_if(conj, breg[k]) * x[tx * incx];
			}
			else
			{
				#pragma unroll
				for(int k = 0; k < ept; k++)
					res[k] += conj_if(conj, breg[k]) * x[tx * incx];
			}
		}

	}

	#pragma unroll
	for(int k = 0; k < ept; k++)
		la[(ty*ept + k)*nb + tx] = res[k];
    __syncthreads();

    if(ty == 0 && tx < width)
    {
		T res_1_ = make_zero<T>();
      	#pragma unroll
      	for(int k = tx; k < (tx+nb); k++)
			res_1_ += la[tx * nb + k%nb];

      	// use atomics
      	if(mod_c != 0)
      	{
      		if(blkc == gridDim.x-1) {if(tx < mod_c)atomicAdd(&y[tx * incy], (alpha*res_1_));}
      		else atomicAdd(&y[tx * incy], (alpha*res_1_));
      	}
      	else {atomicAdd(&y[tx * incy], (alpha*res_1_));}
    }
}
