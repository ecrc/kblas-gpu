#include "hip/hip_runtime.h"
/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/syhemv_core.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ahmad Abdelfattah
 * @date 2018-11-14
 **/

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas.h>
#include "kblas_operators.h"
#include <stdio.h>


template <class T, int syhemv_bs, int thread_x, int thread_y, int elements_per_thread>
__global__ void
syhemvl_special_d( 	int n, T alpha,
				    T *A, int lda,
				    T *x, int incx,
				    T  beta,
				    T *y, int incy)
{
    const int	tx   = threadIdx.x ;
    const int	ty   = threadIdx.y ;
    const int td  = (thread_x * ty ) + tx;
    const int	blkc = blockIdx.x ;

    T res	= make_zero<T>();
    T yold	= make_zero<T>();
    //make_zero(&res);
    //make_zero(&yold);

    __shared__ T la   [syhemv_bs * syhemv_bs];
    __shared__ T buff [syhemv_bs];
    __shared__ T accum[syhemv_bs * (2 * thread_y)];

	// Advance 'A' to start of diagonal blocks first
	A += syhemv_bs * blkc * (lda + 1);

	// Advance 'A' to start row for each thread inside the diagonal block
	A += ty * lda + tx;

	// handle the case when incx and/or incy is -ve
	//if(incx < 0) x -= (n-1) * incx;
	//if(incy < 0) y -= (n-1) * incy;

	// Advance 'x'
	x += (blkc * syhemv_bs) * incx;

    // Advance 'y'
    y += (blkc * syhemv_bs) * incy;

	if(ty == 0)
	{
		yold = beta * y[incy * tx];
		buff[tx] = x[incx * tx];
	}

	// load first chunk
	#pragma unroll
	for(int k = 0; k < (syhemv_bs/2); k+= thread_y)
		la[td + k * syhemv_bs] = A[k * lda];

	// Advance to second chunk
	A += (syhemv_bs/2) * lda;
	// load second chunk
	if(tx >= (syhemv_bs/2))	// even warps will load un-necessary elements in the 2nd chunck og diagonal block
	{
	  #pragma unroll
	  for(int j = 0; j < (syhemv_bs/2); j+= thread_y)
		la[syhemv_bs * ((syhemv_bs/2) + j + ty) + tx] = A[j * lda];
	}

	__syncthreads();

	// mirror necessary elements in first chunk
	if(ty > tx)
		la[td] = conjugate( la[tx * syhemv_bs + ty] );
	else
		la[td] = la[td];

	#pragma unroll
	for(int k = thread_y; k < (syhemv_bs/2); k+= thread_y)
		if(abs(tx - ty) < k)
			la[tx + (ty + k) * syhemv_bs] = conjugate( la[ty + k + tx * syhemv_bs] );

	// mirror second chunk
	#pragma unroll
	for(int k = 0; k < (syhemv_bs/2); k+= thread_y)
		if(abs(tx-ty) < (k + (syhemv_bs/2)))
			la[syhemv_bs * ((syhemv_bs/2) + k + ty) + tx] = conjugate( la[syhemv_bs * tx + (syhemv_bs/2) + k + ty] );

	if(ty == 0) la[tx * syhemv_bs + tx] = make_real(la[tx * syhemv_bs + tx]);
	__syncthreads();

	// compute first chunk
	#pragma unroll
	for(int k = 0; k < (syhemv_bs/2); k+= thread_y)
		res += la[(ty + k) * syhemv_bs + tx] * buff[k + ty];

	// compute second chunk
	#pragma unroll
	for(int k = (syhemv_bs/2); k < 2 * (syhemv_bs/2); k+= thread_y)
		res += la[(ty + k) * syhemv_bs + tx] * buff[k + ty];

	accum[td] = res;

	__syncthreads();

	if(ty == 0)
	{
		res = make_zero<T>();
	  	#pragma unroll
	  	for(int k = 0; k < thread_y; k++)
			res += accum[k * syhemv_bs + tx];
		res *= alpha;
		res += yold;

		y[incy * tx] = res;
	}
}
/*******************************************************************************/
template <class T, int syhemv_bs, int thread_x, int thread_y, int elements_per_thread>
__global__ void
syhemvl_special_nd( 	int n, T alpha,
				    T *A, int lda,
				    T *x, int incx,
				    T  beta,
				    T *y, int incy)
{
    const int	tx   = threadIdx.x ;
    const int	ty   = threadIdx.y ;
    const int	blkc = blockIdx.x ;
    const int	by 	= blockIdx.y;
    const int	td  = (thread_x * ty ) + tx;
    const int	tx_  = td % (syhemv_bs/2);
    const int	ty_  = td / (syhemv_bs/2);
    T		* xcopy, *ycopy;

    // compute how many matrix blocks to be processed
	int count = (gridDim.x-blkc-1)/gridDim.y;
	//int count = (gridDim.x-blkc-1)/gridDim.y;
	//if(by < (gridDim.x-blkc-1)%gridDim.y) count++;

	T xreg[elements_per_thread];
	T areg[elements_per_thread];
	T treg[elements_per_thread] = { make_zero<T>()};

	//#pragma unroll
	//for(int k = 0; k < elements_per_thread; k++) make_zero(&treg[k]);

    __shared__ T la   [syhemv_bs * (syhemv_bs/2)];
    __shared__ T accum[syhemv_bs * (2 * thread_y)];
    __shared__ T xbuff[syhemv_bs];

    if(blkc == gridDim.x-1)return;

	{
		// compute number of preceding blocks
		//int pr = by*(gridDim.x-blkc-1)/gridDim.y + min(by, (gridDim.x-blkc-1)%gridDim.y);

		// Advance 'A' to start of diagonal blocks first
		A += syhemv_bs * blkc * (lda + 1);
		// divide work among the y-direction of the grid
		A += (by * count) * syhemv_bs;

		// Advance 'x'
		x += (blkc * syhemv_bs) * incx;
		xcopy = x;
    	x += (by * count * syhemv_bs) * incx;

    	if(ty == 0) xbuff[tx] = xcopy[tx * incx];

    	// Advance 'y'
    	y += (blkc * syhemv_bs) * incy;
    	ycopy = y;
    	ycopy += (by * count * syhemv_bs) * incy;
    }
    if(by == gridDim.y-1) count += (gridDim.x-blkc-1)%gridDim.y;
	if(count == 0) return;

	T res_1_	= make_zero<T>();
    T res_2_	= make_zero<T>();
    T x1		= make_zero<T>();
    T x2		= make_zero<T>();
	const int j = ty_ * elements_per_thread * lda + tx_;

	A += syhemv_bs;
    x += syhemv_bs * incx;

	__syncthreads();

	// read upper
	#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
		xreg[k] = A[j + k * lda];


	#pragma unroll
    for(int Vblocks = 0; Vblocks < count /*gridDim.x-blkc-1*/; Vblocks++)
    {

		res_1_	=	make_zero<T>();
		res_2_	=	make_zero<T>();

		x1 = x[incx * tx_];
		x2 = x[incx * (tx_ + (syhemv_bs/2))];

		// read lower
		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
	    	areg[k] = A[(syhemv_bs/2) + j + k * lda];

	    // compute upper
	    #pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
		{
	    	res_1_ += xreg[k] * xbuff[ty_ * elements_per_thread + k];
	    	treg[k] += conjugate( xreg[k] ) * x1;
		}

		A += syhemv_bs;
		x += syhemv_bs * incx;

		// read upper from next block
		if(Vblocks != count-1 /*(gridDim.x-blkc-1)-1*/)
		{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			xreg[k] = A[j + k * lda];
		}

		// compute lower
		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
		{
	  		res_2_ 	+= areg[k] * xbuff[ty_ * elements_per_thread + k]; //xcopy[incx * (ty_ * elements_per_thread + k)];
	  		treg[k] += conjugate( areg[k] ) * x2;
		}

		// Horizontal block should be stored in global memory
		__syncthreads();
		accum[ty_ * syhemv_bs + tx_] = res_1_;
		accum[ty_ * syhemv_bs + tx_ + (syhemv_bs/2)] = res_2_;
		__syncthreads();
		if(ty == 0)
		{
			ycopy += syhemv_bs * incy;
	    	res_1_ = make_zero<T>();
	    	#pragma unroll
	    	for(int k = 0; k < (2 * thread_y); k++)
	      		res_1_ += accum[k * syhemv_bs + tx];

	    	res_1_ *= alpha;
	    	// use atomics
	    	atomicAdd(&ycopy[incy * tx], res_1_);
	    }
	}

	// reduction of treg
	#pragma unroll
	for(int k = 0; k < elements_per_thread; k++)
	  	la[(ty_ * elements_per_thread + k) * (syhemv_bs/2) + tx_] = treg[k];

	__syncthreads();

	if(blkc != gridDim.x-1)
	{
		if(ty == 0)
		{
			treg[0] = make_zero<T>(); 			// as a temporary accumulator
	  		#pragma unroll
	    	for(int j = tx; j < tx+(syhemv_bs/2); j++)
	      		treg[0] += la[tx * (syhemv_bs/2) +  (j % (syhemv_bs/2))];

	      	treg[0] *= alpha;
	      	// use atomics
	      	atomicAdd(&y[incy * tx], treg[0]);
	  	}
	}
}
/*******************************************************************************/
template <class T, int syhemv_bs, int thread_x, int thread_y, int elements_per_thread>
__global__ void
syhemvl_generic_d( int n, T alpha,
				    T *A, int lda,
				    T *x, int incx,
				    T  beta,
				    T *y, int incy,
				    int	    n_mod_syhemv_bs)
{
    const int tx   = threadIdx.x ;
    const int ty   = threadIdx.y ;
    const int blkc = blockIdx.x ;
    const int td  = (thread_x * ty ) + tx;

    T res  = make_zero<T>();
    T yold = make_zero<T>();

    __shared__ T la   [syhemv_bs * syhemv_bs];
    __shared__ T buff [syhemv_bs];
    __shared__ T accum[syhemv_bs * (2 * thread_y)];

	// Advance 'A' to start of diagonal blocks first
	A += syhemv_bs * blkc * (lda + 1);

	// Advance 'A' to start row for each thread inside the diagonal block
	A += ty * lda + tx;

	// handle the case when incx and/or incy is -ve
	//if(incx < 0) x -= (n-1) * incx;
	//if(incy < 0) y -= (n-1) * incy;

	// Advance x
	x += (blkc * syhemv_bs) * incx;

	// Advacne y
	y += (blkc * syhemv_bs) * incy;

	// load part of vector x
	if(blkc == gridDim.x-1)
	{
		if(ty == 0)
		{
	  		if(tx < n_mod_syhemv_bs)
	  		{
	    		buff[tx] = x[incx * tx];
	    		yold = beta * y[tx * incy];
	    	}
	    	else
	    	{
	    		buff[tx] = make_zero<T>();
	    		yold = make_zero<T>();
	    	}
	  	}
	}
	else
	{
	  	if(ty == 0)
	  	{
			buff[tx] = x[incx * tx];
			yold = beta * y[tx * incy];
		}
	} // end of load part of vector x

	// init shmem (last TB only)
	if(blkc == gridDim.x-1)
	{
		#pragma unroll
		for(int j = 0; j < syhemv_bs; j+= thread_y)
			la[j * syhemv_bs + td ] = make_zero<T>();
		__syncthreads();

		if(tx >= n_mod_syhemv_bs) return; 	// these threads should not read any useful data
	}

	// load a bock of data
	if(blkc == gridDim.x-1)
	{
		int j;
		#pragma unroll
		for(j = 0; j < n_mod_syhemv_bs/thread_y; j++)
			la[(j*thread_y) * syhemv_bs + td] = A[(j*thread_y) * lda];

		if(ty < (n_mod_syhemv_bs%thread_y))
			la[(j*thread_y) * syhemv_bs + td] = A[(j*thread_y) * lda];
	}
	else
	{
		#pragma unroll
		for(int j = 0; j < syhemv_bs; j+= thread_y)
			la[j * syhemv_bs + td] = A[j * lda];
	}
	// end of reading a diagonal block of data

	__syncthreads();

	// mirror necessary elements in first chunk
	if(ty > tx)
		la[td] = conjugate( la[tx * syhemv_bs + ty] );
	else
		la[td] = la[td];

	#pragma unroll
	for(int j = thread_y; j < (syhemv_bs/2); j+= thread_y)
		if(abs(tx - ty) < j)
			la[tx + (ty + j) * syhemv_bs] = conjugate( la[ty + j + tx * syhemv_bs] );

	// mirror second chunk
	#pragma unroll
	for(int j = 0; j < (syhemv_bs/2); j+= thread_y)
		if(abs(tx-ty) < (j + (syhemv_bs/2)))
			la[syhemv_bs * ((syhemv_bs/2) + j + ty) + tx] = conjugate( la[syhemv_bs * tx + (syhemv_bs/2) + j + ty] );

	// ignore imaginary part of diagonal elements
	if(ty == 0) la[tx * syhemv_bs + tx] = make_real(la[tx * syhemv_bs + tx]);

	__syncthreads();

	// compute first chunk
	#pragma unroll
	for(int j = 0; j < (syhemv_bs/2); j+= thread_y)
		res += la[(ty + j) * syhemv_bs + tx] * buff[j + ty];

	// compute second chunk
	#pragma unroll
	for(int j = (syhemv_bs/2); j < 2 * (syhemv_bs/2); j+= thread_y)
		res += la[(ty + j) * syhemv_bs + tx] * buff[j + ty];

	accum[td] = res;
	__syncthreads();
	if(ty == 0)
	{
		res = make_zero<T>();
	  	#pragma unroll
	  	for(int j = 0; j < thread_y; j++)
			res += accum[j * syhemv_bs + tx];
	  	res *= alpha;
	  	res += yold;
	  	if(blkc == gridDim.x-1){if(tx < n_mod_syhemv_bs)y[tx * incy] = res;}
	  	else{y[tx * incy] = res;}
	}
}
/*****************************************************************************************/
template <class T, int syhemv_bs, int thread_x, int thread_y, int elements_per_thread >
__global__ void
syhemvl_generic_nd( int n, T alpha,
                               T *A, int lda,
                               T *x, int incx,
                               T  beta,
                               T *y, int incy,
								int     n_mod_syhemv_bs)
{
    const int tx   = threadIdx.x ;
    const int ty   = threadIdx.y ;
    const int blkc = blockIdx.x ;
    const int by 	= blockIdx.y;
    const int td  = (thread_x * ty ) + tx;
    const int tx_  = td % (syhemv_bs/2);
    const int ty_  = td / (syhemv_bs/2);
    T *xcopy, *ycopy;

    int count = (gridDim.x-blkc-1-1)/gridDim.y;

    T xreg[elements_per_thread];
    T areg[elements_per_thread];
    T treg[elements_per_thread] = {make_zero<T>()};

    T res_1_	= make_zero<T>();
    T res_2_	= make_zero<T>();
    T x1		= make_zero<T>();
    T x2		= make_zero<T>();

    __shared__ T la   [syhemv_bs * (syhemv_bs/2)];
    __shared__ T accum[syhemv_bs * (2 * thread_y)];
    __shared__ T xbuff[syhemv_bs];

    if(blkc == gridDim.x - 1)return;

    // Advance 'A' to start of diagonal blocks first
    A += syhemv_bs * blkc * (lda + 1);
    // divide work among the y-direction of the grid
	A += (by * count) * syhemv_bs;

    // Advance 'x'
    x += (blkc * syhemv_bs) * incx;
    xcopy = x;
    x += (by * count * syhemv_bs) * incx;

    if(ty == 0) xbuff[tx] = xcopy[incx * tx];

    //Advance 'y'
	y += (blkc * syhemv_bs) * incy;
    ycopy = y;
    ycopy += (by * count * syhemv_bs) * incy;

    if(by == gridDim.y-1) count += ((gridDim.x-blkc-1-1)%gridDim.y) ;//- 1;	// -1 for the generic block at the bottom
    if(by != gridDim.y-1){if(count == 0) return;}

	int j = ty_ * elements_per_thread * lda + tx_;

	__syncthreads();

    A += syhemv_bs;
    x += syhemv_bs * incx;

    if(blkc < gridDim.x-2)		// to prevent out of bound access
    {
    	#pragma unroll
    	for(int k = 0; k < elements_per_thread; k++)
			xreg[k] = A[j + k * lda];
    	x1 = x[incx * tx_];
    }

    A -= syhemv_bs;
    x -= syhemv_bs * incx;

    #pragma unroll
    for(int Vblocks = 0; Vblocks < count /*(gridDim.x-blkc-1)-1*/; Vblocks++)
    {
		A += syhemv_bs;
		x += syhemv_bs * incx;

		res_1_ = make_zero<T>();
		res_2_ = make_zero<T>();

		x2 = x[incx * (tx_ + (syhemv_bs/2))];

		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
	    	areg[k] = A[(syhemv_bs/2) + j + k * lda];

		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
		{
	    	res_1_ += xreg[k] * xbuff[ty_ * elements_per_thread + k]; //xcopy[incx* (ty_ * elements_per_thread + k)];
	    	treg[k] += conjugate( xreg[k] ) * x1;
		}

		A += syhemv_bs;
		x += syhemv_bs * incx;

		if(Vblocks != count-1 /*((gridDim.x-blkc-1)-1)-1*/)
		{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			xreg[k] = A[j + k * lda];
	  		x1 = x[incx * tx_];
	  	}

		A -= syhemv_bs;
		x -= syhemv_bs * incx;

		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
		{
	  		res_2_ += areg[k] * xbuff[ty_ * elements_per_thread + k]; //xcopy[incx * (ty_ * elements_per_thread + k)];
	  		treg[k] += conjugate( areg[k] ) * x2;
		}

		// Horizontal block should be stored in global memory
		__syncthreads();
		accum[ty_ * syhemv_bs + tx_] = res_1_;
		accum[ty_ * syhemv_bs + tx_ + (syhemv_bs/2)] = res_2_;
		__syncthreads();
		if(ty == 0)
		{
			ycopy += syhemv_bs * incy;
	   		res_1_ = make_zero<T>();
	   		#pragma unroll
	   		for(int k = 0; k < (2 * thread_y); k++)
	    		res_1_ += accum[k * syhemv_bs + tx];

	    	res_1_ *= alpha;
	    	// use atomics
	    	atomicAdd(&ycopy[incy * tx], res_1_);
		}
    }// end of for loop on blocks

    //////////////////////////////////////////////////
    // last irregular tile
    if(by == gridDim.y-1)
    {
    	res_1_ = make_zero<T>();
    	res_2_ = make_zero<T>();

		A += syhemv_bs;
		x += syhemv_bs * incx;

    	#pragma unroll
    	for(int k = 0; k < elements_per_thread; k++)
    	{
    		xreg[k] = make_zero<T>();
    		areg[k] = make_zero<T>();
    	}

    	if(tx_ < n_mod_syhemv_bs)
    	{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
				xreg[k] = A[j + k * lda];

			x1 = x[incx * tx_];
		}

		if( (tx_+(syhemv_bs/2)) < n_mod_syhemv_bs)
		{
			#pragma unroll
    		for(int k = 0; k < elements_per_thread; k++)
				areg[k] = A[(syhemv_bs/2) + j + k * lda];

			x2 = x[incx * (tx_ + (syhemv_bs/2))];
		}

    	#pragma unroll
    	for(int k = 0; k < elements_per_thread; k++)
    	{
			res_1_ 	+= xreg[k] * xbuff[ty_ * elements_per_thread + k]; //xcopy[incx * (ty_ * elements_per_thread + k)];
			treg[k] += conjugate( xreg[k] ) * x1;
		}

		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
		{
			res_2_	+= areg[k] * xbuff[ty_ * elements_per_thread + k]; //xcopy[incx * (ty_ * elements_per_thread + k)];
			treg[k] += conjugate( areg[k] ) * x2;
		}

    	// Horizontal block reduction
		__syncthreads();
		accum[ty_ * syhemv_bs + tx_] = res_1_;
		accum[ty_ * syhemv_bs + tx_ + (syhemv_bs/2)] = res_2_;
		__syncthreads();
		if(ty == 0)
		{
			ycopy += syhemv_bs * incy;
	   		res_1_ = make_zero<T>();;
	   		#pragma unroll
	   		for(int k = 0; k < (2 * thread_y); k++)
	    		res_1_ += accum[k * syhemv_bs + tx];

	    	res_1_ *= alpha;
	    	// use atomics
	    	if(tx < n_mod_syhemv_bs)atomicAdd(&ycopy[incy * tx], res_1_);
		}
	}

	#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
    	la[(ty_ * elements_per_thread + k) * (syhemv_bs/2) + tx_] = treg[k];

    __syncthreads();		// important

    if(ty == 0)
    {
		treg[0] = make_zero<T>(); // tmp accumulator
		#pragma unroll
		for(int j = tx; j < tx+(syhemv_bs/2); j++)
	  		treg[0] += la[tx * (syhemv_bs/2) +  (j % (syhemv_bs/2))];

	   	treg[0] *= alpha;
	   	atomicAdd(&y[incy * tx], treg[0]);
	}
}
/*******************************************************************************/
template <class T, int syhemv_bs, int thread_x, int thread_y, int elements_per_thread>
__global__ void
syhemvu_special_d( int n, T alpha,
				    T *A, int lda,
				    T *x, int incx,
				    T  beta,
				    T *y, int incy)
{
    const int tx   = threadIdx.x ;
    const int ty   = threadIdx.y ;
    const int blkc = blockIdx.x ;
	const int td  = (thread_x * ty ) + tx;

    T res  = make_zero<T>();
	T yold = make_zero<T>();

    __shared__ T la   [syhemv_bs * syhemv_bs];
    __shared__ T buff [syhemv_bs];
    __shared__ T accum[syhemv_bs * (2 * thread_y)];

	// Advance 'A' to start of diagonal blocks first
	A += syhemv_bs * blkc * (lda + 1);
	// Advance 'A' to start row for each thread inside the diagonal block
	A += ty * lda + tx;

	// handle the case when incx and/or incy is -ve
	//if(incx < 0) x -= (n-1) * incx;
	//if(incy < 0) y -= (n-1) * incy;

	// Advance 'x'
	x += (blkc * syhemv_bs) * incx;

	// Advance 'y'
	y += (blkc * syhemv_bs) * incy;

	if(ty == 0)
	{
		buff[tx] = x[incx * tx];
		yold = beta * y[incy * tx];
	}


	// load first chunk
	if(tx < (syhemv_bs/2))
	{
		#pragma unroll
		for(int j = 0; j < (syhemv_bs/2); j+= thread_y)
			la[td + j * syhemv_bs] = A[j * lda];
	}

	// Advance to second chunk
	A += (syhemv_bs/2) * lda;

	// load second chunk first
	#pragma unroll
	for(int j = 0; j < (syhemv_bs/2); j+= thread_y)
		la[syhemv_bs * ((syhemv_bs/2) + j + ty) + tx] = A[j * lda];

	__syncthreads();

	// mirror second chunk
	#pragma unroll
	for(int j = 0; j < (syhemv_bs/2); j+= thread_y)
		if(abs(tx-ty) > (j + (syhemv_bs/2)))
			la[syhemv_bs * ((syhemv_bs/2) + j + ty) + tx] = conjugate( la[syhemv_bs * tx + (syhemv_bs/2) + j + ty] );

	// mirror first chunk
	if(ty > tx)
		la[td] = la[td];
	else
		la[td] = conjugate( la[tx * syhemv_bs + ty] );

	#pragma unroll
	for(int j = thread_y; j < (syhemv_bs/2); j+= thread_y)
		if(abs(tx - ty) > j)
			la[tx + (ty + j) * syhemv_bs] = conjugate( la[ty + j + tx * syhemv_bs] );

	// ignore imaginary part of diagonal elements
	if(ty == 0) la[tx * syhemv_bs + tx] = make_real(la[tx * syhemv_bs + tx]);

	__syncthreads();

	// compute first chunk
	#pragma unroll
	for(int j = 0; j < (syhemv_bs/2); j+= thread_y)
		res += la[(ty + j) * syhemv_bs + tx] * buff[j + ty];

	// compute second chunk
	#pragma unroll
	for(int j = (syhemv_bs/2); j < 2 * (syhemv_bs/2); j+= thread_y)
		res += la[(ty + j) * syhemv_bs + tx] * buff[j + ty];

	accum[td] = res;
	__syncthreads();

	if(ty == 0)
	{
		res = make_zero<T>();
	  	#pragma unroll
	  	for(int j = 0; j < thread_y; j++)
			res += accum[j * syhemv_bs + tx];

		res *= alpha;
		y[tx * incy] = yold + res;
	}
}
/***************************************************************************************/
template <class T, int syhemv_bs, int thread_x, int thread_y, int elements_per_thread>
__device__ __inline__ void
fn_syhemvu_special_nd( int n, T alpha,
                               T *A, int lda,
                               T *x, int incx,
                               T  beta,
                               T *y, int incy)
{
    const int tx   = threadIdx.x ;
    const int ty   = threadIdx.y ;
    const int td  = (thread_x * ty ) + tx;
    const int tx_  = td % (syhemv_bs/2);
    const int ty_  = td / (syhemv_bs/2);
    const int blkc = blockIdx.x;
    const int by	= blockIdx.y;

    // compute how many matrix blocks to be processed
	int count = blkc/gridDim.y;

    T res_1_  	= make_zero<T>();
    T res_2_	= make_zero<T>();
    T x1, x2;

    T* xcopy, *ycopy;

    __shared__ T la   [syhemv_bs * (syhemv_bs/2)];
    __shared__ T accum[syhemv_bs * (2 * thread_y)];
    __shared__ T xbuff[syhemv_bs];

    T xreg[elements_per_thread];
    T areg[elements_per_thread];
    T treg[elements_per_thread] = {make_zero<T>()};

    // Advance 'A' to the start a non-diagonal block
    A += syhemv_bs * blkc * lda;
    // divide the work among y-direction of the grid
    A += (by * count) * syhemv_bs;

    xcopy = x + (blkc * syhemv_bs) * incx;
	x += (by * count * syhemv_bs) * incx;

	if(blkc == 0)return;

	if(ty == 0) xbuff[tx] = xcopy[tx * incx];

	// Advance 'y'
	ycopy = y;
	y += (blkc * syhemv_bs) * incy;
    ycopy += (by * count * syhemv_bs) * incy;

	if(by == gridDim.y-1) count += blkc%gridDim.y;
	if(count == 0) return;
    const int j = ty_ * elements_per_thread * lda + tx_;

	__syncthreads();

	// prefetch upper
    #pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
		xreg[k] = A[j + k * lda];

    x1 = x[incx * tx_];

    #pragma unroll
    for(int Vblocks = 0; Vblocks < count /*blkc*/; Vblocks++)
    {

		res_1_ = make_zero<T>();
		res_2_ = make_zero<T>();

		x2 = x[incx * (tx_ + (syhemv_bs/2))];

		// prefetch lower
		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
	    	areg[k] = A[(syhemv_bs/2) + j + k * lda];

		// compute upper
		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
		{
	    	res_1_ += xreg[k] * xbuff[ty_ * elements_per_thread + k]; //xcopy[incx* (ty_ * elements_per_thread + k)];
	    	treg[k] += conjugate( xreg[k] ) * x1;
		}

		// Advance to next block
		A += syhemv_bs;
		x += syhemv_bs * incx;

		if(Vblocks != count-1 /*blkc-1*/)
		{
			// prefetch upper of next block
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			xreg[k] = A[j + k * lda];

			x1 = x[incx * tx_];
		}

		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
		{
	  		res_2_ += areg[k] * xbuff[ty_ * elements_per_thread + k]; //xcopy[incx * (ty_ * elements_per_thread + k)];
	  		treg[k] += conjugate( areg[k] ) * x2;
		}

		// Horizontal block should be stored in global memory
		__syncthreads();
		accum[ty_ * syhemv_bs + tx_] = res_1_;
		accum[ty_ * syhemv_bs + tx_ + (syhemv_bs/2)] = res_2_;
		__syncthreads();
		if(ty == 0)
		{
	    	res_1_ = make_zero<T>();;
	    	#pragma unroll
	    	for(int k = 0; k < (2 * thread_y); k++)
	      		res_1_ += accum[k * syhemv_bs + tx];

	    	res_1_ *= alpha;
	    	// use atomics
	    	atomicAdd(&ycopy[incy * tx], res_1_);
	    	ycopy += syhemv_bs * incy;
	    }
    }// end of for loop on blocks


	#pragma unroll
	for(int k = 0; k < elements_per_thread; k++)
	  la[(ty_ * elements_per_thread + k) * (syhemv_bs/2) + tx_] = treg[k];

	__syncthreads();		// important

	if(ty == 0)
	{
		treg[0] = make_zero<T>();
		#pragma unroll
		for(int j = tx; j < tx+(syhemv_bs/2); j++)
			treg[0] += la[tx * (syhemv_bs/2) +  (j % (syhemv_bs/2))];

		treg[0] *= alpha;
		// use atomics
		atomicAdd(&y[tx * incy], treg[0]);
	}
}
/***************************************************************************************/
template <class T, int syhemv_bs, int thread_x, int thread_y, int elements_per_thread>
__global__ void
syhemvu_special_nd( int n, T alpha,
                               T *A, int lda,
                               T *x, int incx,
                               T  beta,
                               T *y, int incy)
{
	fn_syhemvu_special_nd<T, syhemv_bs, thread_x, thread_y, elements_per_thread>(n, alpha, A, lda, x, incx, beta, y, incy);
}
/***************************************************************************************/
template <class T, int syhemv_bs, int thread_x, int thread_y, int elements_per_thread>
__global__ void
syhemvu_generic_d( int n, T alpha,
				    T *A, int lda,
				    T *x, int incx,
				    T  beta,
				    T *y, int incy,
				    int	    n_mod_syhemv_bs)
{
    const int tx   = threadIdx.x ;
    const int ty   = threadIdx.y ;
    const int blkc = blockIdx.x ;
	const int td  = (thread_x * ty ) + tx;

   	T res  = make_zero<T>();
	T yold = make_zero<T>();

	__shared__ T la   [syhemv_bs * syhemv_bs];
    __shared__ T buff [syhemv_bs];
    __shared__ T accum[syhemv_bs * (2 * thread_y)];

	// Advance 'A' to start of diagonal blocks first
	A += syhemv_bs * blkc * (lda + 1);

	// Advance 'A' to start row for each thread inside the diagonal block
	A += ty * lda + tx;

	// handle the case when incx and/or incy is -ve
	//if(incx < 0) x -= (n-1) * incx;
	//if(incy < 0) y -= (n-1) * incy;

	// Advance 'x'
	x += (blkc * syhemv_bs) * incx;

	// Advance 'y'
	y += (blkc * syhemv_bs) * incy;

	// load part of vector 'x'
	if(blkc == gridDim.x-1)		// last TB
	{
		if(ty == 0)
		{
			if (tx < n_mod_syhemv_bs)
	  		{
	  			buff[tx] = x[incx * tx];
	  			yold = beta * y[incy * tx];
	  		}
	  		else
	  		{buff[tx] = make_zero<T>();}
	  	}
	}
	else		// not the last TB
	{
		if(ty == 0)
		{
			buff[tx] = x[incx * tx];
			yold = beta * y[incy * tx];
		}
	}

	// init shmem (last TB only)
	if(blkc == gridDim.x-1)
	{
		#pragma unroll
		for(int j = 0; j < syhemv_bs; j+= thread_y)
			la[j * syhemv_bs + td ] = make_zero<T>();
		__syncthreads();

		if(tx >= n_mod_syhemv_bs) return; 	// these threads should not read any useful data
	}

	// load a bock of data
	if(blkc == gridDim.x-1)
	{
		int j;
		#pragma unroll
		for(j = 0; j < n_mod_syhemv_bs/thread_y; j++)
			la[(j*thread_y) * syhemv_bs + td] = A[(j*thread_y) * lda];

		if(ty < (n_mod_syhemv_bs%thread_y))
			la[(j*thread_y) * syhemv_bs + td] = A[(j*thread_y) * lda];
	}
	else
	{
		#pragma unroll
		for(int j = 0; j < syhemv_bs; j+= thread_y)
			la[j * syhemv_bs + td] = A[j * lda];
	}
	// end of reading a diagonal block of data

	__syncthreads();

	// mirror second chunk
	#pragma unroll
	for(int j = 0; j < (syhemv_bs/2); j+= thread_y)
		if(abs(tx-ty) > (j + (syhemv_bs/2)))
			la[syhemv_bs * ((syhemv_bs/2) + j + ty) + tx] = conjugate( la[syhemv_bs * tx + (syhemv_bs/2) + j + ty] );

	// mirror elements in first chunk
	if(ty > tx)
		la[td] = la[td];
	else
		la[td] = conjugate( la[tx * syhemv_bs + ty] );

	#pragma unroll
	for(int j = thread_y; j < (syhemv_bs/2); j+= thread_y)
		if(abs(tx - ty) > j)
			la[tx + (ty + j) * syhemv_bs] = conjugate( la[ty + j + tx * syhemv_bs] );

	// ignore imaginary part of diagonal elements
	if(ty == 0) la[tx * syhemv_bs + tx] = make_real(la[tx * syhemv_bs + tx]);

	__syncthreads();

	// compute first chunk
	#pragma unroll
	for(int j = 0; j < (syhemv_bs/2); j+= thread_y)
		res += la[(ty + j) * syhemv_bs + tx] * buff[j + ty];

	// compute second chunk
	#pragma unroll
	for(int j = (syhemv_bs/2); j < 2 * (syhemv_bs/2); j+= thread_y)
		res += la[(ty + j) * syhemv_bs + tx] * buff[j + ty];

	accum[td] = res;
	__syncthreads();

	if(ty == 0)
	{
		res  = make_zero<T>();
		#pragma unroll
		for(int j = 0; j < thread_y; j++)
			res += accum[j * syhemv_bs + tx];
		res *= alpha;
		res += yold;
		if(blkc == gridDim.x-1){if(tx < n_mod_syhemv_bs)y[tx * incy] = res;}
	  	else{y[tx * incy] = res;}
	}
}
/*****************************************************************************************/
template <class T, int syhemv_bs, int thread_x, int thread_y, int elements_per_thread, int irregular_part>
__device__ __inline__ void
fn_syhemvu_generic_nd(int n, T alpha,
                            T *A, int lda,
                        	T *x, int incx,
                        	T  beta,
                        	T *y, int incy,
							int n_mod_syhemv_bs)
{
    const int tx   = threadIdx.x ;
    const int ty   = threadIdx.y ;
    const int td  = (thread_x * ty ) + tx;
    const int tx_  = td % (syhemv_bs/2);
    const int ty_  = td / (syhemv_bs/2);
	const int blkc = blockIdx.x;
	const int by = blockIdx.y;

    // compute how many matrix blocks to be processed
	int count = blkc/gridDim.y;

    T res_1_ = make_zero<T>();
    T res_2_ = make_zero<T>();
    T x1, x2;

    T *xcopy, *ycopy;

    __shared__ T la   [syhemv_bs * (syhemv_bs/2)];
    __shared__ T accum[syhemv_bs * (2 * thread_y)];
    __shared__ T xbuff[syhemv_bs];

    T xreg[elements_per_thread] = {make_zero<T>()};
    T areg[elements_per_thread] = {make_zero<T>()};
    T treg[elements_per_thread] = {make_zero<T>()};

    // Advance 'A' to the start a non-diagonal block
    A += syhemv_bs * blkc * lda;
	// divide the work among y-direction of the grid
    A += (by * count) * syhemv_bs;

	// Advance 'x'
    xcopy = x + (blkc * syhemv_bs) * incx;
	x += (by * count * syhemv_bs) * incx;

	// Advance 'y'
	ycopy = y;
	y += (blkc * syhemv_bs) * incy;
	ycopy += (by * count * syhemv_bs) * incy;

	if(blkc == 0) return;

	if(by == gridDim.y-1) count += blkc%gridDim.y;
	if(count == 0) return;

	// useful for ltb (last thread block only)
	const int num_active_thread_cols = n_mod_syhemv_bs/elements_per_thread;

	// cache part of 'x' needed for computing res_1_ and res_2_
	if(blkc == gridDim.x-1)		//last TB
	{
		if(ty == 0)
		{
			if(tx < n_mod_syhemv_bs)xbuff[tx] = xcopy[tx * incx];
			else xbuff[tx] = make_zero<T>();
		}
		// init shmem arrays to zeros
		accum[ty_ * syhemv_bs + tx_] = make_zero<T>();
		accum[ty_ * syhemv_bs + tx_ + (syhemv_bs/2)] = make_zero<T>();
		for(int k = 0; k < elements_per_thread; k++)
			la[(ty_ * elements_per_thread + k) * (syhemv_bs/2) + tx_] = make_zero<T>();
		int tmp = max(2, num_active_thread_cols);
		if(ty_ > tmp) return;		// we need at least two thread columns to do final reduction
	}
	else		// not the last TB
	{
		if(ty == 0) xbuff[tx] = xcopy[tx * incx];
	}

	__syncthreads();

    const int j = ty_ * elements_per_thread * lda + tx_;

	// prefetch upper
	if(blkc == gridDim.x-1)		// last TB "irregular"
	{
		if(ty_ < num_active_thread_cols)
		{
			#pragma unroll
    		for(int k = 0; k < elements_per_thread; k++)
				xreg[k] = A[j + k * lda];
		}
		else if(ty_ == num_active_thread_cols)
		{
			#pragma unroll
    		for(int k = 0; k < irregular_part; k++)
				xreg[k] = A[j + k * lda];
		}
	}
	else	// not last TB
	{
    	#pragma unroll
    	for(int k = 0; k < elements_per_thread; k++)
			xreg[k] = A[j + k * lda];
	}
    x1 = x[incx * tx_];

    #pragma unroll
    for(int Vblocks = 0; Vblocks < count /*blkc*/; Vblocks++)
    {

		res_1_ = make_zero<T>();
		res_2_ = make_zero<T>();

		x2 = x[incx * (tx_ + (syhemv_bs/2))];

		// prefetch lower
		if(blkc == gridDim.x-1)
		{
			if(ty_ < num_active_thread_cols)
			{
				#pragma unroll
				for(int k = 0; k < elements_per_thread; k++)
	    			areg[k] = A[(syhemv_bs/2) + j + k * lda];
	    	}
	    	else if(ty_ == num_active_thread_cols)
	    	{
	    		#pragma unroll
				for(int k = 0; k < irregular_part; k++)
	    			areg[k] = A[(syhemv_bs/2) + j + k * lda];
	    	}
		}
		else
		{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	    		areg[k] = A[(syhemv_bs/2) + j + k * lda];
	    }// end of prefetch lower

		// compute upper
		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
		{
	    	res_1_  += xreg[k] * xbuff[ty_ * elements_per_thread + k]; //xcopy[incx* (ty_ * elements_per_thread + k)];
	    	treg[k] += conjugate( xreg[k] ) * x1;
		}

		// Advance to next block
		A += syhemv_bs;
		x += syhemv_bs * incx;

		// prefetch upper of next block
		if(Vblocks != count-1 /*blkc-1*/)
		{
			if(blkc == gridDim.x-1)		// last TB "irregular"
			{
				if(ty_ < num_active_thread_cols)
				{
					#pragma unroll
    				for(int k = 0; k < elements_per_thread; k++)
						xreg[k] = A[j + k * lda];
				}
				else if(ty_ == num_active_thread_cols)
				{
					#pragma unroll
    				for(int k = 0; k < irregular_part; k++)
						xreg[k] = A[j + k * lda];
				}
			}
			else	// not last TB
			{
    			#pragma unroll
    			for(int k = 0; k < elements_per_thread; k++)
					xreg[k] = A[j + k * lda];
			}
			x1 = x[incx * tx_];
		}

		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
		{
	  		res_2_  += areg[k] * xbuff[ty_ * elements_per_thread + k]; //xcopy[incx * (ty_ * elements_per_thread + k)];
	  		treg[k] += conjugate( areg[k] ) * x2;
		}

		// Horizontal block should be stored in global memory
		__syncthreads();
		accum[ty_ * syhemv_bs + tx_] = res_1_;
		accum[ty_ * syhemv_bs + tx_ + (syhemv_bs/2)] = res_2_;
		__syncthreads();

		if(ty == 0)
		{
	    	res_1_ = make_zero<T>();
	    	#pragma unroll
	    	for(int k = 0; k < (2 * thread_y); k++)
	      		res_1_ += accum[k * syhemv_bs + tx];
	    	res_1_ *= alpha;
	    	// use atomics
	    	atomicAdd(&ycopy[incy * tx], res_1_);
	    	ycopy += syhemv_bs * incy;
	    }
	}// end of for loop on blocks


	#pragma unroll
	for(int k = 0; k < elements_per_thread; k++)
		la[(ty_ * elements_per_thread + k) * (syhemv_bs/2) + tx_] = treg[k];

	__syncthreads();		// important

	if(ty == 0)
	{
	  	treg[0] = make_zero<T>();
	  	#pragma unroll
		for(int j = tx; j < tx+(syhemv_bs/2); j++)
			treg[0] += la[tx * (syhemv_bs/2) +  (j % (syhemv_bs/2))];

		treg[0] *= alpha;
		// use atomics
		if(blkc == gridDim.x-1){if(tx < n_mod_syhemv_bs)atomicAdd(&y[tx * incy], treg[0]);}
		else atomicAdd(&y[tx * incy], treg[0]);
	}
}
/*****************************************************************************************/
template <class T, int syhemv_bs, int thread_x, int thread_y, int elements_per_thread, int irregular_part>
__global__ void
syhemvu_generic_nd(	int n, T alpha,
                        	T *A, int lda,
                            T *x, int incx,
                            T  beta,
                            T *y, int incy,
							int n_mod_syhemv_bs)
{
	fn_syhemvu_generic_nd<T, syhemv_bs, thread_x, thread_y, elements_per_thread, irregular_part>(n, alpha, A, lda, x, incx, beta, y, incy, n_mod_syhemv_bs);
}
/*****************************************************************************************/
