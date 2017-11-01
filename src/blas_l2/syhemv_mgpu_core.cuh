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

#ifndef	SYHEMV_MGPU_HEADER
#define SYHEMV_MGPU_HEADER

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "operators.h"
#include <stdio.h>

template <class T, int syhemv_bs, int thread_x, int thread_y, int elements_per_thread>
__global__ void
syhemvl_mgpu_special_d( int n, T alpha,
				    T *A, int lda,
				    T *x, int incx,
				    T  beta,
				    T *y, int incy, 
				    int gpu_gid, int ngpus, int nstripes)
{
    const int	tx   = threadIdx.x ;
    const int	ty   = threadIdx.y ;
    const int 	td  = (thread_x * ty ) + tx;
    const int	blkc = blockIdx.x ;
    
    T res	= make_zero<T>();
    T yold	= make_zero<T>();
    
    __shared__ T la   [syhemv_bs * syhemv_bs];
    __shared__ T buff [syhemv_bs];
    __shared__ T accum[syhemv_bs * (2 * thread_y)];
    
    // compute the stripe id
    const int sid = gpu_gid + blkc * ngpus; 
    
	// Advance 'A' to start of diagonal blocks first
	A += syhemv_bs * (blkc * lda + sid);
	
	// Advance 'A' to start row for each thread inside the diagonal block	
	A += ty * lda + tx;	
	
	// Advance 'x'
	x += (sid * syhemv_bs) * incx;
	
    // Advance 'y'
    y += (sid * syhemv_bs) * incy;
    
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
	
	// ignore imaginary part of diagonal elements
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
syhemvl_mgpu_special_nd( 	int n, T alpha,
				    T *A, int lda,
				    T *x, int incx,
				    T  beta,
				    T *y, int incy, 
				    int gpu_gid, int ngpus, int nstripes)
{
    const int	tx   = threadIdx.x ;
    const int	ty   = threadIdx.y ;
    const int	blkc = blockIdx.x ;
    const int	by	 = blockIdx.y;
    const int	td  = (thread_x * ty ) + tx;
    const int	tx_  = td % (syhemv_bs/2);
    const int	ty_  = td / (syhemv_bs/2);
    T		* xcopy, *ycopy;

	const int sid = gpu_gid + blkc * ngpus;
	
	// compute how many matrix blocks to be processed
	int count = (nstripes-sid-1)/gridDim.y;
	
	T xreg[elements_per_thread];
	T areg[elements_per_thread];
	T treg[elements_per_thread] = { make_zero<T>()};
	
    __shared__ T la   [syhemv_bs * (syhemv_bs/2)];
    __shared__ T accum[syhemv_bs * (2 * thread_y)];
    __shared__ T xbuff[syhemv_bs];
    
    if(sid == nstripes-1)return;
    
	// Advance 'A' to the first off-diagonal block
	A += syhemv_bs * (blkc * lda + sid);
	// divide work among the y-direction of the grid
	A += (by * count) * syhemv_bs; 
	
	// Advance 'x'
	x += (sid * syhemv_bs) * incx;
	xcopy = x;
	x += (by * count * syhemv_bs) * incx; 
    
    if(ty == 0) xbuff[tx] = xcopy[tx * incx];

    // Advance 'y'
    y += (sid * syhemv_bs) * incy;
    ycopy = y;
    ycopy += (by * count * syhemv_bs) * incy; 
    
	if(by == gridDim.y-1) count += (nstripes-sid-1)%gridDim.y; 
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
	x1 = x[incx * tx_];
	
	A -= syhemv_bs;
    x -= syhemv_bs * incx;
    
	#pragma unroll
    for(int Vblocks = 0; Vblocks < count /*nstripes-sid-1*/ ; Vblocks++)
    {
		A += syhemv_bs;
		x += syhemv_bs * incx;
		
		res_1_	=	make_zero<T>();
		res_2_	=	make_zero<T>();
		
		x2 = x[incx * (tx_ + (syhemv_bs/2))];
		
		// read lower
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
		
		A += syhemv_bs;
		x += syhemv_bs * incx;
		
		// read upper from next block
		if(Vblocks != count-1 /*(nstripes-sid-1)-1*/)
		{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			xreg[k] = A[j + k * lda];
			x1 = x[incx * tx_];
		}

		A -= syhemv_bs;
		x -= syhemv_bs * incx;

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
	
	if(sid != nstripes-1)
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
syhemvl_mgpu_generic_d( int n, T alpha,
				    T *A, int lda,
				    T *x, int incx,
				    T  beta,
				    T *y, int incy,
				    int	    n_mod_syhemv_bs, 
				    int gpu_gid, int ngpus, int nstripes)
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
    
    const int sid = gpu_gid + blkc * ngpus;
    
    //printf("NGPUS= %d, GPU= %d, Block= %d, sid= %d, nstripes= %d \n", ngpus, gpu_gid, blkc, sid, nstripes);
    
    // Advance 'A' to start of diagonal blocks first
	A += syhemv_bs * (blkc * lda + sid);
	
	// Advance 'A' to start row for each thread inside the diagonal block	
	A += ty * lda + tx;	
	
	// Advance x
	x += (sid * syhemv_bs) * incx;
	
	// Advacne y
	y += (sid * syhemv_bs) * incy;
	
	// load part of vector x
	if(sid == nstripes-1)
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
	if(sid == nstripes-1)
	{
		#pragma unroll
		for(int j = 0; j < syhemv_bs; j+= thread_y)	 
			la[j * syhemv_bs + td ] = make_zero<T>();
		__syncthreads();
		
		if(tx >= n_mod_syhemv_bs) return; 	// these threads should not read any useful data
	}
	
	// load a bock of data
	if(sid == nstripes-1)
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
	  	if(sid == nstripes-1){if(tx < n_mod_syhemv_bs)y[tx * incy] = res;}
	  	else{y[tx * incy] = res;}
	}
}
/*****************************************************************************************/
template <class T, int syhemv_bs, int thread_x, int thread_y, int elements_per_thread >
__global__ void 
syhemvl_mgpu_generic_nd( int n, T alpha,
                               T *A, int lda,
                               T *x, int incx,
                               T  beta,
                               T *y, int incy,
							   int n_mod_syhemv_bs, 
							   int gpu_gid, int ngpus, int nstripes)
{
    const int tx   = threadIdx.x ;
    const int ty   = threadIdx.y ;
    const int blkc = blockIdx.x ;
    const int by   = blockIdx.y;
    const int td  = (thread_x * ty ) + tx;
    const int tx_  = td % (syhemv_bs/2);
    const int ty_  = td / (syhemv_bs/2);
    
    const int sid = gpu_gid + blkc * ngpus;
    int count = (nstripes-sid-1-1)/gridDim.y;
    
    T *xcopy, *ycopy; 
    
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
    
    if(sid == nstripes - 1)return;
    
    // Advance 'A' to start of diagonal blocks first
    A += syhemv_bs * (blkc * lda + sid);
    // divide work among the y-direction of the grid
	A += (by * count) * syhemv_bs; 
	
    // Advance 'x'
    x += (sid * syhemv_bs) * incx;
    xcopy = x;
    x += (by * count * syhemv_bs) * incx; 
    
    if(ty == 0) xbuff[tx] = xcopy[incx * tx];
    
    //Advance 'y'
	y += (sid * syhemv_bs) * incy;
    ycopy = y;
    ycopy += (by * count * syhemv_bs) * incy; 
    if(by == gridDim.y-1) count += ((nstripes-sid-1-1)%gridDim.y); //- 1;	// -1 for the generic block at the bottom
    if(by != gridDim.y-1){if(count == 0) return;}
    
	int j = ty_ * elements_per_thread * lda + tx_;
	
	__syncthreads(); 
	
    A += syhemv_bs;
    x += syhemv_bs * incx;
    
    if(sid < nstripes-2)		// to prevent out of bound access
    {
    	#pragma unroll
    	for(int k = 0; k < elements_per_thread; k++)
			xreg[k] = A[j + k * lda];
    	x1 = x[incx * tx_];
    }
    
    A -= syhemv_bs;
    x -= syhemv_bs * incx;
	
    #pragma unroll
    for(int Vblocks = 0; Vblocks < count /*(nstripes-sid-1)-1*/; Vblocks++)
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
		
		if(Vblocks != count-1 /*((nstripes-sid-1)-1)-1*/)
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
	} // end if by == gridDim.y-1
	
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
syhemvu_mgpu_special_d( int n, T alpha,
				    T *A, int lda,
				    T *x, int incx,
				    T  beta,
				    T *y, int incy, 
				    int gpu_gid, int ngpus, int nstripes)
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
	
	const int sid = gpu_gid + blkc * ngpus; 
	
	// Advance 'A' to start of diagonal blocks first
	A += syhemv_bs * (blkc * lda + sid);
	// Advance 'A' to start row for each thread inside the diagonal block	
	A += ty * lda + tx;	
	
	// Advance 'x'
	x += (sid * syhemv_bs) * incx;

	// Advance 'y'
	y += (sid * syhemv_bs) * incy;
	
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
fn_syhemvu_mgpu_special_nd( int n, T alpha,
                               T *A, int lda,
                               T *x, int incx,
                               T  beta,
                               T *y, int incy, 
                               int gpu_gid, int ngpus, int nstripes)
{
    const int tx   = threadIdx.x ;
    const int ty   = threadIdx.y ;
    const int td  = (thread_x * ty ) + tx;
    const int tx_  = td % (syhemv_bs/2);
    const int ty_  = td / (syhemv_bs/2);
    const int blkc = blockIdx.x;
    const int by   = blockIdx.y; 
    	
    const int sid = gpu_gid + blkc * ngpus; 
    // compute how many matrix blocks to be processed
	int count = sid/gridDim.y;
    
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
    
    xcopy = x + (sid * syhemv_bs) * incx;
	x += (by * count * syhemv_bs) * incx;
	
	if(sid == 0)return;
	
	if(ty == 0) xbuff[tx] = xcopy[tx * incx];	

	// Advance 'y'
	ycopy = y;
	ycopy += (by * count * syhemv_bs) * incy;
	y += (sid * syhemv_bs) * incy;
    
    if(by == gridDim.y-1) count += sid%gridDim.y; 
	if(count == 0) return; 	  
    const int j = ty_ * elements_per_thread * lda + tx_;

	__syncthreads();
	
	// prefetch upper
    #pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
		xreg[k] = A[j + k * lda];
    
    x1 = x[incx * tx_];

    #pragma unroll
    for(int Vblocks = 0; Vblocks < count; Vblocks++)
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
	
		if(Vblocks != count-1)
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
syhemvu_mgpu_special_nd( int n, T alpha,
                               T *A, int lda,
                               T *x, int incx,
                               T  beta,
                               T *y, int incy, 
                               int gpu_gid, int ngpus, int nstripes)
{
	fn_syhemvu_mgpu_special_nd<T, syhemv_bs, thread_x, thread_y, elements_per_thread>(n, alpha, A, lda, x, incx, beta, y, incy, gpu_gid, ngpus, nstripes);
}
/***************************************************************************************/
template <class T, int syhemv_bs, int thread_x, int thread_y, int elements_per_thread>
__global__ void
syhemvu_mgpu_generic_d( int n, T alpha,
				    T *A, int lda,
				    T *x, int incx,
				    T  beta,
				    T *y, int incy,
				    int n_mod_syhemv_bs, 
				    int gpu_gid, int ngpus, int nstripes)
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
        
    const int sid = gpu_gid + blkc * ngpus;
    
	// Advance 'A' to start of diagonal blocks first
	A += syhemv_bs * (blkc * lda + sid);
	
	// Advance 'A' to start row for each thread inside the diagonal block	
	A += ty * lda + tx;	
	
	// Advance 'x'
	x += (sid * syhemv_bs) * incx;
	
	// Advance 'y'
	y += (sid * syhemv_bs) * incy;
	
	// load part of vector 'x'
	if(sid == nstripes-1)		// last TB
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
	if(sid == nstripes-1)
	{
		#pragma unroll
		for(int j = 0; j < syhemv_bs; j+= thread_y)	 
			la[j * syhemv_bs + td ] = make_zero<T>();
		__syncthreads();
		
		if(tx >= n_mod_syhemv_bs) return; 	// these threads should not read any useful data
	}
	
	// load a bock of data
	if(sid == nstripes-1)
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
		if(sid == nstripes-1){if(tx < n_mod_syhemv_bs)y[tx * incy] = res;}
	  	else{y[tx * incy] = res;}
	}
}
/*****************************************************************************************/
template <class T, int syhemv_bs, int thread_x, int thread_y, int elements_per_thread, int irregular_part>
__device__ __inline__ void
fn_syhemvu_mgpu_generic_nd(int n, T alpha,
                            T *A, int lda,
                        	T *x, int incx,
                        	T  beta,
                        	T *y, int incy,
							int n_mod_syhemv_bs, 
							int gpu_gid, int ngpus, int nstripes)
{
    const int tx   = threadIdx.x ;
    const int ty   = threadIdx.y ;
    const int td  = (thread_x * ty ) + tx;
    const int tx_  = td % (syhemv_bs/2);
    const int ty_  = td / (syhemv_bs/2);
	const int blkc = blockIdx.x;
	const int by   = blockIdx.y;
	
	const int sid = gpu_gid + blkc * ngpus; 
	// compute how many matrix blocks to be processed
	int count = sid/gridDim.y;
	
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
    xcopy = x + (sid * syhemv_bs) * incx;    
	x += (by * count * syhemv_bs) * incx;
	
	// Advance 'y'
	ycopy = y;
	ycopy += (by * count * syhemv_bs) * incy;
	y += (sid * syhemv_bs) * incy;
	
	if(sid == 0) return;
	
	if(by == gridDim.y-1) count += sid%gridDim.y;  
	if(count == 0) return; 	 
	
	// useful for ltb (last thread block only)
	const int num_active_thread_cols = n_mod_syhemv_bs/elements_per_thread; 

	// cache part of 'x' needed for computing res_1_ and res_2_ 
	if(sid == nstripes-1)		//last TB
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
	if(sid == nstripes-1)		// last TB "irregular"
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
    for(int Vblocks = 0; Vblocks < count /*sid*/; Vblocks++)
    {
	
		res_1_ = make_zero<T>();
		res_2_ = make_zero<T>();
	
		x2 = x[incx * (tx_ + (syhemv_bs/2))];	
	
		// prefetch lower
		if(sid == nstripes-1)
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
		if(Vblocks != count-1 /*sid-1*/)
		{
			if(sid == nstripes-1)		// last TB "irregular"
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
		if(sid == nstripes-1){if(tx < n_mod_syhemv_bs)atomicAdd(&y[tx * incy], treg[0]);}
		else atomicAdd(&y[tx * incy], treg[0]);
	}
}
/*****************************************************************************************/
template <class T, int syhemv_bs, int thread_x, int thread_y, int elements_per_thread, int irregular_part>
__global__ void
syhemvu_mgpu_generic_nd(	int n, T alpha,
                        	T *A, int lda,
                            T *x, int incx,
                            T  beta,
                            T *y, int incy,
							int n_mod_syhemv_bs, 
							int gpu_gid, int ngpus, int nstripes)
{
	fn_syhemvu_mgpu_generic_nd<T, syhemv_bs, thread_x, thread_y, elements_per_thread, irregular_part>(n, alpha, A, lda, x, incx, beta, y, incy, n_mod_syhemv_bs, gpu_gid, ngpus, nstripes);
}
/*****************************************************************************************/

#endif