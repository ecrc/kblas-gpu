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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "operators.h"
#include <stdio.h>

template <class T, int gemv_bs, int thread_x, int thread_y, int elements_per_thread>
__global__ void
gemvn_mgpu_special(	int rows, int cols, 
				T alpha, T *A, int lda, 
				T *x, int incx, 
				T  beta, T *y, int incy, 
			   	int gpu_gid, int ngpus)
{
    const int	tx   = threadIdx.x ;
    const int	ty   = threadIdx.y ;
    const int	blkc = blockIdx.x ;
    const int	by	=	blockIdx.y; 
    const int	td  = (thread_x * ty ) + tx;
    const int	tx_  = td % (gemv_bs/2);
    const int	ty_  = td / (gemv_bs/2);
    
    T res_1_	= make_zero<T>();
    T res_2_	= make_zero<T>();
	T xreg[elements_per_thread];
	T areg[elements_per_thread];
	
    __shared__ T la[gemv_bs * (2 * thread_y)];
    
    int count = (cols/gemv_bs)/gridDim.y + (by < (cols/gemv_bs)%gridDim.y);
    
    {
    	int start = by * ((cols/gemv_bs)/gridDim.y) + min(by, (cols/gemv_bs)%gridDim.y);
    	
    	// Advance 'A' to start of diagonal blocks first
		A += gemv_bs * blkc;
		A += start * gemv_bs * lda;
		
    	// Advance 'x'
    	x += (gpu_gid * gemv_bs) * incx;	// start offset
    	x += (start * ngpus) * gemv_bs * incx; 
    
    	// Advance 'y'
    	y += (blkc * gemv_bs) * incy;
    }
    
    //if(by == gridDim.y-1) count += (cols/gemv_bs) % gridDim.y; 
    if(count == 0) return; 
    
    T res = make_zero<T>();
    
    const int j = ty_ * elements_per_thread * lda + tx_;
	
	//if(td == 0)printf("block %d, count = %d \n", blkc, count); __syncthreads();
	
	// read upper
	#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
		xreg[k] = A[j + k * lda];
	
	#pragma unroll
    for(int Vblocks = 0; Vblocks < count; Vblocks++)
    {
		// read lower
		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
	    	areg[k] = A[(gemv_bs/2) + j + k * lda];
	    
	    // compute upper
	    #pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
			res_1_ += xreg[k] * x[(ty_ * elements_per_thread + k) * incx]; 
	    	
		A += gemv_bs * lda;
		
		// read upper from next block
		if(Vblocks != count-1)
		{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			xreg[k] = A[j + k * lda];
		}

		// compute lower
		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
	  		res_2_ 	+= areg[k] * x[(ty_ * elements_per_thread + k) * incx]; 
		
		x += ngpus * gemv_bs * incx;	
	}
	
	la[ty_ * gemv_bs + tx_] = res_1_;
    la[ty_ * gemv_bs + tx_ + (gemv_bs/2)] = res_2_;
    __syncthreads();
    
    if(ty == 0)
    {
		res_1_ = make_zero<T>();
      	#pragma unroll
      	for(int k = 0; k < 2*thread_y; k++)
			res_1_ += la[k * gemv_bs + tx];
      	// use atomics
      	atomicAdd(&y[tx * incy], (alpha*res_1_ + res));
      	//y[tx] = alpha * res_1_ + res;
    }
}
//--------------------------------------------------------------------------------------------------------//

template <class T, int gemv_bs, int thread_x, int thread_y, int elements_per_thread, int irregular_cols>
__global__ void
gemvn_mgpu_generic(	int rows, int cols, 
				T alpha, T *A, int lda, 
				T *x, int incx, 
				T  beta, T *y, int incy, 
				int rows_mod_gemv_bs, int cols_mod_gemv_bs, 
				int gpu_gid, int ngpus)
{
    const int	tx   = threadIdx.x ;
    const int	ty   = threadIdx.y ;
    const int	blkc = blockIdx.x ;
    const int 	by 	= blockIdx.y;
    const int	td  = (thread_x * ty ) + tx;
    const int	tx_  = td % (gemv_bs/2);
    const int	ty_  = td / (gemv_bs/2);
    
	T res_1_	= make_zero<T>();
    T res_2_	= make_zero<T>();
	T xreg[elements_per_thread] = {make_zero<T>()};
	T areg[elements_per_thread] = {make_zero<T>()};
	
    __shared__ T la[gemv_bs * (2 * thread_y)];
    __shared__ T xbuff[gemv_bs];				// used for the last irregular part of x
    
    int count = (cols/gemv_bs)/gridDim.y + (by < (cols/gemv_bs)%gridDim.y);
    {
    	int start = by * ((cols/gemv_bs)/gridDim.y) + min(by, (cols/gemv_bs)%gridDim.y);
    	
    	// Advance 'A' 
		A += gemv_bs * blkc;
		A += start * gemv_bs * lda; 
	
		// Advance 'x'
		//T *x_ = x;;
		x += (gpu_gid * gemv_bs) * incx;	// start offset
    	x += (start * ngpus) * gemv_bs * incx; 
    	
    	// Advance 'y'
    	y += (blkc * gemv_bs) * incy;
    }
    T *A_ = A;
    //if(by == gridDim.y-1) count += (cols/gemv_bs)%gridDim.y; 
    
    if(by != gridDim.y-1){if(count == 0) return;}
    
    // test special case, when rows mod block_size is zero
    if(rows_mod_gemv_bs == 0)
    {
    	if(blkc == gridDim.x-1)return;
    }

	//la[ty_ * gemv_bs + tx_] = make_zero<T>();
    //la[ty_ * gemv_bs + tx_ + (gemv_bs/2)] = make_zero<T>();
    
    // load the last segment of x
    //if(by == gridDim.y-1)
    //{
    	if(cols_mod_gemv_bs != 0)
   		{ 	
   			if(ty == 0)
    		{
    			if(tx < cols_mod_gemv_bs) xbuff[tx] = x[(count*ngpus*gemv_bs + tx)*incx];
    			else xbuff[tx] = make_zero<T>();
    			
    			/*if(blkc == 0)
    				for(int l = 0; l < cols_mod_gemv_bs; l++)
    				{
    					if(tx == l) printf("gpu_dig: %d - xbuff[%d] = %.2f\n",gpu_gid,  l, xbuff[l]);
    					__syncthreads();
    				}*/
    		}
    	}
    //}
    
    //T res = make_zero<T>();
    /* -- the term beta * y has to be done in a separate kernel
    if(blkc == gridDim.x-1)
    {
    	if(ty == 0)
    	{
    		if(tx < rows_mod_gemv_bs)
    			y[tx] = res = beta * y[tx]; 
    	}
   	}
    else
    {if(ty == 0) y[tx] = res = beta * y[tx];}
    */
    
    //if(td == 0)printf("gpu %d: block (%d, %d) .. count = %d \n", gpu_gid, blkc, by, count);
    
    const int j = ty_ * elements_per_thread * lda + tx_;
	
	__syncthreads(); 
	
	if(count > 0)
	{
		// read upper
		if(blkc == gridDim.x-1)
		{
			if(tx_ < rows_mod_gemv_bs)
			{
				#pragma unroll
    			for(int k = 0; k < elements_per_thread; k++)
					xreg[k] = A[j + k * lda];	
			}
		}
		else
		{
			#pragma unroll
    		for(int k = 0; k < elements_per_thread; k++)
				xreg[k] = A[j + k * lda];
		}
	}
	
	// -- Main Loop
	int Vblocks;
    #pragma unroll
	for(Vblocks = 0; Vblocks < count; Vblocks++)
    {
		// read lower
		if(blkc == gridDim.x-1)
		{
			if(tx_+(gemv_bs/2) < rows_mod_gemv_bs)
			{
				#pragma unroll
				for(int k = 0; k < elements_per_thread; k++)
	    			areg[k] = A[(gemv_bs/2) + j + k * lda];
			}
		}
		else
		{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	    		areg[k] = A[(gemv_bs/2) + j + k * lda];
	    }
	    
	    // compute upper
	    if(blkc == gridDim.x-1)
	    {
	    	if(tx_ < rows_mod_gemv_bs)
	    	{
	    		#pragma unroll
				for(int k = 0; k < elements_per_thread; k++)
					res_1_ += xreg[k] * x[(ty_ * elements_per_thread + k) * incx]; 	
	    	}
	    }
	    else
	    {
	    	#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
				res_1_ += xreg[k] * x[(ty_ * elements_per_thread + k) * incx]; 
	    }
	    
		A += gemv_bs * lda;
		
		// read upper from next block
		if(Vblocks != count-1)
		{
			if(blkc == gridDim.x-1)
			{
				if(tx_ < rows_mod_gemv_bs)
				{
					#pragma unroll
    				for(int k = 0; k < elements_per_thread; k++)
						xreg[k] = A[j + k * lda];	
				}
			}
			else
			{
				#pragma unroll
				for(int k = 0; k < elements_per_thread; k++)
	  				xreg[k] = A[j + k * lda];
	  		}		
		}

		// compute lower
		if(blkc == gridDim.x-1)
		{
			if(tx_+(gemv_bs/2) < rows_mod_gemv_bs)
			{
				#pragma unroll
				for(int k = 0; k < elements_per_thread; k++)
	  				res_2_ 	+= areg[k] * x[(ty_ * elements_per_thread + k) * incx]; 
			}
		}
		else
		{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			res_2_ 	+= areg[k] * x[(ty_ * elements_per_thread + k) * incx]; 
		}
		x += ngpus * gemv_bs * incx;	
	} // end of main loop
	
	//////////////////
	// process last irregular tile
	
	if( (cols_mod_gemv_bs != 0) && (by == gridDim.y-1) )
	{
		//if
		{
			int offset = count*gemv_bs*lda;
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
			{
				xreg[k] = make_zero<T>();
				areg[k] = make_zero<T>();
			}
	
			const int num_active_thread_cols = cols_mod_gemv_bs/elements_per_thread;
		
			//load upper 
			if(blkc == gridDim.x-1)
			{
				if(ty_ < num_active_thread_cols)
				{
					if(tx_ < rows_mod_gemv_bs)
					{
						#pragma unroll
    					for(int k = 0; k < elements_per_thread; k++)
							//xreg[k] = A[j + k * lda];
							xreg[k] = A_[offset+j+k*lda];
					}
				}
				else if (ty_ == num_active_thread_cols)
				{
					if(tx_ < rows_mod_gemv_bs)
					{
						#pragma unroll
    					for(int k = 0; k < irregular_cols; k++)
							//xreg[k] = A[j + k * lda];
							xreg[k] = A_[offset+j+k*lda];
					}
				}
			}
			else
			{
				if(ty_ < num_active_thread_cols)
				{
					#pragma unroll
    				for(int k = 0; k < elements_per_thread; k++)
						//xreg[k] = A[j + k * lda];
						xreg[k] = A_[offset+j+k*lda];
				}
				else if (ty_ == num_active_thread_cols)
				{
					#pragma unroll
    				for(int k = 0; k < irregular_cols; k++)
						//xreg[k] = A[j + k * lda];
						xreg[k] = A_[offset+j+k*lda]; 
				}
			}
		
			// load lower
			if( blkc == gridDim.x-1)
			{
				if(ty_ < num_active_thread_cols)
				{
					if(tx_+(gemv_bs/2) < rows_mod_gemv_bs)
					{
						#pragma unroll
    					for(int k = 0; k < elements_per_thread; k++)
							//areg[k] = A[(gemv_bs/2) + j + k * lda];
							areg[k] = A_[offset+j+k*lda+(gemv_bs/2)];
					}
				}
				else if (ty_ == num_active_thread_cols)
				{
					if(tx_+(gemv_bs/2) < rows_mod_gemv_bs)
					{
						#pragma unroll
    					for(int k = 0; k < irregular_cols; k++)
							//areg[k] = A[(gemv_bs/2) + j + k * lda];
							areg[k] = A_[offset+j+k*lda+(gemv_bs/2)];
					}
				}
			}
			else
			{
				if(ty_ < num_active_thread_cols)
				{
					#pragma unroll
					for(int k = 0; k < elements_per_thread; k++)
	    				//areg[k] = A[(gemv_bs/2) + j + k * lda];
	    				areg[k] = A_[offset+j+k*lda+(gemv_bs/2)]; 
				}
				else if (ty_ == num_active_thread_cols)
				{	
					#pragma unroll
					for(int k = 0; k < irregular_cols; k++)
	    				//areg[k] = A[(gemv_bs/2) + j + k * lda];
	    				areg[k] = A_[offset+j+k*lda+(gemv_bs/2)]; 
				}
			}
		} // end of if by == gridDim.x-1
		
		// compute upper
		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
			res_1_ += xreg[k] * xbuff[ty_ * elements_per_thread + k]; // x[(ty_ * elements_per_thread + k) * incx]; 
	
		// compute lower
		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
			res_2_ 	+= areg[k] * xbuff[ty_ * elements_per_thread + k]; // x[(ty_ * elements_per_thread + k) * incx];

	} // end of if  cols_mod_gemv_bs != 0
	
	// final reduction
	la[ty_ * gemv_bs + tx_] = res_1_;
    la[ty_ * gemv_bs + tx_ + (gemv_bs/2)] = res_2_;
    __syncthreads();
    
    if(ty == 0)
    {
		res_1_ = make_zero<T>();
      	#pragma unroll
      	for(int k = 0; k < 2*thread_y; k++)
			res_1_ += la[k * gemv_bs + tx];
		res_1_ *= alpha;
		//res_1_ += res;
      	if(blkc == gridDim.x-1){if(tx < rows_mod_gemv_bs) atomicAdd(&y[tx * incy], res_1_);}
      	else {atomicAdd(&y[tx * incy], res_1_);}
    }
}
//--------------------------------------------------------------------------------------------------------//

template <class T, int gemv_bs, int thread_x, int thread_y, int elements_per_thread>
__global__ void
gemvt_mgpu_special(	int rows, int cols, 
				T alpha, T *A, int lda, 
				T *x, int incx, T  beta, T *y, int incy, 
				int gpu_gid, int ngpus, int conj)
{
    const int	tx   = threadIdx.x ;
    const int	ty   = threadIdx.y ;
    const int	blkc = blockIdx.x ;
    const int 	by	=	blockIdx.y; 
    const int	td  = (thread_x * ty ) + tx;
    const int	tx_  = td % (gemv_bs/2);
    const int	ty_  = td / (gemv_bs/2);
    
    __shared__ T la[gemv_bs * (thread_x/2)];
    
	T xreg[elements_per_thread];
	T areg[elements_per_thread];
	T treg[elements_per_thread] = {make_zero<T>()};
	
    int count = (rows/gemv_bs)/gridDim.y + (by < (rows/gemv_bs)%gridDim.y);
    {
    	int start = by * ((rows/gemv_bs)/gridDim.y) + min(by, (rows/gemv_bs)%gridDim.y);
    	
    	// Advance 'A' to start a block column
		A += gemv_bs * blkc * lda;
		A += start * gemv_bs; 
		
		// Advance 'x'
		x += start * gemv_bs * incx; 
		
    	// Advance 'y'
    	y += (gpu_gid + blkc * ngpus) * gemv_bs * incy;
    	//y += (blkc * gemv_bs) * incy;
    }
    
    T res = make_zero<T>();
    
    
    //if(by == gridDim.y-1) count += (rows/gemv_bs)%gridDim.y;
    
    if(count == 0) return;
    
    // beta*y should be handled through a separate kernel
    //if(ty == 0)res = beta * y[tx];
     
    const int j = ty_ * elements_per_thread * lda + tx_;
	
	// read upper
	#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
		xreg[k] = A[j + k * lda];
	
	#pragma unroll
    for(int Vblocks = 0; Vblocks < count; Vblocks++)
    {
		// read lower
		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
	    	areg[k] = A[(gemv_bs/2) + j + k * lda];
	    
	    // compute upper
		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
	  		treg[k] += conj_if(conj, xreg[k]) * x[tx_ * incx];
	
	    A += gemv_bs;
		
		// read upper from next block
		if(Vblocks != count-1)
		{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			xreg[k] = A[j + k * lda];
		}
		
		//compute lower
		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
	  		treg[k] += conj_if(conj, areg[k]) * x[(tx_ + (gemv_bs/2)) * incx];
	  
		x += gemv_bs * incx;	
	}
	
	// final reduction
	#pragma unroll
	for(int k = 0; k < elements_per_thread; k++)
      la[(ty_ * elements_per_thread + k) * (gemv_bs/2) + tx_] = treg[k];
    
    __syncthreads();

    if(ty == 0)
    {
		treg[0] = make_zero<T>();
      	#pragma unroll
      	for(int j = tx; j < tx+(gemv_bs/2); j++)
			treg[0] += la[tx * (gemv_bs/2) +  (j % (gemv_bs/2) )];
      
      	treg[0] *= alpha;
      	treg[0] += res;
      	atomicAdd(&y[tx * incy], treg[0]);	//y[tx] = treg[0];
    }
}

//--------------------------------------------------------------------------------------------------------//

template <class T, int gemv_bs, int thread_x, int thread_y, int elements_per_thread, int irregular_cols>
__global__ void
gemvt_mgpu_generic(	int rows, int cols, 
				T alpha, T *A, int lda, 
				T *x, int incx, 
				T  beta, T *y, int incy, 
				int rows_mod_gemv_bs, int cols_mod_gemv_bs, 
				int gpu_gid, int ngpus, int conj)
{
    const int	tx   = threadIdx.x ;
    const int	ty   = threadIdx.y ;
    const int	blkc = blockIdx.x ;
    const int	by = blockIdx.y;
    const int	td  = (thread_x * ty ) + tx;
    const int	tx_  = td % (gemv_bs/2);
    const int	ty_  = td / (gemv_bs/2);
    
    __shared__ T la[gemv_bs * (thread_x/2)];
    __shared__ T xbuff[gemv_bs];
    
	T xreg[elements_per_thread] = {make_zero<T>()};
	T areg[elements_per_thread] = {make_zero<T>()};
	T treg[elements_per_thread] = {make_zero<T>()};
	
    int count = (rows/gemv_bs)/gridDim.y + (by < (rows/gemv_bs)%gridDim.y);
    {
		int start = by * ((rows/gemv_bs)/gridDim.y) + min(by, (rows/gemv_bs)%gridDim.y);
		
    	// Advance 'A' to start a block column
		A += gemv_bs * blkc * lda;
		A += start * gemv_bs; 
		
		// Advance 'x'
		x += start * gemv_bs * incx;
		
    	// Advance 'y'
    	y += (gpu_gid + blkc * ngpus) * gemv_bs * incy;
    	//y += (blkc * gemv_bs) * incy;
    }
    //if(by == gridDim.y-1) count += (rows/gemv_bs)%gridDim.y; 
    
    if(by != gridDim.y-1){if(count == 0) return;}
    
    //if(td == 0)printf("block (%d, %d): count = %d \n", blkc, by, count);
    
    // load irregular segment of x (if any)
    if(by == gridDim.y-1)
    {
    	//if(td == 0)printf("block (%d, %d): load xbuff\n", blkc, by);
    	if(rows_mod_gemv_bs != 0)
    	{
    		if(ty == 0)
    		{
    			if(tx < rows_mod_gemv_bs) xbuff[tx] = x[(count*gemv_bs + tx) * incx];
    			else xbuff[tx] = make_zero<T>();
    		}
    	}
    }
    
    T res = make_zero<T>();
    
    const int num_active_thread_cols = cols_mod_gemv_bs/elements_per_thread; 
    
    if(cols_mod_gemv_bs == 0)
    {
    	if(blkc == gridDim.x-1) return;
    }
    
    if(blkc == gridDim.x-1)
    {
    	// scaling with beta is done through a separate kernel
    	/*if(ty == 0)
    	{
    		if(tx < cols_mod_gemv_bs) res = beta * y[tx];
    	}*/
    	
    	// init shmem to zero
    	#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
      		la[(ty_ * elements_per_thread + k) * (gemv_bs/2) + tx_] = make_zero<T>();
    	
    	// some warps/half-warps will do no useful work
    	int tmp = max(2, num_active_thread_cols);
		if(ty_ > tmp) return;		// we need at least two thread columns to do final reduction
    }
    /*else
    {
    	if(ty == 0)res = beta * y[tx];
    }*/
    
    __syncthreads();
    
    const int j = ty_ * elements_per_thread * lda + tx_;
	
	// read upper
	if(count > 0)
	{
		if(blkc == gridDim.x-1)
		{
			if(ty_ < num_active_thread_cols)
			{
				#pragma unroll
    			for(int k = 0; k < elements_per_thread; k++)
					xreg[k] = A[j + k * lda];
			}
			else if (ty_ == num_active_thread_cols)
			{
				#pragma unroll
    			for(int k = 0; k < irregular_cols; k++)
					xreg[k] = A[j + k * lda];
			}
		}
		else
		{
			#pragma unroll
    		for(int k = 0; k < elements_per_thread; k++)
				xreg[k] = A[j + k * lda];
		}
	}
	
	//--- Main Loop
	#pragma unroll
    for(int Vblocks = 0; Vblocks < count; Vblocks++)
    {
    	//if(td == 0)printf("block (%d, %d): loop # %d \n", blkc, by, Vblocks);
    	
		// read lower
		if(blkc == gridDim.x-1)
		{
			if(ty_ < num_active_thread_cols)
			{
				#pragma unroll
    			for(int k = 0; k < elements_per_thread; k++)
					areg[k] = A[(gemv_bs/2) + j + k * lda];
			}
			else if (ty_ == num_active_thread_cols)
			{
				#pragma unroll
    			for(int k = 0; k < irregular_cols; k++)
					areg[k] = A[(gemv_bs/2) + j + k * lda];
			}	
		}
		else
		{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	    		areg[k] = A[(gemv_bs/2) + j + k * lda];
	    }
	    
	    // compute upper
		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
	  		treg[k] += conj_if(conj, xreg[k]) * x[tx_ * incx];
	
	    A += gemv_bs;
		
		// read upper from next block
		if(Vblocks != count-1)
		{
			if(blkc == gridDim.x-1)
			{
				if(ty_ < num_active_thread_cols)
				{
					#pragma unroll
    				for(int k = 0; k < elements_per_thread; k++)
						xreg[k] = A[j + k * lda];
				}
				else if (ty_ == num_active_thread_cols)
				{
					#pragma unroll
    				for(int k = 0; k < irregular_cols; k++)
						xreg[k] = A[j + k * lda];
				}
			}
			else
			{
				#pragma unroll
				for(int k = 0; k < elements_per_thread; k++)
	  				xreg[k] = A[j + k * lda];
			}
		}
		
		//compute lower
		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
	  		treg[k] += conj_if(conj, areg[k]) * x[(tx_ + (gemv_bs/2)) * incx];
	  
		x += gemv_bs * incx;	
	}
	
	/////////////////////////////////
	// process the last irregular tile
	
	if(by == gridDim.y-1)
	{
		//if(td == 0)printf("block (%d, %d): irregular tile processing \n", blkc, by);
		
		if(rows_mod_gemv_bs != 0)
		{
			// process last irregular tile
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
			{
				xreg[k] = make_zero<T>();
				areg[k] = make_zero<T>();
			}
		
			//read upper
			if(blkc == gridDim.x-1)
			{
				if(tx_ < rows_mod_gemv_bs)
				{
					if(ty_ < num_active_thread_cols)
					{
						#pragma unroll
    					for(int k = 0; k < elements_per_thread; k++)
							xreg[k] = A[j + k * lda];
					}
					else if (ty_ == num_active_thread_cols)
					{
						#pragma unroll
    					for(int k = 0; k < irregular_cols; k++)
							xreg[k] = A[j + k * lda];
					}
				}
			}
			else
			{
				if(tx_ < rows_mod_gemv_bs)
				{
					#pragma unroll
    				for(int k = 0; k < elements_per_thread; k++)
						xreg[k] = A[j + k * lda];
				}
			}
	
			//read lower
			if(blkc == gridDim.x-1)
			{
				if(tx_+(gemv_bs/2) < rows_mod_gemv_bs)
				{
					if(ty_ < num_active_thread_cols)
					{
						#pragma unroll
    					for(int k = 0; k < elements_per_thread; k++)
							areg[k] = A[(gemv_bs/2) + j + k * lda];
					}
					else if (ty_ == num_active_thread_cols)
					{
						#pragma unroll
    					for(int k = 0; k < irregular_cols; k++)
							areg[k] = A[(gemv_bs/2) + j + k * lda];
					}
				}
			}
			else
			{
				if(tx_+(gemv_bs/2) < rows_mod_gemv_bs)
				{
					#pragma unroll
					for(int k = 0; k < elements_per_thread; k++)
	    				areg[k] = A[(gemv_bs/2) + j + k * lda];
				}
			}
		
			//compute upper 
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			treg[k] += conj_if(conj, xreg[k]) * xbuff[tx_]; //x[tx_ * incx];
	
			//compute lower 
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			treg[k] += conj_if(conj, areg[k]) * xbuff[tx_ + (gemv_bs/2)]; //x[(tx_ + (gemv_bs/2)) * incx];
	  		
		}	// end of if rows_mode_gemv_bs != 0 		
	} // end of if by == gridDim.y-1
	
	//if(td == 0)printf("block (%d, %d): final reduction\n", blkc, by);
	
	// final reduction
	#pragma unroll
	for(int k = 0; k < elements_per_thread; k++)
      la[(ty_ * elements_per_thread + k) * (gemv_bs/2) + tx_] = treg[k];
    
    __syncthreads();

    if(ty == 0)
    {
		treg[0] = make_zero<T>();
      	#pragma unroll
      	for(int j = tx; j < tx+(gemv_bs/2); j++)
			treg[0] += la[tx * (gemv_bs/2) +  (j % (gemv_bs/2) )];
      
      	treg[0] *= alpha;
      	treg[0] += res;
      	if(blkc == gridDim.x-1){if(tx < cols_mod_gemv_bs) atomicAdd(&y[tx * incy], treg[0]);}
      	else {atomicAdd(&y[tx * incy], treg[0]);}
    }
}
//--------------------------------------------------------------------------------------------------------//