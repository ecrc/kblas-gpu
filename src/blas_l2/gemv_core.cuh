/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/gemv_core.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ahmad Abdelfattah
 * @date 2018-11-14
 **/

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "kblas_operators.h"
#include <stdio.h>

template <class T, int gemv_bs, int thread_x, int thread_y, int elements_per_thread>
__global__ void
gemvn_special(int rows, int cols, T alpha, T *A, int lda, T *x, int incx, T  beta, T *y, int incy)
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

    	// Advance 'A'
		A += gemv_bs * blkc;
		A += start * gemv_bs * lda;

    	// Advance 'x'
    	x += start * gemv_bs * incx;

    	// Advance 'y'
    	y += (blkc * gemv_bs) * incy;

    	//if(by == gridDim.y-1) count += (cols/gemv_bs) % gridDim.y;
    }

    if(count == 0) return;

    /*if(by == gridDim.y-1)
    {
    	if(td == 0)printf("block(%d, %d): count = %d\n", blkc, by, count);
    	__syncthreads();
    }*/
    T res = make_zero<T>();

    // the term beta*y has to be done through another kernel
    //if(ty == 0)y[tx] = make_zero<T>();

    const int j = ty_ * elements_per_thread * lda + tx_;

	// read upper
	#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
		xreg[k] = A[j + k * lda];

	int Vblocks = 0;
	#pragma unroll
    for(Vblocks = 0; Vblocks < count; Vblocks++)
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

		x += gemv_bs * incx;
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
template <class T, int gemv_bs, int thread_x, int thread_y, int elements_per_thread>
__global__ void
gemvn_special_(int rows, int cols, T alpha, T *A, int lda, T *x, int incx, T  beta, T *y, int incy)
{
    const int	tx   = threadIdx.x ;
    const int	ty   = threadIdx.y ;
    const int	blkc = blockIdx.x ;
    const int	by	=	blockIdx.y;

    T res_1_	= make_zero<T>();
    T xreg[elements_per_thread];
	T areg[elements_per_thread];

    __shared__ T la[gemv_bs * thread_y];

    int count = (cols/gemv_bs) / gridDim.y;

    // Advance 'A' to start of diagonal blocks first
	A += gemv_bs * blkc;
	A += (by * count) * gemv_bs * lda;

    // Advance 'x'
    x += (by * count) * gemv_bs * incx;

    // Advance 'y'
    y += (blkc * gemv_bs) * incy;

    if(by == gridDim.y-1) count += (cols/gemv_bs) % gridDim.y;
    if(count == 0) return;

    /*if(by == gridDim.y-1)
    {
    	if(td == 0)printf("block(%d, %d): count = %d\n", blkc, by, count);
    	__syncthreads();
    }*/
    T res = make_zero<T>();

    // the term beta*y has to be done through another kernel
    //if(ty == 0)y[tx] = make_zero<T>();

    const int j = ty * elements_per_thread * lda + tx;

	// read left
	#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
		xreg[k] = A[j + k * lda];

	int Vblocks = 0;
	#pragma unroll
    for(Vblocks = 0; Vblocks < count; Vblocks++)
    {
		// read right
		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
	    	areg[k] = A[j + (k + (gemv_bs/2)) * lda];

	    // compute left
	    #pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
			res_1_ += xreg[k] * x[(ty * elements_per_thread + k) * incx];

		A += gemv_bs * lda;
		x += (gemv_bs/2) * incx;

		// read left from next block
		if(Vblocks != count-1)
		{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			xreg[k] = A[j + k * lda];
		}

		// compute right
		#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
	  		res_1_ 	+= areg[k] * x[(ty * elements_per_thread + k) * incx];

		x += (gemv_bs/2) * incx;
	}

	la[ty * gemv_bs + tx] = res_1_;
    __syncthreads();

    if(ty == 0)
    {
		res_1_ = make_zero<T>();
      	#pragma unroll
      	for(int k = 0; k < thread_y; k++)
			res_1_ += la[k * gemv_bs + tx];
      	// use atomics
      	atomicAdd(&y[tx * incy], (alpha*res_1_ + res));
      	//y[tx] = alpha * res_1_ + res;
    }
}
//--------------------------------------------------------------------------------------------------------//
template <class T, int gemv_bs, int thread_x, int thread_y, int elements_per_thread, int irregular_cols>
__global__ void
gemvn_generic(int rows, int cols, T alpha, T *A, int lda, T *x, int incx, T  beta, T *y, int incy, int rows_mod_gemv_bs, int cols_mod_gemv_bs)
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
		//if(blkc == 0)if(td == 0)printf("A = %x , A_ = %x \n", A, A_);

		// Advance 'x'
		x += start * gemv_bs * incx;

    	// Advance 'y'
    	y += (blkc * gemv_bs) * incy;

    	//if(by == gridDim.y-1) count += (cols/gemv_bs)%gridDim.y;
    }
    T *A_ = A;
    if(by != gridDim.y-1){if(count == 0) return;}

    // test special case, when rows mod block_size is zero
    if(rows_mod_gemv_bs == 0)
    {
    	if(blkc == gridDim.x-1)return;
    }

    // load the last segment of x
    if(by == gridDim.y-1)
    {
    	if(cols_mod_gemv_bs != 0)
   		{
   			if(ty == 0)
    		{
    			if(tx < cols_mod_gemv_bs) xbuff[tx] = x[(count*gemv_bs + tx)*incx];
    			else xbuff[tx] = make_zero<T>();
    		}
    	}
    }

    T res = make_zero<T>();
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
	int Vblocks = 0;
    //#pragma unroll
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
		x += gemv_bs * incx;
	} // end of main loop

	//////////////////
	// process last irregular tile

	if( (cols_mod_gemv_bs != 0) && (by == gridDim.y-1) )
	{
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
		res_1_ += res;
      	if(blkc == gridDim.x-1){if(tx < rows_mod_gemv_bs) atomicAdd(&y[tx * incy], res_1_);}
      	else {atomicAdd(&y[tx * incy], res_1_);}
    }
}
//--------------------------------------------------------------------------------------------------------//

template <class T, int gemv_bs, int thread_x, int thread_y, int elements_per_thread>
__global__ void
gemvt_special(int rows, int cols, T alpha, T *A, int lda, T *x, int incx, T  beta, T *y, int incy, int conj)
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
    	y += (blkc * gemv_bs) * incy;
    }

    //if(by == gridDim.y-1) count += (rows/gemv_bs)%gridDim.y;

    T res = make_zero<T>();

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
gemvt_generic(int rows, int cols, T alpha, T *A, int lda, T *x, int incx, T  beta, T *y, int incy, int rows_mod_gemv_bs, int cols_mod_gemv_bs, int conj)
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
    	y += (blkc * gemv_bs) * incy;
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
    else
    {
    	/*if(ty == 0)res = beta * y[tx];*/
    }

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
