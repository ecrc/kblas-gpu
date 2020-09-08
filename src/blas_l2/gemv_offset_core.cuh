/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/gemv_offset_core.cuh

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

/*int gpu_gid, int ngpus,*/
#define ngpus   (1)
#define gpu_gid (0)

template <class T, int gemv_bs, int thread_x, int thread_y, int elements_per_thread>
__global__ void
gemvn_special_offset(	int rows, int cols,
				T alpha, T *A, int lda,
				T *x, int incx,
				T  beta, T *y, int incy,
			   	int nstripes,
			   	int offset_r, int offset_c)
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
    __shared__ T xbuff[gemv_bs];		// stores the first segment to be processed of x

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

    // the firsy segment of x needs special treatment because of the offset
    if(gpu_gid == 0 && by == 0)
    {
    	if(ty == 0)
    	{
    		if(tx >= offset_c) xbuff[tx] = x[tx * incx];
    		else xbuff[tx] = make_zero<T>();
    	}
    }
    else
    {
    	if(ty == 0) xbuff[tx] = x[tx * incx];
    }

    //if(by == gridDim.y-1) count += (cols/gemv_bs) % gridDim.y;
    if(count == 0) return;

    T res = make_zero<T>();

    const int j = ty_ * elements_per_thread * lda + tx_;

	__syncthreads();

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
	    if(Vblocks == 0)
	    {
	    	#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
				res_1_ += xreg[k] * xbuff[ty_ * elements_per_thread + k];
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
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			xreg[k] = A[j + k * lda];
		}

		// compute lower
		if(Vblocks == 0)
		{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			res_2_ 	+= areg[k] * xbuff[ty_ * elements_per_thread + k];
		}
		else
		{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			res_2_ 	+= areg[k] * x[(ty_ * elements_per_thread + k) * incx];
		}

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
      	if(blkc == 0){if(tx >= offset_r) atomicAdd(&y[tx * incy], (alpha*res_1_ + res));}
      	else atomicAdd(&y[tx * incy], (alpha*res_1_ + res));
      	//y[tx] = alpha * res_1_ + res;
    }
}
//--------------------------------------------------------------------------------------------------------//
template <class T, int gemv_bs, int thread_x, int thread_y, int elements_per_thread, int irregular_cols>
__global__ void
gemvn_generic_offset(	int rows, int cols,
				T alpha, T *A, int lda,
				T *x, int incx,
				T  beta, T *y, int incy,
				int rows_mod_gemv_bs, int cols_mod_gemv_bs,
				int nstripes,
				int offset_r, int offset_c)
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
    __shared__ T xbuff_[gemv_bs];				// used for handling the offset

    // special case test
    if(cols == 0)return;

    // identify which block will process the last block
    const int nstripes_local = (nstripes/ngpus) + (gpu_gid < (nstripes%ngpus));
    const int last_active_tb = min(nstripes_local, gridDim.y) - 1;
    const int gpu_last = (nstripes+ngpus-1)%ngpus;

    int start = by * (nstripes_local/gridDim.y) + min(by, nstripes_local%gridDim.y);
    int count = nstripes_local/gridDim.y + (by < nstripes_local%gridDim.y);

    if(count == 0) return;

    {
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

    // test special case, when rows mod block_size is zero
    if(rows_mod_gemv_bs == 0){if(blkc == gridDim.x-1)return;}

	{
		int sid_start = gpu_gid + start * ngpus;
		int sid_end = sid_start + (count-1) * ngpus;

		if(sid_start == 0)
		{
			if(sid_start == nstripes-1)
			{
				if(ty == 0)
				{
					if(tx >= offset_c && tx < cols_mod_gemv_bs) xbuff_[tx] = x[tx * incx];
					else xbuff_[tx] = make_zero<T>();
				}
			}
			else
			{
				if(ty == 0)
				{
					if(tx >= offset_c) xbuff_[tx] = x[tx * incx];
					else xbuff_[tx] = make_zero<T>();
				}
			}
		}
		else if(sid_start != nstripes-1) {if(ty == 0) xbuff_[tx] = x[tx * incx];}

		if(cols_mod_gemv_bs != 0)
		{
			if(sid_end == nstripes-1)
			{
				if(sid_end == 0)	// corner case, handle mod and offset
				{
					if(ty == 0)
					{
						if(tx >= offset_c && tx < cols_mod_gemv_bs) xbuff[tx] = x[(tx + (count-1)*gemv_bs*ngpus)*incx];
						else xbuff[tx] = make_zero<T>();
					}
				}
				else
				{
					if(ty == 0)
					{
						if(tx < cols_mod_gemv_bs) xbuff[tx] = x[(tx + (count-1)*gemv_bs*ngpus)*incx];
						else xbuff[tx] = make_zero<T>();
					}
				}
			}
		}
	}

    const int j = ty_ * elements_per_thread * lda + tx_;

	// last irregular tile is processed outside the main loop
    if(gpu_gid == gpu_last && by == last_active_tb) count -= 1;

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
	    if(Vblocks == 0)
	    {
	    	if(blkc == gridDim.x-1)
	    	{
	    		if(tx_ < rows_mod_gemv_bs)
	    		{
	    			#pragma unroll
					for(int k = 0; k < elements_per_thread; k++)
						res_1_ += xreg[k] * xbuff_[ty_ * elements_per_thread + k];
	    		}
	    	}
	    	else
	    	{
	    		#pragma unroll
				for(int k = 0; k < elements_per_thread; k++)
					res_1_ += xreg[k] * xbuff_[ty_ * elements_per_thread + k];
	    	}
	    }
	    else
	    {
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
		if(Vblocks == 0)
		{
			if(blkc == gridDim.x-1)
			{
				if(tx_+(gemv_bs/2) < rows_mod_gemv_bs)
				{
					#pragma unroll
					for(int k = 0; k < elements_per_thread; k++)
	  					res_2_ 	+= areg[k] * xbuff_[ty_ * elements_per_thread + k];
				}
			}
			else
			{
				#pragma unroll
				for(int k = 0; k < elements_per_thread; k++)
	  				res_2_ 	+= areg[k] * xbuff_[ty_ * elements_per_thread + k];
			}
		}
		else
		{
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
		}
		x += ngpus * gemv_bs * incx;
	} // end of main loop

	//////////////////
	// process last irregular tile

	if( (cols_mod_gemv_bs != 0) && (by == last_active_tb) && (gpu_gid == gpu_last))
	{
		//if
		//if(tx == 0 && ty == 0)printf("gpu %d hi from block %d, %d\n", gpu_gid, blkc, by);

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

		//printf("tx = %d .. xbuff = %f \n", tx, xbuff[tx]);

		//res_1_ += res;
		if(blkc == 0)
		{
			// a very corner case .. only one thread block on one gpu is active
			if(blkc == gridDim.x-1){if(tx >= offset_r && tx < rows_mod_gemv_bs) {atomicAdd(&y[tx * incy], res_1_);}}
			else {if(tx >= offset_r) atomicAdd(&y[tx * incy], res_1_);}
		}
      	else if(blkc == gridDim.x-1){if(tx < rows_mod_gemv_bs) atomicAdd(&y[tx * incy], res_1_);}
      	else {atomicAdd(&y[tx * incy], res_1_);}
    }
}
//--------------------------------------------------------------------------------------------------------//

template <class T, int gemv_bs, int thread_x, int thread_y, int elements_per_thread>
__global__ void
gemvt_special_offset(	int rows, int cols,
				T alpha, T *A, int lda,
				T *x, int incx, T  beta, T *y, int incy,
				int nstripes,
				int offset_r, int offset_c, int conj)
{
    const int	tx   = threadIdx.x ;
    const int	ty   = threadIdx.y ;
    const int	blkc = blockIdx.x ;
    const int 	by	=	blockIdx.y;
    const int	td  = (thread_x * ty ) + tx;
    const int	tx_  = td % (gemv_bs/2);
    const int	ty_  = td / (gemv_bs/2);

    __shared__ T la[gemv_bs * (thread_x/2)];
    __shared__ T xbuff[gemv_bs];

	T xreg[elements_per_thread] = {make_zero<T>()};
	T areg[elements_per_thread] = {make_zero<T>()};
	T treg[elements_per_thread] = {make_zero<T>()};

	const int sid = gpu_gid + blkc * ngpus;

    int count = (rows/gemv_bs)/gridDim.y + (by < (rows/gemv_bs)%gridDim.y);
    if(count == 0) return;

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

    // first segment of x needs special handling due to offset
    {
    	if(by == 0)
    	{
    		if(ty == 0)
    		{
    			if(tx >= offset_r) xbuff[tx] = x[tx * incx];
    			else xbuff[tx] = make_zero<T>();
    		}
    	}
    	else
    	{
    		if(ty == 0){xbuff[tx] = x[tx * incx];}
    	}
    }

    T res = make_zero<T>();

    const int j = ty_ * elements_per_thread * lda + tx_;

	__syncthreads();

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
	    if(Vblocks == 0)
	    {
	    	#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			treg[k] += conj_if(conj, xreg[k]) * xbuff[tx_];
	    }
	    else
	    {
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			treg[k] += conj_if(conj, xreg[k]) * x[tx_ * incx];
		}

	    A += gemv_bs;

		// read upper from next block
		if(Vblocks != count-1)
		{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			xreg[k] = A[j + k * lda];
		}

		//compute lower
		if(Vblocks == 0)
		{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			treg[k] += conj_if(conj, areg[k]) * xbuff[(tx_ + (gemv_bs/2))];
		}
		else
		{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			treg[k] += conj_if(conj, areg[k]) * x[(tx_ + (gemv_bs/2)) * incx];
	  	}
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

      	if(sid == 0) {if(tx >= offset_c) atomicAdd(&y[tx * incy], treg[0]);}
      	else
      	{
      		atomicAdd(&y[tx * incy], treg[0]);
      	}	//y[tx] = treg[0];
    }
}

//--------------------------------------------------------------------------------------------------------//

template <class T, int gemv_bs, int thread_x, int thread_y, int elements_per_thread, int irregular_cols>
__global__ void
gemvt_generic_offset(	int rows, int cols,
				T alpha, T *A, int lda,
				T *x, int incx,
				T  beta, T *y, int incy,
				int rows_mod_gemv_bs, int cols_mod_gemv_bs,
				int nstripes,
				int offset_r, int offset_c, int conj)
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
    __shared__ T xbuff_[gemv_bs];

	T xreg[elements_per_thread] = {make_zero<T>()};
	T areg[elements_per_thread] = {make_zero<T>()};
	T treg[elements_per_thread] = {make_zero<T>()};

	//special case test
	if(cols == 0) return;

     // identify which block will process the last block
    const int ntiles = (rows/gemv_bs) + (rows_mod_gemv_bs != 0);
    const int last_active_tb = min(ntiles, gridDim.y) - 1;
    const int gpu_last = (nstripes+ngpus-1)%ngpus;

    int start = by * (ntiles/gridDim.y) + min(by, ntiles%gridDim.y);
    int count = ntiles/gridDim.y + (by < ntiles%gridDim.y);
    if(count == 0) return;

    // special case if col % gemv_bs is zero
    if(cols_mod_gemv_bs == 0) {if(gpu_gid == gpu_last && blkc == gridDim.x-1) return;}

    {
		// Advance 'A' to start a block column
		A += gemv_bs * blkc * lda;
		A += start * gemv_bs;

		// Advance 'x'
		x += start * gemv_bs * incx;

    	// Advance 'y'
    	y += (gpu_gid + blkc * ngpus) * gemv_bs * incy;
    }


    // load irregular segment of x (if any)
    // init xbuff and xbuff_
    {
    	if(by == 0)
    	{
    		// corner case
    		if(by == last_active_tb && count == 1)
    		{
    			if(ty == 0)
    			{
    				if(tx >= offset_r && tx < rows_mod_gemv_bs) xbuff_[tx] = x[tx * incx];
    				else xbuff_[tx] = make_zero<T>();
    			}
    		}
    		else
    		{
    			if(ty == 0)
    			{
    				if(tx >= offset_r) xbuff_[tx] = x[tx * incx];
    				else xbuff_[tx] = make_zero<T>();
    			}
    		}
    	}
    	else if(by == last_active_tb && count == 1)
    	{
    		if(ty == 0)
    		{
    			if(tx >= offset_r && tx < rows_mod_gemv_bs) xbuff_[tx] = x[tx * incx];
    			else xbuff_[tx] = make_zero<T>();
    		}
    	}
    	else {xbuff_[tx] = x[tx * incx];}

    	if(by == last_active_tb)
    	{
    		// corner case
    		if(by == 0 && count == 1)
    		{
    			if(ty == 0)
    			{
    				if(tx >= offset_r && tx < rows_mod_gemv_bs) xbuff[tx] = x[((count-1)*gemv_bs + tx)*incx];
    				else xbuff[tx] = make_zero<T>();
    			}
    		}
    		else
    		{
    			if(ty == 0)
    			{
    				if(tx < rows_mod_gemv_bs) xbuff[tx] = x[((count-1)*gemv_bs + tx)*incx];
    				else xbuff[tx] = make_zero<T>();
    			}
    		}
    	}
    }

    T res = make_zero<T>();

    const int num_active_thread_cols = cols_mod_gemv_bs/elements_per_thread;

    if(blkc == gridDim.x-1 && gpu_gid == gpu_last)
    {
    	// init shmem to zero
    	#pragma unroll
		for(int k = 0; k < elements_per_thread; k++)
      		la[(ty_ * elements_per_thread + k) * (gemv_bs/2) + tx_] = make_zero<T>();

    	// some warps/half-warps will do no useful work
    	int tmp = max(2, num_active_thread_cols);
		if(ty_ > tmp) return;		// we need at least two thread columns to do final reduction
    }

    // the last irregular tile is processed separately
    if(by == last_active_tb) count -= 1;


    __syncthreads();

    //if(gpu_gid == 0 && ty == 0){printf("gpu %d - block(%d, %d) - [%-2d]: xbuff_ = %f -- xbuff = %f \n", blkc, by, gpu_gid, tx, xbuff_[tx], xbuff[tx]);}
    //if(gpu_gid == 0 && ty == 0){printf("gpu %d - block(%d, %d) - [%-2d]:  num_active_thread_cols = %d\n", blkc, by, gpu_gid, tx, num_active_thread_cols);}

    const int j = ty_ * elements_per_thread * lda + tx_;

	// read upper
	if(count > 0)
	{
		if(blkc == gridDim.x-1 && gpu_gid == gpu_last)
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
    	// read lower
		if(blkc == gridDim.x-1 && gpu_gid == gpu_last)
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
	    if(Vblocks == 0)
	    {
	    	#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			treg[k] += conj_if(conj, xreg[k]) * xbuff_[tx_];
	    }
	    else
	    {
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			treg[k] += conj_if(conj, xreg[k]) * x[tx_ * incx];
		}

	    A += gemv_bs;

		// read upper from next block
		if(Vblocks != count-1)
		{
			if(blkc == gridDim.x-1 && gpu_gid == gpu_last)
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
		if(Vblocks == 0)
		{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			treg[k] += conj_if(conj, areg[k]) * xbuff_[(tx_ + (gemv_bs/2))];
		}
		else
		{
			#pragma unroll
			for(int k = 0; k < elements_per_thread; k++)
	  			treg[k] += conj_if(conj, areg[k]) * x[(tx_ + (gemv_bs/2)) * incx];
	  	}

		x += gemv_bs * incx;
	}

	/////////////////////////////////
	// process the last irregular tile

	if(by == last_active_tb)
	{
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
			if(blkc == gridDim.x-1 && gpu_gid == gpu_last)
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
			if(blkc == gridDim.x-1 && gpu_gid == gpu_last)
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

      	const int sid = gpu_gid + blkc * ngpus;
      	if(sid == 0)
      	{
      		if(sid == nstripes-1){if(tx >= offset_c && tx < cols_mod_gemv_bs) atomicAdd(&y[tx * incy], treg[0]);}
      		else {if(tx >= offset_c) atomicAdd(&y[tx * incy], treg[0]);}
      	}
      	else if (sid == nstripes-1){if(tx < cols_mod_gemv_bs) atomicAdd(&y[tx * incy], treg[0]);}
      	else {atomicAdd(&y[tx * incy], treg[0]);}
    }
}
//--------------------------------------------------------------------------------------------------------//
