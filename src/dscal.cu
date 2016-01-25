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
#include "scal_core.cuh"

#define dscal_nbx		(128)

int kblas_dscal_driver(int n, double alpha, double *x, int incx, cudaStream_t stream)
{
	int gridx = n / dscal_nbx + (n % dscal_nbx != 0);
	
	dim3 dimBlock(dscal_nbx, 1);
	dim3 dimGrid(gridx, 1);
	
	scal<double><<<dimGrid, dimBlock, 0, stream>>>(n, alpha, x, incx);
	
	return 0;
}

extern "C"    
int kblas_dscal(int n, double alpha, double *x, int incx)
{
	return kblas_dscal_driver(n, alpha, x, incx, 0);
}

extern "C"    
int kblas_dscal_async(int n, double alpha, double *x, int incx, cudaStream_t stream)
{
	return kblas_dscal_driver(n, alpha, x, incx, stream);
}
