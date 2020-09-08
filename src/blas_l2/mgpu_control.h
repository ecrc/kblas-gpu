/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/mgpu_control.h

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
#include <stdio.h>
#include "kblas_defs.h"

#ifndef MGPU_CONTROL
#define MGPU_CONTROL

/*****************************************************************************************/
template <class T>
void kblas_malloc_mgpu_1D(	int rows, int cols, T** dA, int ngpus, int ldb, int block_size)
{
    const int ngpus_local = ngpus;

	int current_device;
	cudaGetDevice(&current_device);

	int regular_stripes = (cols/block_size);
	int irregular_stripe_dim = (cols%block_size);

	int extra_stripes = (regular_stripes%ngpus) + (irregular_stripe_dim != 0);
	int min_stripes =  (regular_stripes)/ ngpus;		// the min # of stipes per gpu

	for(int i = 0; i < ngpus_local; i++)
	{
		cudaSetDevice(gpu_lid[i]);
		// decide height and width
		int h = ldb;
		int w = min_stripes * block_size;
		if(gpu_gid[i] == extra_stripes-1)
		{
			if(irregular_stripe_dim != 0)w += irregular_stripe_dim;
			else w += block_size;
		}
		else if (gpu_gid[i] < extra_stripes-1)
			w += block_size;

		int size = h * w;
		//printf("size of gpu %d is %.2f KB \n", gpu_lid[i], (float)(size*sizeof(T))/(1024.0));
		cudaError_t e = cudaMalloc((void**)&dA[i], size * sizeof(T));
		if(e != cudaSuccess){printf("kblas: error allocating memory on device %d: %s \n", gpu_lid[i], cudaGetErrorString(e)); exit(0);}
	}
	cudaSetDevice(current_device);
}
/*****************************************************************************************/
template <class T>
void kblas_setmatrix_mgpu_1D(int rows, int cols, T* A, int LDA, T** dA, int LDB, int ngpus, int block_size)
{
    const int ngpus_local = ngpus;

	int current_device;
	cudaGetDevice(&current_device);

	T* ha = A;
	int lda = LDA;
	int ldb = LDB;

	// maintain copy offsets on gpu memory
	int* offset = (int*)malloc(ngpus_local * sizeof(int));
	for(int k = 0; k < ngpus_local; k++)offset[k] = 0;

	int stride_a = block_size * lda;
	int stride_b = block_size * ldb;
	int cols_ = (cols/block_size)*block_size;;

	// loop over the host matrix continuously
	int i = 0;
	for(int b = 0; b < cols_; b+= block_size)
	{
		cudaSetDevice(gpu_lid[i]);
		cublasSetMatrixAsync(rows, block_size, sizeof(T), (void*)ha, lda, (void*)(dA[i]+offset[i]), ldb, 0);
		offset[i] += stride_b;
		i++;
		if(i == ngpus_local) i = 0;
		ha += stride_a;
	}

	int rem  = cols %block_size;
	if(rem != 0)
	{
		cudaSetDevice(gpu_lid[i]);
		cublasSetMatrixAsync(rows, rem, sizeof(T), (void*)ha, lda, (void*)(dA[i]+offset[i]), ldb, 0);
	}

	// wait for copy to finish on all devices
	for(int k = 0; k < ngpus_local; k++)
	{cudaSetDevice(gpu_lid[k]); cudaDeviceSynchronize();}

	if(offset)free(offset);
	cudaSetDevice(current_device);
}
/*****************************************************************************************/
template <class T>
void kblas_setvector_mgpu_1D(int n, T* Y, T** dY, int ngpus, int block_size)
{
    const int ngpus_local = ngpus;

	int current_device;
	cudaGetDevice(&current_device);

	T* hy = Y;

	int i = 0;
	int b;
	int n_ = (n/block_size)*block_size;

	// init dY to zeros
	for(int i = 0; i < ngpus; i++)
	{
		cudaSetDevice(gpu_lid[i]);
		cudaMemset(dY[i], 0, n * sizeof(T));
	}

	for(b = 0; b < n_; b+= block_size)
	{
		cudaSetDevice(gpu_lid[i]);
		cudaMemcpyAsync(dY[i]+b, hy, block_size*sizeof(T), cudaMemcpyHostToDevice, 0);
		i++;
		if(i == ngpus_local) i = 0;
		hy += block_size;
	}

	int rem  = n%block_size;
	if(rem != 0)
	{
		cudaSetDevice(gpu_lid[i]);
		cudaMemcpyAsync((void*)(dY[i]+b), (void*)hy, rem*sizeof(T), cudaMemcpyHostToDevice, 0);
	}

	for(int k = 0; k < ngpus_local; k++)
	{cudaSetDevice(gpu_lid[k]); cudaDeviceSynchronize();}

	cudaSetDevice(current_device);
}

#endif
