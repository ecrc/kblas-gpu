/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/sgemm_mgpu.cu

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
#include <stdio.h>
#include "gemm_aux.cuh"
#include "kblas_operators.h"


#define SGEMM_MAX_TILE	(8192)

extern "C"
void kblas_sgemm_mgpu(	char transa, char transb, long m, long n, long k,
                		float alpha, const float* A, long lda,
                		const float* B, long ldb,
                		float beta, float* C, long ldc,
                		long ngpus, long* gpu_id,
                		long *tile)
{

	hipblasStatus_t se;
	int current_gpu;
	hipGetDevice(&current_gpu);

	long tile_size = (*tile);
	if(tile_size == -1)
	{
		tile_size = recommend_tile(m, n, k, ngpus, SGEMM_MAX_TILE);
		(*tile) = tile_size;
	}
	// set to 1 to print info
	long pflag = 0;

	// compute #waves of full stripes
	long stripes = (m + tile_size-1)/tile_size; //(m / tile_size) + (m%tile_size != 0);
	long full_waves = stripes / ngpus;
	long remaining_stripes = stripes % ngpus;

	// compute the memory space required per gpu
	// first, wrap up k to be multiple of tile_size
	long k__ = ( (k + tile_size-1)/tile_size ) * tile_size;
	long width = tile_size;
	long height = k__; 						// height of a h-stripe of A or v-stripe of B
	height += 2 * tile_size;				// 2 extra tiles for multiplication
	height += 2 * tile_size;				// 2 output tiles
	height = ( (height+31)/32 ) * 32;		// for coalesced memory access
	long mem_space = height * width;

	// gpu pointers/worspace
	float* gpu_ws[MAX_NGPUS];
	float* a[MAX_NGPUS];
	float* b[MAX_NGPUS][2];
	float* c[MAX_NGPUS][2];
	float* a_[MAX_NGPUS];
	float* b_[MAX_NGPUS][2];
	float* c_[MAX_NGPUS][2];

	// streams
	hipStream_t	stream[MAX_NGPUS][4];

	// events
	long nevents = (max(n, k)+tile_size-1) / tile_size;
	hipEvent_t _ain_[MAX_NGPUS][MAX_EVENTS];
	hipEvent_t _bin_[MAX_NGPUS][MAX_EVENTS];

	hipEvent_t _afree_[MAX_NGPUS][MAX_EVENTS];
	hipEvent_t _bfree_[MAX_NGPUS][MAX_EVENTS];

	hipEvent_t _cin_[MAX_NGPUS][MAX_EVENTS];
	hipEvent_t _cout_[MAX_NGPUS][MAX_EVENTS];
	hipEvent_t _compute_[MAX_NGPUS][MAX_EVENTS];

	// allocate gpu memory
	{
		if(pflag)printf("memory allocation\n");
		hipError_t e;
		for(long i = 0; i <  ngpus; i++)
		{
			hipSetDevice(gpu_id[i]);
			e = hipMalloc((void**)&gpu_ws[i], mem_space * sizeof(float));
			if(e != hipSuccess)
			{
				printf("ERROR: failed to allocate memory on gpu %ld \n", i);
				for(long j = 0; j <= i; j++) { if(gpu_ws[i]) hipFree(gpu_ws[i]); }
				exit(1);
			}
		}
	}

	// aux host pointers
	// aux pointers
	float	*A_[MAX_NGPUS], *B_[MAX_NGPUS], *C_[MAX_NGPUS];

	// Adjust pointers
	{
		if(pflag)printf("adjust pointers\n");
		// host
		for(long i = 0; i < ngpus; i++)
		{
			A_[i] = (float*)A;
			B_[i] = (float*)B;
			C_[i] = (float*)C;
		}
		// device
		for(long i = 0; i < ngpus; i++)
		{
			a[i] = gpu_ws[i];
			b[i][0] = a[i] 		+ tile_size * k__;
			b[i][1] = b[i][0] 	+ tile_size * tile_size;
			c[i][0] = b[i][1]	+ tile_size * tile_size;
			c[i][1] = c[i][0]	+ tile_size * tile_size;
		}
	}

	// create streams and events
	{
		if(pflag)printf("stream create\n");
		for(long i = 0; i < ngpus; i++)
		{
			hipSetDevice(gpu_id[i]);

			hipStreamCreate(&stream[i][0]);	// compute
			hipStreamCreate(&stream[i][1]);	// copy a in and c out
			hipStreamCreate(&stream[i][2]);	// copy b in
			hipStreamCreate(&stream[i][3]);	// copy c in


			for(long j = 0; j < nevents; j++)
			{
				hipEventCreateWithFlags(&_ain_[i][j], hipEventDisableTiming);
				hipEventCreateWithFlags(&_bin_[i][j], hipEventDisableTiming);
				hipEventCreateWithFlags(&_afree_[i][j], hipEventDisableTiming);
				hipEventCreateWithFlags(&_bfree_[i][j], hipEventDisableTiming);
				hipEventCreateWithFlags(&_compute_[i][j], hipEventDisableTiming);
				hipEventCreateWithFlags(&_cin_[i][j], hipEventDisableTiming);
				hipEventCreateWithFlags(&_cout_[i][j], hipEventDisableTiming);
			}
		}
	}

	// set stream for the gemm calls
	for(long id = 0; id < ngpus; id++)
	{
		if(pflag)printf("set kernel stream\n");
		hipSetDevice(gpu_id[id]);
        hipblasSetStream(stream[id][0]);
	}


	// compute stepping in A and B
	long step_a, step_b;
	if(transa == 'n' || transa == 'N') step_a = tile_size * lda;
	else step_a = tile_size;

	if(transb == 'n' || transb == 'N') step_b = tile_size;
	else step_b = tile_size * ldb;

	// selector to switch between 2 gpu buffers
	long bselect[MAX_NGPUS] = {0};
	long cselect[MAX_NGPUS] = {0};

	// variables that store the actual tile sizes from A, B, and C for every GPU
	long ra[MAX_NGPUS] = {0};
	long ca[MAX_NGPUS] = {0};
	long rb[MAX_NGPUS] = {0};
	long cb[MAX_NGPUS] = {0};
	long rc[MAX_NGPUS] = {0};
	long cc[MAX_NGPUS] = {0};

	//main loop
	{
		if(pflag)printf("main loop\n");
		long total_iterations = full_waves + (remaining_stripes!=0);
		long ngpus_active;
		long n_ = (n + tile_size-1) / tile_size;
		long k_ = (k + tile_size-1) / tile_size;

		// i - loop over full waves (m)
		for(long i = 0; i < total_iterations; i++)
		{
			ngpus_active = ngpus;
			if(i == total_iterations-1){if(remaining_stripes != 0) ngpus_active = remaining_stripes;}

			// advance A_
			if(pflag)printf("i = %ld, advance A_\n", i);
			 if(transa == 'n' || transa == 'N') for(long id = 0; id < ngpus_active; id++) {A_[id] = (float*)A + (i *ngpus + id) * tile_size;}
			else for(long id = 0; id < ngpus_active; id++) { A_[id] = (float*)A + (i * ngpus + id) * tile_size * lda; }

			// compute #rows of current tiles in A and C
			for(long id = 0; id < ngpus_active; id++) rc[id] = min(m - (i*ngpus+id)*tile_size , tile_size);
			if(transa == 'n' || transa == 'N')
				for(long id = 0; id < ngpus_active; id++) ra[id] = min(m - (i*ngpus+id)*tile_size , tile_size);
			else
				for(long id = 0; id < ngpus_active; id++) ca[id] = min(m - (i*ngpus+id)*tile_size , tile_size);

			// j - loop over (n) -
			for(long j = 0; j < n_ ; j++)
			{
				if(pflag)printf("\t j = %ld, advance B_ and C_\n", j);

				// compute #cols in current tiles in B and C
				for(long id = 0; id < ngpus_active; id++) cc[id] = min(n - j*tile_size , tile_size);
				if(transb == 'n' || transb == 'N')
					for(long id = 0; id < ngpus_active; id++) cb[id] = min(n - j*tile_size , tile_size);
				else
					for(long id = 0; id < ngpus_active; id++) rb[id] = min(n - j*tile_size , tile_size);

				// Advance B_
				if(transb == 'n' || transb == 'N') for(long id = 0; id < ngpus_active; id++) {B_[id] = (float*)B + j * tile_size * ldb;}
				else for(long id = 0; id < ngpus_active; id++) {B_[id] = (float*)B + j * tile_size;}

				// Advance C_
				for(long id = 0; id < ngpus_active; id++)
				{
					//C_[id] = (float*)C + ( (i *ngpus + id) * tile_size ) + ( j * tile_size * ldc);
					C_[id] = (float*)C;
					//if(transa == 'n' || transa == 'N')
					C_[id] += (i *ngpus + id) * tile_size;
					//else C_[id] += (i * ngpus + id) * tile_size * ldc;

					//if(transb == 'n' || transb == 'N')
					C_[id] += j * tile_size * ldc;
					//else C_[id] += j * tile_size;
				}

				// copy device pointers
				for(long id = 0; id < ngpus_active; id++)
				{
					a_[id] = a[id];
					b_[id][0] = b[id][0];
					b_[id][1] = b[id][1];
					c_[id][0] = c[id][0];
					c_[id][1] = c[id][1];
				}

				// if starting to compute new row of tiles in C
				// copy the first tile of C in the row into devices
				if(j == 0)
				{
					for(long id = 0; id < ngpus_active; id++)
					{
						hipSetDevice(gpu_id[id]);
						hipStreamWaitEvent(stream[id][3], _cout_[id][cselect[id]], 0);
						se = hipblasSetMatrixAsync(rc[id], cc[id], sizeof(float), C_[id], ldc, c[id][cselect[id]], tile_size, stream[id][3]);
						process_error(se, "copy cin new row of tiles");
						hipEventRecord(_cin_[id][cselect[id]], stream[id][3]);
					}
				}

				if(pflag)printf("\t j = %ld, copy a, b tile in\n", j);
				// prepare a first input offload
				for(long id = 0; id < ngpus_active; id ++)
				{
					// as if p = 0 (first iteration in the inner-most loop)
					if(transa == 'n' || transa == 'N') ca[id] = min(k - 0*tile_size , tile_size);
					else ra[id] = min(k - 0*tile_size , tile_size);

					if(transb == 'n' || transb == 'N') rb[id] = min(k - 0*tile_size , tile_size);
					else cb[id] = min(k - 0*tile_size , tile_size);



					hipSetDevice(gpu_id[id]);
					if(j == 0)
					{
						hipStreamWaitEvent(stream[id][1], _afree_[id][0], 0);
						se = hipblasSetMatrixAsync(ra[id], ca[id], sizeof(float), A_[id], lda, a_[id], tile_size, stream[id][1]);
						char ss[100];
						sprintf(ss, " i =%ld, j = %ld copy ain new row of tiles: [%ld]x[%ld]", i, j, ra[id], ca[id]);
						process_error(se, ss);
						hipEventRecord(_ain_[id][0], stream[id][1]);
					}
					hipStreamWaitEvent(stream[id][2], _bfree_[id][bselect[id]], 0);
					se = hipblasSetMatrixAsync(rb[id], cb[id], sizeof(float), B_[id], ldb, b_[id][bselect[id]], tile_size, stream[id][2]);
					process_error(se, "copy bin new row of tiles");
					hipEventRecord(_bin_[id][bselect[id]], stream[id][2]);
				}

				// init b selector
				//for(long id = 0; id < ngpus; id++) bselect[id] = 0;

				// p - loop over k
				long p = 0;
				for(p = 0;  p < k_; p++)
				{
					float beta_;
					if(p == 0)beta_ = beta; else beta_ = 1;

					for(long id = 0; id < ngpus_active; id++)
					{
						hipSetDevice(gpu_id[id]);

						if(pflag)printf("\t\t p = %ld, wait for communication\n", p);

						if(transa == 'n' || transa == 'N') ca[id] = min(k - p*tile_size , tile_size);
						else ra[id] = min(k - p*tile_size , tile_size);

						if(transb == 'n' || transb == 'N') rb[id] = min(k - p*tile_size , tile_size);
						else cb[id] = min(k - p*tile_size , tile_size);



						// wait for communication
						//if(p == 0)hipStreamSynchronize(stream[id][3]);
						//if(j == 0)hipStreamSynchronize(stream[id][1]);
						//hipStreamSynchronize(stream[id][2]);
						if(p == 0) hipStreamWaitEvent(stream[id][0], _cin_[id][cselect[id]], 0);
						if(j == 0) hipStreamWaitEvent(stream[id][0], _ain_[id][p], 0);
						hipStreamWaitEvent(stream[id][0], _bin_[id][bselect[id]], 0);

						if(pflag)printf("\t\t p = %ld, gpu = %ld, invoke sgemm\n", p, id);
						if(pflag)printf("\t\t ------------------------------\n");
						if(pflag)printf("\t\t cselect[%ld] = %ld \n", id, cselect[id]);
						long msmall = rc[id];
						long nsmall = cc[id];
						long ksmall;
						if(transa == 'n' || transa == 'N') ksmall = ca[id];
						else ksmall = ra[id];

						//{
						//	printf("\n");
						//	printf("gpu %ld: [%ld][%ld] x [%ld][%ld] = [%ld][%ld]\n", id, msmall, ksmall, ksmall, nsmall, msmall, nsmall);
						//	//print a
						//	printf("A\n--------\n");
						//	myprint_matrix(transa, msmall, ksmall, a_[id], tile_size);
						//	//print b
						//	printf("B\n--------\n");
						//	myprint_matrix(transb, ksmall, nsmall, b_[id][bselect[id]], tile_size);
						//}
						// invoke sgemm
						cublasSgemm(transa, transb,
									msmall, nsmall, ksmall,
									alpha, a_[id], tile_size,
									b_[id][bselect[id]], tile_size,
									beta_, c_[id][cselect[id]], tile_size);
						hipEventRecord(_bfree_[id][bselect[id]], stream[id][0]);
						if(j == n_-1) hipEventRecord(_afree_[id][p], stream[id][0]);
						if(p == k_-1) hipEventRecord(_compute_[id][j], stream[id][0]);

						// prepare next input
						bselect[id] = 1 - bselect[id];
						a_[id] += tile_size * tile_size;
						if(p != k_-1)
						{
							if(pflag)printf("\t\t p = %ld, prepare next input\n", p);
							if(j == 0)
							{
								A_[id] += step_a;

								if(transa == 'n' || transa == 'N')ca[id] = min(k - (p+1)*tile_size, tile_size);
								else ra[id] = min(k - (p+1)*tile_size, tile_size);

								hipStreamWaitEvent(stream[id][1], _afree_[id][p+1], 0);
								se = hipblasSetMatrixAsync(ra[id], ca[id], sizeof(float), A_[id], lda, a_[id], tile_size, stream[id][1]);
								process_error(se, "prefetch ain");
								hipEventRecord(_ain_[id][p+1], stream[id][1]);

								if(transa == 'n' || transa == 'N')ca[id] = min(k - (p)*tile_size, tile_size);
								else ra[id] = min(k - (p)*tile_size, tile_size);

							}
							B_[id] += step_b;

							if(transb == 'n' || transb == 'N') rb[id] = min(k - (p+1)*tile_size, tile_size);
							else cb[id] = min(k - (p+1)*tile_size, tile_size);


							hipStreamWaitEvent(stream[id][2], _bfree_[id][bselect[id]], 0);
							se = hipblasSetMatrixAsync(rb[id], cb[id], sizeof(float), B_[id], ldb, b_[id][bselect[id]], tile_size, stream[id][2]);
							process_error(se, "prefetch bin");
							hipEventRecord(_bin_[id][bselect[id]], stream[id][2]);


							if(transb == 'n' || transb == 'N') rb[id] = min(k - (p)*tile_size, tile_size);
							else cb[id] = min(k - (p)*tile_size, tile_size);
						}
						if( p == 0)
						{
							if(j != n_-1)
							{
								// early copy of the next tile of C
								float* Ctmp = C_[id] + tile_size * ldc;
								cselect[id] = 1 - cselect[id];
								// rc[id] is the same, but we need to compute cc
								cc[id] = min(n - (j+1)*tile_size, tile_size);
								if(pflag)printf("\t\t cselect[%ld] = %ld \n", id, cselect[id]);
								hipStreamWaitEvent(stream[id][3], _cout_[id][cselect[id]], 0);
								se = hipblasSetMatrixAsync(rc[id], cc[id], sizeof(float), Ctmp, ldc, c_[id][cselect[id]], tile_size, stream[id][3]);
								char ss[100];
								sprintf(ss, "gpu[%ld]: prefetch cin [%ld]x[%ld]", id, rc[id], cc[id]);

								process_error(se, ss);
								hipEventRecord(_cin_[id][cselect[id]], stream[id][3]);
								cselect[id] = 1 - cselect[id];
								// restore cc
								cc[id] = min(n - j*tile_size, tile_size);
							}
						}
						if(pflag)printf("\n");
					}
				}// p - loop over k

				// copy c into cpu
				for(long id = 0; id < ngpus_active; id++)
				{
					if(pflag)printf("i = %ld, j = %ld, gpu = %ld, copy c output\n", i, j, id);
					hipSetDevice(gpu_id[id]);
					//hipStreamSynchronize(stream[id][0]);
					hipStreamWaitEvent(stream[id][3], _compute_[id][j], 0);
					se = hipblasGetMatrixAsync(rc[id], cc[id], sizeof(float), c_[id][cselect[id]], tile_size, C_[id], ldc, stream[id][3]);
					process_error(se, "read output c");
					hipEventRecord(_cout_[id][cselect[id]], stream[id][3]);
					cselect[id] = 1 - cselect[id];
				}
			}// j - loop over (n)
		} // i - loop over full waves (n)
	}// main compute part

	// global sync point
	{
		if(pflag)printf("global sync\n");
		for(long id = 0; id < ngpus; id++)
		{
			hipSetDevice(gpu_id[id]);
			hipDeviceSynchronize();
		}
	}

	// switch cublas streams to the default one
	for(long id = 0; id < ngpus; id ++)
	{
		hipSetDevice(gpu_id[id]);
		hipSetStream(0);
	}

	// destroy streams
	{
		if(pflag)printf("destroy stream\n");
		for(long i = 0; i < ngpus; i++)
		{
			hipSetDevice(gpu_id[i]);
			hipStreamDestroy(stream[i][0]);
			hipStreamDestroy(stream[i][1]);
			hipStreamDestroy(stream[i][2]);
			hipStreamDestroy(stream[i][3]);
		}
	}

	// destroy events
	{
		for(long i = 0; i < ngpus; i++)
		{
			for(long j = 0; j < nevents; j++)
			{
				hipEventDestroy(_ain_[i][j]);
				hipEventDestroy(_bin_[i][j]);
				hipEventDestroy(_afree_[i][j]);
				hipEventDestroy(_bfree_[i][j]);
				hipEventDestroy(_compute_[i][j]);
				hipEventDestroy(_cin_[i][j]);
				hipEventDestroy(_cout_[i][j]);
			}
		}
	}

	// free resources
	{
		if(pflag)printf("free resources\n");
		for(long i = 0; i < ngpus; i++)
			if(gpu_ws[i]) hipFree(gpu_ws[i]);
	}

	// retrieve current gpu
	hipSetDevice(current_gpu);
}
