/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/batch_triangular/test_Xgeqrf_batch.cpp

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Wajih Halim Boukaram
 * @date 2020-12-10
 **/

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>      
#include <algorithm>
#include <vector>


#include "testing_helper.h"
#include <batch_getrf.h>
#include <kblas_operators.h>

#ifdef PREC_d
typedef double Scalar;
typedef double Real;
#define generateArrayOfPointers 		generateDArrayOfPointers
#define generateArrayOfPointersHost		generateDArrayOfPointersHost
#define generateRealArrayOfPointersHost	generateDArrayOfPointersHost
#define generateArrayOffsetPointers     generateDArrayOffsetPointers
#define generateArrayOffsetPointersHost generateDArrayOffsetPointersHost
#define generate_randomMatricesArray	generateDrandomMatricesArray
#define LAPACKE_Xgetrf					LAPACKE_dgetrf
#endif 

#ifdef PREC_s
typedef float Scalar;
typedef float Real;
#define generateArrayOfPointers 		generateSArrayOfPointers
#define generateArrayOfPointersHost		generateSArrayOfPointersHost
#define generateRealArrayOfPointersHost	generateSArrayOfPointersHost
#define generateArrayOffsetPointers     generateSArrayOffsetPointers
#define generateArrayOffsetPointersHost generateSArrayOffsetPointersHost
#define generate_randomMatricesArray	generateSrandomMatricesArray
#define LAPACKE_Xgetrf					LAPACKE_sgetrf
#endif 

#ifdef PREC_c
typedef std::complex<float> Scalar;
typedef float Real;
#define generateArrayOfPointers 		generateCArrayOfPointers
#define generateArrayOfPointersHost		generateCArrayOfPointersHost
#define generateRealArrayOfPointersHost	generateSArrayOfPointersHost
#define generateArrayOffsetPointers     generateCArrayOffsetPointers
#define generateArrayOffsetPointersHost generateCArrayOffsetPointersHost
#define generate_randomMatricesArray	generateCrandomMatricesArray
#define LAPACKE_Xgetrf					LAPACKE_cgetrf
#endif

#ifdef PREC_z
typedef std::complex<double> Scalar;
typedef double Real;
#define generateArrayOfPointers 		generateZArrayOfPointers
#define generateArrayOfPointersHost		generateZArrayOfPointersHost
#define generateRealArrayOfPointersHost	generateDArrayOfPointersHost
#define generateArrayOffsetPointers     generateZArrayOffsetPointers
#define generateArrayOffsetPointersHost generateZArrayOffsetPointersHost
#define generate_randomMatricesArray	generateZrandomMatricesArray
#define LAPACKE_Xgetrf					LAPACKE_zgetrf
#endif

typedef Scalar*	 ScalarArray;
typedef Scalar** ScalarPtrArray;
typedef Real*	 RealArray;
typedef Real**   RealPtrArray;
typedef int*	 IntArray;
typedef int**	 IntPtrArray;

typedef unsigned char WS_Byte;

#define SYNC_TIMERS(timers, run_time) \
	max_time = -1;	\
	for(int g = 0; g < num_gpus; g++) \
	{ \
		cudaSetDevice(opts.devices[g]); \
		double gpu_time = gpuTimerToc(timers[g]); \
		if(max_time < gpu_time)	\
			max_time = gpu_time; \
	} \
	run_time[i] = max_time;


void syncGPUs(kblas_opts* opts)
{
	for(int g = 0; g < opts->ngpu; g++)
	{
		cudaSetDevice(opts->devices[g]);
		cudaDeviceSynchronize();
	}
}

void cblas_gemm(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const int m, const int n, const int k, const float alpha, const float *a, const int lda, const float *b, const int ldb, const float beta, float *c, const int ldc)
{ cblas_sgemm(CblasColMajor, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }

void cblas_gemm(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const int m, const int n, const int k, const double alpha, const double *a, const int lda, const double *b, const int ldb, const double beta, double *c, const int ldc)
{ cblas_dgemm(CblasColMajor, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }

void cblas_gemm(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const int m, const int n, const int k, const float alpha, const std::complex<float> *a, const int lda, const std::complex<float> *b, const int ldb, const float beta, std::complex<float>*c, const int ldc)
{ cblas_cgemm(CblasColMajor, transa, transb, m, n, k, (void*)&alpha, (const void*)a, lda, (const void*)b, ldb, (void*)&beta, (void*)c, ldc); }

void cblas_gemm(const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const int m, const int n, const int k, const double alpha, const std::complex<double> *a, const int lda, const std::complex<double> *b, const int ldb, const double beta, std::complex<double>*c, const int ldc)
{ cblas_zgemm(CblasColMajor, transa, transb, m, n, k, (void*)&alpha, (const void*)a, lda, (const void*)b, ldb, (void*)&beta, (void*)c, ldc); }

template<class T>
double batch_lu_cpu(T **A_batch, int *lda_batch, int **ipiv_batch, int *m_batch, int *n_batch, int num_ops, int num_threads)
{
	double total_time = gettime();

	#pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i < num_ops; i++)
	{
		int info = LAPACKE_Xgetrf(LAPACK_COL_MAJOR, m_batch[i], n_batch[i], A_batch[i], lda_batch[i], ipiv_batch[i]);
		if(info != 0)
			printf("Failed to factorize operation %d (info = %d)\n", i, info);
	}

	total_time = gettime() - total_time;
	return total_time;
}

template<class T>
double solution_error(T **LU_batch, int *ldlu_batch, T **A_batch, int *lda_batch, int **ipiv_batch, int *m_batch, int *n_batch, int num_ops, int num_threads)
{
	double err = 0;
	
	#pragma omp parallel for num_threads(num_threads)
    for(int op = 0; op < num_ops; op++)
	{
		int m = m_batch[op], n = n_batch[op];
		T* LU = LU_batch[op], *A = A_batch[op];
		int ldlu = ldlu_batch[op], lda = lda_batch[op];
		int* ipiv = ipiv_batch[op];
		
		int rank = std::min(m, n);
		std::vector<T> L(m * rank, 0), U(rank * n, 0), prod(m * n, 0);
		std::vector<int> P(m, 0);
		for(int i = 0; i < m; i++)
			P[i] = i;
		for(int i = 0; i < rank; i++)
			std::swap(P[ipiv[i]-1], P[i]);

		for(int i = 0; i < m; i++)
			for(int j = 0; j <= std::min(i, n - 1); j++)
				L[i + j * m] = (i == j ? (T)1.0 : LU[i + j * ldlu]);

		for(int i = 0; i < rank; i++)
			for(int j = i; j < n; j++)
				U[i + j * rank] = LU[i + j * ldlu];
		
		cblas_gemm(CblasNoTrans, CblasNoTrans, m, n, rank, 1, &L[0], m, &U[0], rank, 0, &prod[0], m);
	
		double op_err = 0;
		for(int i = 0; i < m; i++)
		{
			for(int j = 0; j < n; j++)
			{
				double entry_err = fabs(prod[i + j * m] - A[P[i] + j * lda]);
				op_err += entry_err * entry_err;
			}
		}
		
		op_err = sqrt(op_err);
		
		if(op_err > 1e-5)
			printf("Error of %e at operation %d\n", op_err, op);
		
		err += op_err;
	}
	
	return err / num_ops;
}

void printResults(Scalar **A_batch, int *lda_batch, int* m_batch, int* n_batch, int batchCount, const char *label)
{
	for(int i = 0; i < batchCount; i++)
	{
		printf("%s_%d = [\n", label, i);
		for(int r = 0; r < m_batch[i]; r++)
		{
			for(int c = 0; c < n_batch[i]; c++)
				std::cout << std::setprecision(15) << A_batch[i][r + c * lda_batch[i]] << " ";
			std::cout << std::endl;
		}
		std::cout << "];" << std::endl;
    }
}

void printResults(int **ipiv_ptrs, int* m_batch, int* n_batch, int batchCount, const char *label)
{
	for(int i = 0; i < batchCount; i++)
	{
		printf("%s_%d = [\n", label, i);
		for(int j = 0; j < std::min(m_batch[i], n_batch[i]); j++)
			printf("%d ", ipiv_ptrs[i][j]);
		printf("];\n");
	}
}

int main(int argc, char** argv)
{
	kblas_opts opts;
	parse_opts(argc, argv, &opts);
	int num_gpus = opts.ngpu;
	int max_rows, max_cols;
	int batchCount, batchCount_gpu;
	int nruns = 1; //opts.nruns;
	double max_time;
	
	// Host stuff
	ScalarArray h_M, h_M_copy;
	ScalarPtrArray h_M_ptrs, h_M_copy_ptrs;
	IntPtrArray h_ipiv_ptrs;
	IntArray h_rows_batch, h_cols_batch, h_ipiv, h_ldm_batch;
	size_t* h_offsets, h_subset_sizes[num_gpus], h_subset_offsets[num_gpus];
	RealArray h_svals;
	RealPtrArray h_svals_ptrs;
	
	// GPU stuff
	ScalarArray d_M[num_gpus];
	ScalarPtrArray d_M_ptrs[num_gpus];
	IntPtrArray d_ipiv_ptrs[num_gpus];
	IntArray d_rows_batch[num_gpus], d_cols_batch[num_gpus], d_ipiv[num_gpus], d_info_batch[num_gpus], d_ldm_batch[num_gpus];
	size_t* d_offsets[num_gpus];
	
	GPU_Timer_t magma_timers[num_gpus], kblas_timers[num_gpus];
	kblasHandle_t kblas_handles[num_gpus];
	kblasRandState_t rand_state[num_gpus];
	
	double kblas_getrf_time[nruns], magma_time[nruns];
	
	magma_init();
	magma_queue_t queues[num_gpus];
	
	for(int g = 0; g < num_gpus; g++)
	{
		cudaSetDevice(opts.devices[g]);
		
		#ifdef PREC_d
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		#endif

		// Init kblas handles and timers
		kblasCreate(&kblas_handles[g]);
		kblas_timers[g] = newGPU_Timer(kblasGetStream(kblas_handles[g]));
		kblasInitRandState(kblas_handles[g], &rand_state[g], 16384*2, 0);
		kblasEnableMagma(kblas_handles[g]);
		
		// Init magma queue and timer
		magma_queue_create(opts.devices[g], &queues[g]);
		magma_timers[g] = newGPU_Timer(magma_queue_get_cuda_stream(queues[g]));
    }

	// printf("%-15s%-10s%-10s%-15s%-15s%-15s%-15s%-15s%-15s%-15s\n", "batchCount", "N", "K", "rank", "svdTime", "araTime", "ARA_GFlops", "avgError", "avgRankDiff", "sdevRankDiff");
	// printf("=====================================================================================================================================\n");

	int sgen_mode = 2;
	Real decay_min = 0.2, decay_max = 0.3;
	
	for(int itest = 0; itest < opts.ntest; ++itest)
	{
		max_rows = opts.msize[itest];
		max_cols = opts.nsize[itest];
		int max_rank = kmin(max_rows, max_cols);
		
		for(int iter = 0; iter < opts.niter; ++iter)
		{
			for(int btest = 0; btest < opts.btest; ++btest)
			{
				batchCount = opts.batchCount;
				if(opts.btest > 1)
					batchCount = opts.batch[btest];

				batchCount_gpu = batchCount / num_gpus;
				
				// Generate random matrix dimensions 
				TESTING_MALLOC_CPU(h_rows_batch, int, batchCount);
				TESTING_MALLOC_CPU(h_cols_batch, int, batchCount);
				TESTING_MALLOC_CPU(h_ldm_batch,  int, batchCount);
				
				generateRandDimensions(max_rows / 2, max_rows, h_rows_batch, 1234, batchCount);
				generateRandDimensions(max_cols / 2, max_cols, h_cols_batch, 5678, batchCount);
				
				// h_rows_batch[0] = h_cols_batch[0] = 18;
				
				// Pad to multiple of 32
				// for(size_t i = 0; i < batchCount; i++) 
				// 	h_ldm_batch[i] = iDivUp(h_rows_batch[i], 32) * 32;
				
				// Same ld as rows
				for(size_t i = 0; i < batchCount; i++) 
					h_ldm_batch[i] = h_rows_batch[i];
				
				// Determine total number of entries in the batch for each ld[i]*cols[i]
				size_t host_entries = getDimProductEntriesHost(h_ldm_batch, h_cols_batch, batchCount);
				
				////////////////////////////////////////////////////////////////////////
				// Set up the data
				////////////////////////////////////////////////////////////////////////
				TESTING_MALLOC_CPU(h_M, Scalar, host_entries);
				TESTING_MALLOC_CPU(h_M_copy, Scalar, host_entries);
				TESTING_MALLOC_CPU(h_M_ptrs,  Scalar*, batchCount);
				TESTING_MALLOC_CPU(h_M_copy_ptrs,  Scalar*, batchCount);
				TESTING_MALLOC_CPU(h_svals_ptrs,  Real*, batchCount);
				TESTING_MALLOC_CPU(h_ipiv_ptrs,  int*, batchCount);
				TESTING_MALLOC_CPU(h_svals, Real, batchCount * max_rank);
				TESTING_MALLOC_CPU(h_ipiv, int, batchCount * max_rank);
				TESTING_MALLOC_CPU(h_offsets, size_t, batchCount);

				fillIntArray(h_ipiv, 0, batchCount * max_rank);
				
				generateDimProductScanOffsetsHost(h_offsets, h_ldm_batch, h_cols_batch, batchCount);
				generateArrayOffsetPointersHost(h_M, h_M_ptrs, h_offsets, batchCount);
				generateArrayOffsetPointersHost(h_M_copy, h_M_copy_ptrs, h_offsets, batchCount);
				generateRealArrayOfPointersHost(h_svals, h_svals_ptrs, max_rank, batchCount);
				generateIArrayOfPointersHost(h_ipiv, h_ipiv_ptrs, max_rank, batchCount);
				
				generate_randomMatricesArray(
					h_M_ptrs, h_ldm_batch, h_svals_ptrs, h_rows_batch, h_cols_batch,
					sgen_mode, 0, decay_min, decay_max, 0, batchCount, 16
				);
				
				// FILE * pFile = fopen("F18.bin", "rb");
				// size_t read_entries = fread(h_M_ptrs[0], sizeof(double), 18 * 18, pFile);
				// printf("%zu bytes read\n", read_entries);
				// fclose(pFile);
				// 
				// printResults(h_M_ptrs, h_ldm_batch, h_rows_batch, h_cols_batch, batchCount, "M");
				
				std::copy(h_M, h_M + host_entries, h_M_copy);
			
				// Allocate on the GPUs 
				h_subset_offsets[0] = 0;
				for(int g = 0; g < num_gpus; g++)
				{
					cudaSetDevice(opts.devices[g]);
					
					TESTING_MALLOC_DEV(d_rows_batch[g], int , batchCount_gpu);
					TESTING_MALLOC_DEV(d_cols_batch[g], int , batchCount_gpu);
					TESTING_MALLOC_DEV(d_ldm_batch[g],  int , batchCount_gpu);
					TESTING_MALLOC_DEV(d_info_batch[g], int , batchCount_gpu);
					TESTING_MALLOC_DEV(d_ipiv_ptrs[g],  int*, batchCount_gpu);
					TESTING_MALLOC_DEV(d_offsets[g],  size_t, batchCount_gpu);
					TESTING_MALLOC_DEV(d_M_ptrs[g], Scalar* , batchCount_gpu);
					
					h_subset_sizes[g] = getDimProductEntriesHost(h_ldm_batch + batchCount_gpu * g, h_cols_batch + batchCount_gpu * g, batchCount_gpu);
					if(g > 1) h_subset_offsets[g] = h_subset_offsets[g - 1] + h_subset_sizes[g];
					
					check_error( cudaMemcpy(d_rows_batch[g], h_rows_batch + batchCount_gpu * g, batchCount_gpu * sizeof(int), cudaMemcpyHostToDevice) );
					check_error( cudaMemcpy(d_cols_batch[g], h_cols_batch + batchCount_gpu * g, batchCount_gpu * sizeof(int), cudaMemcpyHostToDevice) );
					check_error( cudaMemcpy(d_ldm_batch[g],  h_ldm_batch  + batchCount_gpu * g, batchCount_gpu * sizeof(int), cudaMemcpyHostToDevice) );
				}
		
				for(int g = 0; g < num_gpus; g++)
				{
					cudaSetDevice(opts.devices[g]);
					
					TESTING_MALLOC_DEV(d_M[g], Scalar, h_subset_sizes[g]);
					TESTING_MALLOC_DEV(d_ipiv[g], int, batchCount_gpu * max_rank);
					
					generateDimProductScanOffsets(d_offsets[g], d_ldm_batch[g], d_cols_batch[g], batchCount_gpu, 0);
					generateArrayOffsetPointers(d_M[g], d_M_ptrs[g], d_offsets[g], batchCount_gpu, 0);
					generateIArrayOfPointers(d_ipiv[g], d_ipiv_ptrs[g], max_rank, batchCount_gpu, 0);
					fillGPUIntArray(d_ipiv[g], 0, batchCount_gpu * max_rank, 0);
					fillGPUIntArray(d_info_batch[g], 0, batchCount_gpu, 0);
					
					kblas_gemm_batch_nonuniform_wsquery(kblas_handles[g]);
					kblasAllocateWorkspace(kblas_handles[g]);
				}
				
				// Test on the GPU 
				for(int i = 0; i < nruns; i++)
				{	
					for(int g = 0; g < num_gpus; g++)
					{
						check_error( cudaMemcpy(d_M[g], h_M_copy + h_subset_offsets[g], h_subset_sizes[g] * sizeof(Scalar), cudaMemcpyHostToDevice) );
						check_error( cudaMemcpy(d_ipiv[g], h_ipiv + max_rank * batchCount_gpu * g, max_rank * batchCount_gpu * sizeof(int), cudaMemcpyHostToDevice) );
					}
					syncGPUs(&opts);
					
					for(int g = 0; g < num_gpus; g++)
					{
						typedef typename kblas_base_type<Scalar>::device_type kblas_type;
						printf("Allocating workspace...\n");
						// size_t lu_ws_bytes = kblas_getrf_vbatch_workspace<kblas_type>(kblas_handles[g], d_rows_batch[g], d_cols_batch[g], batchCount_gpu);
						size_t lu_ws_bytes = kblas_getrf_vbatch_workspace_host<kblas_type>(kblas_handles[g], h_rows_batch + batchCount_gpu * g, h_cols_batch + batchCount_gpu * g, batchCount_gpu);
						printf("%zu ws bytes needed\n", lu_ws_bytes);
						
						void* lu_ws_ptr;
						TESTING_MALLOC_DEV(lu_ws_ptr, WS_Byte, lu_ws_bytes);
						
						cudaSetDevice(opts.devices[g]);
						gpuTimerTic(kblas_timers[g]);
						
						check_kblas_error( kblas_getrf_vbatch(
							kblas_handles[g], d_rows_batch[g], d_cols_batch[g], max_rows, max_cols, 
							(kblas_type**)d_M_ptrs[g], d_ldm_batch[g], d_ipiv_ptrs[g], lu_ws_ptr, lu_ws_bytes,
							d_info_batch[g], batchCount_gpu
						) );
						
						gpuTimerRecordEnd(kblas_timers[g]);
						
						TESTING_FREE_DEV(lu_ws_ptr);
					}
					SYNC_TIMERS(kblas_timers, kblas_getrf_time);
					
					for(int g = 0; g < num_gpus; g++)
						printf("KBLAS time = %.2f ms\n", kblas_getrf_time[g] * 1000);
					
					for(int g = 0; g < num_gpus; g++)
					{
						check_error( cudaMemcpy(h_M + h_subset_offsets[g], d_M[g], h_subset_sizes[g] * sizeof(Scalar), cudaMemcpyDeviceToHost) );
						check_error( cudaMemcpy(h_ipiv + max_rank * batchCount_gpu * g, d_ipiv[g], max_rank * batchCount_gpu * sizeof(int), cudaMemcpyDeviceToHost) );
					}
					syncGPUs(&opts);
					
					double gpu_err = solution_error<Scalar>(h_M_ptrs, h_ldm_batch, h_M_copy_ptrs, h_ldm_batch, h_ipiv_ptrs, h_rows_batch, h_cols_batch, batchCount, 16);
					printf("KBLAS err = %e\n", gpu_err);
					
					// printResults(h_M_ptrs, h_ldm_batch, h_rows_batch, h_cols_batch, batchCount, "LU");
				}
				
				// Run on the CPU 
				std::copy(h_M_copy, h_M_copy + host_entries, h_M);
				
				double cpu_time = batch_lu_cpu(h_M_ptrs, h_ldm_batch, h_ipiv_ptrs, h_rows_batch, h_cols_batch, batchCount, 16);
				printf("CPU LU time = %.2f ms\n", cpu_time * 1000);
				
				double err = solution_error<Scalar>(h_M_ptrs, h_ldm_batch, h_M_copy_ptrs, h_ldm_batch, h_ipiv_ptrs, h_rows_batch, h_cols_batch, batchCount, 16);
				printf("CPU err = %e\n", err);
				
				// Free the data
				TESTING_FREE_CPU(h_M);
				TESTING_FREE_CPU(h_M_copy);
				TESTING_FREE_CPU(h_svals);
				TESTING_FREE_CPU(h_ipiv);
				TESTING_FREE_CPU(h_rows_batch);
				TESTING_FREE_CPU(h_cols_batch);
				TESTING_FREE_CPU(h_ldm_batch);
				TESTING_FREE_CPU(h_M_ptrs);
				TESTING_FREE_CPU(h_M_copy_ptrs);
				TESTING_FREE_CPU(h_svals_ptrs);
				TESTING_FREE_CPU(h_ipiv_ptrs);
				TESTING_FREE_CPU(h_offsets);
				
				for(int g = 0; g < num_gpus; g++)
				{
					cudaSetDevice(opts.devices[g]);
					TESTING_FREE_DEV(d_M[g]);
					TESTING_FREE_DEV(d_ipiv[g]);
					TESTING_FREE_DEV(d_rows_batch[g]);
					TESTING_FREE_DEV(d_cols_batch[g]);
					TESTING_FREE_DEV(d_ldm_batch[g]);
					TESTING_FREE_DEV(d_info_batch[g]);
					TESTING_FREE_DEV(d_M_ptrs[g]);
					TESTING_FREE_DEV(d_ipiv_ptrs[g]);
				}
			}
		}
	}
	
    return 0;
}

