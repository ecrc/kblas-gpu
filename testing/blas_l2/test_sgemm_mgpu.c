/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_l2/test_sgemm_mgpu.c

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ahmad Abdelfattah
 * @date 2020-12-10
 **/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "kblas.h"
#include "testing_utils.h"

#define FMULS_GEMM(m_, n_, k_) ((m_) * (n_) * (k_))
#define FADDS_GEMM(m_, n_, k_) ((m_) * (n_) * (k_))

#define FLOPS_ZGEMM(m_, n_, k_) (6. * FMULS_GEMM((double)(m_), (double)(n_), (double)(k_)) + 2.0 * FADDS_GEMM((double)(m_), (double)(n_), (double)(k_)) )
#define FLOPS_CGEMM(m_, n_, k_) (6. * FMULS_GEMM((double)(m_), (double)(n_), (double)(k_)) + 2.0 * FADDS_GEMM((double)(m_), (double)(n_), (double)(k_)) )
#define FLOPS_DGEMM(m_, n_, k_) (     FMULS_GEMM((double)(m_), (double)(n_), (double)(k_)) +       FADDS_GEMM((double)(m_), (double)(n_), (double)(k_)) )
#define FLOPS_SGEMM(m_, n_, k_) (     FMULS_GEMM((double)(m_), (double)(n_), (double)(k_)) +       FADDS_GEMM((double)(m_), (double)(n_), (double)(k_)) )

#define MAX_NGPUS	(16)

int main(int argc, char** argv)
{
	if(argc < 9)
	{
		printf("USAGE: %s <ngpus-start> <ngpus-end> <transa> <transb> <start-dim> <stop-dim> <step-dim> <tile-size>\n", argv[0]);
		printf("==> <ngpus-start> <ngpus-end>: Run the test on <ngpus-start> up to <ngpus-end>, adding 1 GPU each time\n");
		printf("==> <transa>: 'n' or 'N' (no-traspose) or 't' or 'T' (transpose) or 'c' or 'C' (conjugate) \n");
		printf("==> <transb>: 'n' or 'N' (no-traspose) or 't' or 'T' (transpose) or 'c' or 'C' (conjugate) \n");
		printf("==> <start-dim> <stop-dim> <step-dim>: test for dimensions in the range start-dim : stop-dim with step step-dim \n");
		printf("==> <tile-size>: must divide every dimension to be tested\n");
		exit(-1);
	}

	long ngpus1 = atoi(argv[1]);
	long ngpus2 = atoi(argv[2]);
	char transa = *argv[3];
	char transb = *argv[4];
	long istart = atoi(argv[5]);
	long istop = atoi(argv[6]);
	long istep = atoi(argv[7]);
	long tile_size = atoi(argv[8]);

	int gpus_avail;
	cudaGetDeviceCount(&gpus_avail);
	if(ngpus1 > gpus_avail)
	{printf("Error: Can't run on %ld gpus, only %d gpus are available \n", ngpus1, gpus_avail); exit(-1);}

	if(ngpus2 > gpus_avail)
	{printf("Error: Can't run on %ld gpus, only %d gpus are available \n", ngpus2, gpus_avail); exit(-1);}

	if(ngpus1 > ngpus2)
	{printf("Error: ngpus-end is larger than ngpu-start \n"); exit(1);}

	long *gpu_id = (long*)malloc(ngpus2 * sizeof(long));

	// init ids - for now assume one node
	long k;
	for(k = 0; k < ngpus2; k++)
		gpu_id[k] = (long)k;

	long M = istop;
    long N = M;
    long K = M;
    long LDA = M;
    long LDB = K;
    long LDC = M;

    cublasOperation_t transa_, transb_;

	if(transa == 'N' || transa == 'n')
		transa_ = CUBLAS_OP_N;
	else if (transa == 'T' || transa == 't')
		transa_ = CUBLAS_OP_T;
	else if (transa == 'C' || transa == 'c')
		transa_ = CUBLAS_OP_C;
	else
		{printf("wrong parameter transa = %c\n", transa); exit(1);}

	if(transb == 'N' || transb == 'n')
		transb_ = CUBLAS_OP_N;
	else if (transb == 'T' || transb == 't')
		transb_ = CUBLAS_OP_T;
	else if (transb == 'C' || transb == 'c')
		transb_ = CUBLAS_OP_C;
	else
		{printf("wrong parameter transb = %c\n", transb); exit(1);}

	//float alpha = 2.3, beta = -1.0;
	float alpha = 1.0, beta = 0.0;

	cudaError_t err;
	cudaEvent_t start, stop;

	cudaSetDevice(gpu_id[0]);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    // point to host memory
    float* A = NULL;
    float* B = NULL;
    float* C = NULL;
    float* csingle = NULL;
    float* cmgpu = NULL;

    printf("transa = %c - transb = %c .. \n", transa, transb);

    long size_a = M * LDA;
	long size_b = K * LDB;
	long size_c = M * LDC;

	// cpu alloc / init
	{
		// alloc memory cpu
		printf("Allocating matrices on cpu .. \n");

    	//A = (float*)malloc(size_a*sizeof(float));
    	//B = (float*)malloc(size_b*sizeof(float));
    	C = (float*)malloc(size_c*sizeof(float));
    	csingle = (float*)malloc(size_c*sizeof(float));
    	//cmgpu = (float*)malloc(size_c*sizeof(float));

    	cudaMallocHost((void**)&A, size_a*sizeof(float));			// better for mgpu version
    	cudaMallocHost((void**)&B, size_b*sizeof(float));			// better for mgpu version
    	//cudaMallocHost((void**)&C, size_c*sizeof(float));			// better for mgpu version
    	//cudaMallocHost((void**)&csingle, size_c*sizeof(float));	// better for mgpu version
    	cudaMallocHost((void**)&cmgpu, size_c*sizeof(float));		// better for mgpu version


    	printf("Initializing on cpu .. \n");
    	// Initialize matrix and vector on cpu
    	long i, j, m;
    	srand_matrix(M, K, A, LDA);
    	srand_matrix(K, N, B, LDB);
    	srand_matrix(M, N, C, LDC);
	}

	long alloc = 1, alloc_mgpu = 1;
	long ngpus;
	for(ngpus = ngpus1; ngpus <= ngpus2; ngpus++)
	{
		printf("-------------------------------- Testing SGEMM -------------------------------\n");
		printf("  Matrix         1 GPU        Tile Size      KBLAS %-2ld GPUs          Max.    \n", ngpus);
		printf(" Dimension     (Gflop/s)         Used     (Gflop/s)     Time(s)      Error    \n");
		printf("-----------   -------------   ---------   ----------   ----------   ----------\n");

		long dim;
		for(dim = istart; dim <= istop; dim += istep)
		{
			float elapsedTime;
			float dimf = (float)dim;
			float flops = FLOPS_SGEMM(dimf, dimf, dimf) / 1e6;
			float single_gpu_perf;
			float mgpu_perf;
			float error;
			long m = dim;
			long n = dim;
			long k = dim;
			long lda = ((k+31)/32)*32;
			long ldb = ((n+31)/32)*32;
			long ldc = ((n+31)/32)*32;

			long sizea = m * lda;
			long sizeb = k * ldb;
			long sizec = m * ldc;

			printf("%-11ld   ", m);
			//printf("check point 1\n");
			// single gpu test
			{
				float* dA_single = NULL;
    			float* dB_single = NULL;
    			float* dC_single = NULL;

    			alloc = 1;
    			// alloc A, B, C on gpus
    			// single gpu test
    			cudaError_t e1, e2, e3;
    			cudaSetDevice(gpu_id[0]);
    			e1 = cudaMalloc((void**)&dA_single, sizea*sizeof(float));
    			e2 = cudaMalloc((void**)&dB_single, sizeb*sizeof(float));
    			e3 = cudaMalloc((void**)&dC_single, sizec*sizeof(float));

    			//printf("check point 1.1\n");

    			if((e1 != cudaSuccess) || (e2 != cudaSuccess) || (e3 != cudaSuccess) )
    			{
    				if(dA_single)cudaFree(dA_single);
    				if(dB_single)cudaFree(dB_single);
    				if(dC_single)cudaFree(dC_single);
    				alloc = 0;
    				printf("-----           ");
    			}
    			//printf("check point 1.2\n");
    			if(alloc == 1)
    			{
    				// offload A, y to gpus
    				cudaSetDevice(gpu_id[0]);
    				//cublasSetKernelStream(0);

    				cublasSetMatrix((long)m, (long)k, sizeof(float), A, (long)LDA, dA_single, (long)lda);
    				cublasSetMatrix((long)k, (long)n, sizeof(float), B, (long)LDB, dB_single, (long)ldb);
    				float time = 0.0;
    				long r;
    				for(r = 0; r < (NRUNS+1); r++)
    				{
    					cublasSetMatrix((long)m, (long)n, sizeof(float), C, (long)LDC, dC_single, (long)ldc);
    					//printf("check point 1.3\n");
    					cudaSetDevice(gpu_id[0]);
      					cudaEventRecord(start, 0);
      					// call cublas sgemm
      					cublasSgemm(transa, transb,
      							(long)m, (long)n, (long)k,
      							alpha, dA_single, (long)lda,
      							dB_single, (long)ldb,
      							beta, dC_single, (long)ldc);
      					//printf("check point 1.4\n");
                		cudaEventRecord(stop, 0);
      					cudaEventSynchronize(stop);
      					elapsedTime = 0.0;
      					cudaEventElapsedTime(&elapsedTime, start, stop);
      					if(r > 0)time += elapsedTime;		// skip first iteration
      				}
      				elapsedTime = time/NRUNS;
      				single_gpu_perf = flops / elapsedTime;
					cublasGetMatrix((long)m, (long)n, sizeof(float), dC_single, (long)ldc, csingle, (long)LDC);

					printf("%-13.2f   ", single_gpu_perf);
					//printf("check point 1.5\n");
					if(dA_single)cudaFree(dA_single);
					if(dB_single)cudaFree(dB_single);
					if(dC_single)cudaFree(dC_single);
					//printf("check point 1.6\n");
				}
			} // end of 1 gpu test
			//printf("check point 2\n");
			// mgpu test
			{
				float time  = 0.0;
				long tile;
				long r;
				for(r = 0; r < (NRUNS+1); r++)
				{
    				// make a copy of C
    				memcpy(cmgpu, C, size_c * sizeof(float));

    				cudaSetDevice(gpu_id[0]);
      				cudaEventRecord(start, 0);
      				// sgemm_mgpu
      				tile = tile_size;
      				kblas_sgemm_mgpu( transa, transb,
      							(long)m, (long)n, (long)k,
      							alpha, A, (long)LDA,
      							B, (long)LDB,
      							beta, cmgpu, (long)LDC,
      							ngpus, gpu_id,
      							(long*)&tile);
      				cudaSetDevice(gpu_id[0]);
      				cudaEventRecord(stop, 0);
      				cudaEventSynchronize(stop);

      				elapsedTime = 0.0;
      				cudaEventElapsedTime(&elapsedTime, start, stop);
      				if(r > 0)time += elapsedTime;		// skip first iteration
      			}
      			elapsedTime = time / NRUNS;

      			printf("%-9ld   ", tile);

      			// flops/s including alloc time
      			mgpu_perf = flops / elapsedTime;
      			printf("%-10.2f   ", mgpu_perf);

      			// print time
      			printf("%-10.2f   ", elapsedTime * 1e-3);
    		} // end of mgpu test
    		//printf("check point 3\n");
    		// testing correctness
    		{
    			if(alloc == 1 && alloc_mgpu == 1)
    			{
    				// testing error -- specify ref. vector and result vector
      				float* cref = csingle;
      				float* cres = cmgpu;
      				//error = get_max_error(cref, cres, m , n, LDC);
    				error = sget_max_error_matrix(cref, cres, m, n, LDC);
    				printf("%-10e;\n", error);
    			}
    			else
    				printf ("N/A       \n");
    		}
    		//printf("check point 4\n");
    		if(alloc == 0 && alloc_mgpu == 0)break;

		} // end of for loop over dim
		printf("\n\n");
	} // end of for loop over ngpus
	// finalize
	cudaSetDevice(gpu_id[0]);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    //if(A)free(A);
    //if(B)free(B);
    if(C)free(C);
    if(csingle)free(csingle);
    //if(cmgpu)free(cmgpu);

    if(A)cudaFreeHost(A);
    if(B)cudaFreeHost(B);
    //if(C)cudaFreeHost(C);
    //if(csingle)cudaFreeHost(csingle);
    if(cmgpu)cudaFreeHost(cmgpu);
    if(gpu_id)free(gpu_id);

    return 0;
}

