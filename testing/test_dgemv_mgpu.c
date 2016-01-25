#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include "kblas.h"
#include "testing_utils.h"

#define FMULS_GEMV(n) ((n) * (n) + 2. * (n))
#define FADDS_GEMV(n) ((n) * (n)           )

#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 6. * FMULS_GEMV(n) + 2. * FADDS_GEMV(n))
#else
#define FLOPS(n) (      FMULS_GEMV(n) +      FADDS_GEMV(n))
#endif

#define EVENT_TIMING

int main(int argc, char** argv)
{
	if(argc < 7)
	{
		printf("USAGE: %s <ngpus> <no-trans'n' or trans't' or conj-trans'c'> <start-dim> <stop-dim> <step-dim> <offset>\n", argv[0]); 
		printf("==> <ngpus>: Number of GPUs to use \n");
		printf("==> <no-trans'n' or trans't' or conj-trans'c'>: Process the matrix in non-transposed,transposed, or conjugate transposed configuration \n");
		printf("==> <start-dim> <stop-dim> <step-dim>: test for dimensions in the range start-dim : stop-dim with step step-dim \n");
		printf("==> <offset>: skip the [0:offset-1] rows and the [0:offset-1] columns \n");
		exit(-1);
	}
	
	int ngpus = atoi(argv[1]);
	char trans = *argv[2];
	int istart = atoi(argv[3]);
	int istop = atoi(argv[4]);
	int istep = atoi(argv[5]);
	int offset = atoi(argv[6]);
	
	const int nruns = NRUNS;
	int k, r, dim, j, i, m;
	
	int gpus_avail; 
	cudaGetDeviceCount(&gpus_avail);
	if(ngpus > gpus_avail){printf("Error: Can't run on %d gpus, only %d gpus are available \n", ngpus, gpus_avail); exit(-1);}
	
	int ngpus_local = ngpus;	// for now (no mpi)
	
	cudaStream_t streams[MAX_NGPUS][MAX_STREAMS];
	
	cublasHandle_t cublas_handle;
	cublasAtomicsMode_t mode = CUBLAS_ATOMICS_ALLOWED;
	cublasCreate(&cublas_handle);
	cublasSetAtomicsMode(cublas_handle, mode);
	
	// create streams
	for(k = 0; k < ngpus; k++)
	{
		cudaSetDevice(k);
		cudaStreamCreate(&streams[k][0]);
	}

	int M = istop;
    int N = M;
    int LDA = M;
    int incx = 1;
	int incy = 1;
	cublasOperation_t trans_;
	int alloc = 1, alloc_mgpu = 1;
	if(trans == 'N' || trans == 'n')
		trans_ = CUBLAS_OP_N;
	else if (trans == 'T' || trans == 't')
		trans_ = CUBLAS_OP_T;
	else if (trans == 'C' || trans == 'c')
		trans_ = CUBLAS_OP_C;
		
	double alpha = 2.3, beta = -1.0;
	
	cudaError_t err;
	cudaEvent_t start, stop; 
	
	cudaSetDevice(0);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	#ifdef EVENT_TIMING
	cudaEvent_t _start[MAX_NGPUS], _stop[MAX_NGPUS];
	for(k = 0; k < ngpus; k++)
	{
		cudaSetDevice(k);
		cudaEventCreate(&_start[k]);
		cudaEventCreate(&_stop[k]);
	}
	#endif
	
    // point to host memory
    double* A = NULL;
    double* x = NULL;
    double* y = NULL;
    double* ysingle = NULL;
    double* ykblas[MAX_NGPUS] = {NULL};
    
    if(trans == 'N' || trans == 'n')printf("Non-transposed test .. \n");
	else if (trans == 'T' || trans == 't') printf("Transposed case test .. \n");
	else if (trans == 'C' || trans == 'c') printf("Conjugate transposed test .. \n");
	else { printf("transpose/non-transpose configuration is not properly specified\n"); exit(-1);}
	
	// cpu alloc / init
	{
		int vecsize_x = N * abs(incx);
		int vecsize_y = M * abs(incy);
		// alloc memory cpu
		printf("Allocating matrices on cpu .. \n");
    	A = (double*)malloc(N*LDA*sizeof(double));
    	x = (double*)malloc(vecsize_x*sizeof(double));
    	y = (double*)malloc(vecsize_y*sizeof(double));
    	ysingle = (double*)malloc(vecsize_y*sizeof(double));
    	for(k = 0; k < ngpus; k++)
    		ykblas[k] = (double*)malloc(vecsize_y*sizeof(double));
    	
    	printf("Initializing on cpu .. \n");
    	// Initialize matrix and vector on cpu
    	for(i = 0; i < M; i++)
    		for(j = 0; j < N; j++)
    			A[j*LDA+i] = kblas_drand();
    
    	for(i = 0; i < vecsize_x; i++)
      		x[i] = kblas_drand();
    
    	for(i = 0; i < vecsize_y; i++)
    		y[i] = kblas_drand();
	}
	
	printf("------------------- Testing DGEMV ------------------------\n");
	printf("  Matrix       KBLAS-1 GPU     KBLAS %-2d GPU(s)    Max.  \n", ngpus);
	printf(" Dimension     (Gflop/s)         (Gflop/s)         Error  \n");
	printf("-----------   -------------   ----------------   ----------\n");

	for(dim = istart; dim <= istop; dim += istep)
	{
		float elapsedTime;
		float flops = FLOPS( (float)(dim-offset) ) / 1e6;
		float single_gpu_perf;
		float mgpu_perf;
		double error;
		int m = dim; 
		int n = m;
		int lda_ = ((m+31)/32)*32;
		
		printf("%-11d   ", m);
		
		// single gpu test
		{
			double* dA_single = NULL;
    		double* dx_single = NULL;
    		double* dy_single = NULL;
    		
    		int vecsize_x = n * abs(incx);
			int vecsize_y = m * abs(incy);
	
			alloc = 1;
    		// alloc A, x, y on gpus
    		// for cublas test
    		cudaError_t e1, e2, e3;
    		cudaSetDevice(0);
    		e1 = cudaMalloc((void**)&dA_single, n*lda_*sizeof(double));
    		e2 = cudaMalloc((void**)&dx_single, vecsize_x*sizeof(double));
    		e3 = cudaMalloc((void**)&dy_single, vecsize_y*sizeof(double));
    	
    		if((e1 != cudaSuccess) || (e2 != cudaSuccess) || (e3 != cudaSuccess) )
    		{
    			if(dA_single)cudaFree(dA_single);
    			if(dx_single)cudaFree(dx_single);
    			if(dy_single)cudaFree(dy_single);
    			alloc = 0;
    			printf("-----           ");
    		}
    		
    		if(alloc == 1)
    		{
    			// offload A, y to gpus
    			cudaSetDevice(0);
    			cublasSetMatrix(m, n, sizeof(double), A, LDA, dA_single, lda_);
    			
    			float time = 0;
    			elapsedTime = 0.0;
    			for(r = 0; r < nruns; r++)
    			{
    				cudaMemcpy(dy_single, y, vecsize_y*sizeof(double), cudaMemcpyHostToDevice);
    				cudaMemcpy(dx_single, x, vecsize_x*sizeof(double), cudaMemcpyHostToDevice);
    				
    				// handle offset
    				double* da = dA_single + (offset * lda_) + offset; 
    				double* dx = dx_single + (offset * incx); 
    				double* dy = dy_single + (offset * incy);
    				int m_ = m-offset; 
    				int n_ = n-offset;
    			
    				cudaSetDevice(0);
      				cudaEventRecord(start, 0);
      				//cublasStatus_t s = cublasDgemv(cublas_handle, trans_, m, n, &alpha, dA_single, lda_, dx_single, incx, &beta, dy_single, incy);
      				kblas_dgemv(trans, m_, n_, alpha, da, lda_, dx, incx, beta, dy, incy);
      				cudaEventRecord(stop, 0);
      				cudaEventSynchronize(stop);
      				time = 0.0;
      				cudaEventElapsedTime(&time, start, stop);
      				elapsedTime += time;
      			}
      			elapsedTime /= nruns;
      			single_gpu_perf = flops / elapsedTime;
	
				cudaMemcpy(ysingle, dy_single, vecsize_y * sizeof(double), cudaMemcpyDeviceToHost);
				
				printf("%-13.2f   ", single_gpu_perf);
				
				if(dA_single)cudaFree(dA_single);
				if(dx_single)cudaFree(dx_single);
				if(dy_single)cudaFree(dy_single);
			}
		} // end of 1 gpu test
		
		// mgpu test
		{
			alloc_mgpu = 1;
    		// point to device memory
    		double* dA[MAX_NGPUS] = {NULL};
    		double* dx[MAX_NGPUS] = {NULL};
    		double* dy[MAX_NGPUS] = {NULL};
    		
    		int vecsize_x = n * abs(incx);
			int vecsize_y = m * abs(incy);

    		// for kblas test
    		// alloc
    		int nb = get_dgemv_mgpu_bs(trans);
    		kblas_dmalloc_mgpu_1D(	m, n, dA, ngpus, lda_, nb);
    		
    		for(k = 0; k < ngpus; k++)
    		{
    			// check allocation
    			if(dA[k] == NULL) {alloc_mgpu = 0; break;} 
    		}
    		
    		cudaError_t e1, e2;
    		for(k = 0; k < ngpus; k++)
    		{
    			cudaSetDevice(k);
    			e1 = cudaMalloc((void**)&dx[k], vecsize_x*sizeof(double));
    			e2 = cudaMalloc((void**)&dy[k], vecsize_y*sizeof(double));
    			if( (e1 != cudaSuccess) || (e2 != cudaSuccess) )
    			{
    				alloc_mgpu = 0; break;
    			}
    			// init y
    			cudaMemset(dy[k], 0, vecsize_y*sizeof(double));
    		}
    		
    		if(alloc_mgpu == 1)
    		{
    			// offload (broadcast) x to all gpus
    			for(k = 0; k < ngpus; k++)
    			{
    				cudaSetDevice(k);
    				cudaError_t e = cudaMemcpy(dx[k], x, vecsize_x*sizeof(double), cudaMemcpyHostToDevice);
					//printf("gpu %d: cudaMemcpy x - %s \n", k, cudaGetErrorString(e));
				}
			
				// offload A and y on gpus
				kblas_dsetmatrix_mgpu_1D(m, n, A, LDA, dA, lda_, ngpus, nb);
				
				float time = 0;
				float _time = 0;
				
				elapsedTime = 0.0;
				for(r = 0; r < nruns; r++)
				{
					// offload (broadcast) y to all gpus
    			    for(k = 0; k < ngpus; k++)
    			    {
    				    cudaSetDevice(k);
    				    cudaError_t e = cudaMemcpy(dy[k], y, vecsize_y*sizeof(double), cudaMemcpyHostToDevice);
				    }
    				
    				// ---- kblas
      				cudaSetDevice(0);
      				cudaEventRecord(start, 0);
      				
      				#ifdef EVENT_TIMING
      				for(k = 0; k < ngpus; k++){cudaSetDevice(k); cudaEventRecord(_start[k], streams[k][0]);}
      				#endif
      				
      				kblas_dgemv_mgpu_async(trans, m, n, alpha, dA, lda_, dx, incx, beta, dy, incy, ngpus, offset, offset, streams);
      				
      				#ifdef EVENT_TIMING
      				for(k = 0; k < ngpus; k++){cudaSetDevice(k); cudaEventRecord(_stop[k], streams[k][0]);}
      				#endif
      				
      				// sync
					for(k = 0; k < ngpus; k++)
					{
						cudaSetDevice(k);
						cudaStreamSynchronize(streams[k][0]);
					}
      				cudaSetDevice(0);
      				cudaEventRecord(stop, 0);
      				cudaEventSynchronize(stop);
      				cudaEventElapsedTime(&time, start, stop);
      				
      				// event timing
      				#ifdef EVENT_TIMING
      				float gpu_time = 0, max_gpu_time = 0;
      				for(k = 0; k < ngpus; k++)
					{
						
						cudaSetDevice(k);
						cudaEventElapsedTime(&gpu_time, _start[k], _stop[k]);
						if(gpu_time > max_gpu_time) max_gpu_time = gpu_time;
					}
      				_time = max_gpu_time;
      				#endif
      				// end of event timing
      				
      				#ifndef EVENT_TIMING
      				elapsedTime += time;
      				#else
      				elapsedTime += _time;
      				#endif
      			}
      			
      			for(k = 0; k < ngpus; k++)
      			{
      				cudaSetDevice(k);
      				cudaMemcpy(ykblas[k], dy[k], vecsize_y * sizeof(double), cudaMemcpyDeviceToHost);
      			}
      			elapsedTime /= nruns;
      			mgpu_perf = flops / elapsedTime;
      			//reduce the result
      			//for(k = 0; k < vecsize_y; k++)
      			//	ykblas[0][k] = ykblas[1][k];
      			for(k = 0; k < vecsize_y; k++)
      				for(j = 1; j < ngpus_local; j++)
      					ykblas[0][k] += ykblas[j][k];
      			
      			printf("%-16.2f   ", mgpu_perf);
      		}
      		else printf("-----           ");
      		
      		for(k = 0; k < ngpus_local; k++)
			{
				cudaSetDevice(k);
				if(dA[k])cudaFree(dA[k]);
				if(dx[k])cudaFree(dx[k]);
				if(dy[k])cudaFree(dy[k]);
			}
    	} // end of mgpu test
    	
    	// testing correctness
    	{
    		if(alloc == 1 && alloc_mgpu == 1)
    		{
    			// testing error -- specify ref. vector and result vector
      			double* yref = ysingle + (offset * incy); 
      			double* yres = ykblas[0] + (offset * incy);
      			error = dget_max_error(yref, yres, m-offset, incy);
    			//print
    			printf("%-10e;\n", error);
    		}
    		else
    			printf ("N/A       \n");
    	}
    	
    	if(alloc == 0 && alloc_mgpu == 0)break;
	} // end of for loop 
	
	// finalize
	cudaSetDevice(0);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	#ifdef EVENT_TIMING
	for(k = 0; k < ngpus; k++)
	{
		cudaSetDevice(k);
		cudaEventDestroy(_start[k]);
		cudaEventDestroy(_stop[k]);
	}
	#endif
	
	// destroy streams
	for(k = 0; k < ngpus; k++)
	{
		cudaSetDevice(k);
		cudaStreamDestroy(streams[k][0]);
	}

    if(A)free(A);
    if(x)free(x);
    if(ysingle)free(ysingle);
    for(k = 0; k < ngpus_local; k++)if(ykblas[k])free(ykblas[k]);
	cublasDestroy(cublas_handle);
    return EXIT_SUCCESS;
}

