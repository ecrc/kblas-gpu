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

#define FMULS_SYMV(n) ((n) * (n) + 2. * (n))
#define FADDS_SYMV(n) ((n) * (n)           )

#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 6. * FMULS_SYMV(n) + 2. * FADDS_SYMV(n))
#else
#define FLOPS(n) (      FMULS_SYMV(n) +      FADDS_SYMV(n))
#endif

#define EVENT_TIMING

int main(int argc, char** argv)
{
	if(argc < 7)
	{
		printf("USAGE: %s <ngpus> <upper'u'-or-lower'l'> <start-dim> <stop-dim> <step-dim> <offset>\n", argv[0]); 
		printf("==> <ngpus>: Number of GPUs to use \n");
		printf("==> <upper'u'-or-lower'l'>: Access either the upper or lower triangular part of the matrix \n");
		printf("==> <start-dim> <stop-dim> <step-dim>: test for dimensions in the range start-dim : stop-dim with step step-dim \n");
		printf("==> <offset>: skip the [0:offset-1] rows and the [0:offset-1] columns\n");
		exit(-1);
	}
	
	const int nruns = NRUNS;
	
	int ngpus = atoi(argv[1]);
	char uplo = *argv[2];
	int istart = atoi(argv[3]);
	int istop = atoi(argv[4]);
	int istep = atoi(argv[5]);
	int offset = atoi(argv[6]);
	
	int i, j, m, k, r, dim;
	
	int gpus_avail; 
	cudaGetDeviceCount(&gpus_avail);
	if(ngpus > gpus_avail){printf("Error: Can't run on %d gpus, only %d gpus are available \n", ngpus, gpus_avail); exit(-1);}
	
	int ngpus_local = ngpus;	// for now (without mpi)
	
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

	int N = istop;
    int M = N;
    int LDA = M;
    int incx = 1;
	int incy = 1;
	
	cublasFillMode_t uplo_;
	if(uplo == 'L' || uplo == 'l')
		uplo_ = CUBLAS_FILL_MODE_LOWER;
	else if (uplo == 'U' || uplo == 'u')
		uplo_ = CUBLAS_FILL_MODE_UPPER;
		
	float alpha = 2.3, beta = -1.0;
	
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
    float* A = NULL;
    float* x = NULL;
    float* y = NULL;
    float* ycuda = NULL;
    float* ykblas[MAX_NGPUS] = {NULL};
    
    if(uplo == 'L' || uplo == 'l')printf("Lower triangular test .. \n");
	else if (uplo == 'U' || uplo == 'u') printf("Upper triangular test .. \n");
	else { printf("upper/lower triangular configuration is not properly specified\n"); exit(-1);}
	
	// cpu alloc / init
	{
		int vecsize_x = N * abs(incx);
		int vecsize_y = N * abs(incy);
		// alloc memory cpu
		printf("Allocating matrices on cpu .. \n");
    	A = (float*)malloc(N*LDA*sizeof(float));
    	x = (float*)malloc(vecsize_x*sizeof(float));
    	y = (float*)malloc(vecsize_x*sizeof(float));
    	ycuda = (float*)malloc(vecsize_y*sizeof(float));
    	for(k = 0; k < ngpus; k++)
    		ykblas[k] = (float*)malloc(vecsize_y*sizeof(float));
    	
    	printf("Initializing on cpu .. \n");
    	// Initialize matrix and vector on cpu
    	for(i = 0; i < M; i++)
    		for(j = 0; j < N; j++)
    			A[j*LDA+i] = kblas_srand();
    
    	for(i = 0; i < M; i++)
      		for(j = i; j < M; j++)
				A[i * N + j] = A[j * M + i];
      
    	for(i = 0; i < vecsize_x; i++)
      		x[i] = kblas_srand();
    
    	for(i = 0; i < vecsize_y; i++)
    		y[i] = kblas_srand();
	}
	
	int alloc = 1, alloc_mgpu = 1;
	
	printf("------------------- Testing SSYMV ------------------------\n");
	printf("  Matrix       KBLAS-1 GPU     KBLAS-%-2d GPU(s)    Max.  \n", ngpus);
	printf(" Dimension     (Gflop/s)         (Gflop/s)         Error  \n");
	printf("-----------   -------------   ----------------   ---------\n");
	
	for(dim = istart; dim <= istop; dim += istep)
	{
		float elapsedTime;
		float flops = FLOPS( (float)(dim-offset) ) / 1e6;
		float single_gpu_perf;
		float mgpu_perf;
		float error;
		int m = dim;
		int n = m;
		int lda_ = ((m+31)/32)*32;
		
		printf("%-11d   ", m);
		
		// single gpu test
		{
			float* dA_single = NULL;
    		float* dx_single = NULL;
    		float* dy_single = NULL;
    		
    		int vecsize_x = m * abs(incx);
			int vecsize_y = m * abs(incy);
	
    		alloc = 1;
    		cudaError_t e1, e2, e3;
    		// alloc A, x, y on gpus
    		// for cublas test
    		cudaSetDevice(0);
    		e1 = cudaMalloc((void**)&dA_single, n*lda_*sizeof(float));
    		e2 = cudaMalloc((void**)&dx_single, vecsize_x*sizeof(float));
    		e3 = cudaMalloc((void**)&dy_single, vecsize_y*sizeof(float));
    		
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
    			cublasSetMatrix(m, m, sizeof(float), A, LDA, dA_single, lda_);
    			cudaMemcpy(dx_single, x, vecsize_x*sizeof(float), cudaMemcpyHostToDevice);
    			
    			elapsedTime  = 0;
    			for(r = 0; r < nruns; r++)
    			{
    				cudaMemcpy(dy_single, y, vecsize_y*sizeof(float), cudaMemcpyHostToDevice);
    				
    				// handle offset
    				float* da = dA_single + (offset * lda_) + offset; 
    				float* dx = dx_single + (offset * incx); 
    				float* dy = dy_single + (offset * incy);
    				int m_ = m-offset; 
    			
    				cudaSetDevice(0);
      				cudaEventRecord(start, 0);
      				//cublasStatus_t s = cublasSsymv(cublas_handle, uplo_, m_, &alpha, da, lda_, dx, incx, &beta, dy, incy);
      				kblas_ssymv(uplo, m_, alpha, da, lda_, dx, incx, beta, dy, incy);
      				cudaEventRecord(stop, 0);
      				cudaEventSynchronize(stop);
      				float time = 0.0;
      				cudaEventElapsedTime(&time, start, stop);
      				elapsedTime += time;
      			}
      			elapsedTime /= nruns;
      			single_gpu_perf = flops / elapsedTime;
				
				cudaMemcpy(ycuda, dy_single, vecsize_y * sizeof(float), cudaMemcpyDeviceToHost);
				
				printf("%-13.2f   ", single_gpu_perf);
				
				if(dA_single)cudaFree(dA_single);
				if(dx_single)cudaFree(dx_single);
				if(dy_single)cudaFree(dy_single);
			}
		} // end of 1 gpu test
		
		// mgpu test
		{
    		// point to device memory
    		float* dA[MAX_NGPUS] = {NULL};
    		float* dx[MAX_NGPUS] = {NULL};
    		float* dy[MAX_NGPUS] = {NULL};
    		
    		int vecsize_x = m * abs(incx);
			int vecsize_y = m * abs(incy);

			alloc_mgpu = 1;
    		// for kblas test
    		// alloc
    		int nb = get_ssymv_mgpu_bs(uplo);
    		kblas_smalloc_mgpu_1D(m, m, dA, ngpus, lda_, nb);
    		
    		for(k = 0; k < ngpus; k++)
    		{
    			// check allocation
    			if(dA[k] == NULL) {alloc_mgpu = 0; break;} 
    		}
    		
    		cudaError_t e1, e2;
    		for(k = 0; k < ngpus; k++)
    		{
    			cudaSetDevice(k);
    			e1 = cudaMalloc((void**)&dx[k], vecsize_x*sizeof(float));
    			e2 = cudaMalloc((void**)&dy[k], vecsize_y*sizeof(float));
    			if( (e1 != cudaSuccess) || (e2 != cudaSuccess) )
    			{
    				alloc_mgpu = 0; break;
    			}
    			// init y
    			cudaMemset(dy[k], 0, vecsize_y*sizeof(float));
    		}
    		
    		if(alloc_mgpu == 1)
    		{
    			// offload (broadcast) x to all gpus
    			for(k = 0; k < ngpus; k++)
    			{
    				cudaSetDevice(k);
    				cudaMemcpy(dx[k], x, vecsize_x*sizeof(float), cudaMemcpyHostToDevice);
				}
				
				// offload A and y on gpus
    			kblas_ssetmatrix_mgpu_1D(m, m, A, LDA, dA, lda_, ngpus, nb);
    			
				float time = 0;
				float _time = 0;
				elapsedTime = 0;
				for(r = 0; r < nruns; r++)
				{
					kblas_ssetvector_mgpu_1D(m, y, dy, ngpus, nb);
					
    				// ---- kblas
      				cudaSetDevice(0);
      				cudaEventRecord(start, 0);
      			
      				#ifdef EVENT_TIMING
      				for(k = 0; k < ngpus; k++){cudaSetDevice(k); cudaEventRecord(_start[k], streams[k][0]);}
      				#endif
      				
      				kblas_ssymv_mgpu_async(uplo, m, alpha, dA, lda_, dx, incx, beta, dy, incy, ngpus, offset, streams);
      			
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
      				time = 0;
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
      				cudaMemcpy(ykblas[k], dy[k], vecsize_y * sizeof(float), cudaMemcpyDeviceToHost);
      			}
      		
      			//reduce the result
      			for(k = 0; k < vecsize_y; k++)
      				for(j = 1; j < ngpus_local; j++)
      					ykblas[0][k] += ykblas[j][k];
      			
      			elapsedTime /= nruns;
      			mgpu_perf = flops / elapsedTime;
      			
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
      			float* yref = ycuda + (offset * incy); 
      			float* yres = ykblas[0] + (offset * incy);
      			
      			error = sget_max_error(yref, yres, m-offset, incy);
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
    if(ycuda)free(ycuda);
    for(k = 0; k < ngpus_local; k++)if(ykblas[k])free(ykblas[k]);
	cublasDestroy(cublas_handle);
    return EXIT_SUCCESS;
}

