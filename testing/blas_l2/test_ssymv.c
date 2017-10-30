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

int main(int argc, char** argv)
{
	if(argc < 6)
	{
		printf("USAGE: %s <device-id> <upper'u'-or-lower'l'> <start-dim> <stop-dim> <step-dim>\n", argv[0]); 
		printf("==> <device-id>: GPU device id to use \n");
		printf("==> <upper'u'-or-lower'l'>: Access either upper or lower triangular part of the matrix \n");
		printf("==> <start-dim> <stop-dim> <step-dim>: test for dimensions in the range start-dim : stop-dim with step step-dim \n");
		exit(-1);
	}
	
	int dev = atoi(argv[1]);
	char uplo = *argv[2];
	int istart = atoi(argv[3]);
	int istop = atoi(argv[4]);
	int istep = atoi(argv[5]);
	
	const int nruns = NRUNS;
	
	cudaError_t ed = cudaSetDevice(dev);
	if(ed != cudaSuccess){printf("Error setting device : %s \n", cudaGetErrorString(ed) ); exit(-1);}
	
	cublasHandle_t cublas_handle;
	cublasAtomicsMode_t mode = CUBLAS_ATOMICS_ALLOWED;
	cublasCreate(&cublas_handle);
	cublasSetAtomicsMode(cublas_handle, mode);
	
	struct cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	
    int M = istop;
    int N = M;
    int LDA = M;
    int LDA_ = ((M+31)/32)*32;

	int incx = 1;
	int incy = 1;
	int vecsize_x = N * abs(incx);
	int vecsize_y = M * abs(incy);
	
	cublasFillMode_t uplo_;
	if(uplo == 'L' || uplo == 'l')
		uplo_ = CUBLAS_FILL_MODE_LOWER;
	else if (uplo == 'U' || uplo == 'u')
		uplo_ = CUBLAS_FILL_MODE_UPPER;
		
	float alpha = 2.3, beta = -0.6;
	
	cudaError_t err;
	cudaEvent_t start, stop; 
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
    // point to host memory
    float* A = NULL;
    float* x = NULL;
    float* ycuda = NULL;
    float* ykblas = NULL;
	
    // point to device memory
    float* dA = NULL;
    float* dx = NULL;
    float* dy = NULL;

    if(uplo == 'L' || uplo == 'l')printf("Lower triangular test .. \n");
	else if (uplo == 'U' || uplo == 'u') printf("Upper triangular test .. \n");
	else { printf("upper/lower triangular configuration is not properly specified\n"); exit(-1);}
	printf("Allocating Matrices\n");
    A = (float*)malloc(LDA*N*sizeof(float));
    x = (float*)malloc(vecsize_x*sizeof(float));
    ycuda = (float*)malloc(vecsize_y*sizeof(float));
    ykblas = (float*)malloc(vecsize_y*sizeof(float));
    
    err = cudaMalloc((void**)&dA, N*LDA_*sizeof(float));
    if(err != cudaSuccess){printf("ERROR: %s \n", cudaGetErrorString(err)); exit(1);}
    err = cudaMalloc((void**)&dx, vecsize_x*sizeof(float));
    if(err != cudaSuccess){printf("ERROR: %s \n", cudaGetErrorString(err)); exit(1);}
    err = cudaMalloc((void**)&dy, vecsize_y*sizeof(float));
	if(err != cudaSuccess){printf("ERROR: %s \n", cudaGetErrorString(err)); exit(1);}
	
    // Initialize matrix and vector
    printf("Initializing on cpu .. \n");
    int i, j, m;
    for(i = 0; i < M; i++)
    	for(j = 0; j < N; j++)
    		A[j*LDA+i] = kblas_srand();
    
    for(i = 0; i < M; i++)
      for(j = i; j < M; j++)
		A[i * N + j] = A[j * M + i];
      
    for(i = 0; i < vecsize_x; i++)
      x[i] = kblas_srand();
    
    cudaMemcpy(dx, x, vecsize_x*sizeof(float), cudaMemcpyHostToDevice);

	printf("------------------- Testing SSYMV ----------------\n");
    printf("  Matrix        CUBLAS       KBLAS          Max.  \n");
    printf(" Dimension     (Gflop/s)   (Gflop/s)       Error  \n");
    printf("-----------   ----------   ----------   ----------\n");
    
    int r;
    for(m = istart; m <= istop; m += istep)
    {
    	float elapsedTime; 
    	
      	int lda = ((m+31)/32)*32;
      	float flops = FLOPS( (float)m ) / 1e6;
		
		cublasSetMatrix(m, m, sizeof(float), A, LDA, dA, lda);
		
		for(i = 0; i < vecsize_y; i++)
    	{
      		ycuda[i] = kblas_srand();
      		ykblas[i] = ycuda[i];
    	}
      
      	// --- cuda test
      	elapsedTime = 0;
      	for(r = 0; r < nruns; r++)
      	{
      		cudaMemcpy(dy, ycuda, vecsize_y * sizeof(float), cudaMemcpyHostToDevice);
      		
      		cudaEventRecord(start, 0);
      		cublasSsymv(cublas_handle, uplo_, m, &alpha, dA, lda, dx, incx, &beta, dy, incy);
      		cudaEventRecord(stop, 0);
      		cudaEventSynchronize(stop);
      		
      		float time = 0.0;
      		cudaEventElapsedTime(&time, start, stop);
      		elapsedTime += time;
      	}
      	elapsedTime /= nruns;
      	float cuda_perf = flops / elapsedTime;
      
      	cudaMemcpy(ycuda, dy, vecsize_y * sizeof(float), cudaMemcpyDeviceToHost);
      	// end of cuda test
      	  	
      	// ---- kblas
      	elapsedTime = 0;
      	for(r = 0; r < nruns; r++)
      	{
      		cudaMemcpy(dy, ykblas, vecsize_y * sizeof(float), cudaMemcpyHostToDevice);
      		
      		cudaEventRecord(start, 0);
      		kblas_ssymv(uplo, m, alpha, dA, lda, dx, incx, beta, dy, incy);
      		cudaEventRecord(stop, 0);
      		cudaEventSynchronize(stop);
      		float time = 0;	
      		cudaEventElapsedTime(&time, start, stop);
      		elapsedTime += time;
      	}
      	elapsedTime /= nruns;
      	float kblas_perf = flops / elapsedTime;
		
		cudaMemcpy(ykblas, dy, vecsize_y * sizeof(float), cudaMemcpyDeviceToHost);
      	
      	// testing error -- specify ref. vector and result vector
      	float* yref = ycuda; 
      	float* yres = ykblas;
      	
      	float error = sget_max_error(yref, yres, m, incy);
      
      	//printf("-----------   ----------   ----------   ----------   ----------   ----------\n");
    	printf("%-11d   %-10.2f   %-10.2f   %-10e;\n", m, cuda_perf, kblas_perf, error);
    
    }
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
    if(dA)cudaFree(dA);
    if(dx)cudaFree(dx);
    if(dy)cudaFree(dy);
    
    if(A)free(A);
    if(x)free(x);
    if(ycuda)free(ycuda);
	if(ykblas)free(ykblas);

	cublasDestroy(cublas_handle);
    return EXIT_SUCCESS;
}

