/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_l3/test_dgemmXt.c

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cublasXt.h"

#include "kblas.h"
#include "testing_utils.h"
#include "testing_Xtr_common.h"


//#define USAGE printf("usage: x-dim y-dim\n");
#define FMULS_GEMM(m_, n_, k_) ((m_) * (n_) * (k_))
#define FADDS_GEMM(m_, n_, k_) ((m_) * (n_) * (k_))
#define FLOPS_DGEMM(m_, n_, k_) (     FMULS_GEMM((double)(m_), (double)(n_), (double)(k_)) +       FADDS_GEMM((double)(m_), (double)(n_), (double)(k_)) )


int main(int argc, char* argv[]){

  double   gflops, cublas_perf, cublas_time, cpu_perf, cpu_time, cublas_error;
  int M, N, K;
  int Am, An, Bm, Bn;
  int sizeA, sizeB, sizeC;
  int lda, ldb, ldc, ldda, lddb, lddc;
  int ione     = 1;
  int ISEED[4] = {0,0,0,1};
  int blockDim;

  double *h_A, *h_B, *h_C, *h_R;
  double alpha = 0.29, beta  = -0.48, c_neg_one = -1;

  kblas_opts opts;
  if(!parse_opts( argc, argv, &opts )){
    USAGE;
    return -1;
  }
  const int nruns = opts.nruns;
  cublasStatus_t status;
  cublasXtHandle_t cublasXt_handle;
  if(cublasXtCreate(&cublasXt_handle) != CUBLAS_STATUS_SUCCESS) {printf("handle create fail\n"); return 1;}
  //int devices[8] = { 0,1,2,3,4, 5, 6, 7 };  // add this line
  if((status = cublasXtDeviceSelect(cublasXt_handle, opts.ngpu, opts.devices)) != CUBLAS_STATUS_SUCCESS) {
    printf("set devices fail with status: %s\n", cublasGetErrorString(status));
    return 1;
  }
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  printf("    M     N     K   CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)  CUBLAS error\n");
  printf("========================================================================\n");
  for( int i = 0; i < opts.ntest; ++i ) {
    M = opts.msize[i];
    N = opts.nsize[i];
    K = opts.ksize[i];
    gflops = FLOPS_DGEMM( M, N, K ) / 1e9;
    printf("%5d %5d %5d   ",
        (int) M, (int) N, (int) K);
    fflush( stdout );

    if ( opts.transA == KBLAS_NoTrans ) {
      lda = Am = M;
      An = K;
    } else {
      lda = Am = K;
      An = M;
    }

    if ( opts.transB == KBLAS_NoTrans ) {
      ldb = Bm = K;
      Bn = N;
    } else {
      ldb = Bm = N;
      Bn = K;
    }
    ldc = M;

    ldda = ((lda+31)/32)*32;
    lddb = ((ldb+31)/32)*32;
    lddc = ((ldc+31)/32)*32;

    sizeA = lda*An;
    sizeB = ldb*Bn;
    sizeC = ldc*N;


    if ( cudaHostAlloc((void**)&h_A, (lda*An)*sizeof( double ),0 ) != cudaSuccess) {
      fprintf( stderr, "!!!! cudaHostAlloc failed for: h_A\n" );
      exit(-1);
    }
    if ( cudaHostAlloc((void**)&h_B, (ldb*Bn)*sizeof( double ),0 ) != cudaSuccess) {
      fprintf( stderr, "!!!! cudaHostAlloc failed for: h_B\n" );
      exit(-1);
    }
    if ( cudaHostAlloc((void**)&h_C, (ldc*N)*sizeof( double ),0 ) != cudaSuccess) {
      fprintf( stderr, "!!!! cudaHostAlloc failed for: h_C\n" );
      exit(-1);
    }
    /*if ( cudaHostAlloc((void**)&h_R, (ldc*N)*sizeof( double ),0 ) != cudaSuccess) {
      fprintf( stderr, "!!!! cudaHostAlloc failed for: h_C\n" );
      exit(-1);
    }*/

    for( int iter = 0; iter < opts.niter; ++iter ) {
      drand_matrix(Am, An, h_A, lda);
      drand_matrix(Bm, Bn, h_B, ldb);
      drand_matrix(ldc, N, h_C, ldc);

      //cublasStatus_t status;
      blockDim = opts.nb;
      cublasXtSetBlockDim(cublasXt_handle, blockDim);

      if(opts.warmup){
        cublasXtDgemm(cublasXt_handle,
                      CUBLAS_OP_T, CUBLAS_OP_N,
                      M, N, K,
                      &alpha, h_A, ldda,
                      h_B, lddb,
                      &beta,  h_C, lddc );
        drand_matrix(ldc, N, h_C, ldc);
      }

      float time = 0, cublas_time = 0;
      for(int r = 0; r < nruns; r++)
      {
        cudaEventRecord(start, 0);

        cublasXtDgemm(cublasXt_handle,
                      CUBLAS_OP_T, CUBLAS_OP_N,
                      M, N, K,
                      &alpha, h_A, ldda,
                              h_B, lddb,
                      &beta,  h_C, lddc );
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        time = 0.0;
        cudaEventElapsedTime(&time, start, stop);
        cublas_time += time/1000.0;//to be in sec
      }
      cublas_time /= nruns;
      cublas_perf = gflops / cublas_time;


      printf("%7.2f (%7.2f)     %d    ---\n",
             cublas_perf, 1000.*cublas_time,
             blockDim );
    }

    cudaFreeHost( h_A );
    cudaFreeHost( h_B );
    cudaFreeHost( h_C );
    //cudaFreeHost( h_R );
    if ( opts.niter > 1 ) {
        printf( "\n" );
    }
  }
  cublasXtDestroy(cublasXt_handle);
}
