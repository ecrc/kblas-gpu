/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_l3/test_gemm.ch

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef _TEST_GEMM_
#define _TEST_GEMM_

#include "l3_common.h"
#include "testing_Xtr_common.h"


//==============================================================================================
template<class T>
int test_gemm(kblas_opts& opts, T alpha, T beta, hipblasHandle_t cublas_handle){


  int nruns = opts.nruns;
  double  gflops,
          ref_perf = 0.0, ref_time = 0.0;
  int M, N, K;
  int Am, An, Bm, Bn, Cm, Cn;
  int sizeA, sizeB, sizeC;
  int lda, ldda, ldb, lddb, ldc, lddc;
  int ione     = 1;
  int ISEED[4] = {0,0,0,1};

  T *h_A, *h_B, *h_C;
  T *d_A, *d_B, *d_C;


  USING
  hipError_t err;

  hipEventCreate(&start);
  hipEventCreate(&stop);

  hipblasOperation_t transA = (opts.transA == KBLAS_Trans ? HIPBLAS_OP_T : HIPBLAS_OP_N);
  hipblasOperation_t transB = (opts.transB == KBLAS_Trans ? HIPBLAS_OP_T : HIPBLAS_OP_N);

  printf("    M     N     cublasGEMM GF/s ms\n");
  printf("====================================================================\n");
  for( int i = 0; i < opts.ntest; ++i ) {
    for( int iter = 0; iter < opts.niter; ++iter ) {
      ref_time = 0.0;
      M = opts.msize[i];
      N = opts.nsize[i];
      K = opts.ksize[i];

      gflops = FLOPS_GEMM<T>(M, N, K ) / 1e9;

      printf("%5d %5d %5d   ",
             (int) M, (int) N, (int) K);
      fflush( stdout );

      if ( opts.transA == KBLAS_Trans ) {
        lda = Am = M;
        An = K;
      } else {
        lda = Am = K;
        An = M;
      }
      if ( opts.transB == KBLAS_Trans ) {
        ldb = Bm = K;
        Bn = N;
      } else {
        ldb = Bm = N;
        Bn = K;
      }
      Cm = ldc = M;
      Cn = N;

      ldda = ((lda+31)/32)*32;
      lddb = ((ldb+31)/32)*32;
      lddc = ((ldc+31)/32)*32;

      sizeA = lda*An;
      sizeB = ldb*Bn;
      sizeC = ldc*N;

      TESTING_MALLOC_PIN( h_A, T, sizeA);
      TESTING_MALLOC_PIN( h_B, T, sizeB);
      TESTING_MALLOC_PIN( h_C, T, sizeC);

      TESTING_MALLOC_DEV( d_A, T, ldda*An);
      TESTING_MALLOC_DEV( d_B, T, lddb*Bn);
      TESTING_MALLOC_DEV( d_C, T, lddc*N);

      if(opts.check)
      {
        nruns = 1;
      }
      // Initialize matrix and vector
      //printf("Initializing on cpu .. \n");
      Xrand_matrix(Am, An, h_A, lda);
      Xrand_matrix(Bm, Bn, h_B, ldb);
      Xrand_matrix(Cm, Cn, h_C, ldc);

      hipStream_t curStream = NULL;
      check_error( hipblasSetStream(cublas_handle, curStream));

      check_error( hipblasSetMatrix( Am, An, sizeof(T), h_A, lda, d_A, ldda ) );
      check_error( hipblasSetMatrix( Bm, Bn, sizeof(T), h_B, ldb, d_B, lddb ) );

      if(opts.warmup){
        check_error( cublasXgemm( cublas_handle,
                                  transA, transB,
                                  M, N, K,
                                  &alpha, d_A, ldda,
                                          d_B, lddb,
                                  &beta,  d_C, lddc) );
      }
      float time = 0;


      for(int r = 0; r < nruns; r++)
      {
        check_error( hipblasSetMatrix( Cm, Cn, sizeof(T), h_C, ldb, d_C, lddc ) );

        start_timing(curStream);
        check_error( cublasXgemm( cublas_handle,
                                  transA, transB,
                                  M, N, K,
                                  &alpha, d_A, ldda,
                                          d_B, lddb,
                                  &beta,  d_C, lddc) );
        time = get_elapsed_time(curStream);
        ref_time += time;//to be in sec
      }
      ref_time /= nruns;
      ref_perf = gflops / (ref_time /1000.0);


      hipHostFree( h_A );
      hipHostFree( h_B );
      hipHostFree( h_C );
      check_error(  hipFree( d_A ) );
      check_error(  hipFree( d_B ) );
      check_error(  hipFree( d_C ) );

      printf(" %7.2f %7.2f    \n",
             ref_perf, ref_time);
    }
    if ( opts.niter > 1 ) {
      printf( "\n" );
    }
  }


  hipEventDestroy(start);
  hipEventDestroy(stop);
}


#endif
