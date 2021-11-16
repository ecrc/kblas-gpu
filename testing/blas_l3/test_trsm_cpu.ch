/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_l3/test_trsm_cpu.ch

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef _TEST_TRSM_
#define _TEST_TRSM_

#include "l3_common.h"
#include "testing_Xtr_common.h"


hipblasStatus_t kblas_xtrsm(hipblasHandle_t handle,
                          hipblasSideMode_t side, hipblasFillMode_t uplo,
                          hipblasOperation_t trans, hipblasDiagType_t diag,
                          int m, int n,
                          const float *alpha,
                          const float *A, int lda,
                                float *B, int ldb){
  return kblas_strsm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
hipblasStatus_t kblas_xtrsm(hipblasHandle_t handle,
                          hipblasSideMode_t side, hipblasFillMode_t uplo,
                          hipblasOperation_t trans, hipblasDiagType_t diag,
                          int m, int n,
                          const double *alpha,
                          const double *A, int lda,
                                double *B, int ldb){
  return kblas_dtrsm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
hipblasStatus_t kblas_xtrsm(hipblasHandle_t handle,
                          hipblasSideMode_t side, hipblasFillMode_t uplo,
                          hipblasOperation_t trans, hipblasDiagType_t diag,
                          int m, int n,
                          const hipComplex *alpha,
                          const hipComplex *A, int lda,
                                hipComplex *B, int ldb){
  return kblas_ctrsm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
hipblasStatus_t kblas_xtrsm(hipblasHandle_t handle,
                          hipblasSideMode_t side, hipblasFillMode_t uplo,
                          hipblasOperation_t trans, hipblasDiagType_t diag,
                          int m, int n,
                          const hipDoubleComplex *alpha,
                          const hipDoubleComplex *A, int lda,
                                hipDoubleComplex *B, int ldb){
  return kblas_ztrsm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}

//==============================================================================================
extern int kblas_trsm_ib_data;
//==============================================================================================
template<class T>
int test_trsm(kblas_opts& opts, T alpha, hipblasHandle_t cublas_handle){


  int nruns = opts.nruns;
  double   gflops, ref_perf = 0.0, ref_time = 0.0, cpu_perf = 0.0, cpu_time = 0.0, gpu_perf = 0.0, gpu_time = 0.0, ref_error = 0.0;
  int M, N;
  int Am, An, Bm, Bn;
  int sizeA, sizeB;
  int lda, ldb, ldda, lddb;
  int ione     = 1;
  int ISEED[4] = {0,0,0,1};

  T *h_A, *h_B, *h_Rc, *h_Rk;
  T *d_A, *d_B;

  kblas_trsm_ib_data = opts.db;
  check_error( hipSetDevice(opts.devices[0]) );

  USING
  hipError_t err;

  check_error( hipEventCreate(&start) );
  check_error( hipEventCreate(&stop) );

  hipblasSideMode_t  side  = (opts.side   == KBLAS_Left  ? HIPBLAS_SIDE_LEFT : HIPBLAS_SIDE_RIGHT);
  hipblasFillMode_t  uplo  = (opts.uplo   == KBLAS_Lower ? HIPBLAS_FILL_MODE_LOWER : HIPBLAS_FILL_MODE_UPPER);
  hipblasOperation_t trans = (opts.transA == KBLAS_Trans ? HIPBLAS_OP_T : HIPBLAS_OP_N);
  hipblasDiagType_t  diag  = (opts.diag   == KBLAS_Unit  ? HIPBLAS_DIAG_UNIT : HIPBLAS_DIAG_NON_UNIT);

  printf("    M     N   kblasTRSM CPU GF/s (ms) kblasTRSM GPU GF/s (ms)   cublasTRSM GPU GF/s (ms)  SP_CPU  SP_GPU   Error\n");
  printf("====================================================================\n");
  for( int i = 0; i < opts.ntest; ++i ) {
    for( int iter = 0; iter < opts.niter; ++iter ) {
      ref_time = cpu_time = gpu_time = 0.0;
      M = opts.msize[i];
      N = opts.nsize[i];

      gflops = FLOPS_TRSM<T>(opts.side, M, N ) / 1e9;

      printf("%5d %5d   ",
             (int) M, (int) N);
      fflush( stdout );

      if ( opts.side == KBLAS_Left ) {
        lda = Am = M;
        An = M;
      } else {
        lda = Am = N;
        An = N;
      }
      ldb = Bm = M;
      Bn = N;

      ldda = ((lda+31)/32)*32;
      lddb = ((ldb+31)/32)*32;

      sizeA = lda*An;
      sizeB = ldb*Bn;

      TESTING_MALLOC_PIN( h_A, T, sizeA);

      TESTING_MALLOC_PIN( h_Rk, T, sizeB);
      TESTING_MALLOC_CPU( h_B, T, sizeB);

      if(opts.time)
      {
        TESTING_MALLOC_PIN( h_Rc, T, sizeB);

      }
      // Initialize matrix and vector
      //printf("Initializing on cpu .. \n");
      Xrand_matrix(Am, An, h_A, lda);
      Xrand_matrix(Bm, Bn, h_B, ldb);
      kblas_make_hpd( Am, h_A, lda );
      if(opts.check){
        nruns = 1;
      }
      hipStream_t curStream;
      //check_error( hipStreamCreateWithFlags( &curStream, hipStreamNonBlocking) );
      //check_error( hipblasSetStream(cublas_handle, curStream));
      check_error( hipblasGetStream(cublas_handle, &curStream ) );

      if(opts.warmup){
        TESTING_MALLOC_DEV( d_A, T, ldda*An);
        TESTING_MALLOC_DEV( d_B, T, lddb*Bn);
        check_error( cublasSetMatrixAsync( Am, An, sizeof(T), h_A, lda, d_A, ldda, curStream) );
        check_error( cublasSetMatrixAsync( Bm, Bn, sizeof(T), h_B, ldb, d_B, lddb, curStream) );
        check_error( cublasXtrsm( cublas_handle,
                                  side, uplo, trans, diag,
                                  M, N,
                                  &alpha, d_A, ldda,
                                  d_B, lddb) );
        check_error( hipStreamSynchronize(curStream) );
        check_error( hipGetLastError() );
        check_error( hipFree( d_A ) );
        check_error( hipFree( d_B ) );
      }
      float time = 0;

      if(opts.time){

        for(int r = 0; r < nruns; r++)
        {
          check_error( hipMemcpyAsync ( (void*)h_Rc, (void*)h_B, sizeB * sizeof(T), hipMemcpyHostToHost, curStream ) );
          start_timing(curStream);

          TESTING_MALLOC_DEV( d_A, T, ldda*An);
          TESTING_MALLOC_DEV( d_B, T, lddb*Bn);

          check_error( cublasSetMatrixAsync( Am, An, sizeof(T), h_A, lda, d_A, ldda, curStream) );
          check_error( cublasSetMatrixAsync( Bm, Bn, sizeof(T), h_Rc, ldb, d_B, lddb, curStream ) );

          check_error( cublasXtrsm( cublas_handle,
                                    side, uplo, trans, diag,
                                    M, N,
                                    &alpha, d_A, ldda,
                                            d_B, lddb) );
          check_error( cublasGetMatrixAsync( M, N, sizeof(T), d_B, lddb, h_Rc, ldb, curStream ) );
          check_error( hipStreamSynchronize(curStream) );
          check_error( hipGetLastError() );
          check_error( hipFree( d_A ) );
          check_error( hipFree( d_B ) );
          check_error( hipGetLastError() );

          time = get_elapsed_time(curStream);
          ref_time += time;
        }
        ref_time /= nruns;
        ref_perf = gflops / (ref_time / 1000.0);

      }

      //check_error( hipblasSetMatrix( Am, An, sizeof(T), h_A, lda, d_A, ldda ) );
      check_error( hipDeviceSynchronize() );


      for(int r = 0; r < nruns; r++)
      {
        check_error( hipMemcpyAsync ( (void*)h_Rk, (void*)h_B, sizeB * sizeof(T), hipMemcpyHostToHost, curStream) );
        check_error( hipGetLastError() );
        //check_error( hipblasSetMatrix( Bm, Bn, sizeof(T), h_B, ldb, d_B, lddb ) );

        start_timing(curStream);
        check_error( kblas_xtrsm(cublas_handle,
                                 side, uplo, trans, diag,
                                 M, N,
                                 &alpha, h_A, lda,
                                         h_Rk, ldb) );
        time = get_elapsed_time(curStream);
        cpu_time += time;
      }
      cpu_time /= nruns;
      cpu_perf = gflops / (cpu_time / 1000.0);
      check_error( hipDeviceSynchronize() );

      /* TODO
      if(opts.check){
        double normA = kblas_lange<T,double>('M', Am, An, h_A, lda);
        double normX = kblas_lange<T,double>('M', Bm, Bn, h_Rk, ldb);

        TESTING_MALLOC_DEV( d_A, T, ldda*An);
        TESTING_MALLOC_DEV( d_B, T, lddb*Bn);

        check_error( cublasSetMatrixAsync( Am, An, sizeof(T), h_A, lda, d_A, ldda, curStream) );
        check_error( cublasSetMatrixAsync( Bm, Bn, sizeof(T), h_Rk, ldb, d_B, lddb, curStream ) );

        T one = make_one<T>();
        T mone = make_zero<T>() - one;
        T invAlpha = one / alpha;
        check_error( kblasXtrmm(cublas_handle,
                                side, uplo, trans, diag,
                                M, N,
                                &invAlpha, d_A, ldda,
                                           d_B, lddb) );
        check_error( hipblasGetMatrix( Bm, Bn, sizeof(T), d_B, lddb, h_Rk, ldb ) );
        kblasXaxpy( Bm * Bn, mone, h_B, 1, h_Rk, 1 );
        double normR = kblas_lange<T,double>('M', Bm, Bn, h_Rk, ldb);
        ref_error = normR / (normX * normA);
        //ref_error = Xget_max_error_matrix(h_Rc, h_Rk, Bm, Bn, ldb);
        check_error(  hipFree( d_A ) );
        check_error(  hipFree( d_B ) );
      }*/

      if(opts.time){
        for(int r = 0; r < nruns; r++)
        {
          check_error( hipMemcpy ( (void*)h_Rk, (void*)h_B, sizeB * sizeof(T), hipMemcpyHostToHost ) );
          check_error( hipGetLastError() );
          start_timing(curStream);

          TESTING_MALLOC_DEV( d_A, T, ldda*An);
          TESTING_MALLOC_DEV( d_B, T, lddb*Bn);

          check_error( cublasSetMatrixAsync( Am, An, sizeof(T), h_A, lda, d_A, ldda, curStream ) );
          check_error( cublasSetMatrixAsync( Bm, Bn, sizeof(T), h_Rk, ldb, d_B, lddb, curStream ) );

          check_error( kblasXtrsm(cublas_handle,
                                  side, uplo, trans, diag,
                                  M, N,
                                  &alpha, d_A, lda,
                                          d_B, ldb) );
          check_error( cublasGetMatrixAsync( M, N, sizeof(T), d_B, lddb, h_Rk, ldb, curStream ) );
          check_error( hipStreamSynchronize(curStream) );
          check_error( hipGetLastError() );
          check_error( hipFree( d_A ) );
          check_error( hipFree( d_B ) );
          check_error( hipGetLastError() );

          time = get_elapsed_time(curStream);
          gpu_time += time;
        }
        gpu_time /= nruns;
        gpu_perf = gflops / (gpu_time / 1000.0);
      }

      if(opts.time){
        check_error( hipHostFree( h_Rc ) );
        check_error( hipGetLastError() );
      }
      check_error( hipHostFree( h_A ) );
      check_error( hipGetLastError() );
      free( h_B );
      check_error( hipHostFree( h_Rk ) );
      check_error( hipGetLastError() );

      printf(" %7.2f %7.2f      %7.2f %7.2f       %7.2f %7.2f     %2.2f   %2.2f   %8.2e\n",
             cpu_perf, cpu_time,
             gpu_perf, gpu_time,
             ref_perf, ref_time,
             ref_time / cpu_time, ref_time / gpu_time,
             ref_error );

      //check_error( hipStreamDestroy( curStream ) );
      check_error( hipDeviceSynchronize() );
      check_error( hipGetLastError() );

    }
    if ( opts.niter > 1 ) {
      printf( "\n" );
    }
  }


  check_error( hipEventDestroy(start) );
  check_error( hipEventDestroy(stop) );
  check_error( hipGetLastError() );
}


#endif
