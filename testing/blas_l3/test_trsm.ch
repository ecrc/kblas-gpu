/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_l3/test_trsm.ch

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

//==============================================================================================
extern bool kblas_trsm_use_custom;
template<class T>
int test_trsm(kblas_opts& opts, T alpha, hipblasHandle_t cublas_handle){


  int nruns = opts.nruns;
  double  gflops,
          ref_perf = 0.0, ref_time = 0.0,
          kblas_perf_rec = 0.0, kblas_time_rec = 0.0,
          kblas_perf_cus = 0.0, kblas_time_cus = 0.0,
          ref_error = 0.0;
  int M, N;
  int Am, An, Bm, Bn;
  int sizeA, sizeB;
  int lda, ldb, ldda, lddb;
  int ione     = 1;
  int ISEED[4] = {0,0,0,1};

  T *h_A, *h_B, *h_R;
  T *d_A, *d_B;


  USING
  hipError_t err;

  check_error( hipEventCreate(&start) );
  check_error( hipEventCreate(&stop) );

  hipblasSideMode_t  side  = (opts.side   == KBLAS_Left  ? HIPBLAS_SIDE_LEFT : HIPBLAS_SIDE_RIGHT);
  hipblasFillMode_t  uplo  = (opts.uplo   == KBLAS_Lower ? HIPBLAS_FILL_MODE_LOWER : HIPBLAS_FILL_MODE_UPPER);
  hipblasOperation_t trans = (opts.transA == KBLAS_Trans ? HIPBLAS_OP_T : HIPBLAS_OP_N);
  hipblasDiagType_t  diag  = (opts.diag   == KBLAS_Unit  ? HIPBLAS_DIAG_UNIT : HIPBLAS_DIAG_NON_UNIT);

  printf("    M     N     kblasTRSM_REC GF/s (ms)  kblasTRSM_CU GF/s (ms)  cublasTRSM GF/s (ms)  SP_REC   SP_CU   Error\n");
  printf("====================================================================\n");
  for( int i = 0; i < opts.ntest; ++i ) {
    for( int iter = 0; iter < opts.niter; ++iter ) {
      ref_time = kblas_time_rec = kblas_time_cus = 0.0;
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

      TESTING_MALLOC_CPU( h_A, T, sizeA);
      TESTING_MALLOC_CPU( h_B, T, sizeB);

      TESTING_MALLOC_DEV( d_A, T, ldda*An);
      TESTING_MALLOC_DEV( d_B, T, lddb*Bn);

      if(opts.check)
      {
        TESTING_MALLOC_CPU( h_R, T, sizeB);
        nruns = 1;
        opts.time = 0;
      }
      // Initialize matrix and vector
      //printf("Initializing on cpu .. \n");
      Xrand_matrix(Am, An, h_A, lda);
      Xrand_matrix(Bm, Bn, h_B, ldb);
      kblas_make_hpd( Am, h_A, lda );

      hipStream_t curStream;
      //check_error( hipStreamCreateWithFlags( &curStream, hipStreamNonBlocking) );
      //check_error(hipblasSetStream(cublas_handle, curStream));
      check_error( hipblasGetStream(cublas_handle, &curStream ) );

      check_error( hipblasSetMatrixAsync( Am, An, sizeof(T), h_A, lda, d_A, ldda, curStream ) );
      check_error( hipGetLastError() );

      if(opts.warmup){
        check_error( hipblasSetMatrixAsync( Bm, Bn, sizeof(T), h_B, ldb, d_B, lddb, curStream ) );
        check_error( cublasXtrsm( cublas_handle,
                                  side, uplo, trans, diag,
                                  M, N,
                                  &alpha, d_A, ldda,
                                  d_B, lddb) );
        check_error( hipStreamSynchronize(curStream) );
        check_error( hipGetLastError() );
      }

      float time = 0;

      kblas_trsm_use_custom = true;
      for(int r = 0; r < nruns; r++)
      {
        check_error( hipblasSetMatrixAsync( Bm, Bn, sizeof(T), h_B, ldb, d_B, lddb, curStream ) );
        check_error( hipGetLastError() );

        start_timing(curStream);
        check_error( kblasXtrsm(cublas_handle,
                                side, uplo, trans, diag,
                                M, N,
                                &alpha, d_A, ldda,
                                        d_B, lddb) );
        time = get_elapsed_time(curStream);
        kblas_time_cus += time;
      }
      kblas_time_cus /= nruns;
      kblas_perf_cus = gflops / (kblas_time_cus / 1000.);

      kblas_trsm_use_custom = false;
      for(int r = 0; r < nruns; r++)
      {
        check_error( hipblasSetMatrixAsync( Bm, Bn, sizeof(T), h_B, ldb, d_B, lddb, curStream ) );
        check_error( hipGetLastError() );

        start_timing(curStream);
        check_error( kblasXtrsm(cublas_handle,
                                side, uplo, trans, diag,
                                M, N,
                                &alpha, d_A, ldda,
                                        d_B, lddb) );
        time = get_elapsed_time(curStream);
        kblas_time_rec += time;
      }
      kblas_time_rec /= nruns;
      kblas_perf_rec = gflops / (kblas_time_rec / 1000.);


      if(opts.check){
        double normA = kblas_lange<T,double>('M', Am, An, h_A, lda);
        check_error( hipblasGetMatrixAsync( Bm, Bn, sizeof(T), d_B, lddb, h_R, ldb, curStream ) );
        check_error( hipGetLastError() );
        check_error( hipStreamSynchronize(curStream) );
        check_error( hipGetLastError() );
        double normX = kblas_lange<T,double>('M', Bm, Bn, h_R, ldb);

        T one = make_one<T>();
        T mone = make_zero<T>() - one;
        T invAlpha = one / alpha;
        check_error( kblasXtrmm(cublas_handle,
                                side, uplo, trans, diag,
                                M, N,
                                &invAlpha, d_A, ldda,
                                           d_B, lddb) );
        check_error( hipblasGetMatrixAsync( Bm, Bn, sizeof(T), d_B, lddb, h_R, ldb, curStream ) );
        check_error( hipGetLastError() );
        check_error( hipStreamSynchronize(curStream) );
        check_error( hipGetLastError() );
        kblasXaxpy( Bm * Bn, mone, h_B, 1, h_R, 1 );
        double normR = kblas_lange<T,double>('M', Bm, Bn, h_R, ldb);
        ref_error = normR / (normX * normA);
        //ref_error = Xget_max_error_matrix(h_B, h_R, Bm, Bn, ldb);
        free( h_R );
      }
      if(opts.check || opts.time){

        for(int r = 0; r < nruns; r++)
        {
          check_error( hipblasSetMatrixAsync( Bm, Bn, sizeof(T), h_B, ldb, d_B, lddb, curStream ) );
          check_error( hipGetLastError() );

          start_timing(curStream);
          check_error( cublasXtrsm( cublas_handle,
                                    side, uplo, trans, diag,
                                    M, N,
                                    &alpha, d_A, ldda,
                                            d_B, lddb) );
          time = get_elapsed_time(curStream);
          ref_time += time;//to be in sec
        }
        ref_time /= nruns;
        ref_perf = gflops / (ref_time / 1000.);
      }

      free( h_A );
      free( h_B );
      check_error( hipFree( d_A ) );
      check_error( hipGetLastError() );
      check_error( hipFree( d_B ) );
      check_error( hipGetLastError() );
     // printf(" %7.2f  %7.2f %7.2f %8.2e\n",
     //        kblas_perf_rec, kblas_perf_cus,
     //        ref_perf,
     //        ref_error );  
       printf(" %7.2f %7.2f      %7.2f %7.2f        %7.2f %7.2f     %2.2f   %2.2f   %8.2e\n",
              kblas_perf_rec, kblas_time_rec,
              kblas_perf_cus, kblas_time_cus,
              ref_perf, ref_time,
              ref_time / kblas_time_rec, ref_time / kblas_time_cus,
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
  return 0;
}


#endif
