#ifndef _TEST_TRMM_
#define _TEST_TRMM_

#include "testing_Xtr_common.h"

cublasStatus_t cublasXtrmm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const float *alpha,
                           const float *A, int lda,
                           const float *B, int ldb,
                                 float *C, int ldc){
  return cublasStrmm(handle,
                     side, uplo, trans, diag,
                     m, n,
                     alpha, A, lda,
                            B, ldb,
                            C, ldc );
}
cublasStatus_t cublasXtrmm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t      diag,
                           int m, int n,
                           const double *alpha,
                           const double *A, int lda,
                           const double *B, int ldb,
                                 double *C, int ldc){
  return cublasDtrmm(handle,
                     side, uplo, trans, diag,
                     m, n,
                     alpha, A, lda,
                            B, ldb,
                            C, ldc );
}
cublasStatus_t cublasXtrmm (cublasHandle_t handle,
                            cublasSideMode_t side, cublasFillMode_t uplo,
                            cublasOperation_t trans, cublasDiagType_t diag,
                            int m, int n,
                            const cuComplex *alpha,
                            const cuComplex *A, int lda,
                            const cuComplex *B, int ldb,
                                  cuComplex *C, int ldc){
  return cublasCtrmm(handle,
                     side, uplo, trans, diag,
                     m, n,
                     alpha, A, lda,
                            B, ldb,
                            C, ldc );
}
cublasStatus_t cublasXtrmm (cublasHandle_t handle,
                            cublasSideMode_t side, cublasFillMode_t uplo,
                            cublasOperation_t trans, cublasDiagType_t diag,
                            int m, int n,
                            const cuDoubleComplex *alpha,
                            const cuDoubleComplex *A, int lda,
                            const cuDoubleComplex *B, int ldb,
                                  cuDoubleComplex *C, int ldc){
  return cublasZtrmm(handle,
                     side, uplo, trans, diag,
                     m, n,
                     alpha, A, lda,
                            B, ldb,
                            C, ldc );
}

//==============================================================================================
extern bool kblas_trmm_use_custom;
template<class T>
int test_trmm(kblas_opts& opts, T alpha, cublasHandle_t cublas_handle){

  
  int nruns = opts.nruns;
  double  gflops, 
          ref_perf = 0.0, ref_time = 0.0, 
          ref_perf_oop = 0.0, ref_time_oop = 0.0,
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
  cudaError_t err;

  check_error( cudaEventCreate(&start) );
  check_error( cudaEventCreate(&stop) );
  
  cublasSideMode_t  side  = (opts.side   == KBLAS_Left  ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);
  cublasFillMode_t  uplo  = (opts.uplo   == KBLAS_Lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER);
  cublasOperation_t trans = (opts.transA == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);
  cublasDiagType_t  diag  = (opts.diag   == KBLAS_Unit  ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT);

  printf("    M     N     kblasTRMM_REC GF/s (ms)  kblasTRMM_CU GF/s (ms)  cublasTRMM GF/s (ms)  SP_REC   SP_CU   Error\n");
  printf("====================================================================\n");
  for( int i = 0; i < opts.ntest; ++i ) {
    for( int iter = 0; iter < opts.niter; ++iter ) {
      ref_time = ref_time_oop = kblas_time_rec = kblas_time_cus = 0.0;
      M = opts.msize[i];
      N = opts.nsize[i];
      
      gflops = FLOPS_TRMM(alpha, opts.side, M, N ) / 1e9;
      
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
      
      cudaStream_t curStream;
      //check_error( cudaStreamCreateWithFlags( &curStream, cudaStreamNonBlocking) );
      //check_error(cublasSetStream(cublas_handle, curStream));
      check_error( cublasGetStream(cublas_handle, &curStream ) );
      
      check_error( cublasSetMatrixAsync( Am, An, sizeof(T), h_A, lda, d_A, ldda, curStream ) );
      check_error( cudaGetLastError() );

      if(opts.warmup){
        kblas_trmm_use_custom = true;
        check_error( cublasSetMatrixAsync( Bm, Bn, sizeof(T), h_B, ldb, d_B, lddb, curStream ) );
        check_error( cudaGetLastError() );
        check_error( kblasXtrmm( cublas_handle,
                                  side, uplo, trans, diag,
                                  M, N,
                                  &alpha, d_A, ldda,
                                          d_B, lddb) );
        check_error( cudaStreamSynchronize(curStream) );
        check_error( cudaGetLastError() );
      }
      float time = 0;

      kblas_trmm_use_custom = true;
      for(int r = 0; r < nruns; r++)
      {
        check_error( cublasSetMatrixAsync( Bm, Bn, sizeof(T), h_B, ldb, d_B, lddb, curStream ) );
        check_error( cudaGetLastError() );
        
        start_timing(curStream);
        check_error( kblasXtrmm(cublas_handle,
                                side, uplo, trans, diag,
                                M, N,
                                &alpha, d_A, ldda,
                                        d_B, lddb) );
        time = get_elapsed_time(curStream);
        kblas_time_cus += time;
      }
      kblas_time_cus /= nruns;
      kblas_perf_cus = gflops / (kblas_time_cus / 1000.0);
      
      kblas_trmm_use_custom = false;
      for(int r = 0; r < nruns; r++)
      {
        check_error( cublasSetMatrixAsync( Bm, Bn, sizeof(T), h_B, ldb, d_B, lddb, curStream ) );
        check_error( cudaGetLastError() );
        
        start_timing(curStream);
        check_error( kblasXtrmm(cublas_handle,
                                side, uplo, trans, diag,
                                M, N,
                                &alpha, d_A, ldda,
                                        d_B, lddb) );
        time = get_elapsed_time(curStream);
        kblas_time_rec += time;
      }
      kblas_time_rec /= nruns;
      kblas_perf_rec = gflops / (kblas_time_rec / 1000.0);


      if(opts.check || opts.time){
        if(opts.check){
          check_error( cublasGetMatrixAsync( M, N, sizeof(T), d_B, lddb, h_R, ldb, curStream ) );
          check_error( cudaGetLastError() );
          check_error( cudaDeviceSynchronize() );
        }
      
        for(int r = 0; r < nruns; r++)
        {
          check_error( cublasSetMatrixAsync( Bm, Bn, sizeof(T), h_B, ldb, d_B, lddb, curStream ) );
          check_error( cudaGetLastError() );
          
          start_timing(curStream);
          check_error( cublasXtrmm( cublas_handle,
                                    side, uplo, trans, diag,
                                    M, N,
                                    &alpha, d_A, ldda,
                                    d_B, lddb) );
          time = get_elapsed_time(curStream);
          ref_time += time;//to be in sec
        }
        ref_time /= nruns;
        ref_perf = gflops / (ref_time /1000.0);
        
        if(opts.check){
          check_error( cublasGetMatrixAsync( M, N, sizeof(T), d_B, lddb, h_B, ldb, curStream ) );
          check_error( cudaGetLastError() );
          check_error( cudaStreamSynchronize(curStream) );
          check_error( cudaGetLastError() );
          //TODO use norm for error checking from plasma
          ref_error = Xget_max_error_matrix(h_B, h_R, Bm, Bn, ldb);
          free( h_R );
        }
      }

      if(opts.time){
        check_error( cudaDeviceSynchronize() );
        check_error( cudaGetLastError() );
        //check_error( cublasSetMatrixAsync( Bm, Bn, sizeof(T), h_B, ldb, d_B, lddb, curStream ) );

        T *h_C, *d_C;
        
        for(int r = 0; r < 1; r++)
        {
          check_error( cublasSetMatrixAsync( Bm, Bn, sizeof(T), h_B, ldb, d_B, lddb, curStream ) );
          check_error( cudaGetLastError() );

          start_timing(curStream);
          TESTING_MALLOC_DEV( d_C, T, lddb*Bn);
          //cudaDeviceSynchronize();
          check_error( cudaGetLastError() );
          check_error( cublasXtrmm( cublas_handle,
                                    side, uplo, trans, diag,
                                    M, N,
                                    &alpha, d_A, ldda,
                                            d_B, lddb,
                                            d_C, lddb) );
          check_error( cudaMemcpyAsync(d_B, d_C, lddb*Bn*sizeof(T), cudaMemcpyDeviceToDevice, curStream) );
          check_error( cudaGetLastError() );
          check_error( cudaStreamSynchronize(curStream) );
          check_error( cudaGetLastError() );
          check_error(  cudaFree( d_C ) );
          check_error( cudaGetLastError() );
          time = get_elapsed_time(curStream);
          ref_time_oop += time;//to be in sec
        }
        //ref_time_oop /= nruns;
        ref_perf_oop = gflops / (ref_time_oop /1000.0);

        //check_error( cublasGetMatrixAsync( M, N, sizeof(T), d_B, lddb, h_B, ldb, curStream ) );

      }

      free( h_A );
      free( h_B );
      check_error( cudaFree( d_A ) );
      check_error( cudaGetLastError() );
      check_error( cudaFree( d_B ) );
      check_error( cudaGetLastError() );
      
      printf(" %7.2f %7.2f      %7.2f %7.2f        %7.2f %7.2f     %7.2f %7.2f    %2.2f   %2.2f   %8.2e\n",
             kblas_perf_rec, kblas_time_rec,
             kblas_perf_cus, kblas_time_cus,
             ref_perf, ref_time,
             ref_perf_oop, ref_time_oop,
             ref_time / kblas_time_rec, ref_time / kblas_time_cus,
             ref_error );
      //check_error( cudaStreamDestroy( curStream ) );
      check_error( cudaDeviceSynchronize() );
      check_error( cudaGetLastError() );
    }
    if ( opts.niter > 1 ) {
      printf( "\n" );
    }
  }
    	

  check_error( cudaEventDestroy(start) );
  check_error( cudaEventDestroy(stop) );
  check_error( cudaGetLastError() );
}


#endif