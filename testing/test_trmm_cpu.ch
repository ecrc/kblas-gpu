#ifndef _TEST_TRMM_
#define _TEST_TRMM_

#include "testing_Xtr_common.h"

cublasStatus_t kblas_xtrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const float *alpha,
                          const float *A, int lda,
                                float *B, int ldb){
  return kblas_strmm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
cublasStatus_t kblas_xtrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const double *alpha,
                          const double *A, int lda,
                                double *B, int ldb){
  return kblas_dtrmm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
cublasStatus_t kblas_xtrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const cuComplex *alpha,
                          const cuComplex *A, int lda,
                                cuComplex *B, int ldb){
  return kblas_ctrmm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
cublasStatus_t kblas_xtrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const cuDoubleComplex *alpha,
                          const cuDoubleComplex *A, int lda,
                                cuDoubleComplex *B, int ldb){
  return kblas_ztrmm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}


//==============================================================================================
extern int kblas_trmm_ib_data;
//==============================================================================================
template<class T>
int test_trmm(kblas_opts& opts, T alpha, cublasHandle_t cublas_handle){

  
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
  
  kblas_trmm_ib_data = opts.db;
  check_error( cudaSetDevice(opts.device) );
  
  USING
  cudaError_t err;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cublasSideMode_t  side  = (opts.side   == KBLAS_Left  ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);
  cublasFillMode_t  uplo  = (opts.uplo   == KBLAS_Lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER);
  cublasOperation_t trans = (opts.transA == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);
  cublasDiagType_t  diag  = (opts.diag   == KBLAS_Unit  ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT);

  printf("    M     N   kblasTRMM CPU GF/s (ms) kblasTRMM GPU GF/s (ms)   cublasTRMM GPU GF/s (ms)  SP_CPU  SP_GPU   Error\n");
  printf("====================================================================\n");
  for( int i = 0; i < opts.ntest; ++i ) {
    for( int iter = 0; iter < opts.niter; ++iter ) {
      ref_time = cpu_time = gpu_time = 0.0;
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

      /*
      if ( (h_A = (T*) malloc( (sizeA)*sizeof( T ) ) ) == NULL) {
        fprintf( stderr, "!!!! malloc_cpu failed for: h_A\n" );
        exit(-1);
      }
      if ( (h_Rk = (T*) malloc( (sizeB)*sizeof( T ) ) ) == NULL) {
        fprintf( stderr, "!!!! malloc_cpu failed for: h_Rk\n" );
        exit(-1);
      }*/
      
      TESTING_MALLOC_PIN( h_A, T, sizeA);
      TESTING_MALLOC_PIN( h_Rk, T, sizeB);
      /*if ( cudaMallocHost((void**)&h_A, (sizeA)*sizeof( T ) ) != cudaSuccess) {
        fprintf( stderr, "!!!! malloc_cpu failed for: h_A\n" );
        exit(-1);
      }
      if ( cudaMallocHost((void**)&h_Rk, (sizeB)*sizeof( T ) ) != cudaSuccess) {
        fprintf( stderr, "!!!! malloc_cpu failed for: h_Rk\n" );
        exit(-1);
      }*/
      TESTING_MALLOC_CPU( h_B, T, sizeB);
      /*if ( (h_B = (T*) malloc( (sizeB)*sizeof( T ) ) ) == NULL) {
        fprintf( stderr, "!!!! malloc_cpu failed for: h_B\n" );
        exit(-1);
      }*/
      
      if(opts.check || opts.time)
      {
        /*if ( (h_Rc = (T*) malloc( (sizeB)*sizeof( T ) ) ) == NULL) {
          fprintf( stderr, "!!!! malloc_cpu failed for: h_Rc\n" );
          exit(-1);
        }
        if ( cudaMallocHost((void**)&h_Rc, (sizeB)*sizeof( T ) ) != cudaSuccess) {
          fprintf( stderr, "!!!! malloc_cpu failed for: h_Rc\n" );
          exit(-1);
        }*/
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
      cudaStream_t curStream;
      check_error( cudaStreamCreateWithFlags( &curStream, cudaStreamNonBlocking) );
      check_error( cublasSetStream(cublas_handle, curStream));
      
      if(opts.warmup){
        check_error( cublasSetMatrix( Am, An, sizeof(T), h_A, lda, d_A, ldda) );
        check_error( cublasSetMatrix( Bm, Bn, sizeof(T), h_B, ldb, d_B, lddb) );
        check_error( cublasXtrmm( cublas_handle,
                                  side, uplo, trans, diag,
                                  M, N,
                                  &alpha, d_A, ldda,
                                          d_B, lddb) );
      }
      
      float time = 0;
            
      if(opts.time || opts.check){

        for(int r = 0; r < nruns; r++)
        {
          check_error( cudaMemcpy ( (void*)h_Rc, (void*)h_B, sizeB * sizeof(T), cudaMemcpyHostToHost ) );
          start_timing(curStream);
          
          TESTING_MALLOC_DEV( d_A, T, ldda*An);
          TESTING_MALLOC_DEV( d_B, T, lddb*Bn);
          
          check_error( cublasSetMatrixAsync( Am, An, sizeof(T), h_A, lda, d_A, ldda, curStream) );
          check_error( cublasSetMatrixAsync( Bm, Bn, sizeof(T), h_Rc, ldb, d_B, lddb, curStream ) );

          check_error( cublasXtrmm( cublas_handle,
                                    side, uplo, trans, diag,
                                    M, N,
                                    &alpha, d_A, ldda,
                                            d_B, lddb) );
          check_error( cublasGetMatrixAsync( M, N, sizeof(T), d_B, lddb, h_Rc, ldb, curStream ) );
          check_error(  cudaFree( d_A ) );
          check_error(  cudaFree( d_B ) );

          time = get_elapsed_time(curStream);
          ref_time += time/1000.0;//to be in sec
        }
        ref_time /= nruns;
        ref_perf = gflops / ref_time;

      }
      
      //check_error( cublasSetMatrix( Am, An, sizeof(T), h_A, lda, d_A, ldda ) );
      cudaDeviceSynchronize();

      
      for(int r = 0; r < nruns; r++)
      {
        check_error( cudaMemcpy ( (void*)h_Rk, (void*)h_B, sizeB * sizeof(T), cudaMemcpyHostToHost ) );
        //check_error( cublasSetMatrix( Bm, Bn, sizeof(T), h_B, ldb, d_B, lddb ) );
        
        start_timing(curStream);
        check_error( kblas_xtrmm(cublas_handle,
                                side, uplo, trans, diag,
                                M, N,
                                &alpha, h_A, lda,
                                        h_Rk, ldb) );
        time = get_elapsed_time(curStream);
        cpu_time += time;
      }
      cpu_time /= nruns;
      cpu_perf = gflops / (cpu_time/1000.0);
      cudaDeviceSynchronize();

      if(opts.check){
        ref_error = Xget_max_error_matrix(h_Rc, h_Rk, Bm, Bn, ldb);
      }
      
      for(int r = 0; r < nruns && (opts.check || opts.time); r++)
      {
        check_error( cudaMemcpy ( (void*)h_Rk, (void*)h_B, sizeB * sizeof(T), cudaMemcpyHostToHost ) );
        start_timing(curStream);
        
        TESTING_MALLOC_DEV( d_A, T, ldda*An);
        TESTING_MALLOC_DEV( d_B, T, lddb*Bn);

        check_error( cublasSetMatrixAsync( Am, An, sizeof(T), h_A, lda, d_A, ldda, curStream ) );
        check_error( cublasSetMatrixAsync( Bm, Bn, sizeof(T), h_Rk, ldb, d_B, lddb, curStream ) );
        
        check_error( kblasXtrmm(cublas_handle,
                                side, uplo, trans, diag,
                                M, N,
                                &alpha, d_A, lda,
                                        d_B, ldb) );
        check_error( cublasGetMatrixAsync( M, N, sizeof(T), d_B, lddb, h_Rk, ldb, curStream ) );
        check_error(  cudaFree( d_A ) );
        check_error(  cudaFree( d_B ) );
        
        time = get_elapsed_time(curStream);
        gpu_time += time;
      }
      gpu_time /= nruns;
      gpu_perf = gflops / (gpu_time / 1000.0);

      if(opts.check || opts.time){
        cudaFreeHost( h_Rc );
//         free( h_Rc );
      }
      cudaFreeHost( h_A );
//       free( h_A );
      free( h_B );
      cudaFreeHost( h_Rk );
//       free( h_Rk );
      printf(" %7.2f %7.2f      %7.2f %7.2f       %7.2f %7.2f     %2.2f    %2.2f   %8.2e\n",
             cpu_perf, 1000.*cpu_time,
             gpu_perf, 1000.*gpu_time,
             ref_perf, 1000.*ref_time,
             ref_time / cpu_time, ref_time / gpu_time,
             ref_error );
      
      check_error( cudaStreamDestroy( curStream ) );
      
    }
    if ( opts.niter > 1 ) {
      printf( "\n" );
    }
  }
    	

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}


#endif