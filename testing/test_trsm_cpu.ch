#ifndef _TEST_TRSM_
#define _TEST_TRSM_

#include "testing_Xtr_common.h"


cublasStatus_t kblas_xtrsm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
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
cublasStatus_t kblas_xtrsm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
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
cublasStatus_t kblas_xtrsm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const cuComplex *alpha,
                          const cuComplex *A, int lda,
                                cuComplex *B, int ldb){
  return kblas_ctrsm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
cublasStatus_t kblas_xtrsm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const cuDoubleComplex *alpha,
                          const cuDoubleComplex *A, int lda,
                                cuDoubleComplex *B, int ldb){
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
int test_trsm(kblas_opts& opts, T alpha, cublasHandle_t cublas_handle){


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
  check_error( cudaSetDevice(opts.device) );

  USING
  cudaError_t err;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cublasSideMode_t  side  = (opts.side   == KBLAS_Left  ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);
  cublasFillMode_t  uplo  = (opts.uplo   == KBLAS_Lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER);
  cublasOperation_t trans = (opts.transA == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);
  cublasDiagType_t  diag  = (opts.diag   == KBLAS_Unit  ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT);

  printf("    M     N   kblasTRSM CPU GF/s (ms) kblasTRSM GPU GF/s (ms)   cublasTRSM GPU GF/s (ms)  SP_CPU  SP_GPU   Error\n");
  printf("====================================================================\n");
  for( int i = 0; i < opts.ntest; ++i ) {
    for( int iter = 0; iter < opts.niter; ++iter ) {
      ref_time = cpu_time = gpu_time = 0.0;
      M = opts.msize[i];
      N = opts.nsize[i];

      gflops = FLOPS_TRSM(alpha, opts.side, M, N ) / 1e9;

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

      if ( cudaMallocHost((void**)&h_A, (sizeA)*sizeof( T ) ) != cudaSuccess) {
        fprintf( stderr, "!!!! malloc_cpu failed for: h_A\n" );
        exit(-1);
      }
      if ( cudaMallocHost((void**)&h_Rk, (sizeB)*sizeof( T ) ) != cudaSuccess) {
        fprintf( stderr, "!!!! malloc_cpu failed for: h_Rk\n" );
        exit(-1);
      }

      if ( (h_B = (T*) malloc( (sizeB)*sizeof( T ) ) ) == NULL) {
        fprintf( stderr, "!!!! malloc_cpu failed for: h_B\n" );
        exit(-1);
      }

      if(opts.check || opts.time)
      {
        /*if ( (h_Rc = (T*) malloc( (sizeB)*sizeof( T ) ) ) == NULL) {
          fprintf( stderr, "!!!! malloc_cpu failed for: h_Rc\n" );
          exit(-1);
        }*/
        if ( cudaMallocHost((void**)&h_Rc, (sizeB)*sizeof( T ) ) != cudaSuccess) {
          fprintf( stderr, "!!!! malloc_cpu failed for: h_Rc\n" );
          exit(-1);
        }

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

      float time = 0;

      if(opts.time){

        for(int r = 0; r < nruns; r++)
        {
          check_error( cudaMemcpy ( (void*)h_Rc, (void*)h_B, sizeB * sizeof(T), cudaMemcpyHostToHost ) );
          start_timing(curStream);
          if ( (err = cudaMalloc( (void**)&d_A, (ldda*An)*sizeof(T) )) != cudaSuccess ) {
            fprintf( stderr, "!!!! cudaMalloc failed for: d_A! Error: %s\n", cudaGetErrorString(err) );
            exit(-1);
          }
          if ( (err = cudaMalloc( (void**)&d_B, (lddb*Bn)*sizeof(T) )) != cudaSuccess ) {
            fprintf( stderr, "!!!! cudaMalloc failed for: d_B! Error: %s\n", cudaGetErrorString(err) );
            exit(-1);
          }
          check_error( cublasSetMatrixAsync( Am, An, sizeof(T), h_A, lda, d_A, ldda, curStream) );
          check_error( cublasSetMatrixAsync( Bm, Bn, sizeof(T), h_Rc, ldb, d_B, lddb, curStream ) );

          check_error( cublasXtrsm( cublas_handle,
                                    side, uplo, trans, diag,
                                    M, N,
                                    &alpha, d_A, ldda,
                                            d_B, lddb) );
          check_error( cublasGetMatrixAsync( M, N, sizeof(T), d_B, lddb, h_Rc, ldb, curStream ) );
          check_error(  cudaFree( d_A ) );
          check_error(  cudaFree( d_B ) );

          time = get_elapsed_time(curStream);
          ref_time += time;
        }
        ref_time /= nruns;
        ref_perf = gflops / (ref_time / 1000.0);

      }

      //check_error( cublasSetMatrix( Am, An, sizeof(T), h_A, lda, d_A, ldda ) );
      cudaDeviceSynchronize();


      for(int r = 0; r < nruns; r++)
      {
        check_error( cudaMemcpy ( (void*)h_Rk, (void*)h_B, sizeB * sizeof(T), cudaMemcpyHostToHost ) );
        //check_error( cublasSetMatrix( Bm, Bn, sizeof(T), h_B, ldb, d_B, lddb ) );

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
      cudaDeviceSynchronize();

      if(opts.check){
        double normA = kblas_lange<T,double>('M', Am, An, h_A, lda);
        check_error( cublasGetMatrix( Bm, Bn, sizeof(T), d_B, lddb, h_Rk, ldb ) );
        double normX = kblas_lange<T,double>('M', Bm, Bn, h_Rk, ldb);

        T one = make_one<T>();
        T mone = make_zero<T>() - one;
        T invAlpha = one / alpha;
        check_error( kblasXtrmm(cublas_handle,
                                side, uplo, trans, diag,
                                M, N,
                                &invAlpha, d_A, ldda,
                                           d_B, lddb) );
        check_error( cublasGetMatrix( Bm, Bn, sizeof(T), d_B, lddb, h_Rk, ldb ) );
        kblasXaxpy( Bm * Bn, mone, h_B, 1, h_Rk, 1 );
        double normR = kblas_lange<T,double>('M', Bm, Bn, h_Rk, ldb);
        ref_error = normR / (normX * normA);
        //ref_error = Xget_max_error_matrix(h_Rc, h_Rk, Bm, Bn, ldb);
      }
      
      if(opts.time){
        for(int r = 0; r < nruns; r++)
        {
          check_error( cudaMemcpy ( (void*)h_Rk, (void*)h_B, sizeB * sizeof(T), cudaMemcpyHostToHost ) );
          start_timing(curStream);
          if ( (err = cudaMalloc( (void**)&d_A, (ldda*An)*sizeof(T) )) != cudaSuccess ) {
            fprintf( stderr, "!!!! cudaMalloc failed for: d_A! Error: %s\n", cudaGetErrorString(err) );
            exit(-1);
          }
          if ( (err = cudaMalloc( (void**)&d_B, (lddb*Bn)*sizeof(T) )) != cudaSuccess ) {
            fprintf( stderr, "!!!! cudaMalloc failed for: d_B! Error: %s\n", cudaGetErrorString(err) );
            exit(-1);
          }
          check_error( cublasSetMatrixAsync( Am, An, sizeof(T), h_A, lda, d_A, ldda, curStream ) );
          check_error( cublasSetMatrixAsync( Bm, Bn, sizeof(T), h_Rk, ldb, d_B, lddb, curStream ) );

          check_error( kblasXtrsm(cublas_handle,
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
      }

      if(opts.check || opts.time){
        cudaFreeHost( h_Rc );
//         free( h_Rc );
      }
      cudaFreeHost( h_A );
//       free( h_A );
      free( h_B );
      cudaFreeHost( h_Rk );
//       free( h_Rk );
      printf(" %7.2f (%7.2f)      %7.2f (%7.2f)       %7.2f (%7.2f)     %2.2f   %2.2f   %8.2e\n",
             cpu_perf, cpu_time,
             gpu_perf, gpu_time,
             ref_perf, ref_time,
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