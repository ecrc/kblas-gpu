#ifndef _TEST_TRMM_
#define _TEST_TRMM_

//==============================================================================================
#define FMULS_TRMM_2(m_, n_) (0.5 * (n_) * (m_) * ((m_)+1))
#define FADDS_TRMM_2(m_, n_) (0.5 * (n_) * (m_) * ((m_)-1))
#define FMULS_TRMM(side_, m_, n_) ( ( (side_) == KBLAS_Left ) ? FMULS_TRMM_2((m_), (n_)) : FMULS_TRMM_2((n_), (m_)) )
#define FADDS_TRMM(side_, m_, n_) ( ( (side_) == KBLAS_Left ) ? FADDS_TRMM_2((m_), (n_)) : FADDS_TRMM_2((n_), (m_)) )


double FLOPS_TRMM(float p, char side, int m, int n){
  return FMULS_TRMM(side, (double)(m), (double)(n)) + FADDS_TRMM(side, (double)(m), (double)(n));
}
double FLOPS_TRMM(double p, char side, int m, int n){
  return FMULS_TRMM(side, (double)(m), (double)(n)) + FADDS_TRMM(side, (double)(m), (double)(n));
}
double FLOPS_TRMM(cuFloatComplex p, char side, int m, int n){
  return 6. * FMULS_TRMM(side, (double)(m), (double)(n)) + 2. * FADDS_TRMM(side, (double)(m), (double)(n));
}
double FLOPS_TRMM(cuDoubleComplex p, char side, int m, int n){
  return 6. * FMULS_TRMM(side, (double)(m), (double)(n)) + 2. * FADDS_TRMM(side, (double)(m), (double)(n));
}

const char* cublasGetErrorString( cublasStatus_t error );
int _kblas_error( cudaError_t err, const char* func, const char* file, int line );
int _kblas_error( cublasStatus_t err, const char* func, const char* file, int line );

#define check_error( err ) \
{ \
  if(!_kblas_error( (err), __func__, __FILE__, __LINE__ )) \
    return 0;\
}

//==============================================================================================
/*int kblas_Xtrmm( char side, char uplo, char trans, char diag,
                 int m, int n,
                 float alpha, const float *A, int incA,
                 float *B, int incB){
  return kblas_cublas_strmm( side, uplo, trans, diag,
                       m, n,
                       alpha, A, incA,
                       B, incB);
}
int kblas_Xtrmm( char side, char uplo, char trans, char diag,
                 int m, int n,
                 double alpha, const double *A, int incA,
                 double *B, int incB){
  return kblas_cublas_dtrmm( side, uplo, trans, diag,
                       m, n,
                       alpha, A, incA,
                       B, incB);
}
int kblas_Xtrmm( char side, char uplo, char trans, char diag,
                 int m, int n,
                 cuFloatComplex alpha, const cuFloatComplex *A, int incA,
                 cuFloatComplex *B, int incB){
  return kblas_cublas_ctrmm( side, uplo, trans, diag,
                       m, n,
                       alpha, A, incA,
                       B, incB);
}
int kblas_Xtrmm( char side, char uplo, char trans, char diag,
                 int m, int n,
                 cuDoubleComplex alpha, const cuDoubleComplex *A, int incA,
                 cuDoubleComplex *B, int incB){
  return kblas_cublas_ztrmm( side, uplo, trans, diag,
                       m, n,
                       alpha, A, incA,
                       B, incB);
}*/
cublasStatus_t kblasXtrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const float *alpha,
                          const float *A, int lda,
                                float *B, int ldb){
  return kblasStrmm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
cublasStatus_t kblasXtrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const double *alpha,
                          const double *A, int lda,
                                double *B, int ldb){
  return kblasDtrmm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
cublasStatus_t kblasXtrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const cuComplex *alpha,
                          const cuComplex *A, int lda,
                                cuComplex *B, int ldb){
  return kblasCtrmm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
cublasStatus_t kblasXtrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const cuDoubleComplex *alpha,
                          const cuDoubleComplex *A, int lda,
                                cuDoubleComplex *B, int ldb){
  return kblasZtrmm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}

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

cublasStatus_t cublasXtrmm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const float *alpha,
                           const float *A, int lda,
                                 float *B, int ldb);
cublasStatus_t cublasXtrmm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t      diag,
                           int m, int n,
                           const double *alpha,
                           const double *A, int lda,
                                 double *B, int ldb);
cublasStatus_t cublasXtrmm (cublasHandle_t handle,
                            cublasSideMode_t side, cublasFillMode_t uplo,
                            cublasOperation_t trans, cublasDiagType_t diag,
                            int m, int n,
                            const cuComplex *alpha,
                            const cuComplex *A, int lda,
                                  cuComplex *B, int ldb);
cublasStatus_t cublasXtrmm (cublasHandle_t handle,
                            cublasSideMode_t side, cublasFillMode_t uplo,
                            cublasOperation_t trans, cublasDiagType_t diag,
                            int m, int n,
                            const cuDoubleComplex *alpha,
                            const cuDoubleComplex *A, int lda,
                                  cuDoubleComplex *B, int ldb);


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
  
  
  check_error( cudaSetDevice(opts.device) );
  
  USING
  cudaError_t err;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  cublasSideMode_t  side  = (opts.side   == KBLAS_Left  ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);
  cublasFillMode_t  uplo  = (opts.uplo   == KBLAS_Lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER);
  cublasOperation_t trans = (opts.transA == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);
  cublasDiagType_t  diag  = (opts.diag   == KBLAS_Unit  ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT);

  printf("    M     N   kblasTRMM CPU GF/s (ms) kblasTRMM GPU GF/s (ms)   cublasTRMM GPU GF/s (ms)  MaxError\n");
  printf("====================================================================\n");
  for( int i = 0; i < opts.ntest; ++i ) {
    for( int iter = 0; iter < opts.niter; ++iter ) {
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
      
      if ( (h_A = (T*) malloc( (sizeA)*sizeof( T ) ) ) == NULL) {
        fprintf( stderr, "!!!! malloc_cpu failed for: h_A\n" );
        exit(-1);
      }
      if ( (h_B = (T*) malloc( (sizeB)*sizeof( T ) ) ) == NULL) {
        fprintf( stderr, "!!!! malloc_cpu failed for: h_B\n" );
        exit(-1);
      }
      if ( (h_Rk = (T*) malloc( (sizeB)*sizeof( T ) ) ) == NULL) {
        fprintf( stderr, "!!!! malloc_cpu failed for: h_Rk\n" );
        exit(-1);
      }
      
      if(opts.check || opts.time)
      {
        if ( (h_Rc = (T*) malloc( (sizeB)*sizeof( T ) ) ) == NULL) {
          fprintf( stderr, "!!!! malloc_cpu failed for: h_R\n" );
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
      cudaStream_t curStream = NULL;
      check_error(cublasSetStream(cublas_handle, curStream));

      float time = 0;
            
      if(opts.time || opts.check){

      
        for(int r = 0; r < nruns; r++)
        {
          check_error( cudaMemcpy ( (void*)h_Rc, (void*)h_B, sizeB * sizeof(T), cudaMemcpyHostToHost ) );
          cudaEventRecord(start, 0);
          if ( (err = cudaMalloc( (void**)&d_A, (ldda*An)*sizeof(T) )) != cudaSuccess ) {
            fprintf( stderr, "!!!! cudaMalloc failed for: d_A! Error: %s\n", cudaGetErrorString(err) );
            exit(-1);
          }
          if ( (err = cudaMalloc( (void**)&d_B, (lddb*Bn)*sizeof(T) )) != cudaSuccess ) {
            fprintf( stderr, "!!!! cudaMalloc failed for: d_B! Error: %s\n", cudaGetErrorString(err) );
            exit(-1);
          }
          check_error( cublasSetMatrix( Am, An, sizeof(T), h_A, lda, d_A, ldda ) );
          check_error( cublasSetMatrix( Bm, Bn, sizeof(T), h_Rc, ldb, d_B, lddb ) );

          check_error( cublasXtrmm( cublas_handle,
                                    side, uplo, trans, diag,
                                    M, N,
                                    &alpha, d_A, ldda,
                                            d_B, lddb) );
          check_error( cublasGetMatrix( M, N, sizeof(T), d_B, lddb, h_Rc, ldb ) );
          check_error(  cudaFree( d_A ) );
          check_error(  cudaFree( d_B ) );

          cudaEventRecord(stop, 0);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&time, start, stop);
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
        
        cudaEventRecord(start, 0);
        check_error( kblas_xtrmm(cublas_handle,
                                side, uplo, trans, diag,
                                M, N,
                                &alpha, h_A, lda,
                                        h_Rk, ldb) );
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        cpu_time += time/1000.0;//to be in sec
      }
      cpu_time /= nruns;
      cpu_perf = gflops / cpu_time;
      cudaDeviceSynchronize();

      if(opts.check){
        ref_error = Xget_max_error_matrix(h_Rc, h_Rk, Bm, Bn, ldb);
      }
      
      for(int r = 0; r < nruns; r++)
      {
        check_error( cudaMemcpy ( (void*)h_Rk, (void*)h_B, sizeB * sizeof(T), cudaMemcpyHostToHost ) );
        cudaEventRecord(start, 0);
        if ( (err = cudaMalloc( (void**)&d_A, (ldda*An)*sizeof(T) )) != cudaSuccess ) {
          fprintf( stderr, "!!!! cudaMalloc failed for: d_A! Error: %s\n", cudaGetErrorString(err) );
          exit(-1);
        }
        if ( (err = cudaMalloc( (void**)&d_B, (lddb*Bn)*sizeof(T) )) != cudaSuccess ) {
          fprintf( stderr, "!!!! cudaMalloc failed for: d_B! Error: %s\n", cudaGetErrorString(err) );
          exit(-1);
        }
        check_error( cublasSetMatrix( Am, An, sizeof(T), h_A, lda, d_A, ldda ) );
        check_error( cublasSetMatrix( Bm, Bn, sizeof(T), h_Rk, ldb, d_B, lddb ) );
        
        check_error( kblasXtrmm(cublas_handle,
                                side, uplo, trans, diag,
                                M, N,
                                &alpha, d_A, lda,
                                        d_B, ldb) );
        check_error( cublasGetMatrix( M, N, sizeof(T), d_B, lddb, h_Rk, ldb ) );
        check_error(  cudaFree( d_A ) );
        check_error(  cudaFree( d_B ) );
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        gpu_time += time/1000.0;//to be in sec
      }
      gpu_time /= nruns;
      gpu_perf = gflops / gpu_time;

      if(opts.check || opts.time){
        free( h_Rc );
      }
      free( h_A );
      free( h_B );
      free( h_Rk );
      printf(" %7.2f (%7.2f)      %7.2f (%7.2f)       %7.2f (%7.2f)        %8.2e\n",
             cpu_perf, 1000.*cpu_time,
             gpu_perf, 1000.*gpu_time,
             ref_perf, 1000.*ref_time,
             ref_error );
    }
    if ( opts.niter > 1 ) {
      printf( "\n" );
    }
  }
    	

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}


#endif