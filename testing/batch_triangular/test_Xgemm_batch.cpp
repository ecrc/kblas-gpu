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
#include "cublas_v2.h"

#include "kblas.h"

#define USE_MKL_BATCH

// #include "Xtr_common.ch"
#include "testing_prec_def.h"
#include "batch_triangular/Xblas_core.ch"

#if ((defined PREC_c) || (defined PREC_z)) && (defined USE_MKL)
//TODO need to handle MKL types properly
#undef USE_MKL
#endif

#ifdef USE_MKL
#include <mkl_lapack.h>
#include <mkl_blas.h>
#endif//USE_MKL

#include "testing_Xtr_common.h"

#ifdef USE_OPENMP
#include "omp.h"
#endif//USE_OPENMP

//==============================================================================================
#ifdef USING
#undef USING
#endif

#define USING printf("side %c, uplo %c, transA %c, transB %c, diag %c , batchCount %d, backDoor %d\n", opts.side, opts.uplo, opts.transA, opts.transB, opts.diag, batchCount, -1);

// extern bool use_magma_gemm;
// extern bool use_cublas_gemm;

template<class T>
int test_Xgemm_batch(kblas_opts& opts, T alpha, T beta){

  kblasHandle_t kblas_handle;
  int nruns = opts.nruns;
  int M, N, K;
  int Am, An, Bm, Bn, Cm, Cn;
  int sizeA, sizeB, sizeC;
  int strideA, strideB, strideC;
  int lda, ldb, ldc, ldda, lddb, lddc;

  int ISEED[4] = {0,0,0,1};

  T *h_R,
    *h_A, *h_B, *h_C,
    *d_A, *d_B, *d_C;
  #ifdef USE_MKL_BATCH
  T **h_A_array, **h_B_array, **h_C_array;
  #endif

  int batchCount = opts.batchCount;
  bool strided = 0;//kblas_back_door[3] > 0;
  //FILE *outK, *outL, *outO;

  USING
  cudaError_t err;

  check_error( cudaSetDevice( opts.devices[0] ));
  kblasCreate(&kblas_handle);


  cublasOperation_t cub_transA = (opts.transA == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);
  cublasOperation_t cub_transB = (opts.transB == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);


  #ifdef USE_MKL_BATCH
  TESTING_MALLOC_CPU( h_A_array, T*, batchCount );
  TESTING_MALLOC_CPU( h_B_array, T*, batchCount );
  TESTING_MALLOC_CPU( h_C_array, T*, batchCount );
  #endif

  #ifdef USE_OPENMP
  int NUM_THREADS = opts.omp_numthreads;
  #endif//USE_OPENMP

  printf("    M     N    K    kblasGEMM GF/s (ms)  cublasGEMM GF/s (ms)  SP      Error\n");
  printf("========================================================================\n");
  for( int itest = 0; itest < opts.ntest; ++itest ) {
    for( int iter = 0; iter < opts.niter; ++iter ) {
      for( int btest = 0; btest < opts.btest; ++btest ) {

      double  gflops, perf,
              ref_avg_perf = 0.0, ref_sdev_perf = 0.0, ref_avg_time = 0.0,
              #ifdef USE_MKL_BATCH
              bat_avg_perf = 0.0, bat_sdev_perf = 0.0, bat_avg_time = 0.0,
              #endif
              kblas_perf = 0.0, kblas_time = 0.0, magma_time = 0.0, cublas_time = 0.0,
              ref_error = 0.0;

      if(opts.btest > 1)
        batchCount = opts.batch[btest];

      ref_avg_time = kblas_time = ref_error = magma_time = cublas_time = 0.0;
      M = opts.msize[itest];
      N = opts.nsize[itest];
      K = opts.ksize[itest];

      printf("%10d %5d %5d %5d   ",
              batchCount, (int) M, (int) N, (int) K);
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
      Cm = ldc = M;
      Cn = N;

      ldda = ((lda+31)/32)*32;
      lddb = ((ldb+31)/32)*32;
      lddc = ((ldc+31)/32)*32;

      sizeA = lda * An;
      sizeB = ldb * Bn;
      sizeC = ldc * Cn;
      strideA = ldda * An;
      strideB = lddb * Bn;
      strideC = lddc * Cn;

      gflops = batchCount * FLOPS_GEMM<T>(M, N, K ) / 1e9;

      TESTING_MALLOC_CPU( h_A, T, sizeA * batchCount);
      TESTING_MALLOC_CPU( h_B, T, sizeB * batchCount);
      TESTING_MALLOC_CPU( h_C, T, sizeC * batchCount);

      TESTING_MALLOC_DEV( d_A, T, strideA * batchCount);
      TESTING_MALLOC_DEV( d_B, T, strideB * batchCount);
      TESTING_MALLOC_DEV( d_C, T, strideC * batchCount);

      if(opts.check || opts.time)
      {
        TESTING_MALLOC_CPU( h_R, T, sizeC * batchCount);

        /*outO = fopen("outO", "a");
        outK = fopen("outK", "a");
        outL = fopen("outL", "a");*/
        if(opts.check)
        {
          opts.time = 0;
          nruns = 1;
        }
      }

      Xrand_matrix(Am, An * batchCount, h_A, lda);
      Xrand_matrix(Bm, Bn * batchCount, h_B, ldb);
      Xrand_matrix(Cm, Cn * batchCount, h_C, ldc);

      #ifdef USE_MKL_BATCH
      for( int i = 0; i < batchCount; ++i ) {
        h_A_array[i] = h_A + lda * An * i;
        h_B_array[i] = h_B + ldb * Bn * i;
        h_C_array[i] = h_C + ldc * Cn * i;
      }
      #endif

      if(opts.time)
        memcpy(h_R, h_C, sizeC * batchCount * sizeof(T));
      //int curDev;
      //cudaGetDevice(&curDev);
      cudaStream_t curStream = kblas_handle->stream;
      /*
      check_error( cudaStreamCreateWithFlags( &curStream, cudaStreamNonBlocking) );
      check_error(cublasSetStream(cublas_handle, curStream));*/

      check_error( cublasSetMatrixAsync( Am, An * batchCount, sizeof(T), h_A, lda, d_A, ldda, curStream ) );
      check_error( cublasSetMatrixAsync( Bm, Bn * batchCount, sizeof(T), h_B, ldb, d_B, lddb, curStream ) );

      if(opts.warmup){
        check_error( cublasSetMatrixAsync( Cm, Cn * batchCount, sizeof(T), h_C, ldc, d_C, lddc, curStream ) );
        check_error( cublasXgemm( kblas_handle->cublas_handle,
                                  cub_transA, cub_transB,
                                  M, N, K,
                                  &alpha, d_A, ldda,
                                          d_B, lddb,
                                  &beta,  d_C, lddc) );
        #ifdef USE_OPENMP
        if(opts.time){
          omp_set_num_threads(NUM_THREADS);
          //omp_set_nested(true);
          #pragma omp parallel shared(h_R, N, lda)// num_threads (NUM_THREADS)
          {
            #pragma omp for //schedule(guided,10)
            for (int s=0; s < batchCount; s++)
            {
              LAPACK_GEMM( (opts.transA == KBLAS_Trans ? "Transpose" : "No Transpose"), (opts.transB == KBLAS_Trans ? "Transpose" : "No Transpose"),
                           &Cm, &Cn, &An,
                           &alpha, h_A + s * lda * An, &lda,
                                   h_B + s * ldb * Bn, &ldb,
                           &beta,  h_C + s * ldc * Cn, &ldc
              );
            }
          }
          memcpy(h_R, h_C, sizeC * batchCount * sizeof(T));
        }
        #endif//USE_OPENMP
      }

      Xgemm_batch_strided_wsquery(kblas_handle, batchCount);
      check_error( kblasAllocateWorkspace(kblas_handle) );

      double time = 0;

      //if(opts.bd >= 0) kblas_back_door[0] = opts.bd;
      #ifdef USE_MAGMA
      // use_magma_gemm = 1; use_cublas_gemm = 0;
      //TODO this is not a safe access
      kblas_handle->use_magma = 1;
      for(int r = 0; r < nruns; r++)
      {
        check_error( cublasSetMatrixAsync( Cm, Cn * batchCount, sizeof(T), h_C, ldc, d_C, lddc, curStream ) );

        kblas_handle->tic();
        check_error( Xgemm_batch_strided( kblas_handle,
                                  opts.transA, opts.transB,
                                  M, N, K,
                                  alpha, d_A, ldda, An*ldda,
                                         d_B, lddb, Bn*lddb,
                                  beta,  d_C, lddc, Cn*lddc,
                                  batchCount) );
        time = kblas_handle->toc();
        magma_time += time;
      }
      magma_time /= nruns;
      magma_time *= 1000.;//convert to ms
      #endif

      // use_magma_gemm = 0; use_cublas_gemm = 1;
      kblas_handle->use_magma = 0;
      /*for(int r = 0; r < nruns; r++)
      {
        check_error( cublasSetMatrixAsync( Cm, Cn * batchCount, sizeof(T), h_C, ldc, d_C, lddc, curStream ) );

        kblas_handle->tic();
        check_error( Xgemm_batch_strided( kblas_handle,
                                  opts.transA, opts.transB,
                                  M, N, K,
                                  alpha, d_A, ldda, An*ldda,
                                         d_B, lddb, Bn*lddb,
                                  beta,  d_C, lddc, Cn*lddc,
                                  batchCount) );
        time = kblas_handle->toc();
        cublas_time += time;
      }
      cublas_time /= nruns;
      cublas_time *= 1000.;//convert to ms*/

      // use_magma_gemm = 0; use_cublas_gemm = 0;
      for(int r = 0; r < nruns; r++)
      {
        check_error( cublasSetMatrixAsync( Cm, Cn * batchCount, sizeof(T), h_C, ldc, d_C, lddc, curStream ) );

        kblas_handle->tic();
        check_error( Xgemm_batch_strided( kblas_handle,
                                  opts.transA, opts.transB,
                                  M, N, K,
                                  alpha, d_A, ldda, An*ldda,
                                         d_B, lddb, Bn*lddb,
                                  beta,  d_C, lddc, Cn*lddc,
                                  batchCount) );
        time = kblas_handle->toc();
        kblas_time += time;
      }
      kblas_time /= nruns;
      kblas_perf = gflops / kblas_time;
      kblas_time *= 1000.;//convert to ms


      #ifdef USE_MKL
      if(opts.check || opts.time){
        if(opts.check){
          check_error( cublasGetMatrixAsync( Cm, Cn * batchCount, sizeof(T), d_C, lddc, h_R, ldc, curStream ) );
          cudaDeviceSynchronize();
        }

        for(int r = 0; r < nruns; r++)
        {
          if(opts.time){
            memcpy(h_C, h_R, sizeC * batchCount * sizeof(T));
            time = -gettime();
          }
          #ifdef USE_OPENMP
          omp_set_num_threads(NUM_THREADS);
          //omp_set_nested(true);
          #pragma omp parallel shared(h_A, N, lda)// num_threads (NUM_THREADS)
          {
          #pragma omp for //schedule(guided,10)
          #endif//USE_OPENMP
          for (int s=0; s < batchCount; s++)
          {

            LAPACK_GEMM( (opts.transA == KBLAS_Trans ? "Transpose" : "No Transpose"), (opts.transB == KBLAS_Trans ? "Transpose" : "No Transpose"),
                        &Cm, &Cn, &An,
                        &alpha, h_A + s * lda * An, &lda,
                                h_B + s * ldb * Bn, &ldb,
                        &beta,  h_C + s * ldc * Cn, &ldc
                      );

            if(opts.check && !opts.time){
              ref_error += Xget_max_error_matrix(h_C + s * ldc * Cn, h_R + s * ldc * Cn, Cm, Cn, ldc);
            }
          }
          #ifdef USE_OPENMP
          }
          #endif//USE_OPENMP
          if(opts.time){
            time += gettime();
            perf = gflops / time;
            ref_avg_perf += perf;
            ref_sdev_perf += perf * perf;
            ref_avg_time += time;
          }
        }
        if(opts.check) ref_error /= batchCount;
        #ifdef USE_MKL_BATCH
        for(int r = 0; r < nruns; r++)
        {
          const char transA_bat[1] = {opts.transA == KBLAS_Trans ? 'T' : 'N'};
          const char transB_bat[1] = {opts.transB == KBLAS_Trans ? 'T' : 'N'};
          int m_bat[1] = {Cm};
          int n_bat[1] = {Cn};
          int k_bat[1] = {An};
          const int grp_size = 1;

          if(opts.time){
            memcpy(h_C, h_R, sizeC * batchCount * sizeof(T));
            time = -gettime();
          }

          LAPACK_GEMM_BATCH( transA_bat, transB_bat,
                      m_bat, n_bat, k_bat,
                      &alpha, (const T**)h_A_array, &lda,
                              (const T**)h_B_array, &ldb,
                      &beta,  h_C_array, &ldc,
                      &grp_size, &batchCount
                    );

          if(opts.time){
            time += gettime();
            perf = gflops / time;
            bat_avg_perf += perf;
            bat_sdev_perf += perf * perf;
            bat_avg_time += time;
          }
        }
        #endif

        /*if(opts.check){
          ref_error = Xget_max_error_matrix(h_A, h_R, N, N, lda);
          //Cnorm = kblas_lange<T,double>( 'M', N, N, h_A, lda);
          //LAPACK_AXPY( &sizeA, &c_neg_one, h_A, &ione, h_R, &ione );
          //ref_error = kblas_lange<T,double>( 'M', N, N, h_R, lda) / Cnorm;
        }*/
        if(opts.time){
          ref_avg_time = (ref_avg_time / nruns) * 1000.;//convert to ms
          #ifdef USE_MKL_BATCH
          bat_avg_time = (bat_avg_time / nruns) * 1000.;//convert to ms
          #endif
        }
      }
      #endif//USE_MKL

      free( h_A );
      free( h_B );
      free( h_C );
      if(opts.check || opts.time)
        free( h_R );
      check_error(  cudaFree( d_A ) );
      check_error(  cudaFree( d_B ) );
      check_error(  cudaFree( d_C ) );

      if(opts.time){
        ref_sdev_perf = sqrt((ref_sdev_perf - (ref_avg_perf * ref_avg_perf / nruns)) / nruns);
        #ifdef USE_MKL_BATCH
        bat_sdev_perf = sqrt((bat_sdev_perf - (bat_avg_perf * bat_avg_perf / nruns)) / nruns);
        #endif
      }

      printf(" %7.4f %7.4f %7.4f %7.4f        %7.4f %7.4f %7.4f %7.4f     ",
             kblas_perf, kblas_time, cublas_time, magma_time,
             ref_avg_perf / nruns, ref_avg_time, ref_sdev_perf,
             ref_avg_time / kblas_time);
      #ifdef USE_MKL_BATCH
      printf(" %7.4f %7.4f %7.4f %7.4f ",
             bat_avg_perf / nruns, bat_avg_time, bat_sdev_perf,
             bat_avg_time / kblas_time);
      #endif
      printf(" %.4e \n", ref_error);
      /*if(opts.check){
        fclose(outO);
        fclose(outL);
        fclose(outK);
      }*/
    }
    }
    if ( opts.niter > 1 ) {
      printf( "\n" );
    }
  }


  #ifdef USE_MKL_BATCH
  free( h_A_array );
  free( h_B_array );
  free( h_C_array );
  #endif


  kblasDestroy(&kblas_handle);
}


//==============================================================================================
int main(int argc, char** argv)
{
  // kblas_init();

  kblas_opts opts;
  if(!parse_opts( argc, argv, &opts )){
    USAGE;
    return -1;
  }

#if (defined PREC_s) || (defined PREC_d)
  TYPE alpha = 0.28;
  TYPE beta = 1.2;
#elif defined PREC_c
  TYPE alpha = make_cuFloatComplex(1.2, -0.6);
  TYPE beta = make_cuFloatComplex(0.2, 1.6);
#elif defined PREC_z
  TYPE alpha = make_cuDoubleComplex(1.2, -0.6);
  TYPE beta = make_cuDoubleComplex(0.2, 1.6);
#endif
  test_Xgemm_batch<TYPE>(opts, alpha, beta);

  // kblas_finalize();
}
