/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/batch_triangular/test_Xgemm_batch.cpp

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

// #define DEBUG_DUMP

#if ((defined PREC_c) || (defined PREC_z)) && (defined USE_MKL)
//TODO need to handle MKL types properly
#undef USE_MKL
#endif

#include "testing_helper.h"

#ifdef check_error
#undef check_error
#endif

#include "kblas_common.h" // TODO: need iset_value_1 from this
#include "testing_prec_def.h"
#include "Xhelper_funcs.ch" // TODO: need Xset_pointer_2 from this
#include "flops.h"

//==============================================================================================
#ifdef USING
#undef USING
#endif

#define USING printf("side %c, uplo %c, transA %c, transB %c, diag %c , batchCount %d, backDoor %d\n", opts.side, opts.uplo, opts.transA, opts.transB, opts.diag, batchCount, -1);


template<class T>
int test_Xgemm_batch(kblas_opts& opts, T alpha, T beta){

  kblasHandle_t kblas_handle;
  GPU_Timer_t kblas_timer;
  int nruns = opts.nruns;
  int nonUniform = opts.nonUniform;
  int M, N, K, max_M, max_N, max_K;
  int Am, An, Bm, Bn, Cm, Cn;
  int h_strideA, h_strideB, h_strideC;
  int d_strideA, d_strideB, d_strideC;
  int lda, ldb, ldc, ldda, lddb, lddc;
  int *h_lda, *h_ldb, *h_ldc, *d_ldda, *d_lddb, *d_lddc;

  int ISEED[4] = {0,0,0,1};

  T *h_R,
    *h_A, *h_B, *h_C,
    *d_A, *d_B, *d_C;
  int *h_M = NULL, *h_N = NULL, *h_K = NULL,
      *d_M = NULL, *d_N = NULL, *d_K = NULL;
  T **d_A_array, **d_B_array, **d_C_array;
  #ifdef USE_MKL_BATCH
  T **h_A_array, **h_B_array, **h_C_array;
  #endif

  int batchCount = opts.batchCount;
  bool strided = opts.strided;
  if(nonUniform) strided = 0;
  #ifdef DEBUG_DUMP
  FILE *outK, *outL, *outO;
  #endif
  USING
  cudaError_t err;

  check_error( cudaSetDevice( opts.devices[0] ));
  kblasCreate(&kblas_handle);
  #ifdef USE_MAGMA
    if(opts.magma == 1){
      magma_init();
      kblasEnableMagma(kblas_handle);
    }
  #endif
  kblas_timer = newGPU_Timer(kblasGetStream(kblas_handle));


  cublasOperation_t cub_transA = (opts.transA == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);
  cublasOperation_t cub_transB = (opts.transB == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);

  bool isTransA = (opts.transA == KBLAS_Trans);
  bool isTransB = (opts.transB == KBLAS_Trans);

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

      if(nonUniform){
        max_M = max_N = max_K = 0;
        TESTING_MALLOC_CPU( h_M, int, batchCount);
        TESTING_MALLOC_CPU( h_N, int, batchCount);
        TESTING_MALLOC_CPU( h_K, int, batchCount);
        // TESTING_MALLOC_CPU( h_lda, int, batchCount);
        // TESTING_MALLOC_CPU( h_ldb, int, batchCount);
        // TESTING_MALLOC_CPU( h_ldc, int, batchCount);

        for(int k = 0; k < batchCount; k++){
            h_M[k] = 1 + (rand() % M);
            h_N[k] = 1 + (rand() % N);
            h_K[k] = 1 + (rand() % K);
            max_M = kmax( max_M, h_M[k] );
            max_N = kmax( max_N, h_N[k] );
            max_K = kmax( max_K, h_K[k] );
            // h_lda[k] = (isTransA ? K : M);
            // h_ldb[k] = (isTransB ? N : K);
            // h_ldc[k] = M;
            gflops += FLOPS_GEMM<T>(h_M[k], h_N[k], h_K[k]) / 1e9;
        }
      }else{
        gflops = batchCount * FLOPS_GEMM<T>(M, N, K) / 1e9;
      }
      lda = Am = (isTransA ? K : M);
            An = (isTransA ? M : K);
      ldb = Bm = (isTransB ? N : K);
            Bn = (isTransB ? K : N);
      Cm = ldc = M;
      Cn = N;

      ldda = kblas_roundup(lda, 32);
      lddb = kblas_roundup(ldb, 32);
      lddc = kblas_roundup(ldc, 32);

      h_strideA = lda * An;
      h_strideB = ldb * Bn;
      h_strideC = ldc * Cn;
      d_strideA = ldda * An;
      d_strideB = lddb * Bn;
      d_strideC = lddc * Cn;


      TESTING_MALLOC_CPU( h_A, T, h_strideA * batchCount);
      TESTING_MALLOC_CPU( h_B, T, h_strideB * batchCount);
      TESTING_MALLOC_CPU( h_C, T, h_strideC * batchCount);

      TESTING_MALLOC_DEV( d_A, T, d_strideA * batchCount);
      TESTING_MALLOC_DEV( d_B, T, d_strideB * batchCount);
      TESTING_MALLOC_DEV( d_C, T, d_strideC * batchCount);
      if(!strided){
        TESTING_MALLOC_DEV( d_A_array, T*, batchCount);
        TESTING_MALLOC_DEV( d_B_array, T*, batchCount);
        TESTING_MALLOC_DEV( d_C_array, T*, batchCount);
      }
      if(nonUniform){
        TESTING_MALLOC_DEV( d_M, int, batchCount);
        TESTING_MALLOC_DEV( d_N, int, batchCount);
        TESTING_MALLOC_DEV( d_K, int, batchCount);
        TESTING_MALLOC_DEV( d_ldda, int, batchCount);
        TESTING_MALLOC_DEV( d_lddb, int, batchCount);
        TESTING_MALLOC_DEV( d_lddc, int, batchCount);
      }

      if(opts.check || opts.time)
      {
        TESTING_MALLOC_CPU( h_R, T, h_strideC * batchCount);

        #ifdef DEBUG_DUMP
        //outO = fopen("outO", "a");
        outK = fopen("outK.csv", "a");
        outL = fopen("outL.csv", "a");
        #endif
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
        memcpy(h_R, h_C, h_strideC * batchCount * sizeof(T));
      //int curDev;
      //cudaGetDevice(&curDev);
      cudaStream_t curStream = kblasGetStream(kblas_handle);

      check_cublas_error( cublasSetMatrixAsync( Am, An * batchCount, sizeof(T), h_A, lda, d_A, ldda, curStream ) );
      check_cublas_error( cublasSetMatrixAsync( Bm, Bn * batchCount, sizeof(T), h_B, ldb, d_B, lddb, curStream ) );
      if(!strided){
        check_kblas_error( Xset_pointer_3(d_A_array, d_A, ldda, d_strideA,
                                          d_B_array, d_B, lddb, d_strideB,
                                          d_C_array, d_C, lddc, d_strideC,
                                          batchCount, kblasGetStream(kblas_handle)) );
      }

      if(nonUniform){
        check_cublas_error( cublasSetVectorAsync(batchCount, sizeof(int),
                                                 h_M, 1,
                                                 d_M, 1, curStream) );
        check_cublas_error( cublasSetVectorAsync(batchCount, sizeof(int),
                                                 h_N, 1,
                                                 d_N, 1, curStream ) );
        check_cublas_error( cublasSetVectorAsync(batchCount, sizeof(int),
                                                 h_K, 1,
                                                 d_K, 1, curStream ) );
        check_kblas_error(iset_value_1( d_ldda, ldda, batchCount, curStream));
        check_kblas_error(iset_value_1( d_lddb, lddb, batchCount, curStream));
        check_kblas_error(iset_value_1( d_lddc, lddc, batchCount, curStream));
        // check_cublas_error( cublasSetVectorAsync(batchCount, sizeof(int),
        //                                          h_lda, 1,
        //                                          d_ldda, 1, curStream) );
        // check_cublas_error( cublasSetVectorAsync(batchCount, sizeof(int),
        //                                          h_ldb, 1,
        //                                          d_lddb, 1, curStream ) );
        // check_cublas_error( cublasSetVectorAsync(batchCount, sizeof(int),
        //                                          h_ldc, 1,
        //                                          d_lddc, 1, curStream ) );
      }
      if(opts.warmup){
        check_cublas_error( cublasSetMatrixAsync( Cm, Cn * batchCount, sizeof(T), h_C, ldc, d_C, lddc, curStream ) );
        check_cublas_error( cublasXgemm( kblasGetCublasHandle(kblas_handle),
                                        cub_transA, cub_transB,
                                        M, N, K,
                                        &alpha, d_A, ldda,
                                                d_B, lddb,
                                        &beta,  d_C, lddc) );
        #if (defined USE_OPENMP) && (defined USE_MKL)
        if(opts.time){
          omp_set_num_threads(NUM_THREADS);
          //omp_set_nested(true);
          #pragma omp parallel shared(h_R, N, lda)// num_threads (NUM_THREADS)
          {
            #pragma omp for //schedule(guided,10)
            for (int s=0; s < batchCount; s++)
            {
              LAPACK_GEMM( CblasColMajor,
                           (opts.transA == KBLAS_Trans ? CblasTrans : CblasNoTrans), 
                           (opts.transB == KBLAS_Trans ? CblasTrans : CblasNoTrans),
                           Cm, Cn, An,
                           alpha, h_A + s * lda * An, lda,
                                  h_B + s * ldb * Bn, ldb,
                           beta,  h_C + s * ldc * Cn, ldc
              );
            }
          }
          memcpy(h_R, h_C, h_strideC * batchCount * sizeof(T));
        }
        #endif//USE_OPENMP
      }

      if(nonUniform)
        kblas_gemm_batch_nonuniform_wsquery(kblas_handle);
      else
      if(strided)
        kblas_gemm_batch_strided_wsquery(kblas_handle, batchCount);
      check_kblas_error( kblasAllocateWorkspace(kblas_handle) );
      double time = 0;

      //if(opts.bd >= 0) kblas_back_door[0] = opts.bd;

      for(int r = 0; r < nruns; r++)
      {
        check_cublas_error( cublasSetMatrixAsync( Cm, Cn * batchCount, sizeof(T), h_C, ldc, d_C, lddc, curStream ) );

        kblasTimerTic(kblas_handle);
        if(nonUniform){
          //*
          check_kblas_error( kblas_gemm_batch(kblas_handle,
                                              opts.transA, opts.transB,
                                              d_M, d_N, d_K,
                                              max_M, max_N, max_K,
                                              alpha, (const T**)d_A_array, d_ldda,
                                                     (const T**)d_B_array, d_lddb,
                                              beta,        (T**)d_C_array, d_lddc,
                                              batchCount ) );
          /*/
          check_kblas_error( kblas_gemm_batch(kblas_handle,
                                              opts.transA, opts.transB,
                                              d_M, d_N, d_K,
                                              alpha, (const T**)d_A_array, d_ldda,
                                                     (const T**)d_B_array, d_lddb,
                                              beta,        (T**)d_C_array, d_lddc,
                                              batchCount ) );//*/
        }else
        if(strided){
          check_kblas_error( kblas_gemm_batch(kblas_handle,
                                              opts.transA, opts.transB,
                                              M, N, K,
                                              alpha, d_A, ldda, An*ldda,
                                                     d_B, lddb, Bn*lddb,
                                              beta,  d_C, lddc, Cn*lddc,
                                              batchCount) );
        }else{
          check_kblas_error( kblas_gemm_batch(kblas_handle,
                                              opts.transA, opts.transB,
                                              M, N, K,
                                              alpha, (const T**)d_A_array, ldda,
                                                     (const T**)d_B_array, lddb,
                                              beta,  (T**)d_C_array, lddc,
                                              batchCount) );
        }
		    kblasTimerRecordEnd(kblas_handle);
        time = kblasTimerToc(kblas_handle);
        kblas_time += time;
      }
      kblas_time /= nruns;
      kblas_perf = gflops / kblas_time;
      kblas_time *= 1000.;//convert to ms


      #ifdef USE_MKL
      if(opts.check || opts.time){
        if(opts.check){
          check_cublas_error( cublasGetMatrixAsync( Cm, Cn * batchCount, sizeof(T), d_C, lddc, h_R, ldc, curStream ) );
          cudaDeviceSynchronize();
        }

        for(int r = 0; r < nruns; r++)
        {
          if(opts.time){
            memcpy(h_C, h_R, h_strideC * batchCount * sizeof(T));
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

            if(nonUniform){
              LAPACK_GEMM( CblasColMajor,
                           (opts.transA == KBLAS_Trans ? CblasTrans : CblasNoTrans), 
                           (opts.transB == KBLAS_Trans ? CblasTrans : CblasNoTrans),
                           h_M[s], h_N[s], h_K[s],
                           alpha, h_A + s * lda * An, lda,
                                  h_B + s * ldb * Bn, ldb,
                           beta,  h_C + s * ldc * Cn, ldc
                      );
            }else{
              LAPACK_GEMM( CblasColMajor,
                           (opts.transA == KBLAS_Trans ? CblasTrans : CblasNoTrans), 
                           (opts.transB == KBLAS_Trans ? CblasTrans : CblasNoTrans),
                           Cm, Cn, An,
                           alpha, h_A + s * lda * An, lda,
                                  h_B + s * ldb * Bn, ldb,
                           beta,  h_C + s * ldc * Cn, ldc
                      );

            }
            if(opts.check && !opts.time){
              if(nonUniform){
                #ifdef DEBUG_DUMP
                printMatrix(h_M[s], h_N[s], h_R + s * ldc * Cn, ldc, outK);
                printMatrix(h_M[s], h_N[s], h_C + s * ldc * Cn, ldc, outL);
                #endif
                ref_error += Xget_max_error_matrix(h_C + s * ldc * Cn, h_R + s * ldc * Cn, h_M[s], h_N[s], ldc);
              }
              else
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
            memcpy(h_C, h_R, h_strideC * batchCount * sizeof(T));
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
          //LAPACK_AXPY( &h_strideA, &c_neg_one, h_A, &ione, h_R, &ione );
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
      if(nonUniform){
        free(h_M);
        free(h_N);
        free(h_K);
        // free(h_lda);
        // free(h_ldb);
        // free(h_ldc);
      }
      if(opts.check || opts.time)
        free( h_R );
      check_error(  cudaFree( d_A ) );
      check_error(  cudaFree( d_B ) );
      check_error(  cudaFree( d_C ) );
      if(nonUniform){
        check_error(  cudaFree( d_M ) );
        check_error(  cudaFree( d_N ) );
        check_error(  cudaFree( d_K ) );
        check_error(  cudaFree( d_ldda ) );
        check_error(  cudaFree( d_lddb ) );
        check_error(  cudaFree( d_lddc ) );
      }
      if(!strided){
        check_error(  cudaFree( d_A_array ) );
        check_error(  cudaFree( d_B_array ) );
        check_error(  cudaFree( d_C_array ) );
      }

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
      #ifdef DEBUG_DUMP
      if(opts.check){
        // fclose(outO);
        fclose(outL);
        fclose(outK);
      }
      #endif
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
  deleteGPU_Timer(kblas_timer);
  #ifdef USE_MAGMA
    if(opts.magma == 1){
      magma_finalize();//TODO is this in the proper place?
    }
  #endif
}


//==============================================================================================
int main(int argc, char** argv)
{
  // kblas_init();

  kblas_opts opts;
  parse_opts( argc, argv, &opts );

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
