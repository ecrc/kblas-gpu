/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/batch_triangular/test_Xtrtri_batch.cpp

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ali Charara
 * @date 2020-12-10
 **/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>

#if ((defined PREC_c) || (defined PREC_z)) && (defined USE_MKL)
//TODO need to handle MKL types properly
#undef USE_MKL
#endif

//==============================================================================================
// #define DBG_MSG
//==============================================================================================

#include "testing_helper.h"
#include "testing_prec_def.h"
#include "flops.h"
#include "batch_rand.h"

#ifdef check_error
#undef check_error
#endif

#include "Xhelper_funcs.ch" // TODO: need Xset_pointer_1 from this
#include "kblas_operators.h" // TODO: this has templates and C++ host/device functions
#include "kblas_common.h" // TODO: this has templates and C++ host/device functions

//==============================================================================================
// #define DEBUG_DUMP
//==============================================================================================

#ifdef USING
#undef USING
#endif

#define USING printf("uplo %c, trans %c, batchCount %d, backDoor %d\n", opts.uplo, opts.transA, batchCount, opts.bd);


template<class T>
int test_Xsvd_full_batch(kblas_opts& opts)
{

  bool strided = opts.strided;
  SVD_method svd_method = (SVD_method)opts.svd;//SVD_random;//
  int nruns = opts.nruns, ngpu = opts.ngpu;
  int nonUniform = opts.nonUniform;
  double tolerance = opts.tolerance; //ECHO_f(tolerance);

  if(nonUniform && tolerance <= 0){
    printf("Please specify required tolerance for non-uniform batch SVD. Aborting...\n");
    exit(0);
  }
  if(!nonUniform && tolerance > 0){
    printf("Uniform SVD does not accept tolerance yet. Ignoring tolerance(%e)...\n", tolerance);
    tolerance = 0.;
    // exit(0);
  }
  int M, N, minMN, rA, max_M, max_N, max_rank;
  int Am, An;
  int h_strideA, h_strideS, h_strideU, h_strideV;
  int d_strideA, d_strideS, d_strideU, d_strideV;
  int h_lda, h_ldu, h_ldv;
  int d_lda, d_ldu, d_ldv;
  int ione     = 1;
  int ISEED[4] = {0,0,0,1};
  kblasHandle_t kblas_handle[ngpu];
  kblasRandState_t rand_state[ngpu];

  T *h_A = NULL, *h_Au = NULL, *h_Av = NULL, *h_R = NULL;
  int *h_M = NULL, *h_N = NULL, *h_rA = NULL;
  T *d_A[ngpu], *d_S[ngpu], *d_U[ngpu], *d_V[ngpu];
  int *d_M[ngpu], *d_N[ngpu], *d_rA[ngpu];
  int *d_lda_array[ngpu], *d_ldu_array[ngpu], *d_ldv_array[ngpu];
  T **d_A_array[ngpu], **d_U_array[ngpu], **d_V_array[ngpu], **d_S_array[ngpu];

  double Cnorm;
  T one = make_one<T>();
  T zero = make_zero<T>();
    // c_neg_one = make_zero<T>()-make_one<T>();
  T work[1];
  #ifdef DEBUG_DUMP
  FILE *outK, *outL, *outO;
  #endif
  if(ngpu > 1)
    opts.check = 0;


  //USING
  cudaError_t err;
  #ifdef USE_MAGMA
    if(opts.magma == 1){
      magma_init();//TODO is this in the proper place?
    }
  #endif

  for(int g = 0; g < ngpu; g++){
    err = cudaSetDevice( opts.devices[g] );
    kblasCreate(&kblas_handle[g]);
    #ifdef USE_MAGMA
      if(opts.magma == 1){
        kblasEnableMagma(kblas_handle[g]);
      }
    #endif
    kblasInitRandState(kblas_handle[g], &rand_state[g], 16384*2, 0);
  }

  #ifdef USE_OPENMP
  int NUM_THREADS = opts.omp_numthreads;
  #endif//USE_OPENMP

  printf("batchCount    M     N     kblasSVD GF/s (ms)       Error\n");
  printf("========================================================\n");
  for( int itest = 0; itest < opts.ntest; ++itest ) {
    for( int iter = 0; iter < opts.niter; ++iter ) {
      for( int btest = 0; btest < opts.btest; ++btest ) {

        double  gflops, perf,
                ref_avg_perf = 0.0, ref_sdev_perf = 0.0, ref_avg_time = 0.0,
                //rec_avg_perf = 0.0, rec_sdev_perf = 0.0, rec_avg_time = 0.0,
                kblas_perf = 0.0, kblas_time = 0.0, kblas_time_1 = 0.0, cublas_perf = 0.0, cublas_time = 0.0,
                ref_error = 0.0;

        int batchCount = opts.batchCount;
        if(opts.btest > 1)
          batchCount = opts.batch[btest];

        int batchCount_gpu = batchCount / ngpu;

        M = opts.msize[itest];
        N = opts.nsize[itest];
        minMN = (M < N ? M : N);
        rA = opts.rank[0];
        max_rank = opts.rank[1];
        if(rA == 0)
          rA = minMN;
        if(max_rank <= 0)
          max_rank = rA;

        printf("%5d   %5d   %5d   %5d   ",
              batchCount, (int) M, (int) N, rA);
        fflush( stdout );

        if(nonUniform){
          max_M = max_N = 0;
          TESTING_MALLOC_CPU( h_M, int, batchCount);
          TESTING_MALLOC_CPU( h_N, int, batchCount);

          for(int k = 0; k < batchCount; k++){
            if(svd_method <= SVD_random){
              h_M[k] = M;
              h_N[k] = N;
            }
            else{
              h_M[k] = 1 + (rand() % M);
              h_N[k] = 1 + (rand() % N);
            }
            max_M = kmax( max_M, h_M[k] );
            max_N = kmax( max_N, h_N[k] );
          }
          minMN = kmin(max_M, max_N);
          h_lda = max_M;
          h_ldu = h_lda;
          h_ldv = max_N;
          h_strideA = h_lda * max_N;
          h_strideU = h_ldu * max_rank;
          h_strideV = h_ldv * max_rank;
          Am = max_M;
          An = max_N;
        }else{
          h_lda = Am = M;
          An = N;
          h_ldu = h_lda;
          h_ldv = An;
          h_strideA = h_lda * An;
          h_strideU = h_ldu * rA;
          h_strideV = h_ldv * rA;
        }
        // ECHO_I(Am); ECHO_I(An);

        h_strideS = minMN;

        d_lda = kblas_roundup(h_lda, 32);
        d_ldu = kblas_roundup(h_ldu, 32);
        d_ldv = kblas_roundup(h_ldv, 32);
        if(nonUniform){
          d_strideA = d_lda * max_N;
          d_strideU = d_ldu * max_rank;
          d_strideV = d_ldv * max_rank;
        }else{
          d_strideA = d_lda * An;
          d_strideU = d_ldu * rA;
          d_strideV = d_ldv * rA;
        }
        d_strideS = minMN;

        TESTING_MALLOC_CPU( h_A, T, h_strideA * batchCount);

        TESTING_MALLOC_CPU( h_Au, T, h_strideU * batchCount);
        TESTING_MALLOC_CPU( h_Av, T, h_strideV * batchCount);
        if(tolerance > 0)
          TESTING_MALLOC_CPU( h_rA, int, batchCount);

        for(int g = 0; g < ngpu; g++){
          check_error( cudaSetDevice( opts.devices[g] ));
          TESTING_MALLOC_DEV( d_A[g], T, d_strideA * batchCount_gpu);
          TESTING_MALLOC_DEV( d_U[g], T, d_strideU * batchCount_gpu);
          TESTING_MALLOC_DEV( d_V[g], T, d_strideV * batchCount_gpu);
          TESTING_MALLOC_DEV( d_S[g], T, minMN * batchCount_gpu);
          if(tolerance > 0){
            TESTING_MALLOC_DEV( d_rA[g], int, batchCount_gpu);
          }
          else
            d_rA[g] = NULL;
          if(nonUniform){
            TESTING_MALLOC_DEV( d_M[g], int, batchCount_gpu);
            TESTING_MALLOC_DEV( d_N[g], int, batchCount_gpu);

            TESTING_MALLOC_DEV( d_lda_array[g], int, batchCount_gpu);
            TESTING_MALLOC_DEV( d_ldv_array[g], int, batchCount_gpu);
            TESTING_MALLOC_DEV( d_ldu_array[g], int, batchCount_gpu);

            iset_value_1( d_lda_array[g], d_lda, batchCount_gpu, kblasGetStream(kblas_handle[g]));
            iset_value_1( d_ldu_array[g], d_ldu, batchCount_gpu, kblasGetStream(kblas_handle[g]));
            iset_value_1( d_ldv_array[g], d_ldv, batchCount_gpu, kblasGetStream(kblas_handle[g]));
          }
          if(!strided || nonUniform){
            TESTING_MALLOC_DEV( d_A_array[g], T*, batchCount_gpu);
            TESTING_MALLOC_DEV( d_U_array[g], T*, batchCount_gpu);
            TESTING_MALLOC_DEV( d_V_array[g], T*, batchCount_gpu);
            TESTING_MALLOC_DEV( d_S_array[g], T*, batchCount_gpu);

            check_kblas_error( Xset_pointer_4(d_A_array[g], d_A[g], d_lda, d_strideA,
                                              d_U_array[g], d_U[g], d_ldu, d_strideU,
                                              d_V_array[g], d_V[g], d_ldv, d_strideV,
                                              d_S_array[g], d_S[g], 1, d_strideS,
                                              batchCount_gpu, kblasGetStream(kblas_handle[g])) );
          }
        }

        if(opts.check){
          TESTING_MALLOC_CPU( h_R, T, h_strideA * batchCount);

          #ifdef DEBUG_DUMP
          outO = fopen("outO.csv", "a");
          outK = fopen("outK.csv", "a");
          outL = fopen("outL.csv", "a");
          #endif
          if(opts.check)
          {
            opts.time = 0;
            opts.warmup = 0;
            nruns = 1;
          }
        }

        // Xrand_matrix(Am, rA * batchCount, h_Au, h_lda);
        // Xrand_matrix(An, rA * batchCount, h_Av, An);
        // Xrand_matrix(Am, An * batchCount, h_A, h_lda);

        if(nonUniform){
          memset( h_A, 0, h_strideA*batchCount*sizeof(T) );
          for(int b = 0; b < batchCount; b++){
            hilbertMatrix(h_M[b], h_N[b], h_A + b * h_strideA, h_lda, T(2*b+2));
          }
        }else{
          for(int b = 0; b < batchCount; b++){
            hilbertMatrix(Am, An, h_A + b * h_strideA, h_lda, T(b+2));
          }
        }

        if(svd_method <= SVD_random){
          for(int g = 0; g < ngpu; g++){
            if(nonUniform){
              kblasXsvd_full_batch_nonUniform_wsquery(kblas_handle[g],
                                                      Am, An, max_rank,
                                                      batchCount_gpu, svd_method);
            }else{
              kblasXsvd_full_batch_wsquery( kblas_handle[g],
                                            Am, An, rA,
                                            batchCount_gpu, svd_method);
            }
            check_kblas_error( kblasAllocateWorkspace(kblas_handle[g]) );
            check_error( cudaGetLastError() );
          }
        }

        double time = 0;

        for(int r = 0; r < (nruns+opts.warmup); r++){
          for(int g = 0; g < ngpu; g++){
            check_error( cudaSetDevice( opts.devices[g] ));
            check_cublas_error( cublasSetMatrixAsync(Am, An * batchCount_gpu, sizeof(T),
                                                     h_A + h_strideA * batchCount_gpu * g, h_lda,
                                                     d_A[g], d_lda, kblasGetStream(kblas_handle[g]) ) );
            if(nonUniform){
              check_cublas_error( cublasSetVectorAsync(batchCount_gpu, sizeof(int),
                                                       h_M + batchCount_gpu * g, 1,
                                                       d_M[g], 1, kblasGetStream(kblas_handle[g]) ) );
              check_cublas_error( cublasSetVectorAsync(batchCount_gpu, sizeof(int),
                                                       h_N + batchCount_gpu * g, 1,
                                                       d_N[g], 1, kblasGetStream(kblas_handle[g]) ) );
            }
          }

          for(int g = 0; g < ngpu; g++){
            check_error( cudaSetDevice( opts.devices[g] ));
            cudaDeviceSynchronize();//TODO sync with streams instead
          }
          //start_timing(curStream);
          time = -gettime();
          for(int g = 0; g < ngpu; g++){
            check_error( cudaSetDevice( opts.devices[g] ));
            if(nonUniform){
              check_kblas_error( kblas_svd_full_batch(kblas_handle[g],
                                                      Am, An,
                                                      d_M[g], d_N[g],       //m_array, n_array: device buffers
                                                      d_A_array[g], d_lda, d_lda_array[g],//A, lda_array: device buffers
                                                      d_S_array[g],                         //S: device buffer
                                                      d_U_array[g], d_ldu, d_ldu_array[g],//U, ldu_array: device buffers
                                                      d_V_array[g], d_ldv, d_ldv_array[g],//V, ldv_array: device buffers
                                                      svd_method,
                                                      rand_state[g],
                                                      batchCount,
                                                      max_rank, tolerance,
                                                      d_rA[g]) );
            }else
            if(strided){
              check_kblas_error( kblas_svd_full_batch(kblas_handle[g],
                                                      Am, An, rA,
                                                      d_A[g], d_lda, d_strideA,
                                                      d_S[g], d_strideS,
                                                      d_U[g], d_ldu, d_strideU,
                                                      d_V[g], d_ldv, d_strideV,
                                                      svd_method,
                                                      rand_state[g],
                                                      batchCount_gpu));
            }else{
              check_kblas_error( kblas_svd_full_batch(kblas_handle[g],
                                                      Am, An, rA,
                                                      d_A_array[g], d_lda,
                                                      d_S_array[g],
                                                      d_U_array[g], d_ldu,
                                                      d_V_array[g], d_ldv,
                                                      svd_method,
                                                      rand_state[g],
                                                      batchCount_gpu));
            }
          }
          for(int g = 0; g < ngpu; g++){
            check_error( cudaSetDevice( opts.devices[g] ));
            cudaDeviceSynchronize();//TODO sync with streams instead
          }
          //time = get_elapsed_time(curStream);
          time += gettime();
          if(!opts.warmup || r > 0)
            kblas_time_1 += time;
        }
        kblas_time_1 /= nruns;
        kblas_perf = gflops / kblas_time_1;
        kblas_time_1 *= 1000.0;



        if(opts.check){
          for(int g = 0; g < ngpu; g++){
            //reconstruct A
            check_error( cudaSetDevice( opts.devices[g] ));
            if(tolerance > 0 || nonUniform){
              if(tolerance > 0){
                check_cublas_error( cublasGetVectorAsync(batchCount_gpu, sizeof(int),
                                                         d_rA[g], 1,
                                                         h_rA + batchCount_gpu * g, 1,
                                                         kblasGetStream(kblas_handle[g]) ) );
              }
              check_cublas_error( cublasGetMatrixAsync(Am, (nonUniform?max_rank:rA) * batchCount_gpu, sizeof(T),
                                                       d_U[g], d_ldu,
                                                       h_Au + h_strideU * batchCount_gpu * g, h_lda,
                                                       kblasGetStream(kblas_handle[g]) ) );

              check_cublas_error( cublasGetMatrixAsync(An, (nonUniform?max_rank:rA) * batchCount_gpu, sizeof(T),
                                                       d_V[g], d_ldv,
                                                       h_Av + h_strideV * batchCount_gpu * g, An,
                                                       kblasGetStream(kblas_handle[g]) ) );
            }else
            if(nonUniform == false){
              kblas_gemm_batch_strided_wsquery(kblas_handle[g], batchCount_gpu );
              check_kblas_error( kblasAllocateWorkspace(kblas_handle[g]) );

              check_kblas_error( kblas_gemm_batch(kblas_handle[g],
                                                  KBLAS_NoTrans, KBLAS_Trans,
                                                  Am, An, rA,
                                                  one,  d_U[g], d_ldu, d_strideU,
                                                        d_V[g], d_ldv, d_strideV,
                                                  zero, d_A[g], d_lda, d_strideA,
                                                  batchCount_gpu) );

              check_cublas_error( cublasGetMatrixAsync(Am, An * batchCount_gpu, sizeof(T),
                                                       d_A[g], d_lda,
                                                       h_R + Am * An * batchCount_gpu * g, h_lda,
                                                       kblasGetStream(kblas_handle[g]) ) );
            }
          }
          for(int g = 0; g < ngpu; g++){
            check_error(cudaSetDevice( opts.devices[g] ));
            cudaDeviceSynchronize();
          }
          // compute relative error for kblas, relative to lapack,
          // |kblas - lapack| / |lapack|
          // LAPACK_AXPY( &sizeA, &c_neg_one, h_R + s * h_strideA, &ione, h_A + s * h_strideA, &ione );
          // double Cnorm = LAPACK_LANSY( "M",
          //                             (( opts.uplo == KBLAS_Lower ) ? "Lower" : "Upper"),
          //                             &An, h_R + s * h_strideA, &h_lda, work );
          // double err = LAPACK_LANSY( "M",
          //                           (( opts.uplo == KBLAS_Lower ) ? "Lower" : "Upper"),
          //                           &An, h_A + s * h_strideA, &h_lda, work )
          //               / Cnorm;
          /*if ( isnan(err) || isinf(err) ) {
            ref_error = err;
            break;
          }*/
          ref_error = 0;//fmax( err, ref_error );
          for(int b = 0; b < batchCount; b++)
          {
            int Amb = (nonUniform ? h_M[b] : Am);
            int Anb = (nonUniform ? h_N[b] : An);
            if(tolerance > 0 || nonUniform){
              int rAb = (tolerance > 0 ? h_rA[b] : rA);
              LAPACK_GEMM( CblasColMajor, CblasNoTrans, CblasTrans,
                            Amb, Anb, rAb,
                            one,  h_Au + b * h_strideU, h_lda,
                                  h_Av + b * h_strideV, An,
                            zero, h_R + b * h_strideA, h_lda);
            }
            double ref_error_b = Xget_max_error_matrix(h_A + b * h_strideA, h_R + b * h_strideA, Amb, Anb, h_lda);
            // printf("%d:%f ", b, ref_error_b);
            ref_error += ref_error_b;
          }
          ref_error /= batchCount;
        }

        // if(opts.time){
        //   ref_sdev_perf = sqrt((ref_sdev_perf - (ref_avg_perf * ref_avg_perf / nruns))/nruns);
        //   //rec_sdev_perf = sqrt((rec_sdev_perf - (rec_avg_perf * rec_avg_perf / nruns))/nruns);
        // }

        //printf(" %7.4f %7.4f       %7.4f %7.4f %7.4f %7.4f    %7.4f %7.4f %7.4f %7.4f    %.4e \n",
        printf(" %7.4f %7.4f %7.4f    %d   %7.4f %7.4f %7.4f %7.4f    %.4e \n",
               kblas_perf, kblas_time,  kblas_time_1, rA,
               ref_avg_perf / nruns, ref_avg_time, ref_sdev_perf, ref_avg_time / kblas_time_1,
               //rec_avg_perf / nruns, rec_avg_time, rec_sdev_perf, rec_avg_time / kblas_time,
               ref_error);
        if(opts.verbose && tolerance > 0 && opts.check){
          printf( "\nranks: " );
          for(int b = 0; b < batchCount; b++)
          {
            printf("%d ", h_rA[b]);
          }
          printf( "\n" );
        }

        free( h_A );
        free( h_Au );
        free( h_Av );
        if(h_rA) free( h_rA );

        if(opts.check)
          free( h_R );
        for(int g = 0; g < ngpu; g++){
          check_error( cudaSetDevice( opts.devices[g] ));
          check_error( cudaFree( d_A[g] ) );
          check_error( cudaFree( d_S[g] ) );
          check_error( cudaFree( d_U[g] ) );
          check_error( cudaFree( d_V[g] ) );
          if(tolerance > 0)
            check_error( cudaFree( d_rA[g] ) );
          if(nonUniform){
            check_error( cudaFree( d_M[g] ) );
            check_error( cudaFree( d_N[g] ) );

            check_error( cudaFree( d_lda_array[g] ) );
            check_error( cudaFree( d_ldu_array[g] ) );
            check_error( cudaFree( d_ldv_array[g] ) );
          }
          if(!strided || nonUniform){
            check_error(  cudaFree( d_A_array[g] ) );
            check_error(  cudaFree( d_U_array[g] ) );
            check_error(  cudaFree( d_V_array[g] ) );
            check_error(  cudaFree( d_S_array[g] ) );
          }
          // if(!strided){
          //   check_error(  cudaFree( d_A_array[g] ) );
          // }
        }
        #ifdef DEBUG_DUMP
        if(opts.check){
          fclose(outO);
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
  for(int g = 0; g < ngpu; g++){
    kblasDestroyRandState(rand_state[g]);
    kblasDestroy(&kblas_handle[g]);
  }
  #ifdef USE_MAGMA
    if(opts.magma == 1){
      magma_finalize();//TODO is this in the proper place?
    }
  #endif
}

//==============================================================================================
int main(int argc, char** argv)
{
  kblas_opts opts;
  parse_opts(argc, argv, &opts);

#if defined PREC_d
  check_error( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) );
#endif

  test_Xsvd_full_batch<TYPE>(opts);
}
