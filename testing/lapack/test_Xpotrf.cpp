/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_plr/test_Xpotrf.cpp

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

#include "testing_helper.h"
#include "testing_prec_def.h"
#include "flops.h"
#include "kblas_potrf.h"

#ifdef USE_OPENMP
#include "omp.h"
#endif//USE_OPENMP

//==============================================================================================
template<typename T>
int lapackXpotrf(char uplo,
                 int n,
                 T *M, int incM)
{

  #define LAP_UPPER "Upper"
  #define LAP_LOWER "Lower"
  int info;


  LAPACK_POTRF( (uplo == KBLAS_Lower ? LAP_LOWER : LAP_UPPER), &n,
                M, &incM,
                &info);

  return 1;
}

//==============================================================================================
template<class T>
int test_Xpotrf(kblas_opts& opts){

  kblasHandle_t kblas_handle;
  GPU_Timer_t kblas_timer;
  int nruns = opts.nruns;
  int N;
  int sizeA;
  int lda, ldda;
  int ISEED[4] = {0,0,0,1};

  T *h_A, *h_R;
  T *d_A, t;
  int info, *d_info;

  // USING
  cudaError_t err;

  T one = make_one<T>(),
    Cnorm = make_zero<T>(),
    work[1],
    mone = make_zero<T>() - one;
  int ione     = 1;

  check_error( cudaSetDevice( opts.devices[0] ));
  kblasCreate(&kblas_handle);
  kblas_timer = newGPU_Timer(kblasGetStream(kblas_handle));

  printf("    N     kblasPOTRF GF/s (ms)  MKLPOTRF GF/s (ms) SP_MKL  Error\n");
  printf("====================================================================\n");
  for( int i = 0; i < opts.ntest; ++i ) {
    for( int iter = 0; iter < opts.niter; ++iter ) {

      double  gflops, perf,
              ref_avg_perf = 0.0, ref_sdev_perf = 0.0, ref_avg_time = 0.0,
              kblas_perf = 0.0, kblas_time = 0.0, kblas_time_tr = 0.0,
              ref_error = 0.0;

      N = opts.nsize[i];

      gflops = FLOPS_POTRF<T>( N ) / 1e9;

      printf("%5d    ",
             (int) N);
      fflush( stdout );

      lda = N;

      ldda = kblas_roundup(lda, 32);

      sizeA = lda*N;

      TESTING_MALLOC_PIN( h_A, T, sizeA);

      TESTING_MALLOC_DEV( d_A, T, ldda*N);
      TESTING_MALLOC_DEV( d_info, int, 1);

      // Initialize matrix and vector
      //printf("Initializing on cpu .. \n");
      Xrand_matrix(N, N, h_A, lda);
      Xmatrix_make_hpd( N, h_A, lda);
      
      if(opts.check || opts.time)
      {
        TESTING_MALLOC_PIN( h_R, T, sizeA);
        if(opts.check)
        {
          opts.time = 0;
          nruns = 1;
        }
      }

      cudaStream_t curStream = kblasGetStream(kblas_handle);


      /*if(opts.warmup){
        check_error( cublasSetMatrixAsync( Bm, Bn, sizeof(T), h_B, ldb, d_B, lddb, curStream ) );
        check_error( cublasXtrsm( cublas_handle,
                                  side, uplo, trans, diag,
                                  M, N,
                                  &alpha, d_A, ldda,
                                  d_B, lddb) );
      }*/
      double time = 0, time_tr = 0.;

      for(int r = 0; r < nruns; r++)
      {
        // if(opts.time)
        //   gpuTimerTic(kblas_timer);
        //check_error( cublasSetMatrixAsync( N, N, sizeof(T), h_M, ldm, d_M, lddm, curStream ) );
        check_cublas_error( cublasSetMatrixAsync( N, N, sizeof(T), h_A, lda, d_A, ldda, curStream ) );

        gpuTimerTic(kblas_timer);

        check_kblas_error( kblas_potrf(kblas_handle,
                                       KBLAS_Lower,
                                       N,
                                       d_A, ldda,
                                       d_info) );
        gpuTimerRecordEnd(kblas_timer);
        time = gpuTimerToc(kblas_timer);
        kblas_time += time;
        // if(opts.time) {
        //   check_error( cublasGetMatrix( N, N, sizeof(T), d_A, ldda, h_R, lda ) );
        //   time_tr += gettime();
        //   kblas_time_tr += time_tr;
        // }
        // Make sure the operation did not fail 
        check_error( cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost) );
        if(info != 0)
            printf("Potrf failed at column %d\n", -info);
      }
      kblas_time /= nruns;
      kblas_perf = gflops / kblas_time;
      // if(opts.time) {
      //   kblas_time_tr /= nruns;
      // }
      kblas_time *= 1000.;//convert to ms

      #ifdef USE_MKL
      if(opts.check || opts.time){
        if(opts.check){
          check_cublas_error( cublasGetMatrix( N, N, sizeof(T), d_A, ldda, h_R, lda ) );
          cudaDeviceSynchronize();
        }else
        if(opts.time){
          memcpy(h_R, h_A, sizeA * sizeof(T));
        }

        for(int r = 0; r < nruns; r++)
        {
          if(opts.time){
             if(r > 0)
               memcpy(h_A, h_R, sizeA * sizeof(T));
            time = -gettime();
          }

          lapackXpotrf(KBLAS_Lower,
                       N,
                       h_A, lda);

          if(opts.check && !opts.time){
            //ref_error += Xget_max_error_matrix(h_A, h_R, N, N, lda);
            Cnorm = LAPACK_LANGE( "f", &N, &N, h_A, &lda, work );
            LAPACK_AXPY( &sizeA, &mone, h_A, &ione, h_R, &ione );
            ref_error = LAPACK_LANGE( "f", &N, &N, h_R, &lda, work ) / Cnorm;
          }
          if(opts.time){
            time += gettime();
            perf = gflops / time;
            ref_avg_perf += perf;
            ref_sdev_perf += perf * perf;
            ref_avg_time += time;
          }
        }
        if(opts.time)
          ref_avg_time = (ref_avg_time / nruns) * 1000.;//convert to ms
        cudaFreeHost( h_R );
      }
      #endif

      cudaFreeHost( h_A );
      check_error(  cudaFree( d_A ) );
      check_error(  cudaFree( d_info ) );

      printf(" %7.2f %7.2f  %7.2f     %7.2f %7.2f     %2.2f  %8.2e\n",
             kblas_perf, kblas_time, (kblas_time_tr+kblas_time),
             ref_avg_perf / nruns, ref_avg_time,
             ref_avg_time / kblas_time,
             ref_error );

    }
    if ( opts.niter > 1 ) {
      printf( "\n" );
    }
  }

  kblasDestroy(&kblas_handle);
}

//==============================================================================================
int main(int argc, char** argv)
{

  kblas_opts opts;
  if(!parse_opts( argc, argv, &opts )){
    // USAGE;
    return -1;
  }

  test_Xpotrf<TYPE>(opts);

}


