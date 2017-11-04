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

#include "testing_prec_def.h"
#include "batch_triangular/Xblas_core.ch"
#include "batch_triangular/Xhelper_funcs.ch"

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
// #define DEBUG_DUMP
//==============================================================================================

#ifdef USING
#undef USING
#endif

#define USING printf("uplo %c, trans %c, batchCount %d, backDoor %d\n", opts.uplo, opts.transA, batchCount, opts.bd);

template<class T>
int test_Xsyrk_batch(kblas_opts& opts, T alpha, T beta)
{

  int nruns = opts.nruns, ngpu = opts.ngpu;
  int N, K;
  int Am, An, Cm, Cn;
  int sizeA, sizeC;
  int lda, ldc, ldda, lddc;
  int ione     = 1;
  int ISEED[4] = {0,0,0,1};
  kblasHandle_t kblas_handle[ngpu];

  T *h_A, *h_C, *h_R;
  T *d_A[ngpu], *d_C[ngpu];
  T **d_A_array[ngpu], **d_C_array[ngpu];

  double Cnorm;
  T c_one = make_one<T>(),
    c_neg_one = make_zero<T>()-make_one<T>();
  T work[1];
  #ifdef DEBUG_DUMP
  FILE *outK, *outL, *outO;
  #endif
  if(ngpu > 1)
    opts.check = 0;


  //USING
  cudaError_t err;

  for(int g = 0; g < ngpu; g++){
    err = cudaSetDevice( opts.devices[g] );
    kblasCreate(&kblas_handle[g]);
  }

  #ifdef USE_OPENMP
  int NUM_THREADS = opts.omp_numthreads;
  #endif//USE_OPENMP

  printf("batchCount    N     K     kblasSYRK GF/s (ms)  lapackSYRK GF/s (ms)  SP      Error\n");
  printf("==================================================================================\n");
  for( int itest = 0; itest < opts.ntest; ++itest ) {
    for( int iter = 0; iter < opts.niter; ++iter ) {
      for( int btest = 0; btest < opts.btest; ++btest ) {

        double  gflops, perf,
                ref_avg_perf = 0.0, ref_sdev_perf = 0.0, ref_avg_time = 0.0,
                //rec_avg_perf = 0.0, rec_sdev_perf = 0.0, rec_avg_time = 0.0,
                kblas_perf = 0.0, kblas_time = 0.0, kblas_time_1 = 0.0,
                ref_error = 0.0;

        int batchCount = opts.batchCount;
        if(opts.btest > 1)
          batchCount = opts.batch[btest];

        int batchCount_gpu = batchCount / ngpu;
        bool strided = 1;///*kblas_back_door[0] > 0 &&*/ batchCount_gpu > 1;//TODO make strided a user option

        N = opts.msize[itest];
        K = opts.nsize[itest];

        int depth = 0, s = 16;
        while(s < N){
          s = s << 1;
          depth++;
        }
        s = 1 << (depth-1);

        printf("%5d   %5d %5d   ",
              batchCount, (int) N, (int) K);
        fflush( stdout );

        if ( opts.transA == KBLAS_Trans ) {
          lda = Am = K;
          An = N;
        } else {
          lda = Am = N;
          An = K;
        }
        ldc = Cm = N;
        Cn = N;

        ldda = ((lda+31)/32)*32;
        lddc = ((ldc+31)/32)*32;

        sizeA = lda * An;
        sizeC = ldc * Cn;
        TESTING_MALLOC_PIN( h_A, T, lda * An * batchCount);
        TESTING_MALLOC_PIN( h_C, T, ldc * Cn * batchCount);

        for(int g = 0; g < ngpu; g++){
          check_error( cudaSetDevice( opts.devices[g] ));
          TESTING_MALLOC_DEV( d_A[g], T, ldda * An * batchCount_gpu);
          TESTING_MALLOC_DEV( d_C[g], T, lddc * Cn * batchCount_gpu);

          if(!strided){
            TESTING_MALLOC_DEV( d_A_array[g], T*, batchCount_gpu);
            TESTING_MALLOC_DEV( d_C_array[g], T*, batchCount_gpu);
          }
        }

        gflops = batchCount * FLOPS_SYRK<T>(K, N ) / 1e9;

        if(opts.check || opts.time){
          TESTING_MALLOC_CPU( h_R, T, ldc * Cn * batchCount);

          #ifdef DEBUG_DUMP
          outO = fopen("outO.csv", "a");
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
        Xrand_matrix(Cm, Cn * batchCount, h_C, ldc);
        for (int i=0; i < batchCount; i++){
          kblas_make_hpd( Cm, h_C + i * Cn * ldc, ldc );
        }
        if(opts.time)
          memcpy(h_R, h_C, sizeC * batchCount * sizeof(T));

        for(int g = 0; g < ngpu; g++){
          check_error( cudaSetDevice( opts.devices[g] ));
          check_error( cublasSetMatrixAsync( Am, An * batchCount_gpu, sizeof(T),
                                             h_A + Am * An * batchCount_gpu * g, lda,
                                             d_A[g], ldda, kblas_handle[g]->stream ) );
          if(!strided){
            check_error( Xset_pointer_2(d_A_array[g], d_A[g], ldda, An*ldda,
                                        d_C_array[g], d_C[g], lddc, Cn*lddc,
                                        batchCount_gpu, kblas_handle[g]->stream) );
          }
        }

        for(int g = 0; g < ngpu; g++){
          kblasWorkspace_t work_space = &(kblas_handle[g]->work_space);
          if(strided){
            kblasXsyrk_batch_strided_wsquery(N, batchCount_gpu, work_space);
          }else{
            kblasXsyrk_batch_wsquery(N, batchCount_gpu, work_space);
          }
          check_error( work_space->allocate() );
          check_error( cudaGetLastError() );
        }

        if(opts.warmup){
          for(int g = 0; g < ngpu; g++){
            check_error( cudaSetDevice( opts.devices[g] ));
            check_error( cublasSetMatrixAsync( Cm, Cn * batchCount_gpu, sizeof(T),
                                               h_C + Cm * Cn * batchCount_gpu * g, ldc,
                                               d_C[g], lddc, kblas_handle[g]->stream) );
          }

          for(int g = 0; g < ngpu; g++){
            check_error( cudaSetDevice( opts.devices[g] ));
            //check_error( cublasSetStream(cublas_handle, streams[g]) );
            if(strided){
              check_error( kblasXsyrk_batch_strided( kblas_handle[g],
                                                opts.uplo, opts.transA,
                                                N, K,
                                                alpha, d_A[g], ldda, An*ldda,
                                                beta,  d_C[g], lddc, Cn*lddc,
                                                batchCount_gpu) );
            }else{
              check_error( kblasXsyrk_batch( kblas_handle[g],
                                        opts.uplo, opts.transA,
                                        N, K,
                                        alpha, (const T**)(d_A_array[g]), ldda,
                                        beta,  d_C_array[g], lddc,
                                        batchCount_gpu));
            }
          }

          #ifdef USE_OPENMP
          if(opts.time){
            //memcpy(h_R, h_B, sizeB * batchCount * sizeof(T));
            omp_set_num_threads(NUM_THREADS);
            //omp_set_nested(true);
            #pragma omp parallel shared(h_R, K, lda)// num_threads (NUM_THREADS)
            {
              #pragma omp for //schedule(guided,10)
              for (int s=0; s < batchCount; s++)
              {
                LAPACK_SYRK( (( opts.uplo == KBLAS_Lower ) ? "Lower" : "Upper"),
                             (( opts.transA == KBLAS_NoTrans ) ? "No Transpose" : "Transpose"),
                             &N, &K, &alpha, h_A + s * lda * An, &lda, &beta, h_R + s * ldc * Cn, &ldc );
              }
            }
            memcpy(h_R, h_C, sizeC * batchCount * sizeof(T));
          }
          #endif//USE_OPENMP
        }
        double time = 0;

        #ifdef USE_MAGMA
          use_magma_gemm = 1; use_cublas_gemm = 0;
          for(int r = 0; r < nruns; r++){
            for(int g = 0; g < ngpu; g++){
              check_error( cudaSetDevice( opts.devices[g] ));
              check_error( cublasSetMatrixAsync( Cm, Cn * batchCount_gpu, sizeof(T),
                                                 h_C + Cm * Cn * batchCount_gpu * g, ldc,
                                                 d_C[g], lddc, kblas_handle[g]->stream ) );
            }


            #ifdef USE_NVPROF
            cudaProfilerStart();
            #endif//USE_NVPROF
            for(int g = 0; g < ngpu; g++){
              check_error( cudaSetDevice( opts.devices[g] ));
              cudaDeviceSynchronize();//TODO sync with streams instead
            }
            //start_timing(curStream);
            time = -gettime();
            for(int g = 0; g < ngpu; g++){
              check_error( cudaSetDevice( opts.devices[g] ));
              //check_error( cublasSetStream(cublas_handle, streams[g]) );
              if(strided){
                check_error( kblasXsyrk_batch_strided( kblas_handle[g],
                                                  opts.uplo, opts.transA,
                                                  N, K,
                                                  alpha, d_A[g], ldda, An*ldda,
                                                  beta,  d_C[g], lddc, Cn*lddc,
                                                  batchCount_gpu) );
              }else{
                check_error( kblasXsyrk_batch( kblas_handle[g],
                                          opts.uplo, opts.transA,
                                          N, K,
                                          alpha, (const T**)(d_A_array[g]), ldda,
                                          beta,  d_C_array[g], lddc,
                                          batchCount_gpu));
              }
            }
            for(int g = 0; g < ngpu; g++){
              check_error( cudaSetDevice( opts.devices[g] ));
              cudaDeviceSynchronize();//TODO sync with streams instead
            }
            //time = get_elapsed_time(curStream);
            time += gettime();
            kblas_time += time;
            #ifdef USE_NVPROF
            cudaProfilerStop();
            #endif//USE_NVPROF
          }
          kblas_time /= nruns;
          kblas_perf = gflops / kblas_time;
          kblas_time *= 1000.0;
        #endif

        #if 1
        use_magma_gemm = 0; use_cublas_gemm = 1;
        for(int r = 0;  r < nruns; r++){
          for(int g = 0; g < ngpu; g++){
            check_error( cudaSetDevice( opts.devices[g] ));
            check_error( cublasSetMatrixAsync( Cm, Cn * batchCount_gpu, sizeof(T),
                                               h_C + Cm * Cn * batchCount_gpu * g, ldc,
                                               d_C[g], lddc, kblas_handle[g]->stream ) );
          }

          #ifdef USE_NVPROF
          cudaProfilerStart();
          #endif//USE_NVPROF
          for(int g = 0; g < ngpu; g++){
            check_error( cudaSetDevice( opts.devices[g] ));
            cudaDeviceSynchronize();//TODO sync with streams instead
          }
          //start_timing(curStream);
          time = -gettime();
          for(int g = 0; g < ngpu; g++){
            check_error( cudaSetDevice( opts.devices[g] ));
            //check_error( cublasSetStream(cublas_handle, streams[g]) );
            if(strided){
              check_error( kblasXsyrk_batch_strided( kblas_handle[g],
                                                opts.uplo, opts.transA,
                                                N, K,
                                                alpha, d_A[g], ldda, An*ldda,
                                                beta,  d_C[g], lddc, Cn*lddc,
                                                batchCount_gpu) );
            }else{
              check_error( kblasXsyrk_batch( kblas_handle[g],
                                        opts.uplo, opts.transA,
                                        N, K,
                                        alpha, (const T**)(d_A_array[g]), ldda,
                                        beta,  d_C_array[g], lddc,
                                        batchCount_gpu));
            }
          }
          for(int g = 0; g < ngpu; g++){
            check_error( cudaSetDevice( opts.devices[g] ));
            cudaDeviceSynchronize();//TODO sync with streams instead
          }
          //time = get_elapsed_time(curStream);
          time += gettime();
          kblas_time_1 += time;
          #ifdef USE_NVPROF
          cudaProfilerStop();
          #endif//USE_NVPROF
        }
        kblas_time_1 /= nruns;
        kblas_perf = gflops / kblas_time_1;
        kblas_time_1 *= 1000.0;
        #endif

        #ifdef USE_MKL
        if(opts.check || (opts.time && opts.lapack)){
          if(opts.check){
            for(int g = 0; g < ngpu; g++){
              check_error( cudaSetDevice( opts.devices[g] ));
              check_error( cublasGetMatrixAsync( Cm, Cn * batchCount_gpu, sizeof(T),
                                                 d_C[g], lddc,
                                                 h_R + Cm * Cn * batchCount_gpu * g, ldc,
                                                 kblas_handle[g]->stream ) );
            }
            for(int g = 0; g < ngpu; g++){
              check_error(cudaSetDevice( opts.devices[g] ));
              cudaDeviceSynchronize();
            }
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
            #pragma omp parallel shared(h_A, K, lda)// num_threads (NUM_THREADS)
            {
            #pragma omp for //schedule(guided,10)
            #endif//USE_OPENMP
            //batchCount = 1;
            //printf("testing dsyrk with MKL.\n");
            for (int s=0; s < batchCount; s++)
            {
              //if(opts.check && !opts.time)
              //  printMatrix(Bm, Bn, h_B + s * ldb * Bn, ldb, outO);

              LAPACK_SYRK( (( opts.uplo == KBLAS_Lower ) ? "Lower" : "Upper"),
                          (( opts.transA == KBLAS_NoTrans ) ? "No Transpose" : "Transpose"),
                          &N, &K, &alpha, h_A + s * lda * An, &lda, &beta, h_C + s * ldc * Cn, &ldc );

              if(opts.check && !opts.time){
                // compute relative error for kblas, relative to lapack,
                // |kblas - lapack| / |lapack|
                double kblas_error = 0.;
                // LAPACK_AXPY( &sizeC, &c_neg_one, h_C + s * ldc * Cn, &ione, h_R + s * ldc * Cn, &ione );
                // double Cnorm = LAPACK_LANSY( "fro",
                //                             (( opts.uplo == KBLAS_Lower ) ? "Lower" : "Upper"),
                //                             &Cn, h_C + s * ldc * Cn, &ldc, work );
                // double err = LAPACK_LANSY( "fro",
                //                           (( opts.uplo == KBLAS_Lower ) ? "Lower" : "Upper"),
                //                           &Cn, h_R + s * ldc * Cn, &ldc, work )
                //               / Cnorm;
                // /*if ( isnan(err) || isinf(err) ) {
                //   ref_error = err;
                //   break;
                // }*/
                // ref_error = fmax( err, ref_error );
                ref_error += Xget_max_error_matrix(h_C + s * ldc * Cn, h_R + s * ldc * Cn, Cm, Cn, ldc, opts.uplo);
                #ifdef DEBUG_DUMP
                printMatrix(Cm, Cn, h_C + s * ldc * Cn, ldc, outL);
                printMatrix(Cm, Cn, h_R + s * ldc * Cn, ldc, outK);
                #endif
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

          /*if(opts.check){
            ref_error = Xget_max_error_matrix(h_A, h_R, K, K, lda);
            //Cnorm = kblas_lange<T,double>( 'N', K, K, h_A, lda);
            //LAPACK_AXPY( &sizeA, &c_neg_one, h_A, &ione, h_R, &ione );
            //ref_error = kblas_lange<T,double>( 'N', K, K, h_R, lda) / Cnorm;
          }*/
          if(opts.time){
            ref_avg_time = (ref_avg_time / nruns) * 1000.;//convert to ms
            //rec_avg_time = (rec_avg_time / nruns) * 1000.;//convert to ms
          }
        }
        #endif//USE_MKL

        cudaFreeHost( h_A );
        cudaFreeHost( h_C );

        if(opts.check || opts.time)
          free( h_R );
        for(int g = 0; g < ngpu; g++){
          check_error( cudaSetDevice( opts.devices[g] ));
          check_error( cudaFree( d_A[g] ) );
          check_error( cudaFree( d_C[g] ) );
          if(!strided){
            check_error(  cudaFree( d_A_array[g] ) );
            check_error(  cudaFree( d_C_array[g] ) );
          }
        }

        if(opts.time){
          ref_sdev_perf = sqrt((ref_sdev_perf - (ref_avg_perf * ref_avg_perf / nruns))/nruns);
          //rec_sdev_perf = sqrt((rec_sdev_perf - (rec_avg_perf * rec_avg_perf / nruns))/nruns);
        }

        //printf(" %7.4f %7.4f       %7.4f %7.4f %7.4f %7.4f    %7.4f %7.4f %7.4f %7.4f    %.4e \n",
        printf(" %7.4f %7.4f %7.4f       %7.4f %7.4f %7.4f %7.4f    %.4e \n",
               kblas_perf, kblas_time,  kblas_time_1,
               ref_avg_perf / nruns, ref_avg_time, ref_sdev_perf, ref_avg_time / kblas_time_1,
               //rec_avg_perf / nruns, rec_avg_time, rec_sdev_perf, rec_avg_time / kblas_time,
               ref_error);
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
    kblasDestroy(&kblas_handle[g]);
  }
}

//==============================================================================================
int main(int argc, char** argv)
{

  kblas_opts opts;
  if(!parse_opts( argc, argv, &opts )){
    USAGE;
    return -1;
  }

#if defined PREC_d
  cudaError_t error = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  if(cudaSuccess != error)
  {
    printf("cudaDeviceSetSharedMemConfig returned error: %s\n", cudaGetErrorString(error));
    exit(1);
  }
#endif

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
  test_Xsyrk_batch<TYPE>(opts, alpha, beta);

}