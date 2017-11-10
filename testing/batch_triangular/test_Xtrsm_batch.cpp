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

#include "batch_triangular/Xhelper_funcs.ch" // TODO: need Xset_pointer_2 from this
#include "operators.h" // TODO: this has templates and C++ host/device functions (make_one and make_zero)


//==============================================================================================
// #define DEBUG_DUMP
//==============================================================================================

#ifdef USING
#undef USING
#endif

#define USING printf("uplo %c, trans %c, batchCount %d, backDoor %d\n", opts.uplo, opts.transA, batchCount, opts.bd);

template<class T>
int test_Xtrsm_batch(kblas_opts& opts, T alpha)
{

  bool strided = opts.strided;
  int nruns = opts.nruns, ngpu = opts.ngpu;
  int M, N;
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
                kblas_perf = 0.0, kblas_time = 0.0, kblas_time_1 = 0.0, cublas_perf = 0.0, cublas_time = 0.0,
                ref_error = 0.0;

        int batchCount = opts.batchCount;
        if(opts.btest > 1)
          batchCount = opts.batch[btest];

        int batchCount_gpu = batchCount / ngpu;

        M = opts.msize[itest];
        N = opts.nsize[itest];

        printf("%5d   %5d %5d   ",
              batchCount, (int) M, (int) N);
        fflush( stdout );

        if ( opts.side == KBLAS_Left ) {
          lda = Am = M;
          An = M;
        } else {
          lda = Am = N;
          An = N;
        }
        ldc = Cm = M;
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

        gflops = batchCount * FLOPS_TRSM<T>(opts.side, M, N ) / 1e9;

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
          Xmatrix_make_hpd( Am, h_A + i * An * lda, lda );
        }
        if(opts.time)
          memcpy(h_R, h_C, sizeC * batchCount * sizeof(T));

        for(int g = 0; g < ngpu; g++){
          check_error( cudaSetDevice( opts.devices[g] ));
          check_cublas_error( cublasSetMatrixAsync( Am, An * batchCount_gpu, sizeof(T),
                                             h_A + Am * An * batchCount_gpu * g, lda,
                                             d_A[g], ldda, kblasGetStream(kblas_handle[g]) ) );
          if(!strided){
            check_kblas_error( Xset_pointer_2(d_A_array[g], d_A[g], ldda, An*ldda,
                                        d_C_array[g], d_C[g], lddc, Cn*lddc,
                                        batchCount_gpu, kblasGetStream(kblas_handle[g])) );
          }
        }

        for(int g = 0; g < ngpu; g++){
          if(strided){
            kblas_trsm_batch_strided_wsquery(kblas_handle[g], batchCount_gpu, opts.side, M, N);
          }else{
            kblas_trsm_batch_wsquery(kblas_handle[g], batchCount_gpu, opts.side, M, N);
          }
          check_kblas_error( kblasAllocateWorkspace(kblas_handle[g]) );
          check_error( cudaGetLastError() );
        }

        if(opts.warmup){
          for(int g = 0; g < ngpu; g++){
            check_error( cudaSetDevice( opts.devices[g] ));
            check_cublas_error( cublasSetMatrixAsync( Cm, Cn * batchCount_gpu, sizeof(T),
                                               h_C + Cm * Cn * batchCount_gpu * g, ldc,
                                               d_C[g], lddc, kblasGetStream(kblas_handle[g])) );
          }

          for(int g = 0; g < ngpu; g++){
            check_error( cudaSetDevice( opts.devices[g] ));
            //check_error( cublasSetStream(cublas_handle, streams[g]) );
            if(strided){
              check_kblas_error( kblas_trsm_batch(kblas_handle[g],
                                            opts.side, opts.uplo, opts.transA, opts.diag,
                                            M, N,
                                            alpha, d_A[g], ldda, An*ldda,
                                                   d_C[g], lddc, Cn*lddc,
                                            batchCount_gpu) );
            }else{
              check_kblas_error( kblas_trsm_batch(kblas_handle[g],
                                            opts.side, opts.uplo, opts.transA, opts.diag,
                                            M, N,
                                            alpha, (const T**)(d_A_array[g]), ldda,
                                                               d_C_array[g], lddc,
                                            batchCount_gpu));
            }
          }

          #if (defined USE_OPENMP) && (defined USE_MKL)
          if(opts.time){
            //memcpy(h_R, h_B, sizeB * batchCount * sizeof(T));
            omp_set_num_threads(NUM_THREADS);
            //omp_set_nested(true);
            #pragma omp parallel shared(h_R, K, lda)// num_threads (NUM_THREADS)
            {
              #pragma omp for //schedule(guided,10)
              for (int s=0; s < batchCount; s++)
              {
                LAPACK_TRSM( (( opts.side == KBLAS_Right ) ? "Right" : "Left"),
                             (( opts.uplo == KBLAS_Lower ) ? "Lower" : "Upper"),
                             (( opts.transA == KBLAS_NoTrans ) ? "No Transpose" : "Transpose"),
                             "Non-unit",
                             &Cm, &Cn, &alpha, h_A + s * lda * An, &lda, h_R + s * ldc * Cn, &ldc );
              }
            }
            memcpy(h_R, h_C, sizeC * batchCount * sizeof(T));
          }
          #endif//USE_OPENMP
        }
        double time = 0;

        #ifdef USE_MAGMA
          //TODO this is not a safe access
          // kblas_handle->use_magma = 1;
          for(int r = 0; r < nruns; r++){
            for(int g = 0; g < ngpu; g++){
              check_error( cudaSetDevice( opts.devices[g] ));
              check_cublas_error( cublasSetMatrixAsync( Cm, Cn * batchCount_gpu, sizeof(T),
                                                 h_C + Cm * Cn * batchCount_gpu * g, ldc,
                                                 d_C[g], lddc, kblasGetStream(kblas_handle[g]) ) );
            }


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
                check_kblas_error( kblasXtrsm_batch_strided(kblas_handle[g],
                                                      opts.side, opts.uplo, opts.transA, opts.diag,
                                                      M, N,
                                                      alpha, d_A[g], ldda, An*ldda,
                                                             d_C[g], lddc, Cn*lddc,
                                                      batchCount_gpu) );
              }else{
                check_kblas_error( kblasXtrsm_batch(kblas_handle[g],
                                              opts.side, opts.uplo, opts.transA, opts.diag,
                                              M, N,
                                              alpha, (const T**)(d_A_array[g]), ldda,
                                                                 d_C_array[g], lddc,
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
          }
          kblas_time /= nruns;
          kblas_perf = gflops / kblas_time;
          kblas_time *= 1000.0;
        #endif

        for(int r = 0;  r < nruns; r++){
          for(int g = 0; g < ngpu; g++){
            // kblas_handle[g]->use_magma = 0; // TODO: Off by default
            check_error( cudaSetDevice( opts.devices[g] ));
            check_cublas_error( cublasSetMatrixAsync( Cm, Cn * batchCount_gpu, sizeof(T),
                                               h_C + Cm * Cn * batchCount_gpu * g, ldc,
                                               d_C[g], lddc, kblasGetStream(kblas_handle[g]) ) );
          }

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
                check_kblas_error( kblasXtrsm_batch_strided(kblas_handle[g],
                                                      opts.side, opts.uplo, opts.transA, opts.diag,
                                                      M, N,
                                                      alpha, d_A[g], ldda, An*ldda,
                                                             d_C[g], lddc, Cn*lddc,
                                                      batchCount_gpu) );
              }else{
                check_kblas_error( kblasXtrsm_batch(kblas_handle[g],
                                              opts.side, opts.uplo, opts.transA, opts.diag,
                                              M, N,
                                              alpha, (const T**)(d_A_array[g]), ldda,
                                                                 d_C_array[g], lddc,
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
        }
        kblas_time_1 /= nruns;
        kblas_perf = gflops / kblas_time_1;
        kblas_time_1 *= 1000.0;


        if(opts.time){
          for(int g = 0; g < ngpu; g++){
            check_error( cudaSetDevice( opts.devices[g] ));
            Xset_pointer_2( d_A_array[g], d_A[g], ldda, ldda*An,
                            d_C_array[g], d_C[g], lddc, lddc*Cn,
                            batchCount_gpu, kblasGetStream(kblas_handle[g]));
          }
          for(int r = 0; r < nruns; r++)
          {
            for(int g = 0; g < ngpu; g++){
              check_error( cudaSetDevice( opts.devices[g] ));
              check_cublas_error( cublasSetMatrixAsync( Cm, Cn * batchCount_gpu, sizeof(T),
                                                 h_C + Cm * Cn * batchCount_gpu * g, ldc,
                                                 d_C[g], lddc, kblasGetStream(kblas_handle[g]) ) );
            }
            for(int g = 0; g < ngpu; g++){
              check_error( cudaSetDevice( opts.devices[g] ));
              cudaDeviceSynchronize();//TODO sync with streams instead
            }
            time = -gettime();
            for(int g = 0; g < ngpu; g++){
              check_error( cudaSetDevice( opts.devices[g] ));
              cublasXtrsm_batched(kblasGetCublasHandle(kblas_handle[g]),
                                  (cublasSideMode_t)(CUBLAS_SIDE_LEFT + (opts.side == KBLAS_Right)),
                                  (cublasFillMode_t)(CUBLAS_FILL_MODE_LOWER + (opts.uplo == KBLAS_Upper)),
                                  (cublasOperation_t)(CUBLAS_OP_N + (opts.transA == KBLAS_Trans)),
                                  (cublasDiagType_t)(CUBLAS_DIAG_NON_UNIT + (opts.diag == KBLAS_Unit)),
                                  M, N, &alpha,
                                  (const T**)d_A_array[g], ldda,
                                             d_C_array[g], lddc,
                                  batchCount_gpu);
            }
            for(int g = 0; g < ngpu; g++){
              check_error( cudaSetDevice( opts.devices[g] ));
              cudaDeviceSynchronize();//TODO sync with streams instead
            }
            time += gettime();
            cublas_time += time;
          }
          cublas_time /= nruns;
          cublas_perf = gflops / cublas_time;
          cublas_time *= 1000.0;
        }


        #ifdef USE_MKL
        if(opts.check || (opts.time && opts.lapack)){
          if(opts.check){
            for(int g = 0; g < ngpu; g++){
              check_error( cudaSetDevice( opts.devices[g] ));
              check_cublas_error( cublasGetMatrixAsync( Cm, Cn * batchCount_gpu, sizeof(T),
                                                 d_C[g], lddc,
                                                 h_R + Cm * Cn * batchCount_gpu * g, ldc,
                                                 kblasGetStream(kblas_handle[g]) ) );
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
            //printf("testing dtrsm with MKL.\n");
            for (int s=0; s < batchCount; s++)
            {
              //if(opts.check && !opts.time)
              //  printMatrix(Bm, Bn, h_B + s * ldb * Bn, ldb, outO);

              LAPACK_TRSM( (( opts.side == KBLAS_Right ) ? "Right" : "Left"),
                           (( opts.uplo == KBLAS_Lower ) ? "Lower" : "Upper"),
                           (( opts.transA == KBLAS_NoTrans ) ? "No Transpose" : "Transpose"),
                           "Non-unit",
                           &Cm, &Cn, &alpha, h_A + s * lda * An, &lda, h_C + s * ldc * Cn, &ldc );

              if(opts.check && !opts.time){
                // compute relative error for kblas, relative to lapack,
                // |kblas - lapack| / |lapack|
                LAPACK_AXPY( &sizeC, &c_neg_one, h_C + s * ldc * Cn, &ione, h_R + s * ldc * Cn, &ione );
                double Cnorm = LAPACK_LANGE( "f", &Cm, &Cn, h_C + s * ldc * Cn, &ldc, work );
                double err   = LAPACK_LANGE( "f", &Cm, &Cn, h_R + s * ldc * Cn, &ldc, work )
                              / Cnorm;
                /*if ( isnan(err) || isinf(err) ) {
                  ref_error = err;
                  break;
                }*/
                ref_error = fmax( err, ref_error );
                // ref_error += Xget_max_error_matrix(h_C + s * ldc * Cn, h_R + s * ldc * Cn, Cm, Cn, ldc);
                #ifdef DEBUG_DUMP
                printMatrix(Cm, Cn, h_C + s * ldc * Cn, ldc, outL);
                printMatrix(Cm, Cn, h_R + s * ldc * Cn, ldc, outK);
                #endif
              }
            }
            #ifdef USE_OPENMP
            }
            #endif//USE_OPENMP
            // if(opts.check) ref_error /= batchCount;
            if(opts.time){
              time += gettime();
              perf = gflops / time;
              ref_avg_perf += perf;
              ref_sdev_perf += perf * perf;
              ref_avg_time += time;
            }
          }

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
        printf(" %7.4f %7.4f %7.4f %7.4f %7.4f       %7.4f %7.4f %7.4f %7.4f    %.4e \n",
               kblas_perf, kblas_time,  kblas_time_1,
               cublas_perf, cublas_time,
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
  parse_opts( argc, argv, &opts);
  
#if defined PREC_d
  check_error( cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) );
#endif

#if (defined PREC_s) || (defined PREC_d)
  TYPE alpha = 0.28;
#elif defined PREC_c
  TYPE alpha = make_cuFloatComplex(1.2, -0.6);
#elif defined PREC_z
  TYPE alpha = make_cuDoubleComplex(1.2, -0.6);
#endif
  test_Xtrsm_batch<TYPE>(opts, alpha);

}
