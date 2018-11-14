/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_tlr/test_Xgemm_lr.cpp

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

#if ((defined PREC_c) || (defined PREC_z)) && (defined USE_MKL)
//TODO need to handle MKL types properly
#undef USE_MKL
#endif


#include "testing_helper.h"
#include "testing_prec_def.h"
#include "flops.h"

#ifdef check_error
#undef check_error
#endif

#include "Xhelper_funcs.ch"
#include "operators.h"
#include "kblas_common.h"

#include "Xblas_core.ch"

#ifdef USE_OPENMP
#include "omp.h"
#endif//USE_OPENMP




template<class T>
int test_Xgemm_lr(kblas_opts& opts, T alpha, T beta){

  kblasHandle_t kblas_handle;
  GPU_Timer_t kblas_timer;
  int nruns = opts.nruns;
  int TLR_LLL = opts.LR == 't';
  int batchCount = opts.batchCount;
  bool strided = opts.strided;
  double tolerance = opts.tolerance;

  int M, N, K, max_M, max_N, max_K;
  int Am, An,
      Bm, Bn,
      Cm, Cn,
      rA, rB, rC, res_rC,
      max_rank = 0, max_rA, max_rB;
  int sizeA, sizeB, sizeC;
  int strideA, strideB, strideC,
      strideAu, strideAv,
      strideBu, strideBv,
      strideCu, strideCv;
  int lda, ldb, ldc,
      ldda, lddb, lddc,
      ldAu, ldAv,
      ldBu, ldBv,
      ldCu, ldCv,
      lwork, info;

  int ISEED[4] = {0,0,0,1};

  T *h_Au = NULL, *h_Av = NULL,
    *h_Bu = NULL, *h_Bv = NULL,
    *h_Cu = NULL, *h_Cv = NULL,
    *h_Rc = NULL, *h_Rk = NULL,
    *h_Xu = NULL, *h_Xv = NULL,
    *h_Xs = NULL,
    *h_A = NULL, *h_B = NULL, *h_C = NULL,
    *d_A = NULL, *d_B = NULL, *d_C = NULL,
    *d_Au = NULL, *d_Av = NULL,
    *d_Bu = NULL, *d_Bv = NULL,
    *d_Cu = NULL, *d_Cv = NULL;
  T **d_Au_array = NULL, **d_Av_array = NULL,
    **d_Bu_array = NULL, **d_Bv_array = NULL,
    **d_Cu_array = NULL, **d_Cv_array = NULL,
    **d_A_array = NULL, **d_B_array = NULL,
    **d_C_array = NULL;
  T one = make_one<T>(),
    mone = -one,
    zero = make_zero<T>();
  T work_s[1], *work = NULL;
  int *h_rA, *h_rB,
      *d_rA, *d_rB;
  int *h_M = NULL, *h_N = NULL, *h_K = NULL,
      *d_M = NULL, *d_N = NULL, *d_K = NULL;
  int *d_ldAu, *d_ldAv, *d_ldBu, *d_ldBv, *d_lda, *d_ldb, *d_ldc;
  bool  use_magma = opts.magma;
  bool isTransA = (opts.transA == KBLAS_Trans),
       isTransB = (opts.transB == KBLAS_Trans);

  cudaError_t err;

  err = cudaSetDevice( opts.devices[0] );
  kblasCreate(&kblas_handle);
  #ifdef USE_MAGMA
    if(use_magma == 1){
      magma_init();
      kblasEnableMagma(kblas_handle);
    }
  #endif
  kblasCreateStreams(kblas_handle, 2);
  kblas_timer = newGPU_Timer(kblasGetStream(kblas_handle));

  cublasOperation_t cub_transA = (opts.transA == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);
  cublasOperation_t cub_transB = (opts.transB == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);


  #ifdef USE_OPENMP
  int NUM_THREADS = opts.omp_numthreads;
  #endif//USE_OPENMP

  printf("batch         M     N    K   rankA   rankB   rankC    kblasGEMM-GF/s (ms)    cublasGEMM-GF/s (ms)      Speedup");
  if(opts.check)
    printf("      Error");
  printf("\n");
  printf("======================================================================================================================\n");
  for( int itest = 0; itest < opts.ntest; ++itest ) {
    for( int iter = 0; iter < opts.niter; ++iter ) {
      for( int btest = 0; btest < opts.btest; ++btest ) {

      double  gflops, perf, gflops_lr,
              kblas_perf = 0.0, kblas_time = 0.0,
              cublas_perf = 0.0, cublas_time = 0.0,
              ref_error = 0.0;

      if(opts.btest > 1)
        batchCount = opts.batch[btest];

      M = opts.msize[itest];
      N = opts.nsize[itest];
      K = opts.ksize[itest];
      rA = opts.rank[0];
      rB = opts.rank[0];
      res_rC = rC = opts.rank[0];
      max_rank = opts.rank[1];
      int max_rC = kmax(rC, max_rank);

      if(rA <= 0 || rA > (M < K ? M : K) || rB <= 0 || rB > (N < K ? N : K)){
        printf("Input rank(%d) is not compatible with input matrix dimensions, please specify a feasible rank (range: %d-%d). Aborting... \n", rA, 1, (M < K ? M : K));
        exit(0);
      }

      gflops = batchCount * FLOPS_GEMM<T>(M, N, K ) / 1e9;
      if(TLR_LLL)
        gflops_lr = batchCount * FLOPS_GEMM_LR_LLL<T>(M, N, K, rA, rB, rC) / 1e9;
      else
        gflops_lr = batchCount * FLOPS_GEMM_LR_LLD<T>(M, N, K, rA, rB) / 1e9;

      printf("%8d %5d %5d %5d %5d %5d   ",
             batchCount, (int) M, (int) N, (int) K, (int) rA, (int) rB);
      if(TLR_LLL){
        if(!(max_rank > 0)){
          printf("Please specify max_rank for TLR format with --rank X:X option, aborting\n");
          exit(0);
        }
      }
      printf("%5d   ",
             (int) rC);
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

      ldda = kblas_roundup(lda, 32);
      lddb = kblas_roundup(ldb, 32);
      lddc = kblas_roundup(ldc, 32);
      ldAu = ldda;
      ldAv = kblas_roundup( An, 32);
      ldBu = lddb;
      ldBv = kblas_roundup( Bn, 32);
      if(TLR_LLL){
        ldCu = lddc;
        ldCv = kblas_roundup( Cn, 32);
      }

      strideA  = ldda * An;
      strideB  = lddb * Bn;
      strideC  = lddc * Cn;
      strideAu = ldAu * rA;
      strideAv = ldAv * rA;
      strideBu = ldBu * rB;
      strideBv = ldBv * rB;
      if(TLR_LLL){
        strideCu = ldCu * max_rC;
        strideCv = ldCv * max_rC;
      }

      if(TLR_LLL){
        TESTING_MALLOC_CPU( h_A, T, lda * An * batchCount);
        TESTING_MALLOC_CPU( h_B, T, ldb * Bn * batchCount);
      }
      TESTING_MALLOC_CPU( h_C, T, ldc * Cn * batchCount);

      TESTING_MALLOC_CPU( h_Au, T, lda * rA * batchCount);
      TESTING_MALLOC_CPU( h_Av, T, An  * rA * batchCount);
      TESTING_MALLOC_CPU( h_Bu, T, ldb * rB * batchCount);
      TESTING_MALLOC_CPU( h_Bv, T, Bn  * rB * batchCount);
      if(TLR_LLL){
        TESTING_MALLOC_CPU( h_Cu, T, ldc * max_rC * batchCount);
        TESTING_MALLOC_CPU( h_Cv, T, Cn  * max_rC * batchCount);
      }
      TESTING_MALLOC_DEV( d_C, T, strideC * batchCount);

      TESTING_MALLOC_DEV( d_Au, T, strideAu * batchCount);
      TESTING_MALLOC_DEV( d_Av, T, strideAv * batchCount);
      TESTING_MALLOC_DEV( d_Bu, T, strideBu * batchCount);
      TESTING_MALLOC_DEV( d_Bv, T, strideBv * batchCount);
      if(TLR_LLL){
        TESTING_MALLOC_DEV( d_Cu, T, strideCu * batchCount);
        TESTING_MALLOC_DEV( d_Cv, T, strideCv * batchCount);
      }
      if(!strided && batchCount > 1){
        TESTING_MALLOC_DEV( d_Au_array, T*, batchCount);
        TESTING_MALLOC_DEV( d_Av_array, T*, batchCount);
        TESTING_MALLOC_DEV( d_Bu_array, T*, batchCount);
        TESTING_MALLOC_DEV( d_Bv_array, T*, batchCount);
        if(TLR_LLL){
          TESTING_MALLOC_DEV( d_Cu_array, T*, batchCount);
          TESTING_MALLOC_DEV( d_Cv_array, T*, batchCount);
        }
        TESTING_MALLOC_DEV( d_C_array, T*, batchCount);
      }

      if(opts.check || opts.time)
      {
        TESTING_MALLOC_DEV( d_A, T, strideA * batchCount);
        TESTING_MALLOC_DEV( d_B, T, strideB * batchCount);
        TESTING_MALLOC_DEV( d_A_array, T*, batchCount);
        TESTING_MALLOC_DEV( d_B_array, T*, batchCount);
        TESTING_MALLOC_CPU( h_Rc, T, ldc * Cn * batchCount);
        TESTING_MALLOC_CPU( h_Rk, T, ldc * Cn * batchCount);

        if(opts.check)
        {
          opts.time = 0;
          opts.warmup = 0;
          nruns = 1;
        }
      }

      if(TLR_LLL){
        Xrand_matrix(Cm, max_rC * batchCount, h_Cu, ldc);
        Xrand_matrix(Cn, max_rC * batchCount, h_Cv, Cn);//*/
      }
      {
        Xrand_matrix(Am, rA * batchCount, h_Au, lda);
        Xrand_matrix(An, rA * batchCount, h_Av, An);
        Xrand_matrix(Bm, rB * batchCount, h_Bu, ldb);
        Xrand_matrix(Bn, rB * batchCount, h_Bv, Bn);
        Xrand_matrix(Cm, Cn * batchCount, h_C, ldc);
      }

      cudaStream_t curStream = kblasGetStream(kblas_handle);
      cublasHandle_t cublas_handle = kblasGetCublasHandle(kblas_handle);

      check_cublas_error( cublasSetMatrixAsync( Am, rA * batchCount, sizeof(T), h_Au, lda, d_Au, ldAu, curStream ) );
      check_cublas_error( cublasSetMatrixAsync( An, rA * batchCount, sizeof(T), h_Av, An, d_Av, ldAv, curStream ) );
      check_cublas_error( cublasSetMatrixAsync( Bm, rB * batchCount, sizeof(T), h_Bu, ldb, d_Bu, ldBu, curStream ) );
      check_cublas_error( cublasSetMatrixAsync( Bn, rB * batchCount, sizeof(T), h_Bv, Bn, d_Bv, ldBv, curStream ) );

      if(!strided && batchCount > 1){
        check_kblas_error( Xset_pointer_5(d_Au_array, d_Au, ldAu, strideAu,
                                          d_Av_array, d_Av, ldAv, strideAv,
                                          d_Bu_array, d_Bu, ldBu, strideBu,
                                          d_Bv_array, d_Bv, ldBv, strideBv,
                                          d_C_array, d_C, lddc, strideC,
                                          batchCount, curStream) );
        if(TLR_LLL){
          check_kblas_error( Xset_pointer_2(d_Cu_array, d_Cu, ldCu, strideCu,
                                            d_Cv_array, d_Cv, ldCv, strideCv,
                                            batchCount, curStream) );
        }
      }
      if(batchCount == 1){
        if(TLR_LLL)
          kblasXgemm_lr_lll_wsquery(kblas_handle, M, N, rA, rB, rC, max_rank);
        else
          kblasXgemm_lr_lld_wsquery(kblas_handle, N, rA, rB);
      }else
      if(strided){
        if(TLR_LLL)
          kblasXgemm_lr_lll_batch_strided_wsquery(kblas_handle, M, N, rA, rB, rC, max_rank, batchCount);
        else
          kblasXgemm_lr_lld_batch_strided_wsquery(kblas_handle, N, rA, rB, batchCount);
      }else{
        if(TLR_LLL)
          kblasXgemm_lr_lll_batch_wsquery(kblas_handle, M, N, rA, rB, rC, max_rank, batchCount);
        else
          kblasXgemm_lr_lld_batch_wsquery(kblas_handle, N, rA, rB, batchCount);
      }

      check_kblas_error( kblasAllocateWorkspace(kblas_handle) );
      check_error( cudaGetLastError() );

      double time = 0;

      for(int r = 0; r < (nruns+opts.warmup); r++)
      {
        if(TLR_LLL){
          check_cublas_error( cublasSetMatrixAsync( Cm, max_rC * batchCount, sizeof(T), h_Cu, ldc, d_Cu, ldCu, curStream ) );
          check_cublas_error( cublasSetMatrixAsync( Cn, max_rC * batchCount, sizeof(T), h_Cv, Cn, d_Cv, ldCv, curStream ) );
        }
        else
          check_cublas_error( cublasSetMatrixAsync( Cm, Cn * batchCount, sizeof(T), h_C, ldc, d_C, lddc, curStream ) );

        res_rC = rC;

        gpuTimerTic(kblas_timer);
        if(batchCount == 1){
          if(TLR_LLL){
            check_kblas_error( kblas_gemm_lr(kblas_handle,
                                              opts.transA, opts.transB,
                                              M, N, K,
                                              alpha, d_Au, ldAu, d_Av, ldAv, rA,
                                                     d_Bu, ldBu, d_Bv, ldBv, rB,
                                              beta,  d_Cu, ldCu, d_Cv, ldCv, res_rC,
                                              max_rank, tolerance) );
          }else{
            check_kblas_error( kblas_gemm_lr(kblas_handle,
                                              opts.transA, opts.transB,
                                              M, N, K,
                                              alpha, d_Au, ldAu, d_Av, ldAv, rA,
                                                     d_Bu, ldBu, d_Bv, ldBv, rB,
                                              beta,  d_C, lddc) );
          }
          check_error( cudaGetLastError() );
        }else
        if(strided){
          if(TLR_LLL){
            check_kblas_error( kblas_gemm_lr_batch(kblas_handle,
                                                    opts.transA, opts.transB,
                                                    M, N, K,
                                                    alpha,
                                                    d_Au, ldAu, strideAu,
                                                    d_Av, ldAv, strideAv, rA,
                                                    d_Bu, ldBu, strideBu,
                                                    d_Bv, ldBv, strideBv, rB,
                                                    beta, d_Cu, ldCu, strideCu,
                                                          d_Cv, ldCv, strideCv, res_rC,
                                                    max_rank, tolerance,
                                                    batchCount) );
          }else{
            check_kblas_error( kblas_gemm_lr_batch(kblas_handle,
                                                    opts.transA, opts.transB,
                                                    M, N, K,
                                                    alpha,
                                                    d_Au, ldAu, strideAu,
                                                    d_Av, ldAv, strideAv, rA,
                                                    d_Bu, ldBu, strideBu,
                                                    d_Bv, ldBv, strideBv, rB,
                                                    beta,
                                                    d_C, lddc, strideC,
                                                    batchCount) );
          }
        }else{
          if(TLR_LLL){
              check_kblas_error( kblas_gemm_lr_batch(kblas_handle,
                                                    opts.transA, opts.transB,
                                                    M, N, K,
                                                    alpha,
                                                    (const T**)d_Au_array, ldAu, strideAu,
                                                    (const T**)d_Av_array, ldAv, strideAv, rA,
                                                    (const T**)d_Bu_array, ldBu, strideBu,
                                                    (const T**)d_Bv_array, ldBv, strideBv, rB,
                                                    beta, (T**)d_Cu_array, ldCu, strideCu,
                                                          (T**)d_Cv_array, ldCv, strideCv, res_rC,
                                                    max_rank, tolerance,
                                                    batchCount) );
          }else{
            check_kblas_error( kblas_gemm_lr_batch(kblas_handle,
                                                    opts.transA, opts.transB,
                                                    M, N, K,
                                                    alpha,
                                                    (const T**)d_Au_array, ldAu,
                                                    (const T**)d_Av_array, ldAv, rA,
                                                    (const T**)d_Bu_array, ldBu,
                                                    (const T**)d_Bv_array, ldBv, rB,
                                                    beta, d_C_array, lddc,
                                                    batchCount) );
          }
        }

        gpuTimerRecordEnd(kblas_timer);
        time = gpuTimerToc(kblas_timer);
        check_error( cudaGetLastError() );
        if(!opts.warmup || r > 0)
          kblas_time += time;
      }
      kblas_time /= nruns;
      kblas_perf = gflops_lr / kblas_time;
      kblas_time *= 1000.0;//convert to ms

      if(opts.check || opts.time){
        if(batchCount > 1){
          kblas_gemm_batch_strided_wsquery(kblas_handle, batchCount );
          check_kblas_error( kblasAllocateWorkspace(kblas_handle) );
        }
        if(opts.check){
          if(TLR_LLL){
            if(batchCount == 1){
              check_cublas_error( cublasXgemm(cublas_handle,
                                              CUBLAS_OP_N, CUBLAS_OP_T,
                                              Cm, Cn, res_rC,
                                              &one,  d_Cu, ldCu,
                                                     d_Cv, ldCv,
                                              &zero, d_C,  lddc) );
            }else
            {
              check_kblas_error( kblas_gemm_batch(kblas_handle,
                                                  KBLAS_NoTrans, KBLAS_Trans,
                                                  Cm, Cn, res_rC,
                                                  one,  d_Cu, ldCu, strideCu,
                                                        d_Cv, ldCv, strideCv,
                                                  zero, d_C,  lddc, strideC,
                                                  batchCount) );
            }
          }
          check_cublas_error( cublasGetMatrixAsync( Cm, Cn * batchCount, sizeof(T), d_C, lddc, h_Rk, ldc, curStream ) );
          cudaDeviceSynchronize();
        }

        if(TLR_LLL){
          check_cublas_error( cublasSetMatrixAsync( Cm, max_rC * batchCount, sizeof(T), h_Cu, ldc, d_Cu, ldCu, curStream ) );
          check_cublas_error( cublasSetMatrixAsync( Cn, max_rC * batchCount, sizeof(T), h_Cv, Cn, d_Cv, ldCv, curStream ) );
        }

        if(batchCount == 1){
            check_cublas_error( cublasXgemm( cublas_handle,
                                            CUBLAS_OP_N, CUBLAS_OP_T,
                                            Am, An, rA,
                                            &one,  d_Au, ldAu,
                                                   d_Av, ldAv,
                                            &zero, d_A,  ldda) );
            check_cublas_error( cublasXgemm( cublas_handle,
                                            CUBLAS_OP_N, CUBLAS_OP_T,
                                            Bm, Bn, rB,
                                            &one,  d_Bu, ldBu,
                                                   d_Bv, ldBv,
                                            &zero, d_B,  lddb) );
        }else{
          kblas_gemm_batch_strided_wsquery(kblas_handle, batchCount );
          check_kblas_error( kblasAllocateWorkspace(kblas_handle) );

            check_kblas_error( kblas_gemm_batch(kblas_handle,
                                                KBLAS_NoTrans, KBLAS_Trans,
                                                Am, An, rA,
                                                one,  d_Au, ldAu, strideAu,
                                                      d_Av, ldAv, strideAv,
                                                zero, d_A,  ldda, strideA,
                                                batchCount) );
            check_kblas_error( kblas_gemm_batch(kblas_handle,
                                                KBLAS_NoTrans, KBLAS_Trans,
                                                Bm, Bn, rB,
                                                one,  d_Bu, ldBu, strideBu,
                                                      d_Bv, ldBv, strideBv,
                                                zero, d_B,  lddb, strideB,
                                                batchCount) );
        }
        for(int r = 0; r < (nruns+opts.warmup); r++)
        {
          if(TLR_LLL){
            if(batchCount == 1){
              check_cublas_error( cublasXgemm(cublas_handle,
                                              CUBLAS_OP_N, CUBLAS_OP_T,
                                              Cm, Cn, rC,
                                              &one,  d_Cu, ldCu,
                                                     d_Cv, ldCv,
                                              &zero, d_C,  lddc) );
            }else
            {
              check_kblas_error( kblas_gemm_batch(kblas_handle,
                                                  KBLAS_NoTrans, KBLAS_Trans,
                                                  Cm, Cn, rC,
                                                  one,  d_Cu, ldCu, strideCu,
                                                        d_Cv, ldCv, strideCv,
                                                  zero, d_C,  lddc, strideC,
                                                  batchCount) );
            }
          }else
            check_cublas_error( cublasSetMatrixAsync( Cm, Cn * batchCount, sizeof(T), h_C, ldc, d_C, lddc, curStream ) );

          kblasTimerTic(kblas_handle);
          if(batchCount == 1){
            check_cublas_error( cublasXgemm(cublas_handle,
                                            cub_transA, cub_transB,
                                            M, N, K,
                                            &alpha, d_A, ldda,
                                                    d_B, lddb,
                                            &beta,  d_C, lddc) );
          }else{
            check_kblas_error( kblas_gemm_batch(kblas_handle,
                                                opts.transA, opts.transB,
                                                M, N, K,
                                                alpha, d_A, ldda, strideA,
                                                       d_B, lddb, strideB,
                                                beta,  d_C, lddc, strideC,
                                                batchCount) );
          }
          kblasTimerRecordEnd(kblas_handle);
          time = kblasTimerToc(kblas_handle);
          if(!opts.warmup || r > 0)
            cublas_time += time;
        }
        cublas_time /= nruns;
        cublas_perf = gflops / cublas_time;
        cublas_time *= 1000.0;//convert to ms

        if(opts.check){
          check_cublas_error( cublasGetMatrixAsync( Cm, Cn * batchCount, sizeof(T), d_C, lddc, h_Rc, ldc, curStream ) );
          cudaDeviceSynchronize();
          ref_error = 0;
          for(int b = 0; b < batchCount; b++)
          {
              ref_error += Xget_max_error_matrix(h_Rc + b*ldc*Cn, h_Rk + b*ldc*Cn, Cm, Cn, ldc);
          }
        }

      }
      if(opts.check) ref_error /= batchCount;

      kblasFreeWorkspace(kblas_handle);
      free( h_Au );
      free( h_Av );
      free( h_Bu );
      free( h_Bv );
      if(TLR_LLL){
        free( h_Cu );
        free( h_Cv );
        free( h_A );
        free( h_B );
      }
      free( h_C );
      if(opts.check || opts.time){
        free( h_Rk );
        free( h_Rc );
        check_error(  cudaFree( d_A ) );
        check_error(  cudaFree( d_B ) );
        check_error(  cudaFree( d_A_array ) );
        check_error(  cudaFree( d_B_array ) );
      }
      check_error(  cudaFree( d_C ) );

      check_error(  cudaFree( d_Au ) );
      check_error(  cudaFree( d_Av ) );
      check_error(  cudaFree( d_Bu ) );
      check_error(  cudaFree( d_Bv ) );
      if(TLR_LLL){
        check_error(  cudaFree( d_Cu ) );
        check_error(  cudaFree( d_Cv ) );
      }
      if(!strided && batchCount > 1){
        check_error(  cudaFree( d_Au_array ) );
        check_error(  cudaFree( d_Av_array ) );
        check_error(  cudaFree( d_Bu_array ) );
        check_error(  cudaFree( d_Bv_array ) );
        if(TLR_LLL){
          check_error(  cudaFree( d_Cu_array ) );
          check_error(  cudaFree( d_Cv_array ) );
        }
        check_error(  cudaFree( d_C_array ) );
      }

      printf("     %7.4f %7.4f        %7.4f %7.4f         %7.4f     ",
             kblas_perf, kblas_time,
             cublas_perf, cublas_time,
             cublas_time / kblas_time);
      if(opts.check)
        printf(" %.4e", ref_error);
      printf("\n");
    }
    }
    if ( opts.niter > 1 ) {
      printf( "\n" );
    }
  }

  kblasDestroy(&kblas_handle);
  deleteGPU_Timer(kblas_timer);

  #ifdef USE_MAGMA
    if(use_magma){
      magma_finalize();
    }
  #endif
}

//==============================================================================================
int main(int argc, char** argv)
{

  kblas_opts opts;
  if(!parse_opts( argc, argv, &opts )){
    // USAGE;
    return -1;
  }
#if (defined PREC_s) || (defined PREC_d)
  TYPE alpha = 1;//0.28;
  TYPE beta = 1;//1.2;
#elif defined PREC_c
  TYPE alpha = make_cuFloatComplex(1.2, -0.6);
  TYPE beta = make_cuFloatComplex(0.2, 1.6);
#elif defined PREC_z
  TYPE alpha = make_cuDoubleComplex(1.2, -0.6);
  TYPE beta = make_cuDoubleComplex(0.2, 1.6);
#endif
  test_Xgemm_lr<TYPE>(opts, alpha, beta);

}