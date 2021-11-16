/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_tlr/test_Xgemm_tlr.cpp

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
#include <unistd.h>
#ifdef USE_MKL
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#else
#include <lapacke_utils.h>
#endif

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
#include "kblas_operators.h"
#include "kblas_common.h"

#include "Xblas_core.ch"

#define USE_OPENMP
#ifdef USE_OPENMP
#include "omp.h"
#endif//USE_OPENMP


//==============================================================================================
#ifdef USING
#undef USING
#endif

template<class T>
int test_Xgemm_tlr(kblas_opts& opts, T alpha, T beta)
{

  kblasHandle_t kblas_handle;
  GPU_Timer_t kblas_timer;
  int nruns = opts.nruns;
  int nb = opts.nb;
  int TLR_LLL = opts.LR == 't';

  int M, N, K,
      max_M, max_N, max_K,
      MTiles, NTiles, KTiles,
      MTilesA, MTilesB, NTilesA, NTilesB;
  int Am, An,
      Bm, Bn,
      Cm, Cn,
      rA, rB, rC, res_rC,
      max_rA, max_rB, max_rC, max_rank;
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
      info;

  int ISEED[4] = {0,0,0,1};

  T *h_Au = NULL, *h_Av = NULL,
    *h_Bu = NULL, *h_Bv = NULL,
    *h_Cu = NULL, *h_Cv = NULL,
    *h_Rc = NULL, *h_Rk = NULL,
    *h_A = NULL, *h_B = NULL, *h_C = NULL;
  T *d_A = NULL, *d_B = NULL, *d_C = NULL;
  T *d_Au = NULL, *d_Av = NULL,
    *d_Bu = NULL, *d_Bv = NULL,
    *d_Cu = NULL, *d_Cv = NULL;
  int *h_rA, *h_rB,
      *d_rA, *d_rB;
  T **h_Au_array = NULL, **h_Av_array = NULL,
    **h_Bu_array = NULL, **h_Bv_array = NULL,
    **h_Cu_array = NULL, **h_Cv_array = NULL;
  T **d_Au_array = NULL, **d_Av_array = NULL,
    **d_Bu_array = NULL, **d_Bv_array = NULL,
    **d_Cu_array = NULL, **d_Cv_array = NULL;
  int *d_ldAu, *d_ldAv,
      *d_ldBu, *d_ldBv,
      *d_lda, *d_ldb, *d_ldc,
      *nb_array;

  T one = make_one<T>(), mone = -one, zero = make_zero<T>();
  bool transV = 0;
  bool strided = opts.strided;
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

  printf("  M       N       K     rankA   rankB   rankC    NB    kblasGEMM-GF/s (ms)   cublasGEMM-GF/s (ms)     Speedup");
  if(opts.check)
    printf("      Error");
  printf("\n");
  printf("======================================================================================================================\n");
  for( int itest = 0; itest < opts.ntest; ++itest )
  {

    double  gflops, perf, gflops_lr,
            kblas_perf = 0.0, kblas_time = 0.0,
            cublas_perf = 0.0, cublas_time = 0.0,
            ref_error = 0.0;

    M = opts.msize[itest];
    N = opts.nsize[itest];
    K = opts.ksize[itest];
    rA = opts.rank[0];
    rB = opts.rank[0];
    res_rC = rC = opts.rank[0];
    max_rank = opts.rank[1];
    int max_rC = kmax(rC, max_rank);
    if(rA <= 0 || rA > nb || rB <= 0 || rB > nb || rC <= 0 || rC > nb){
      printf("Input rank(%d) is not compatible with input tile dimensions, please specify a feasible rank (range: %d-%d). Aborting... \n", rA, 1, nb);
      exit(0);
    }
    if(M % nb || N % nb || K % nb){
      printf("please make sure matrix size is multiple of tile size\n");
      break;
    }

    MTiles = M / nb;
    NTiles = N / nb;
    KTiles = K / nb;

    printf("%5d  %5d   %5d   %5d   %5d ",
           (int) M, (int) N, (int) K, (int) rA, (int) rB);
    if(TLR_LLL){
      if(!(max_rank > 0)){
        printf("Please specify max_rank for TLR format with --rank X:X option, aborting\n");
        exit(0);
      }
    }
    printf("%5d   ",
           (int) rC);
    printf("%5d   ",
           (int) nb);
    fflush( stdout );

    if(opts.check)
    {
      opts.time = 0;
      opts.warmup = 0;
      nruns = 1;
    }

    if ( opts.transA == KBLAS_NoTrans ) {
      lda = Am = M;
      An = K;
      MTilesA = MTiles;
      NTilesA = KTiles;
    } else {
      lda = Am = K;
      An = M;
      MTilesA = KTiles;
      NTilesA = MTiles;
    }
    if ( opts.transB == KBLAS_NoTrans ) {
      ldb = Bm = K;
      Bn = N;
      MTilesB = KTiles;
      NTilesB = NTiles;
    } else {
      ldb = Bm = N;
      Bn = K;
      MTilesB = NTiles;
      NTilesB = KTiles;
    }
    Cm = ldc = M;
    Cn = N;

    if(TLR_LLL)
      gflops_lr = MTiles * NTiles * KTiles * FLOPS_GEMM_LR_LLL<T>(nb, nb, nb, rA, rB, rC) / 1e9;
    else
      gflops_lr = MTiles * NTiles * KTiles * FLOPS_GEMM_LR_LLD<T>(nb, nb, nb, rA, rB) / 1e9;
    gflops = FLOPS_GEMM<T>(M, N, K ) / 1e9;

    ldda = kblas_roundup(lda, 32);
    lddb = kblas_roundup(ldb, 32);
    lddc = kblas_roundup(ldc, 32);
    ldAu = kblas_roundup( nb, 32);
    ldAv = kblas_roundup( nb, 32);
    ldBu = kblas_roundup( nb, 32);
    ldBv = kblas_roundup( nb, 32);
    if(TLR_LLL){
      ldCu = kblas_roundup( nb, 32);
      ldCv = kblas_roundup( nb, 32);
    }

    sizeA = lda * An;
    sizeB = ldb * Bn;
    sizeC = ldc * Cn;
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

    if(!TLR_LLL || opts.check || opts.time){
      TESTING_MALLOC_CPU( h_C, T, ldc * Cn);
      TESTING_MALLOC_DEV( d_C, T, strideC);
    }
    TESTING_MALLOC_CPU( h_Au, T, MTilesA * NTilesA * nb * rA);
    TESTING_MALLOC_CPU( h_Av, T, MTilesA * NTilesA * nb * rA);
    TESTING_MALLOC_CPU( h_Bu, T, MTilesB * NTilesB * nb * rB);
    TESTING_MALLOC_CPU( h_Bv, T, MTilesB * NTilesB * nb * rB);
    if(TLR_LLL){
      TESTING_MALLOC_CPU( h_Cu, T, MTiles * NTiles * nb * max_rC);
      TESTING_MALLOC_CPU( h_Cv, T, MTiles * NTiles * nb * max_rC);
    }

    TESTING_MALLOC_DEV( d_Au, T, MTilesA * NTilesA * strideAu);
    TESTING_MALLOC_DEV( d_Av, T, MTilesA * NTilesA * strideAv);
    TESTING_MALLOC_DEV( d_Bu, T, MTilesB * NTilesB * strideBu);
    TESTING_MALLOC_DEV( d_Bv, T, MTilesB * NTilesB * strideBv);
    if(TLR_LLL){
      TESTING_MALLOC_DEV( d_Cu, T, MTiles * NTiles * strideCu);
      TESTING_MALLOC_DEV( d_Cv, T, MTiles * NTiles * strideCv);
    }

    {
      TESTING_MALLOC_CPU( h_Au_array, T*, MTilesA * NTilesA);
      TESTING_MALLOC_CPU( h_Av_array, T*, MTilesA * NTilesA);
      TESTING_MALLOC_CPU( h_Bu_array, T*, MTilesB * NTilesB);
      TESTING_MALLOC_CPU( h_Bv_array, T*, MTilesB * NTilesB);
      if(TLR_LLL){
        TESTING_MALLOC_CPU( h_Cu_array, T*, MTiles * NTiles);
        TESTING_MALLOC_CPU( h_Cv_array, T*, MTiles * NTiles);
      }
    }
    if(!strided)
    {
      TESTING_MALLOC_DEV( d_Au_array, T*, MTilesA * NTilesA);
      TESTING_MALLOC_DEV( d_Av_array, T*, MTilesA * NTilesA);
      TESTING_MALLOC_DEV( d_Bu_array, T*, MTilesB * NTilesB);
      TESTING_MALLOC_DEV( d_Bv_array, T*, MTilesB * NTilesB);
      if(TLR_LLL){
        TESTING_MALLOC_DEV( d_Cu_array, T*, MTiles * NTiles);
        TESTING_MALLOC_DEV( d_Cv_array, T*, MTiles * NTiles);
      }
    }

    if(opts.check || opts.time)
    {
      TESTING_MALLOC_DEV( d_A, T, strideA);
      TESTING_MALLOC_DEV( d_B, T, strideB);
      TESTING_MALLOC_CPU( h_Rc, T, ldc * Cn);
      TESTING_MALLOC_CPU( h_Rk, T, ldc * Cn);

    }

    // if(TLR_LLL)
      T *h_Dense = NULL, *h_Xu = NULL, *h_Xv = NULL, *h_Xs = NULL;
    {
      T work_s[1], *work = NULL;
      int lwork, ione = 1, max_MNK = kmax(kmax(M, N), K);
      TESTING_MALLOC_CPU( h_Xs, T, nb);
      TESTING_MALLOC_CPU( h_Xu, T, nb * nb);
      TESTING_MALLOC_CPU( h_Xv, T, nb * nb);
      TESTING_MALLOC_CPU( h_Dense, T, nb * nb);
      hilbertMatrix(nb, nb, h_Dense, nb, T(2));
      int info;
        lwork = -1;
        LAPACK_GESVD( "S", "S",
                      &nb, &nb,
                      h_Dense, &nb,
                      h_Xs,
                      h_Xu, &ldc,
                      h_Xv, &nb,
                      work_s, &lwork,
                      &info);
        lwork = (int)work_s[0];
        TESTING_MALLOC_CPU( work, T, lwork);
      LAPACK_GESVD( "S", "S",
                    &nb, &nb,
                    h_Dense, &nb,
                    h_Xs,
                    h_Xu, &nb,
                    h_Xv, &nb,
                    work, &lwork,
                    &info);
      hilbertMatrix(nb, nb, h_Dense, nb, T(2));

      for(int c = 0; c < nb; c++){
        LAPACK_SCAL(nb, h_Xs[c], h_Xu + c * nb, ione);
      }
      #pragma omp parallel for
      for(int b = 0; b < MTilesA*NTilesA; b++){
        LAPACK_LACPY("A", &nb, &rA, h_Xu, &nb, h_Au + b*nb*rA, &nb);
        #ifdef USE_MKL
        LAPACK_OMATCPY('C', 'T', rA, nb, one, h_Xv, nb, h_Av + b*nb*rA, nb);
        #else
        LAPACK_TRANS(LAPACK_COL_MAJOR, rA, nb, h_Xv, nb, h_Av + b*nb*rA, nb);
        #endif
      }
      #pragma omp parallel for
      for(int b = 0; b < MTilesB*NTilesB; b++){
        LAPACK_LACPY("A", &nb, &rB, h_Xu, &nb, h_Bu + b*nb*rB, &nb);
        #ifdef USE_MKL
        LAPACK_OMATCPY('C', 'T', rB, nb, one, h_Xv, nb, h_Bv + b*nb*rB, nb);
        #else
        LAPACK_TRANS(LAPACK_COL_MAJOR, rB, nb, h_Xv, nb, h_Bv + b*nb*rB, nb);
        #endif
      }
      if(TLR_LLL){
        #pragma omp parallel for
        for(int b = 0; b < MTiles*NTiles; b++){
          LAPACK_LACPY("A", &nb, &rC, h_Xu, &nb, h_Cu + b*nb*rC, &nb);
          #ifdef USE_MKL
          LAPACK_OMATCPY('C', 'T', rC, nb, one, h_Xv, nb, h_Cv + b*nb*rC, nb);
          #else
          LAPACK_TRANS(LAPACK_COL_MAJOR, rC, nb, h_Xv, nb, h_Cv + b*nb*rC, nb);
          #endif
        }
      }
      if(work != NULL) free(work);
      free(h_Xs);
      free(h_Xu);
      free(h_Xv);
      // free(h_Dense);
    }

    for( int iter = 0; iter < opts.niter; ++iter )
    {
      if(!TLR_LLL){
          Xrand_matrix(Cm, Cn, h_C, ldc);
      }

      cudaStream_t curStream = kblasGetStream(kblas_handle);
      cublasHandle_t cublas_handle = kblasGetCublasHandle(kblas_handle);

      check_cublas_error( cublasSetMatrixAsync( nb, MTilesA * NTilesA * rA, sizeof(T), h_Au, nb, d_Au, ldAu, curStream ) );
      check_cublas_error( cublasSetMatrixAsync( nb, MTilesA * NTilesA * rA, sizeof(T), h_Av, nb, d_Av, ldAv, curStream ) );
      check_cublas_error( cublasSetMatrixAsync( nb, MTilesB * NTilesB * rB, sizeof(T), h_Bu, nb, d_Bu, ldBu, curStream ) );
      check_cublas_error( cublasSetMatrixAsync( nb, MTilesB * NTilesB * rB, sizeof(T), h_Bv, nb, d_Bv, ldBv, curStream ) );

      #pragma omp parallel for
      for(int tn = 0; tn < NTilesA; tn++){
        for(int tm = 0; tm < MTilesA; tm++){
          h_Au_array[tm + tn*MTilesA] = &d_Au[tm * strideAu + tn * strideAu * MTilesA];
          h_Av_array[tm + tn*MTilesA] = &d_Av[tm * strideAv + tn * strideAv * MTilesA];
        }
      }
      #pragma omp parallel for
      for(int tn = 0; tn < NTilesB; tn++){
        for(int tm = 0; tm < MTilesB; tm++){
          h_Bu_array[tm + tn*MTilesB] = &d_Bu[tm * strideBu + tn * strideBu * MTilesB];
          h_Bv_array[tm + tn*MTilesB] = &d_Bv[tm * strideBv + tn * strideBv * MTilesB];
        }
      }
      if(TLR_LLL)
      {
        #pragma omp parallel for
        for(int tn = 0; tn < NTiles; tn++){
          for(int tm = 0; tm < MTiles; tm++){
            h_Cu_array[tm + tn*MTiles] = &d_Cu[tm * strideCu + tn * strideCu * MTiles];
            h_Cv_array[tm + tn*MTiles] = &d_Cv[tm * strideCv + tn * strideCv * MTiles];
          }
        }
      }
      if(!strided){
        check_error( cudaMemcpyAsync(d_Au_array,
                                     h_Au_array,
                                     MTilesA * NTilesA * sizeof(T*),
                                     cudaMemcpyHostToDevice,
                                     curStream) );
        check_error( cudaMemcpyAsync(d_Av_array,
                                     h_Av_array,
                                     MTilesA * NTilesA * sizeof(T*),
                                     cudaMemcpyHostToDevice,
                                     curStream) );
        check_error( cudaMemcpyAsync(d_Bu_array,
                                     h_Bu_array,
                                     MTilesB * NTilesB * sizeof(T*),
                                     cudaMemcpyHostToDevice,
                                     curStream) );
        check_error( cudaMemcpyAsync(d_Bv_array,
                                     h_Bv_array,
                                     MTilesB * NTilesB * sizeof(T*),
                                     cudaMemcpyHostToDevice,
                                     curStream) );
        if(TLR_LLL){
          check_error( cudaMemcpyAsync(d_Cu_array,
                                       h_Cu_array,
                                       MTiles * NTiles * sizeof(T*),
                                       cudaMemcpyHostToDevice,
                                       curStream) );
          check_error( cudaMemcpyAsync(d_Cv_array,
                                       h_Cv_array,
                                       MTiles * NTiles * sizeof(T*),
                                       cudaMemcpyHostToDevice,
                                       curStream) );
        }
      }

      info = -1;
      if(!strided){
        if(TLR_LLL)
          kblasXgemm_tlr_lll_wsquery( kblas_handle,
                                        MTiles, NTiles,
                                        rA, rB, max_rC, max_rank,
                                        nb, nb);
        else
          kblasXgemm_plr_dev_tiled_wsquery(kblas_handle,
                                           MTiles, NTiles, rA, rB,
                                           nb, nb);
      }
      else{
        if(TLR_LLL){
          printf("Not implemented yet, aborting...\n");
          exit(0);
        }
        else
          kblasXgemm_tlr_lld_wsquery(kblas_handle,
                                       MTiles, NTiles, rA, rB,
                                       nb, nb);
      }
      check_kblas_error( kblasAllocateWorkspace(kblas_handle) );
      check_error( cudaGetLastError() );

      double time = 0;

      for(int r = 0; r < nruns+opts.warmup; r++)
      {
        if(TLR_LLL){
          check_cublas_error( cublasSetMatrixAsync( nb, MTiles * NTiles * max_rC, sizeof(T), h_Cu, nb, d_Cu, ldCu, curStream ) );
          check_cublas_error( cublasSetMatrixAsync( nb, MTiles * NTiles * max_rC, sizeof(T), h_Cv, nb, d_Cv, ldCv, curStream ) );
        }
        else
          check_cublas_error( cublasSetMatrixAsync( Cm, Cn, sizeof(T), h_C, ldc, d_C, lddc, curStream ) );

        res_rC = rC;

        gpuTimerTic(kblas_timer);
          if(TLR_LLL){
            check_kblas_error( kblas_gemm_tlr(kblas_handle,
                                                    opts.transA, opts.transB,
                                                    MTiles, NTiles, KTiles,
                                                    nb, nb, nb,
                                                    alpha,
                                                    d_Au_array, ldAu,
                                                    d_Av_array, ldAv, MTilesA, rA,
                                                    d_Bu_array, ldBu,
                                                    d_Bv_array, ldBv, MTilesB, rB,
                                                    beta,
                                                    d_Cu_array, ldCu,
                                                    d_Cv_array, ldCv, MTiles, res_rC,
                                                    max_rank, 0) );
          }
          else{
            check_kblas_error( kblas_gemm_tlr(kblas_handle,
                                                    opts.transA, opts.transB,
                                                    MTiles, NTiles, KTiles,
                                                    nb, nb, nb,
                                                    alpha,
                                                    (const T**)d_Au_array, ldAu,
                                                    (const T**)d_Av_array, ldAv, MTilesA, rA,
                                                    (const T**)d_Bu_array, ldBu,
                                                    (const T**)d_Bv_array, ldBv, MTilesB, rB,
                                                    beta, d_C, lddc) );
          }
        gpuTimerRecordEnd(kblas_timer);
        time = gpuTimerToc(kblas_timer);
        check_error( cudaGetLastError() );
        if(!opts.warmup || r>0)
          kblas_time += time;
      }
      kblas_time /= nruns;
      kblas_perf = gflops_lr / kblas_time;
      kblas_time *= 1000.0;

      printf(" %7.4f %7.4f        ",
             kblas_perf, kblas_time);fflush( stdout );

      if(opts.check || opts.time){
        if(opts.check){

          if(TLR_LLL){
            for(int nTile = 0; nTile < NTiles; nTile++){
              for(int mTile = 0; mTile < MTiles; mTile++){
                check_cublas_error( cublasXgemm(cublas_handle,
                                                CUBLAS_OP_N, CUBLAS_OP_T,
                                                nb, nb, res_rC,
                                                &one,  d_Cu + mTile * strideCu + nTile * strideCu * MTiles, ldCu,
                                                       d_Cv + mTile * strideCv + nTile * strideCv * MTiles, ldCv,
                                                &zero, d_C + mTile * nb + nTile * nb * lddc,  lddc) );
              }
            }
            check_cublas_error( cublasSetMatrixAsync( nb, MTiles * NTiles * max_rC, sizeof(T), h_Cu, nb, d_Cu, ldCu, curStream ) );
            check_cublas_error( cublasSetMatrixAsync( nb, MTiles * NTiles * max_rC, sizeof(T), h_Cv, nb, d_Cv, ldCv, curStream ) );
          }
          check_cublas_error( cublasGetMatrixAsync( Cm, Cn, sizeof(T), d_C, lddc, h_Rk, ldc, curStream ) );
          cudaDeviceSynchronize();
        }

        {
          check_cublas_error( cublasSetMatrixAsync( nb, nb, sizeof(T), h_Dense, nb, d_C, lddc, curStream ) );
        }
        for(int mTile = 0; mTile < MTilesA; mTile++){
          for(int kTile = 0; kTile < NTilesA; kTile++){
                T* d_T = &d_A[mTile * nb + kTile * nb * ldda];
                check_error( cudaMemcpy2DAsync ((void*)(d_T), (size_t)(ldda * sizeof(T)),
                                                (const void*)d_C, (size_t)(lddc * sizeof(T)),
                                                (size_t)(nb * sizeof(T)), (size_t)(nb),
                                                cudaMemcpyDeviceToDevice,
                                                curStream) );
          }
        }
        for(int nTile = 0; nTile < NTilesB; nTile++){
          for(int kTile = 0; kTile < MTilesB; kTile++){
                T* d_T = &d_B[kTile * nb + nTile * nb * lddb];
                check_error( cudaMemcpy2DAsync ((void*)(d_T), (size_t)(lddb * sizeof(T)),
                                                (const void*)d_C, (size_t)(lddc * sizeof(T)),
                                                (size_t)(nb * sizeof(T)), (size_t)(nb),
                                                cudaMemcpyDeviceToDevice,
                                                curStream) );
          }
        }

        for(int r = 0; r < nruns; r++)
        {
          if(TLR_LLL){
            for(int nTile = 0; nTile < NTiles; nTile++){
              for(int mTile = 0; mTile < MTiles; mTile++){
                check_error( cudaMemcpy2DAsync ((void*)(d_C + mTile * nb + nTile * nb * lddc), (size_t)(lddc * sizeof(T)),
                                                (const void*)d_A, (size_t)(ldda * sizeof(T)),
                                                (size_t)(nb * sizeof(T)), (size_t)(nb),
                                                cudaMemcpyDeviceToDevice,
                                                curStream) );
              }
            }
          }else{
            check_cublas_error( cublasSetMatrixAsync( Cm, Cn, sizeof(T), h_C, ldc, d_C, lddc, curStream ) );
          }

          gpuTimerTic(kblas_timer);
            check_cublas_error( cublasXgemm( cublas_handle,
                                            cub_transA, cub_transB,
                                            M, N, K,
                                            &alpha, d_A, ldda,
                                                    d_B, lddb,
                                            &beta,  d_C, lddc) );
          gpuTimerRecordEnd(kblas_timer);
          time = gpuTimerToc(kblas_timer);
          cublas_time += time;
        }
        cublas_time /= nruns;
        cublas_perf = gflops / cublas_time;
        cublas_time *= 1000.0;

        if(opts.check){
          check_cublas_error( cublasGetMatrixAsync( Cm, Cn, sizeof(T), d_C, lddc, h_Rc, ldc, curStream ) );
          cudaDeviceSynchronize();

          ref_error = Xget_max_error_matrix(h_Rc, h_Rk, Cm, Cn, ldc);
        }
      }

      printf("%7.4f %7.4f       %7.4f     ",
             cublas_perf, cublas_time,
             cublas_time / kblas_time);
      if(opts.check)
        printf(" %.4e", ref_error);
      printf("\n");
    }

    free( h_Au );
    free( h_Av );
    free( h_Bu );
    free( h_Bv );
    free( h_Au_array );
    free( h_Av_array );
    free( h_Bu_array );
    free( h_Bv_array );
    free(h_Dense);
    if(!TLR_LLL || opts.check || opts.time){
      free( h_C );
    }
    if(opts.check || opts.time){
      free( h_Rk );
      free( h_Rc );
      check_error(  cudaFree( d_A ) );
      check_error(  cudaFree( d_B ) );
    }
    if(!TLR_LLL || opts.check || opts.time){
      check_error(  cudaFree( d_C ) );
    }

    check_error(  cudaFree( d_Au ) );
    check_error(  cudaFree( d_Av ) );
    check_error(  cudaFree( d_Bu ) );
    check_error(  cudaFree( d_Bv ) );
    if(TLR_LLL){
      free( h_Cu );
      free( h_Cv );
      check_error(  cudaFree( d_Cu ) );
      check_error(  cudaFree( d_Cv ) );
    }
    if(!strided){
      check_error(  cudaFree( d_Au_array ) );
      check_error(  cudaFree( d_Av_array ) );
      check_error(  cudaFree( d_Bu_array ) );
      check_error(  cudaFree( d_Bv_array ) );
      if(TLR_LLL){
        check_error(  cudaFree( d_Cu_array ) );
        check_error(  cudaFree( d_Cv_array ) );
      }
    }

    if ( opts.niter > 1 ) {
      printf( "\n" );
    }
    sleep(2.);
  }


  kblasDestroy(&kblas_handle);
  deleteGPU_Timer(kblas_timer);

  #ifdef USE_MAGMA
    if(use_magma){
      magma_finalize();
    }
  #endif
  printf( "\n" );
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
  TYPE alpha = 0.28;
  TYPE beta = 1.2;
#elif defined PREC_c
  TYPE alpha = make_cuFloatComplex(1.2, -0.6);
  TYPE beta = make_cuFloatComplex(0.2, 1.6);
#elif defined PREC_z
  TYPE alpha = make_cuDoubleComplex(1.2, -0.6);
  TYPE beta = make_cuDoubleComplex(0.2, 1.6);
#endif
  test_Xgemm_tlr<TYPE>(opts, alpha, beta);

}


