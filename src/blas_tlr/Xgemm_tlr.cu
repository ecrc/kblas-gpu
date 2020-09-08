/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_tlr/Xgemm_tlr.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include "kblas.h"
#include "kblas_operators.h"

#define DBG_MSG

#include "kblas_struct.h"
#include "kblas_prec_def.h"
#include "kblas_gpu_util.ch"

#include "workspace_queries.ch"

#include "Xblas_core.ch"
#include "Xhelper_funcs.ch"
#include "batch_block_copy.h"
#include "Xgemm_tlr_core.cuh"

//==============================================================================================
// Single GEMM-LR
//==============================================================================================
// workspace needed: device data
// Au, Av, Bu, Bv, C: host pointers to device buffers
int Xgemm_lr(kblasHandle_t handle,
              char transA, char transB,
              const int M, const int N, const int K,
              const TYPE alpha, const TYPE* Au, int ldAu,
                                const TYPE* Av, int ldAv, int kA,
                                const TYPE* Bu, int ldBu,
                                const TYPE* Bv, int ldBv, int kB,
              TYPE beta,  TYPE* C, int ldC)
{
  return Xgemm_LR_core(handle,
                        M, N, K,
                        alpha,
                        (const TYPE*)(transA == KBLAS_NoTrans ? Au : Av), (transA == KBLAS_NoTrans ? ldAu : ldAv ),
                        (const TYPE*)(transA == KBLAS_NoTrans ? Av : Au), (transA == KBLAS_NoTrans ? ldAv : ldAu ), kA,
                        (const TYPE*)(transB == KBLAS_NoTrans ? Bu : Bv), (transB == KBLAS_NoTrans ? ldBu : ldBv ),
                        (const TYPE*)(transB == KBLAS_NoTrans ? Bv : Bu), (transB == KBLAS_NoTrans ? ldBv : ldBu ), kB,
                        beta,  C, ldC);
}

int kblas_gemm_lr(kblasHandle_t handle,
                  char transA, char transB,
                  const int M, const int N, const int K,
                  const TYPE alpha,
                  const TYPE* Au, int ldAu, const TYPE* Av, int ldAv, int kA,
                  const TYPE* Bu, int ldBu, const TYPE* Bv, int ldBv, int kB,
                  const TYPE beta,
                        TYPE* C, int ldC)
{
  return Xgemm_lr( handle,
                    transA, transB,
                    M, N, K,
                    alpha, Au, ldAu, Av, ldAv, kA,
                           Bu, ldBu, Bv, ldBv, kB,
                    beta,  C, ldC);
}

extern "C"
int kblasXgemm_lr_lld( kblasHandle_t handle,
                    char transA, char transB,
                    const int M, const int N, const int K,
                    const TYPE alpha,
                    const TYPE* Au, int ldAu, const TYPE* Av, int ldAv, int kA,
                    const TYPE* Bu, int ldBu, const TYPE* Bv, int ldBv, int kB,
                    const TYPE beta,
                          TYPE* C, int ldC)
{
  return Xgemm_lr( handle,
                    transA, transB,
                    M, N, K,
                    alpha, Au, ldAu, Av, ldAv, kA,
                           Bu, ldBu, Bv, ldBv, kB,
                    beta,  C, ldC);
}

//----------------------------------------------------------------------------------------------
extern "C"
void kblasXgemm_lr_lld_wsquery(kblasHandle_t handle, const int N, int kA, int kB){
  gemm_lr_wsquery_core<TYPE>(N, kA, kB, &(handle->work_space.requested_ws_state));
}


//==============================================================================================
// Single GEMM-LR
//==============================================================================================
// workspace needed: device data
// Au, Av, Bu, Bv, C: host pointers to device buffers
int kblas_gemm_lr(kblasHandle_t handle,
                  char transA, char transB,
                  const int M, const int N, const int K,
                  const TYPE alpha,
                  const TYPE* Au, int ldAu, const TYPE* Av, int ldAv, int kA,
                  const TYPE* Bu, int ldBu, const TYPE* Bv, int ldBv, int kB,
                  const TYPE beta,
                        TYPE* Cu, int ldCu,       TYPE* Cv, int ldCv, int& kC,
                  int max_rk, double max_acc)
{

  //----------------------------
  //validate & prepare workspace
  KBlasWorkspaceState ws_needed;
  gemm_lr_wsquery_core<TYPE>(M, N,
                              kA, kB, kC, max_rk,
                              (kblasWorkspaceState_t)&ws_needed);

  if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) )
    return KBLAS_InsufficientWorkspace;

  TYPE* d_workspace = (TYPE*)(handle->work_space.d_data);

  int TransA = (transA == KBLAS_Trans );
  int TransB = (transB == KBLAS_Trans);

  return Xgemm_LR_core(handle,
                        M, N, K,
                        alpha, (TransA ? Av : Au), (TransA ? ldAv : ldAu),
                               (TransA ? Au : Av), (TransA ? ldAu : ldAv), kA,
                               (TransB ? Bv : Bu), (TransB ? ldBv : ldBu),
                               (TransB ? Bu : Bv), (TransB ? ldBu : ldBv), kB,
                        beta,  Cu, ldCu, Cv, ldCv, kC,
                        max_rk, max_acc,
                        d_workspace);
}

extern "C"
int kblasXgemm_lr_lll( kblasHandle_t handle,
                    char transA, char transB,
                    const int M, const int N, const int K,
                    const TYPE alpha,
                    const TYPE* Au, int ldAu, const TYPE* Av, int ldAv, int kA,
                    const TYPE* Bu, int ldBu, const TYPE* Bv, int ldBv, int kB,
                    const TYPE beta,
                          TYPE* Cu, int ldCu,       TYPE* Cv, int ldCv, int* kC,
                    int max_rk, double max_acc)
{
  return kblas_gemm_lr(handle,
                        transA, transB,
                        M, N, K,
                        alpha, Au, ldAu, Av, ldAv, kA,
                               Bu, ldBu, Bv, ldBv, kB,
                        beta,  Cu, ldCu, Cv, ldCv, *kC,
                        max_rk, max_acc);
}

//----------------------------------------------------------------------------------------------
extern "C"
void kblasXgemm_lr_lll_wsquery(kblasHandle_t handle,
                            const int M, const int N,
                            int kA, int kB, int kC, int max_rk)
{
  gemm_lr_wsquery_core<TYPE>(M, N,
                              kA, kB, kC, max_rk,
                              &(handle->work_space.requested_ws_state));
}

//==============================================================================================
// Batched Strided Uniform GEMM-PLR
//==============================================================================================
// workspace needed: device data & pointers
// Au, Av, Bu, Bv, C: host pointers to device buffers
int Xgemm_lr_batch(kblasHandle_t handle,
                    char transA, char transB,
                    int M, int N, int K,
                    TYPE alpha, TYPE* Au, int ldAu, long strideAu,
                                TYPE* Av, int ldAv, long strideAv, int kA,
                                TYPE* Bu, int ldBu, long strideBu,
                                TYPE* Bv, int ldBv, long strideBv, int kB,
                    TYPE beta,  TYPE* C,  int ldC,  long strideC,
                    int batchCount)
{
  return Xgemm_LR_batch_core<TYPE, TYPE*, true>(
                              handle,
                              M, N, K,
                              alpha,
                              (TYPE*)(transA == KBLAS_NoTrans ? Au : Av), (transA == KBLAS_NoTrans ? ldAu : ldAv ), (transA == KBLAS_NoTrans ? strideAu : strideAv),
                              (TYPE*)(transA == KBLAS_NoTrans ? Av : Au), (transA == KBLAS_NoTrans ? ldAv : ldAu ), (transA == KBLAS_NoTrans ? strideAv : strideAu), kA,
                              (TYPE*)(transB == KBLAS_NoTrans ? Bu : Bv), (transB == KBLAS_NoTrans ? ldBu : ldBv ), (transB == KBLAS_NoTrans ? strideBu : strideBv),
                              (TYPE*)(transB == KBLAS_NoTrans ? Bv : Bu), (transB == KBLAS_NoTrans ? ldBv : ldBu ), (transB == KBLAS_NoTrans ? strideBv : strideBu), kB,
                              beta,
                              (TYPE*)C, ldC, strideC,
                              batchCount);
}

int kblas_gemm_lr_batch(kblasHandle_t handle,
                        char transA, char transB,
                        const int M, const int N, const int K,
                        const TYPE alpha,
                        const TYPE* Au, int ldAu, long strideAu,
                        const TYPE* Av, int ldAv, long strideAv, int kA,
                        const TYPE* Bu, int ldBu, long strideBu,
                        const TYPE* Bv, int ldBv, long strideBv, int kB,
                        const TYPE beta,
                              TYPE* C, int ldC, long strideC,
                        int batchCount)
{
  return Xgemm_lr_batch( handle,
                          transA, transB,
                          M, N, K,
                          alpha,
                          (TYPE*)Au, ldAu, strideAu,
                          (TYPE*)Av, ldAv, strideAv, kA,
                          (TYPE*)Bu, ldBu, strideBu,
                          (TYPE*)Bv, ldBv, strideBv, kB,
                          beta,
                          (TYPE*)C, ldC, strideC,
                          batchCount);
}

extern "C"
int kblasXgemm_lr_lld_batch_strided( kblasHandle_t handle,
                                  char transA, char transB,
                                  const int M, const int N, const int K,
                                  const TYPE alpha,
                                  const TYPE* Au, int ldAu, long strideAu,
                                  const TYPE* Av, int ldAv, long strideAv, int kA,
                                  const TYPE* Bu, int ldBu, long strideBu,
                                  const TYPE* Bv, int ldBv, long strideBv, int kB,
                                  const TYPE beta,
                                        TYPE* C, int ldC, long strideC,
                                  int batchCount)
{
  return Xgemm_lr_batch( handle,
                          transA, transB,
                          M, N, K,
                          alpha,
                          (TYPE*)Au, ldAu, strideAu,
                          (TYPE*)Av, ldAv, strideAv, kA,
                          (TYPE*)Bu, ldBu, strideBu,
                          (TYPE*)Bv, ldBv, strideBv, kB,
                          beta,
                          (TYPE*)C, ldC, strideC,
                          batchCount);
}

//----------------------------------------------------------------------------------------------
extern "C"
void kblasXgemm_lr_lld_batch_strided_wsquery(kblasHandle_t handle, const int N, int kA, int kB, int batchCount){
  gemm_lr_batch_wsquery_core<TYPE, true>(N, kA, kB, batchCount, &(handle->work_space.requested_ws_state));
}

//==============================================================================================
// Batched Strided Uniform GEMM-TLR
//==============================================================================================
int Xgemm_lr_batch(kblasHandle_t handle,
                    char transA, char transB,
                    int M, int N, int K,
                    TYPE alpha,
                    TYPE* Au, int ldAu, long strideAu,
                    TYPE* Av, int ldAv, long strideAv, int kA,
                    TYPE* Bu, int ldBu, long strideBu,
                    TYPE* Bv, int ldBv, long strideBv, int kB,
                    TYPE beta,
                    TYPE* Cu, int ldCu, long strideCu,
                    TYPE* Cv, int ldCv, long strideCv, int& kC,
                    int max_rk, double max_acc,
                    int batchCount)
{
  int TransA = (transA == KBLAS_Trans );
  int TransB = (transB == KBLAS_Trans);

  return Xgemm_LR_batch_uniform_core<TYPE, TYPE*, true>(
                                      handle,
                                      M, N, K,
                                      alpha,
                                      (TYPE*)(TransA ? Av : Au), (TransA ? ldAv : ldAu), (TransA ? strideAv : strideAu),
                                      (TYPE*)(TransA ? Au : Av), (TransA ? ldAu : ldAv), (TransA ? strideAu : strideAv), kA,
                                      (TYPE*)(TransB ? Bv : Bu), (TransB ? ldBv : ldBu), (TransB ? strideBv : strideBu),
                                      (TYPE*)(TransB ? Bu : Bv), (TransB ? ldBu : ldBv), (TransB ? strideBu : strideBv), kB,
                                      beta, (TYPE*)Cu, ldCu, strideCu,
                                            (TYPE*)Cv, ldCv, strideCv, kC,
                                      max_rk, max_acc, batchCount);
}

int kblas_gemm_lr_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int M, const int N, const int K,
                          const TYPE alpha,
                          const TYPE* Au, int ldAu, long strideAu,
                          const TYPE* Av, int ldAv, long strideAv, int kA,
                          const TYPE* Bu, int ldBu, long strideBu,
                          const TYPE* Bv, int ldBv, long strideBv, int kB,
                          const TYPE beta,
                                TYPE* Cu, int ldCu, long strideCu,
                                TYPE* Cv, int ldCv, long strideCv, int& kC,
                          int max_rk, double max_acc,
                          int batchCount)
{
  return Xgemm_lr_batch( handle,
                          transA, transB,
                          M, N, K,
                          alpha,
                          (TYPE*)Au, ldAu, strideAu,
                          (TYPE*)Av, ldAv, strideAv, kA,
                          (TYPE*)Bu, ldBu, strideBu,
                          (TYPE*)Bv, ldBv, strideBv, kB,
                          beta,
                          (TYPE*)Cu, ldCu, strideCu,
                          (TYPE*)Cv, ldCv, strideCv, kC,
                          max_rk, max_acc,
                          batchCount);
}

extern "C"
int kblasXgemm_lr_lll_batch_strided( kblasHandle_t handle,
                          char transA, char transB,
                          const int M, const int N, const int K,
                          const TYPE alpha,
                          const TYPE* Au, int ldAu, long strideAu,
                          const TYPE* Av, int ldAv, long strideAv, int kA,
                          const TYPE* Bu, int ldBu, long strideBu,
                          const TYPE* Bv, int ldBv, long strideBv, int kB,
                          const TYPE beta,
                                TYPE* Cu, int ldCu, long strideCu,
                                TYPE* Cv, int ldCv, long strideCv, int* kC,
                          int max_rk, double max_acc,
                          int batchCount)
{
  return Xgemm_lr_batch( handle,
                          transA, transB,
                          M, N, K,
                          alpha,
                          (TYPE*)Au, ldAu, strideAu,
                          (TYPE*)Av, ldAv, strideAv, kA,
                          (TYPE*)Bu, ldBu, strideBu,
                          (TYPE*)Bv, ldBv, strideBv, kB,
                          beta,
                          (TYPE*)Cu, ldCu, strideCu,
                          (TYPE*)Cv, ldCv, strideCv, *kC,
                          max_rk, max_acc,
                          batchCount);
}

//----------------------------------------------------------------------------------------------
extern "C"
void kblasXgemm_lr_lll_batch_strided_wsquery(kblasHandle_t handle,
                                          const int M, const int N,
                                          int kA, int kB, int kC, int max_rk,
                                          int batchCount)
{
  gemm_lr_batch_wsquery_core<TYPE, true>(M, N,
                                    kA, kB, kC, max_rk,
                                    batchCount,
                                    &(handle->work_space.requested_ws_state));
}

//==============================================================================================
// Batched Non-Strided Uniform GEMM-TLR
//==============================================================================================
// workspace needed: device pointers + device data
// Au_array, Av_array, Bu_array, Bv_array, C_array: host pointer to device array of pointers to device buffers
int Xgemm_lr_batch(kblasHandle_t handle,
                    char transA, char transB,
                    int M, int N, int K,
                    TYPE alpha, TYPE** Au, int ldAu, long strideAu,
                                TYPE** Av, int ldAv, long strideAv, int kA,
                                TYPE** Bu, int ldBu, long strideBu,
                                TYPE** Bv, int ldBv, long strideBv, int kB,
                    TYPE beta,  TYPE** C,  int ldC,  long strideC,
                    int batchCount)
{
  return Xgemm_LR_batch_core<TYPE, TYPE**, false>(
                              handle,
                              M, N, K,
                              alpha,
                              (TYPE**)(transA == KBLAS_NoTrans ? Au : Av), (transA == KBLAS_NoTrans ? ldAu : ldAv ), (transA == KBLAS_NoTrans ? strideAu : strideAv),
                              (TYPE**)(transA == KBLAS_NoTrans ? Av : Au), (transA == KBLAS_NoTrans ? ldAv : ldAu ), (transA == KBLAS_NoTrans ? strideAv : strideAu), kA,
                              (TYPE**)(transB == KBLAS_NoTrans ? Bu : Bv), (transB == KBLAS_NoTrans ? ldBu : ldBv ), (transB == KBLAS_NoTrans ? strideBu : strideBv),
                              (TYPE**)(transB == KBLAS_NoTrans ? Bv : Bu), (transB == KBLAS_NoTrans ? ldBv : ldBu ), (transB == KBLAS_NoTrans ? strideBv : strideBu), kB,
                              beta,
                              (TYPE**)C,  ldC,  (long)0,
                              batchCount);
}

int kblas_gemm_lr_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int M, const int N, const int K,
                          const TYPE alpha,
                          const TYPE** Au_array, int ldAu,
                          const TYPE** Av_array, int ldAv, int kA,
                          const TYPE** Bu_array, int ldBu,
                          const TYPE** Bv_array, int ldBv, int kB,
                          const TYPE beta,
                                TYPE** C_array, int ldC,
                          int batchCount)
{
  return Xgemm_lr_batch( handle,
                          transA, transB,
                          M, N, K,
                          alpha,
                          (TYPE**)Au_array, ldAu, (long)0,
                          (TYPE**)Av_array, ldAv, (long)0, kA,
                          (TYPE**)Bu_array, ldBu, (long)0,
                          (TYPE**)Bv_array, ldBv, (long)0, kB,
                          beta,
                          (TYPE**)C_array,  ldC,  (long)0,
                          batchCount);
}

extern "C"
int kblasXgemm_lr_lld_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int M, const int N, const int K,
                          const TYPE alpha,
                          const TYPE** Au_array, int ldAu,
                          const TYPE** Av_array, int ldAv, int kA,
                          const TYPE** Bu_array, int ldBu,
                          const TYPE** Bv_array, int ldBv, int kB,
                          const TYPE beta,
                                TYPE** C_array, int ldC,
                          int batchCount)
{
  return Xgemm_lr_batch( handle,
                          transA, transB,
                          M, N, K,
                          alpha,
                          (TYPE**)Au_array, ldAu, (long)0,
                          (TYPE**)Av_array, ldAv, (long)0, kA,
                          (TYPE**)Bu_array, ldBu, (long)0,
                          (TYPE**)Bv_array, ldBv, (long)0, kB,
                          beta,
                          (TYPE**)C_array,  ldC,  (long)0,
                          batchCount);
}

//----------------------------------------------------------------------------------------------
extern "C"
void kblasXgemm_lr_lld_batch_wsquery(kblasHandle_t handle, const int N, int kA, int kB, int batchCount){
  gemm_lr_batch_wsquery_core<TYPE, false>(N, kA, kB, batchCount, &(handle->work_space.requested_ws_state));
}

//==============================================================================================
// Batched Non-Strided Uniform GEMM-TLR
//==============================================================================================
int Xgemm_lr_batch(kblasHandle_t handle,
                    char transA, char transB,
                    int M, int N, int K,
                    TYPE alpha,
                    TYPE** Au, int ldAu, long strideAu,
                    TYPE** Av, int ldAv, long strideAv, int kA,
                    TYPE** Bu, int ldBu, long strideBu,
                    TYPE** Bv, int ldBv, long strideBv, int kB,
                    TYPE beta,
                    TYPE** Cu, int ldCu, long strideCu,
                    TYPE** Cv, int ldCv, long strideCv, int& kC,
                    int max_rk, double max_acc,
                    int batchCount)
{
  int TransA = (transA == KBLAS_Trans );
  int TransB = (transB == KBLAS_Trans);

  return Xgemm_LR_batch_uniform_core<TYPE, TYPE**, false>(
                                      handle,
                                      M, N, K,
                                      alpha,
                                      (TYPE**)(TransA ? Av : Au), (TransA ? ldAv : ldAu), (TransA ? strideAv : strideAu),
                                      (TYPE**)(TransA ? Au : Av), (TransA ? ldAu : ldAv), (TransA ? strideAu : strideAv), kA,
                                      (TYPE**)(TransB ? Bv : Bu), (TransB ? ldBv : ldBu), (TransB ? strideBv : strideBu),
                                      (TYPE**)(TransB ? Bu : Bv), (TransB ? ldBu : ldBv), (TransB ? strideBu : strideBv), kB,
                                      beta, (TYPE**)Cu, ldCu, strideCu,
                                            (TYPE**)Cv, ldCv, strideCv, kC,
                                      max_rk, max_acc, batchCount);
}

int kblas_gemm_lr_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int M, const int N, const int K,
                          const TYPE alpha,
                          const TYPE** Au, int ldAu, long strideAu,
                          const TYPE** Av, int ldAv, long strideAv, int kA,
                          const TYPE** Bu, int ldBu, long strideBu,
                          const TYPE** Bv, int ldBv, long strideBv, int kB,
                          const TYPE beta,
                                TYPE** Cu, int ldCu, long strideCu,
                                TYPE** Cv, int ldCv, long strideCv, int& kC,
                          int max_rk, double max_acc,
                          int batchCount)
{
  return Xgemm_lr_batch( handle,
                          transA, transB,
                          M, N, K,
                          alpha,
                          (TYPE**)Au, ldAu, strideAu,
                          (TYPE**)Av, ldAv, strideAv, kA,
                          (TYPE**)Bu, ldBu, strideBu,
                          (TYPE**)Bv, ldBv, strideBv, kB,
                          beta,
                          (TYPE**)Cu, ldCu, strideCu,
                          (TYPE**)Cv, ldCv, strideCv,  kC,
                          max_rk, max_acc,
                          batchCount);
}


extern "C"
int kblasXgemm_lr_lll_batch( kblasHandle_t handle,
                          char transA, char transB,
                          const int M, const int N, const int K,
                          const TYPE alpha,
                          const TYPE** Au, int ldAu, long strideAu,
                          const TYPE** Av, int ldAv, long strideAv, int kA,
                          const TYPE** Bu, int ldBu, long strideBu,
                          const TYPE** Bv, int ldBv, long strideBv, int kB,
                          const TYPE beta,
                                TYPE** Cu, int ldCu, long strideCu,
                                TYPE** Cv, int ldCv, long strideCv, int& kC,
                          int max_rk, double max_acc,
                          int batchCount)
{
  return Xgemm_lr_batch( handle,
                          transA, transB,
                          M, N, K,
                          alpha,
                          (TYPE**)Au, ldAu, strideAu,
                          (TYPE**)Av, ldAv, strideAv, kA,
                          (TYPE**)Bu, ldBu, strideBu,
                          (TYPE**)Bv, ldBv, strideBv, kB,
                          beta,
                          (TYPE**)Cu, ldCu, strideCu,
                          (TYPE**)Cv, ldCv, strideCv,  kC,
                          max_rk, max_acc,
                          batchCount);
}

//----------------------------------------------------------------------------------------------
extern "C"
void kblasXgemm_lr_lll_batch_wsquery(kblasHandle_t handle,
                                  const int M, const int N,
                                  int kA, int kB, int kC, int max_rk,
                                  int batchCount)
{
  gemm_lr_batch_wsquery_core<TYPE, false>( M, N,
                                            kA, kB, kC, max_rk,
                                            batchCount,
                                            &(handle->work_space.requested_ws_state));
}



//==============================================================================================
// Tiled Non-Strided Uniform GEMM-TLR device pointers
//==============================================================================================
// workspace needed: device data + pointers
// d_Au, d_Av, d_Bu, d_Bv, C: host pointer to array of device pointers to device buffers
int kblas_gemm_tlr( kblasHandle_t handle,
                          char transA, char transB,
                          const int MTiles, const int NTiles, const int KTiles,
                          const int mb, const int nb, const int kb,
                          const TYPE alpha,
                          const TYPE** d_Au, int ldAu,
                          const TYPE** d_Av, int ldAv, int ld_Aptrs, int kA,
                          const TYPE** d_Bu, int ldBu,
                          const TYPE** d_Bv, int ldBv, int ld_Bptrs, int kB,
                          const TYPE beta,
                                TYPE* C, int ldC)
{
  return Xgemm_TLR_core<TYPE>(
                              handle,
                              transA, transB,
                              MTiles, NTiles, KTiles,
                              mb, nb, kb,
                              alpha,
                              d_Au, ldAu,
                              d_Av, ldAv, ld_Aptrs, kA,
                              d_Bu, ldBu,
                              d_Bv, ldBv, ld_Bptrs, kB,
                              beta, C, ldC);
}

//----------------------------------------------------------------------------------------------
extern "C"
void kblasXgemm_tlr_lld_wsquery(kblasHandle_t handle,
                                  const int MTiles, const int NTiles, int kA, int kB,
                                  const int mb, const int nb)
{
  gemm_tlr_wsquery_core<TYPE, true, true>( MTiles, NTiles, kA, kB, mb, nb, &(handle->work_space.requested_ws_state) );
}

//----------------------------------------------------------------------------------------------
extern "C"
void kblasXgemm_plr_dev_tiled_wsquery(kblasHandle_t handle,
                                      const int MTiles, const int NTiles, int kA, int kB,
                                      const int mb, const int nb)
{
  gemm_tlr_wsquery_core<TYPE, false, true>( MTiles, NTiles, kA, kB, mb, nb, &(handle->work_space.requested_ws_state) );
}


//==============================================================================================
// Tiled Non-Strided Uniform GEMM-TLR
//==============================================================================================
// d_Au, d_Av, d_Bu, d_Bv, C: host pointer to array of device pointers to device buffers
int Xgemm_tlr(kblasHandle_t handle,
                    char transA, char transB,
                    int MTiles, int NTiles, int KTiles,
                    int mb, int nb, int kb,
                    TYPE alpha,
                    TYPE** d_Au, int ldAu,
                    TYPE** d_Av, int ldAv, int ld_Aptrs, int kA,
                    TYPE** d_Bu, int ldBu,
                    TYPE** d_Bv, int ldBv, int ld_Bptrs, int kB,
                    TYPE beta,
                    TYPE** d_Cu, int ldCu,
                    TYPE** d_Cv, int ldCv, int ld_Cptrs, int& kC,
                    int max_rk, double max_acc)
{
  return Xgemm_TLR_core<TYPE>(
                              handle,
                              transA, transB,
                              MTiles, NTiles, KTiles,
                              mb, nb, kb,
                              alpha,
                              d_Au, ldAu,
                              d_Av, ldAv, ld_Aptrs, kA,
                              d_Bu, ldBu,
                              d_Bv, ldBv, ld_Bptrs, kB,
                              beta,
                              d_Cu, ldCu,
                              d_Cv, ldCv, ld_Cptrs, kC,
                              max_rk, max_acc);
}

int kblas_gemm_tlr( kblasHandle_t handle,
                          char transA, char transB,
                          int MTiles, int NTiles, int KTiles,
                          int mb, int nb, int kb,
                          TYPE alpha,
                          TYPE** d_Au, int ldAu,
                          TYPE** d_Av, int ldAv, int ld_Aptrs, int kA,
                          TYPE** d_Bu, int ldBu,
                          TYPE** d_Bv, int ldBv, int ld_Bptrs, int kB,
                          TYPE beta,
                          TYPE** d_Cu, int ldCu,
                          TYPE** d_Cv, int ldCv, int ld_Cptrs, int& kC,
                          int max_rk, double max_acc)
{
  return Xgemm_tlr( handle,
                          transA, transB,
                          MTiles, NTiles, KTiles,
                          mb, nb, kb,
                          alpha,
                          d_Au, ldAu,
                          d_Av, ldAv, ld_Aptrs, kA,
                          d_Bu, ldBu,
                          d_Bv, ldBv, ld_Bptrs, kB,
                          beta,
                          d_Cu, ldCu,
                          d_Cv, ldCv, ld_Cptrs, kC,
                          max_rk, max_acc);
}

//----------------------------------------------------------------------------------------------
extern "C"
void kblasXgemm_tlr_lll_wsquery(kblasHandle_t handle,
                                  const int MTiles, const int NTiles,
                                  int kA, int kB, int kC, int max_rk,
                                  const int mb, const int nb)
{
  gemm_tlr_wsquery_core<TYPE, true>(MTiles, NTiles,
                                          kA, kB, kC, max_rk,
                                          mb, nb,
                                          &(handle->work_space.requested_ws_state) );
}
