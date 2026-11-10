/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_svd/Xaca_batch.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ali Charara
 * @date 2020-12-10
 **/

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include "kblas.h"
#include "kblas_operators.h"

#include "kblas_struct.h"
#include "kblas_prec_def.h"
#include "kblas_gpu_util.ch"

#include "workspace_queries.ch"
#include "Xblas_core.ch"

#include "Xaca_batch_core.cuh"

//==============================================================================================

extern "C"
int kblas_ACAf( int m, int n,
                TYPE* A, int lda,
                TYPE* U, int ldu,
                TYPE* V, int ldv,
                TYPE* S,
                double maxacc, int maxrk,
                double* acc, int* rk)
{
  return ACAf1(m, n,
              A, lda,
              U, ldu,
              V, ldv,
              S,
              maxacc, maxrk,
              acc, rk);
}

// extern "C"
int kblas_acaf_gpu( kblasHandle_t handle,
                    int m, int n,
                    TYPE* A, int lda,
                    TYPE* U, int ldu,
                    TYPE* V, int ldv,
                    TYPE* S,
                    double maxacc, int maxrk,
                    double* acc, TYPE* rk)
{
  int mWarps = (m+31)/32, nWarps = (n+31)/32;
  int shared_size = (m + mWarps + 2)*sizeof(int);

  dim3 block(kmax(mWarps,nWarps)*32,1);
  dim3 grid(1,1);


  kernel_ACAf<TYPE><<<grid, block, shared_size, handle->stream>>>
              (m, n,
              A, lda,
              U, ldu,
              V, ldv,
              S,
              maxacc, maxrk,
              acc, rk);
  return KBLAS_Success;
}

//==============================================================================================
int Xacaf_batch(kblasHandle_t handle,
                int m, int n,
                TYPE** A, int lda, long strideA,
                TYPE** U, int ldu, long strideU,
                TYPE** V, int ldv, long strideV,
                TYPE** S, int lds, long strideS,
                double maxacc, int maxrk,
                double* acc, int* rk,
                int batchCount)
{
  int mWarps = (m+31)/32, nWarps = (n+31)/32;
  int shared_size = (m + mWarps + 2)*sizeof(int);

  dim3 block(kmax(mWarps,nWarps)*32,1);
  dim3 grid(batchCount,1);


  kernel_ACAf_batch_U<TYPE, TYPE**><<<grid, block, shared_size, handle->stream>>>
                                  (m, n,
                                  A, lda, strideA,
                                  U, ldu, strideU,
                                  V, ldv, strideV,
                                  S, lds, strideS,
                                  maxacc, maxrk,
                                  acc, rk);
  return KBLAS_Success;
}

int kblas_acaf_batch( kblasHandle_t handle,
                      int m, int n,
                      TYPE** A, int lda,
                      TYPE** U, int ldu,
                      TYPE** V, int ldv,
                      TYPE** S, int lds,
                      double maxacc, int maxrk,
                      double* acc, int* rk,
                      int batchCount)
{
  return Xacaf_batch( handle,
                      m, n,
                      A, lda, 0,
                      U, ldu, 0,
                      V, ldv, 0,
                      S, lds, 0,
                      maxacc, maxrk,
                      acc, rk,
                      batchCount);
}

extern "C"
int kblas_Xacaf_batch(kblasHandle_t handle,
                      int m, int n,
                      TYPE** A, int lda,
                      TYPE** U, int ldu,
                      TYPE** V, int ldv,
                      TYPE** S, int lds,
                      double maxacc, int maxrk,
                      double* acc, int* rk,
                      int batchCount)
{
  return Xacaf_batch( handle,
                      m, n,
                      A, lda, 0,
                      U, ldu, 0,
                      V, ldv, 0,
                      S, lds, 0,
                      maxacc, maxrk,
                      acc, rk,
                      batchCount);
}


//==============================================================================================
int Xacaf_batch(kblasHandle_t handle,
                int* m, int* n,
                int max_m, int max_n,
                TYPE** A, int* lda, long strideA,
                TYPE** U, int* ldu, long strideU,
                TYPE** V, int* ldv, long strideV,
                TYPE** S, int* lds, long strideS,
                double maxacc, int maxrk,
                double* acc, int* rk,
                int batchCount)
{
  int mWarps = (max_m+31)/32, nWarps = (max_n+31)/32;
  int shared_size = (max_m + mWarps + 2)*sizeof(int);

  dim3 block(kmax(mWarps,nWarps)*32,1);
  dim3 grid(batchCount,1);

  kernel_ACAf_batch_N<TYPE, TYPE**, int*><<<grid, block, shared_size, handle->stream>>>
                                        (m, n,
                                        A, lda, strideA,
                                        U, ldu, strideU,
                                        V, ldv, strideV,
                                        S, lds, strideS,
                                        maxacc, maxrk,
                                        acc, rk);
  return KBLAS_Success;
}

int kblas_acaf_vbatch(kblasHandle_t handle,
                      int* m, int* n,
                      int max_m, int max_n,
                      TYPE** A, int* lda,
                      TYPE** U, int* ldu,
                      TYPE** V, int* ldv,
                      TYPE** S, int* lds,
                      double maxacc, int maxrk,
                      double* acc, int* rk,
                      int batchCount)
{
  return Xacaf_batch( handle,
                      m, n,
                      max_m, max_n,
                      A, lda, 0,
                      U, ldu, 0,
                      V, ldv, 0,
                      S, lds, 0,
                      maxacc, maxrk,
                      acc, rk,
                      batchCount);
}

extern "C"
int kblas_Xacaf_vbatch( kblasHandle_t handle,
                        int* m, int* n,
                        int max_m, int max_n,
                        TYPE** A, int* lda,
                        TYPE** U, int* ldu,
                        TYPE** V, int* ldv,
                        TYPE** S, int* lds,
                        double maxacc, int maxrk,
                        double* acc, int* rk,
                        int batchCount)
{
  return Xacaf_batch( handle,
                      m, n,
                      max_m, max_n,
                      A, lda, 0,
                      U, ldu, 0,
                      V, ldv, 0,
                      S, lds, 0,
                      maxacc, maxrk,
                      acc, rk,
                      batchCount);
}


//==============================================================================================
int Xacaf_batch(kblasHandle_t handle,
                int m, int n,
                TYPE* A, int lda, long strideA,
                TYPE* U, int ldu, long strideU,
                TYPE* V, int ldv, long strideV,
                TYPE* S, int lds, long strideS,
                double maxacc, int maxrk,
                double* acc, int* rk,
                int batchCount)
{
  int mWarps = (m+31)/32, nWarps = (n+31)/32;
  int shared_size = (m + mWarps + 2)*sizeof(int);

  dim3 block(kmax(mWarps,nWarps)*32,1);
  dim3 grid(batchCount,1);

  kernel_ACAf_batch_U<TYPE, TYPE*><<<grid, block, shared_size, handle->stream>>>
                                  (m, n,
                                  A, lda, strideA,
                                  U, ldu, strideU,
                                  V, ldv, strideV,
                                  S, lds, strideS,
                                  maxacc, maxrk,
                                  acc, rk);
  return KBLAS_Success;
}

int kblas_acaf_batch( kblasHandle_t handle,
                      int m, int n,
                      TYPE* A, int lda, long strideA,
                      TYPE* U, int ldu, long strideU,
                      TYPE* V, int ldv, long strideV,
                      TYPE* S, int lds, long strideS,
                      double maxacc, int maxrk,
                      double* acc, int* rk,
                      int batchCount)
{
  return Xacaf_batch( handle,
                      m, n,
                      A, lda, strideA,
                      U, ldu, strideU,
                      V, ldv, strideV,
                      S, lds, strideS,
                      maxacc, maxrk,
                      acc, rk,
                      batchCount);
}

extern "C"
int kblas_Xacaf_batch_strided(kblasHandle_t handle,
                              int m, int n,
                              TYPE* A, int lda, long strideA,
                              TYPE* U, int ldu, long strideU,
                              TYPE* V, int ldv, long strideV,
                              TYPE* S, int lds, long strideS,
                              double maxacc, int maxrk,
                              double* acc, int* rk,
                              int batchCount)
{
  return Xacaf_batch( handle,
                      m, n,
                      A, lda, strideA,
                      U, ldu, strideU,
                      V, ldv, strideV,
                      S, lds, strideS,
                      maxacc, maxrk,
                      acc, rk,
                      batchCount);
}

//==============================================================================================
int Xacaf_batch(kblasHandle_t handle,
                int* m, int* n,
                int max_m, int max_n,
                TYPE* A, int lda, long strideA,
                TYPE* U, int ldu, long strideU,
                TYPE* V, int ldv, long strideV,
                TYPE* S, int lds, long strideS,
                double maxacc, int maxrk,
                double* acc, int* rk,
                int batchCount)
{
  int mWarps = (max_m+31)/32, nWarps = (max_n+31)/32;
  int shared_size = (max_n + mWarps + 2)*sizeof(int);

  dim3 block(kmax(mWarps,nWarps)*32,1);
  dim3 grid(batchCount,1);

  kernel_ACAf_batch_N<TYPE, TYPE*, int><<<grid, block, shared_size, handle->stream>>>
                                      (m, n,
                                      A, lda, strideA,
                                      U, ldu, strideU,
                                      V, ldv, strideV,
                                      S, lds, strideS,
                                      maxacc, maxrk,
                                      acc, rk);
  return KBLAS_Success;
}

int kblas_acaf_vbatch(kblasHandle_t handle,
                              int* m, int* n,
                              int max_m, int max_n,
                              TYPE* A, int lda, long strideA,
                              TYPE* U, int ldu, long strideU,
                              TYPE* V, int ldv, long strideV,
                              TYPE* S, int lds, long strideS,
                              double maxacc, int maxrk,
                              double* acc, int* rk,
                              int batchCount)
{
  return Xacaf_batch( handle,
                      m, n,
                      max_m, max_n,
                      A, lda, strideA,
                      U, ldu, strideU,
                      V, ldv, strideV,
                      S, lds, strideS,
                      maxacc, maxrk,
                      acc, rk,
                      batchCount);
}

extern "C"
int kblas_Xacaf_vbatch_strided(kblasHandle_t handle,
                              int* m, int* n,
                              int max_m, int max_n,
                              TYPE* A, int lda, long strideA,
                              TYPE* U, int ldu, long strideU,
                              TYPE* V, int ldv, long strideV,
                              TYPE* S, int lds, long strideS,
                              double maxacc, int maxrk,
                              double* acc, int* rk,
                              int batchCount)
{
  return Xacaf_batch( handle,
                      m, n,
                      max_m, max_n,
                      A, lda, strideA,
                      U, ldu, strideU,
                      V, ldv, strideV,
                      S, lds, strideS,
                      maxacc, maxrk,
                      acc, rk,
                      batchCount);
}
