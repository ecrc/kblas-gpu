 /**
  - -* (C) Copyright 2013 King Abdullah University of Science and Technology
  Authors:
  Ali Charara (ali.charara@kaust.edu.sa)
  David Keyes (david.keyes@kaust.edu.sa)
  Hatem Ltaief (hatem.ltaief@kaust.edu.sa)
  
  Redistribution  and  use  in  source and binary forms, with or without
  modification,  are  permitted  provided  that the following conditions
  are met:
  
  * Redistributions  of  source  code  must  retain  the above copyright
  * notice,  this  list  of  conditions  and  the  following  disclaimer.
  * Redistributions  in  binary  form must reproduce the above copyright
  * notice,  this list of conditions and the following disclaimer in the
  * documentation  and/or other materials provided with the distribution.
  * Neither  the  name of the King Abdullah University of Science and
  * Technology nor the names of its contributors may be used to endorse
  * or promote products derived from this software without specific prior
  * written permission.
  * 
  THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
  LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  **/
#include <stdlib.h>
#include <stdio.h>
#include <set>
#include <cublas.h>
#include "kblas.h"
#include "Xtr_common.cuh"
#include "operators.h"

//==============================================================================================

cublasStatus_t cublasXtrmm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const float *alpha,
                           const float *A, int lda,
                                 float *B, int ldb){
  return cublasStrmm(handle,
                     side, uplo, transa, diag,
                     m, n,
                     alpha, A, lda,
                            B, ldb );
}

cublasStatus_t cublasXtrmm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t      diag,
                           int m, int n,
                           const double *alpha,
                           const double *A, int lda,
                                 double *B, int ldb){
  return cublasDtrmm(handle,
                     side, uplo, transa, diag,
                     m, n,
                     alpha, A, lda,
                            B, ldb );
}
cublasStatus_t cublasXtrmm (cublasHandle_t handle,
                            cublasSideMode_t side, cublasFillMode_t uplo,
                            cublasOperation_t trans, cublasDiagType_t diag,
                            int m, int n,
                            const cuComplex *alpha,
                            const cuComplex *A, int lda,
                                  cuComplex *B, int ldb){
  return cublasCtrmm(handle,
                     side, uplo, transa, diag,
                     m, n,
                     alpha, A, lda,
                            B, ldb );
}
cublasStatus_t cublasXtrmm (cublasHandle_t handle,
                            cublasSideMode_t side, cublasFillMode_t uplo,
                            cublasOperation_t trans, cublasDiagType_t diag,
                            int m, int n,
                            const cuDoubleComplex *alpha,
                            const cuDoubleComplex *A, int lda,
                                  cuDoubleComplex *B, int ldb){
  return cublasCtrmm(handle,
                     side, uplo, transa, diag,
                     m, n,
                     alpha, A, lda,
                            B, ldb );
}

#define Xgemm cublasXgemm
#define Xtrmm cublasXtrmm

//==============================================================================================
int kblas_trmm_ib_custom = 128;
int kblas_trmm_ib_cublas = 128;
bool kblas_trmm_use_custom = 0;
#define SIMPLE_SIZE_CUSTOM(n) ( ((n)<32) || ((n) % 32 == 0 && (n) <= kblas_trmm_ib_custom) )
#define SIMPLE_SIZE(n) ( (n) <= kblas_trmm_ib_cublas) )
//==============================================================================================
#define WARP 32
#define WARP1 33
#define tx threadIdx.x
#define ty threadIdx.y
//==============================================================================================
__device__ __inline__ float shfl(float x, int lane, int ws = 32)
{
  return __shfl(x, lane, ws);
}
__device__ __inline__ double shfl(double x, int lane, int ws = 32)
{
  // Split the double number into 2 32b registers.
  int lo = __double2loint(x), hi = __double2hiint(x);
  // Shuffle the two 32b registers.
  lo = __shfl(lo, lane, ws);
  hi = __shfl(hi, lane, ws);
  // Recreate the 64b number.
  return __hiloint2double(hi,lo);
}
__device__ __inline__ cuComplex shfl(cuComplex x, int lane, int ws = 32)
{
  return make_cuFloatComplex( __shfl(x.x, lane, ws), __shfl(x.y, lane, ws) );
}
__device__ __inline__ cuDoubleComplex shfl(cuDoubleComplex x, int lane, int ws = 32)
{
  return make_cuDoubleComplex( shfl(x.x, lane, ws), shfl(x.y, lane, ws) );
}
//==============================================================================================
template<typename T, int WARPS_PER_BLOCK, int B_COLS_PER_WARP/*TODO*/, bool LEFT, bool LOWER, bool TRANS, bool UNIT/*TODO*/, bool CONJG/*TODO*/>
__global__ void __launch_bounds__(256)
trmm_mul32_sb(int M, int N, T alpha, const T* A, int incA, T* B, int incB, int mb){
  
  int txyw = tx + ty*WARP1, txyiA = tx + ty*incA, txyiB = tx + ty*incB;
  
  //setup shared memory
  __shared__ T sA[WARP * WARP1];//strided to avoid bank conflict
  T rB, s;
  int b = 0, c, j, a;
  const A_COL_PER_WARP = WARP / WARPS_PER_BLOCK;
  if(LEFT){/*TODO*/
    B += blockIdx.x * WARPS_PER_BLOCK * incB;
    const bool forward = (LEFT && (LOWER == TRANS)) || (!LEFT && (LOWER != TRANS));
    const bool active = true/*TODO*/;


    for( c = (forward ? 0 : mb-1); (forward && (c < mb)) || (!forward && (c > -1)); c += (forward : 1 : -1))
    {
      s = make_zero<T>();
      //load A(c,c) from global to shared mem
      #pragma unroll
      for(l = 0; l < A_COL_PER_WARP; l++){
        sA[txyw + l * WARPS_PER_BLOCK * WARP1] = A[txyiA + WARP * c * (incA+1) + l * WARPS_PER_BLOCK * incA];
      }
      //load B(c) into registers
      rB = B[txyiB + WARP * c];
      __syncthreads();

      //perform trmm on shared mem
      if(LOWER == TRANS){
        #pragma unroll
        for(j = 0; j < WARP; j++){
          if(j >= tx){
            s = FMA( sA[j + tx * WARP1], shfl(rB, j), s);/*TODO*/
          }
        }
      }else{
        #pragma unroll
        for(j = WARP-1; j > -1; j--){
          if(j <= tx){
            s = FMA( sA[tx + j * WARP1], shfl(rB, j), s);
          }
        }
      }
      __syncthreads();

      for(r = (forward ? c+1 : 0); (forward && (r < mb)) || (!forward && (r > c)); r++){
        #pragma unroll
        for(l = 0; l < A_COL_PER_WARP; l++){
          if(TRANS)//load A(r,c)
            sA[txyw + l * WARPS_PER_BLOCK * WARP1] = A[txyiA + WARP * (r + c * incA) + l * WARPS_PER_BLOCK * incA];
          else//load A(c,r)
            sA[txyw + l * WARPS_PER_BLOCK * WARP1] = A[txyiA + WARP * (c + r * incA) + l * WARPS_PER_BLOCK * incA];
        }
        //load B(r)
        rB = B[txyiB + WARP * r];
        __syncthreads();

        //gemm A(r,c)|A(c,r) & B(r) onto B(c) held at s
        #pragma unroll
        for(j = 0; j < WARP; j++){
          if(TRANS)
            s = FMA( sA[j + tx * WARP1], shfl(rB, j), s);/*TODO*/
          else
            s = FMA( sA[tx + j * WARP1], shfl(rB, j), s);
        }
        __syncthreads();
      }
      //store back B(c) to global mem
      B[txyiB + WARP * c] = alpha * s;
    }
  }
}
//==============================================================================================
template<class T>
cublasStatus_t Xtrmm(cublasHandle_t handle,
                     cublasSideMode_t side, cublasFillMode_t uplo,
                     cublasOperation_t trans, cublasDiagType_t diag,
                     int m, int n,
                     const T *alpha, const T *A, int incA,
                                           T *B, int incB){
  return cublasXtrmm(handle,
                     side, uplo, trans, diag,
                     m, n,
                     alpha, A, incA,
                            B, incB );
}
cublasStatus_t Xtrmm(cublasHandle_t handle,
                     cublasSideMode_t side, cublasFillMode_t uplo,
                     cublasOperation_t trans, cublasDiagType_t diag,
                     int m, int n,
                     const double *alpha, const double *A, int incA,
                                                double *B, int incB){

  void (*trmm_kernel)(int M, int N, T alpha, const T* A, int incA, T* B, int incB, int mb);

  #define WARPS_PER_BLOCK 8
  #define B_COLS_PER_WARP 1
  trmm_kernel trmm_kernels[4] = {// T, WARPS_PER_BLOCK, B_COLS_PER_WARP, LEFT, LOWER, TRANS, UNIT, CONJG
    trmm_mul32_sb<double, WARPS_PER_BLOCK, B_COLS_PER_WARP, true,  true,  true, true, true>,
    trmm_mul32_sb<double, WARPS_PER_BLOCK, B_COLS_PER_WARP, true,  true, false, true, true>,
    trmm_mul32_sb<double, WARPS_PER_BLOCK, B_COLS_PER_WARP, true, false,  true, true, true>,
    trmm_mul32_sb<double, WARPS_PER_BLOCK, B_COLS_PER_WARP, true, false, false, true, true>
  };
  
  cudaStream_t curStream;
  cublasStatus_t status;
  if(!kblas_trmm_use_custom){
    return cublasXtrmm(handle,
                      side, uplo, trans, diag,
                      m, n,
                      &alpha, A, incA,
                              B, incB );
  }

  if((status = cublasGetStream( handle, &curStream )) != CUBLAS_STATUS_SUCCESS ) return status;
  
  if(side == CUBLAS_SIDE_RIGHT){
    return cublasXtrmm(handle,
                       side, uplo, trans, diag,
                       m, n,
                       alpha, A, incA,
                              B, incB );
  }else
  if(side == CUBLAS_SIDE_LEFT){
    if(m % WARP == 0 && n % WARPS_PER_BLOCK == 0)
    {
      dim3 dimBlock( WARP, WARPS_PER_BLOCK );
      trmm_kernels<<< dim3( n / WARPS_PER_BLOCK, 1), dimBlock, 0, curStream>>> (m, n, *alpha, A, incA, B, incB, m / WARP);
      if(!_kblas_error( (cudaGetLastError()), __func__, __FILE__, __LINE__ ))
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }else{/*TODO*/}
  }
  return CUBLAS_STATUS_SUCCESS;
}

//==============================================================================================

#include "Xtrmm.hxx"

//==============================================================================================

extern "C" {
  /*
int kblas_strmm_async(
  char side, char uplo, char trans, char diag,
  int m, int n,
  float alpha, const float *A, int incA,
                       float *B, int incB,
  cudaStream_t    stream){

  return kblasXtrmm(
    side, uplo, trans, diag,
    m, n,
    alpha, A, incA,
           B, incB,
    stream);
}
int kblas_dtrmm_async(
  char side, char uplo, char trans, char diag,
  int m, int n,
  double alpha, const double *A, int incA,
                        double *B, int incB,
  cudaStream_t    stream){

  return kblasXtrmm(
    side, uplo, trans, diag,
    m, n,
    alpha, A, incA,
           B, incB,
    stream);
}
int kblas_ctrmm_async(
  char side, char uplo, char trans, char diag,
  int m, int n,
  cuComplex alpha, const cuComplex *A, int incA,
                          cuComplex *B, int incB,
  cudaStream_t    stream){

  return kblasXtrmm(
    side, uplo, trans, diag,
    m, n,
    alpha, A, incA,
           B, incB,
    stream);
}
int kblas_ztrmm_async(
  char side, char uplo, char trans, char diag,
  int m, int n,
  cuDoubleComplex alpha, const cuDoubleComplex *A, int incA,
                                cuDoubleComplex *B, int incB,
  cudaStream_t    stream){

  return kblasXtrmm(
    side, uplo, trans, diag,
    m, n,
    alpha, A, incA,
           B, incB,
    stream);
}* /
  
int kblas_strmm(
  char side, char uplo, char trans, char diag,
  int m, int n,
  float alpha, const float *A, int incA,
                      float *B, int incB){
  return kblasXtrmm(
    side, uplo, trans, diag,
    m, n,
    alpha, A, incA,
           B, incB,
    0);
}
int kblas_dtrmm(
  char side, char uplo, char trans, char diag,
  int m, int n,
  double alpha, const double *A, int incA,
                        double *B, int incB){
  return kblasXtrmm(
    side, uplo, trans, diag,
    m, n,
    alpha, A, incA,
           B, incB,
    0);
}
int kblas_ctrmm(
  char side, char uplo, char trans, char diag,
  int m, int n,
  cuComplex alpha, const cuComplex *A, int incA,
                          cuComplex *B, int incB){
  return kblasXtrmm(
    side, uplo, trans, diag,
    m, n,
    alpha, A, incA,
           B, incB,
    0);
}
int kblas_ztrmm(
  char side, char uplo, char trans, char diag,
  int m, int n,
  cuDoubleComplex alpha, const cuDoubleComplex *A, int incA,
                                cuDoubleComplex *B, int incB){
  return kblasXtrmm(
    side, uplo, trans, diag,
    m, n,
    alpha, A, incA,
           B, incB,
    0);
}*/

cublasStatus_t kblasStrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const float *alpha,
                          const float *A, int lda,
                                float *B, int ldb){
  return kblasXtrmm(handle
                    side, uplo, trans, diag,
                    m, n,
                    *alpha, A, lda,
                            B, ldb);
}
cublasStatus_t kblasDtrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const double *alpha,
                          const double *A, int lda,
                                double *B, int ldb){
  return kblasXtrmm(handle
                    side, uplo, trans, diag,
                    m, n,
                    *alpha, A, lda,
                            B, ldb);
}
cublasStatus_t kblasCtrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const cuComplex *alpha,
                          const cuComplex *A, int lda,
                                cuComplex *B, int ldb){
  return kblasXtrmm(handle
                    side, uplo, trans, diag,
                    m, n,
                    *alpha, A, lda,
                            B, ldb);
}
cublasStatus_t kblasZtrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const cuDoubleComplex *alpha,
                          const cuDoubleComplex *A, int lda,
                                cuDoubleComplex *B, int ldb){
  return kblasXtrmm(handle
                    side, uplo, trans, diag,
                    m, n,
                    *alpha, A, lda,
                            B, ldb);
}

}


