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
  T *HIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
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
#ifdef SUPPORT_CUBLAS

void cublasXtrmm( char side, char uplo, char transa, char diag,
                 int m, int n,
                 float alpha, const float *A, int lda,
                                     float *B, int ldb ){
  cublasStrmm(side, uplo, transa, diag,
              m, n,
              alpha, A, lda,
                     B, ldb );
}

void cublasXtrmm( char side, char uplo, char transa, char diag,
                 int m, int n,
                 double alpha, const double *A, int lda,
                                       double *B, int ldb ){
  cublasDtrmm(side, uplo, transa, diag,
              m, n,
              alpha, A, lda,
                     B, ldb );
}
void cublasXtrmm ( char side, char uplo, char transa, char diag,
                  int m, int n,
                  cuComplex alpha, const cuComplex *A, int lda,
                                          cuComplex *B, int ldb){
  cublasCtrmm(side, uplo, transa, diag,
              m, n,
              alpha, A, lda,
                     B, ldb );
}
void cublasXtrmm ( char side, char uplo, char transa, char diag,
                  int m, int n,
                  cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                                                cuDoubleComplex *B, int ldb){
  cublasZtrmm(side, uplo, transa, diag,
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
#define SIMPLE_SIZE(n) ( ((n)<32) || ((n) % 32 == 0 && (n) <= kblas_trmm_ib_cublas) )
//==============================================================================================
#define WARP 32
#define WARP1 33
#define tx threadIdx.x
#define ty threadIdx.y
//==============================================================================================
__device__ __inline__ double shfl(double x, int lane, int ws = 32)
{
  // Split the double number into 2 32b registers.
  //*
  int lo = __double2loint(x), hi = __double2hiint(x);
  //asm volatile( "mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x));
  // Shuffle the two 32b registers.
  lo = __shfl(lo, lane, ws);
  hi = __shfl(hi, lane, ws);
  // Recreate the 64b number.
  //asm volatile( "mov.b64 %0, {%1,%2};" : "=d(x)" : "r"(lo), "r"(hi));
  return __hiloint2double(hi,lo);/*/
  int2 a = *reinterpret_cast<int2*>(&x);
  a.x = __shfl(a.x, lane);
  a.y = __shfl(a.y, lane);
  return *reinterpret_cast<double*>(&a);//*/
}
//==============================================================================================
//expects 8 warps
//does not handle matrices with more than 32 rows
__global__ void
kernel_dtrmm_LLTN_32_x32_8(int M, int N, double alpha, const double* A, int incA, double* B, int incB){
  
  int txyw = tx + ty*WARP1, txyiB = tx + ty*incB, txiB = tx*incB;//, jtxw;//, txy32i = tx + (ty+32)*inc;
  double rA0, rA1, s0, s1;
  //int b = 0;
  
  // read 32x32 block of B into shared memory
  //setup shared memory
  __shared__ double sB[WARP*WARP1];
  B += blockIdx.x * WARP * incB;
  
  sB[txyw] = B[txyiB];
  sB[txyw+ 8*WARP1] = B[txyiB +  8*incB];
  sB[txyw+16*WARP1] = B[txyiB + 16*incB];
  sB[txyw+24*WARP1] = B[txyiB + 24*incB];
  __syncthreads();
  
  //repeat untill
  {
    //read A(:,ty) into registers
    rA0 = A[tx +  2*ty   *incA];
    rA1 = A[tx + (2*ty+1)*incA];
    s0 = s1 = 0.0;
    //each warp computes values of B(ty,:) by gemv A(:,ty) on B
    //#pragma unroll
    int j = 2*ty;
    for(; j < WARP-1; j++){
      s0 += sB[j   + tx*WARP1] * shfl(rA0,j  );
      s1 += sB[j+1 + tx*WARP1] * shfl(rA1,j+1);
    }
    s0 += sB[j + tx*WARP1] * shfl(rA0,j);
    
    //B[2*ty  +txiB] = alpha * s0;
    //B[2*ty+1+txiB] = alpha * s1;
    ((double2*)(B+txiB))[ty] = make_double2(alpha * s0, alpha * s1);
  }{
    //read A(:,ty) into registers
    rA0 = A[tx + (30-2*ty)*incA];
    rA1 = A[tx + (31-2*ty)*incA];
    s0 = s1 = 0.0;
    //each warp computes values of B(ty,:) by gemv A(:,ty) on B
    //#pragma unroll
    int j = WARP-1;
    s0 += sB[j + tx*WARP1] * shfl(rA0, j);
    for(; j >= 31-2*ty; j--){
      s0 += sB[j-1 + tx*WARP1] * shfl(rA0, j-1);
      s1 += sB[j   + tx*WARP1] * shfl(rA1, j);
    }
    
    //B[(30-2*ty)+txiB] = alpha * s0;
    //B[(31-2*ty)+txiB] = alpha * s1;
    ((double2*)(B+txiB))[(15-ty)] = make_double2(alpha * s0, alpha * s1);
  }
}
//==============================================================================================
//fastest so far
__global__ void __launch_bounds__(256)
kernel_dtrmm3_mul32x8_sb(int M, int N, double alpha, const double* A, int incA, double* B, int incB, int mb){
  
  int txyw = tx + ty*WARP1, txyiA = tx + ty*incA, txyiB = tx + ty*incB, jtxw;//, txy32i = tx + (ty+32)*inc;
  //int txx = 31-tx, tyy = 31-ty;
  
  //setup shared memory
  __shared__ double sA[32*33];
  double rB, s;
  int b = 0;
  B += blockIdx.x * 8 * incB;
  
  for(b = 0; b < mb; b++)
  {
    s = 0.0;
    //load A(b,b) from global to shared mem
    sA[txyw] = A[txyiA + 32*b*(incA+1)];
    sA[txyw +  8*WARP1] = A[txyiA + 32*b*(incA+1) +  8*incA];
    sA[txyw + 16*WARP1] = A[txyiA + 32*b*(incA+1) + 16*incA];
    sA[txyw + 24*WARP1] = A[txyiA + 32*b*(incA+1) + 24*incA];
    
    //load B(b) into registers
    rB = B[txyiB + 32*b];
    __syncthreads();
    
    //perform trmm on shared mem
    jtxw = tx*WARP1;
    #pragma unroll
    for(int j = 0; j < WARP; j++){
      if(j>=tx)
        s += sA[jtxw]*shfl(rB, j);
      jtxw++;
    }
    __syncthreads();
    
    for(int a = b+1; a < mb; a++){
      //load A(a,b)
      sA[txyw] = A[txyiA + 32*(a + b*incA)];
      sA[txyw +  8*WARP1] = A[txyiA + 32*(a + b*incA) +  8*incA];
      sA[txyw + 16*WARP1] = A[txyiA + 32*(a + b*incA) + 16*incA];
      sA[txyw + 24*WARP1] = A[txyiA + 32*(a + b*incA) + 24*incA];
      //load B(a)
      rB = B[txyiB + 32*a];
      __syncthreads();
      
      //gemm A(a,b) & B(a) onto B(b) held at s
      jtxw = tx*WARP1;
      #pragma unroll
      for(int j = 0; j < WARP; j++)
        s += sA[jtxw++]*shfl(rB, j);
      __syncthreads();
    }
    //store back B(b) to global mem
    B[txyiB + 32*b] = alpha * s;
  }
}
//==============================================================================================
int trmm_custom(
  char side, char uplo, char transa, char diag,
  int m, int n,
  float alpha, const float *A, int incA,
  float *B, int incB,
  cudaStream_t& curStream
){
  cublasStrmm(side, uplo, transa, diag,
              m, n,
              alpha, A, incA,
              B, incB );
  return 1;
}
int trmm_custom(
  char side, char uplo, char transa, char diag,
  int m, int n,
  double alpha, const double *A, int incA,
  double *B, int incB,
  cudaStream_t& curStream
){
  if(side == KBLAS_Right){
    if(uplo == KBLAS_Upper){
      //Right / Upper / NoTrans
      if(transa == KBLAS_NoTrans){
        cublasDtrmm(side, uplo, transa, diag,
                    m, n,
                    alpha, A, incA,
                    B, incB );
        //return 0;
      }else{
      //Right / Upper / [Conj]Trans
        cublasDtrmm(side, uplo, transa, diag,
                    m, n,
                    alpha, A, incA,
                    B, incB );
      }
    }else{
      //Right / Lower / NoTrans
      //Right / Lower / [Conj]Trans
      cublasDtrmm(side, uplo, transa, diag,
                  m, n,
                  alpha, A, incA,
                  B, incB );
    }
  }else
  if(side == KBLAS_Left){
    if(uplo == KBLAS_Upper){
      //Left / Upper / NoTrans
      //Left / Upper / [Conj]Trans
      cublasDtrmm(side, uplo, transa, diag,
                  m, n,
                  alpha, A, incA,
                  B, incB );
    }else
    if(uplo == KBLAS_Lower){
      //Left / Lower / NoTrans
      if(transa == KBLAS_NoTrans){
        cublasDtrmm(side, uplo, transa, diag,
                    m, n,
                    alpha, A, incA,
                    B, incB );
      }else
      //Left / Lower / [Conj]Trans
      if(transa == KBLAS_Trans){
          
        if(m == 32 && n % 32 == 0)
        {
          dim3 dimBlock(32,8);
          kernel_dtrmm_LLTN_32_x32_8<<< dim3(n/32,1), dimBlock, 0, curStream>>> (m, n, alpha, A, incA, B, incB);
          check_error(cudaGetLastError());
        }else
        if(m % 32 == 0 && n % 8 == 0)
        {
          dim3 dimBlock(32,8);
          kernel_dtrmm3_mul32x8_sb<<< dim3(n/8,1), dimBlock, 0, curStream>>> (m, n, alpha, A, incA, B, incB, m/32);
          check_error(cudaGetLastError());
        }
      }
    }
  }
  return 1;
}
int trmm_custom(
  char side, char uplo, char transa, char diag,
  int m, int n,
  cuComplex alpha, const cuComplex *A, int incA,
  cuComplex *B, int incB,
  cudaStream_t& curStream
){
  cublasCtrmm(side, uplo, transa, diag,
              m, n,
              alpha, A, incA,
                     B, incB );
  return 1;
}
int trmm_custom(
  char side, char uplo, char transa, char diag,
  int m, int n,
  cuDoubleComplex alpha, const cuDoubleComplex *A, int incA,
  cuDoubleComplex *B, int incB,
  cudaStream_t& curStream
){
  cublasZtrmm(side, uplo, transa, diag,
              m, n,
              alpha, A, incA,
              B, incB );
  return 1;
}
//==============================================================================================

#include "Xtrmm.hxx"

//==============================================================================================

extern "C" {
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
}
  
int kblas_cublas_strmm(
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
int kblas_cublas_dtrmm(
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
int kblas_cublas_ctrmm(
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
int kblas_cublas_ztrmm(
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
}

}

#endif// SUPPORT_CUBLAS


