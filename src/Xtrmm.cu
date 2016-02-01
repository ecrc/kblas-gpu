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
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include "kblas.h"
#include "Xtr_common.ch"
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
                     side, uplo, trans, diag,
                     m, n,
                     alpha, A, lda,
                            B, ldb,
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
                     side, uplo, trans, diag,
                     m, n,
                     alpha, A, lda,
                            B, ldb,
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
                     side, uplo, trans, diag,
                     m, n,
                     alpha, A, lda,
                            B, ldb,
                            B, ldb );
}
cublasStatus_t cublasXtrmm (cublasHandle_t handle,
                            cublasSideMode_t side, cublasFillMode_t uplo,
                            cublasOperation_t trans, cublasDiagType_t diag,
                            int m, int n,
                            const cuDoubleComplex *alpha,
                            const cuDoubleComplex *A, int lda,
                                  cuDoubleComplex *B, int ldb){
  return cublasZtrmm(handle,
                     side, uplo, trans, diag,
                     m, n,
                     alpha, A, lda,
                            B, ldb,
                            B, ldb );
}


//==============================================================================================
#define WARP 32
#define WARP1 33
#define tx threadIdx.x
#define ty threadIdx.y
//==============================================================================================
int kblas_trmm_ib_custom = 128;
int kblas_trmm_ib_cublas = 128;
bool kblas_trmm_use_custom = 0;
//#define SIMPLE_SIZE_CUSTOM(n) ( ((n)<32) || ((n) % 32 == 0 && (n) <= kblas_trmm_ib_custom) )
#define SIMPLE_SIZE(n) ( ((n) < WARP) || ( ((n) % WARP == 0) && ( (n) <= kblas_trmm_ib_cublas ) ) )
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
/*template<typename T, int WARPS_PER_BLOCK, int B_COLS_PER_WARP, bool LEFT, bool LOWER, bool TRANS, bool UNIT, bool CONJG>
__global__ void //__launch_bounds__(256)
trmm_mul32_sb(int M, int N, T alpha, const T* A, int incA, T* B, int incB, int mb){
  
  int txyw = tx + ty*WARP1, txyiA = tx + ty*incA, txyiB = tx + ty*incB;
  
  //setup shared memory
  __shared__ T sA[WARP * WARP1];//strided to avoid bank conflict
  T rB, rBj, s, a[4], b[4], *sAA;
  int c, j, r, l;
  const int A_COL_PER_WARP = WARP / WARPS_PER_BLOCK;
  if(LEFT){/ *TODO* /
    B += blockIdx.x * WARPS_PER_BLOCK * incB;
    const bool forward = (LEFT && (LOWER == TRANS)) || (!LEFT && (LOWER != TRANS));
    const bool active = true/ *TODO* /;


    for( c = (forward ? 0 : mb-1); (forward && (c < mb)) || (!forward && (c > -1)); c += (forward ? 1 : -1))
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
          rBj = shfl(rB, j);
          if(j >= tx){
            s = FMA( CONJG ? conjugate(sA[j + tx * WARP1]) : sA[j + tx * WARP1], rBj, s);
          }
        }
      }else{
        #pragma unroll
        for(j = WARP-1; j > -1; j--){
          if(j == tx){
            s = rB;
            if(!UNIT){
              s *= CONJG ? conjugate(sA[tx + j * WARP1]) : sA[tx + j * WARP1];
            }
          }
          rBj = shfl(rB, j);
          if(j < tx)
            s = FMA( CONJG ? conjugate(sA[tx + j * WARP1]) : sA[tx + j * WARP1], rBj, s);
        }
      }
      __syncthreads();

      for(r = (forward ? c+1 : 0); (forward && (r < mb)) || (!forward && (r < c)); r++){
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
        if(TRANS)
          sAA = sA + tx*WARP1;
        else
          sAA = sA + tx;
        #pragma unroll
        for(j = 0; j < WARP; j+=4){
          if(TRANS){
            //s = FMA( sA[j + tx * WARP1], shfl(rB, j), s);            
            a[0] = CONJG ? conjugate(sAA[j + 0]) : sAA[j + 0];
            a[1] = CONJG ? conjugate(sAA[j + 1]) : sAA[j + 1];
            a[2] = CONJG ? conjugate(sAA[j + 2]) : sAA[j + 2];
            a[3] = CONJG ? conjugate(sAA[j + 3]) : sAA[j + 3];
          }
          else{
            //s = FMA( sA[tx + j * WARP1], shfl(rB, j), s);
            a[0] = sAA[(j + 0)*WARP1];
            a[1] = sAA[(j + 1)*WARP1];
            a[2] = sAA[(j + 2)*WARP1];
            a[3] = sAA[(j + 3)*WARP1];
          }
          
          b[0] = shfl(rB, j + 0);
          b[1] = shfl(rB, j + 1);
          b[2] = shfl(rB, j + 2);
          b[3] = shfl(rB, j + 3);
          s = FMA( a[0], b[0], s );
          s = FMA( a[1], b[1], s );
          s = FMA( a[2], b[2], s );
          s = FMA( a[3], b[3], s );
        }
        __syncthreads();
      }
      //store back B(c) to global mem
      //if(LOWER == TRANS)
        B[txyiB + WARP * c] = alpha * s;
      //else
        //B[txyiB + WARP * c] = s;
    }
  }
}*/
//==============================================================================================
template<typename T, int WARPS_PER_BLOCK, int B_COLS_PER_WARP, bool LOWER, bool TRANS, bool UNIT, bool CONJG>
__global__ void //__launch_bounds__(256)
trmm_mul32_L(int M, int N, T alpha, const T* A, int incA, T* B, int incB, int mb){
  
  const int A_COL_PER_WARP = WARP / WARPS_PER_BLOCK;
  
  int txyw = tx + ty*WARP1/*, tyxw = ty + tx*WARP1*/, txyiA = tx + ty*incA, txyiB = tx + ty*incB;
  
  //setup shared memory
  __shared__ T sA[WARP * WARP1];//strided to avoid bank conflict
  T rB[B_COLS_PER_WARP], rBj[B_COLS_PER_WARP], s[B_COLS_PER_WARP], a[4], b[4], *sAA;
  int c, j, r, l, i;
  {
    B += blockIdx.x * (WARPS_PER_BLOCK * B_COLS_PER_WARP) * incB;
    const bool forward = (LOWER == TRANS);
    int active_col = 0;//an inactive warp will still contribute to data fetching but not to computation
    
    #pragma unroll
    for(l = 0; l < B_COLS_PER_WARP; l++)
      active_col += ((blockIdx.x * (WARPS_PER_BLOCK * B_COLS_PER_WARP) + ty + l * WARPS_PER_BLOCK) < N);

    for( c = (forward ? 0 : mb-1); (forward && (c < mb)) || (!forward && (c > -1)); c += (forward ? 1 : -1))
    {
      #pragma unroll
      for(l = 0; l < B_COLS_PER_WARP; l++)
        s[l] = make_zero<T>();
      //load A(c,c) from global to shared mem
      #pragma unroll
      for(l = 0; l < A_COL_PER_WARP; l++)
        sA[txyw + l * WARPS_PER_BLOCK * WARP1] = A[txyiA + WARP * c * (incA+1) + l * WARPS_PER_BLOCK * incA];
      
      //load B(c) into registers
      #pragma unroll
      for(l = 0; l < B_COLS_PER_WARP; l++)
        if(active_col > l)
          rB[l] = B[txyiB + WARP * c + l * WARPS_PER_BLOCK * incB];
      
      /*__syncthreads();
      if(forward){
        #pragma unroll
        for(l = 0; l < A_COL_PER_WARP; l++)
          if(tx < (ty + l * WARPS_PER_BLOCK))
            sA[(ty + l * WARPS_PER_BLOCK) + WARP1 * tx] = sA[tx + WARP1 * (ty + l * WARPS_PER_BLOCK)];
      }*/
      __syncthreads();

      //perform trmm on shared mem
      if(active_col > 0){
        if(forward){
          #pragma unroll
          for(j = 0; j < WARP; j++){
            #pragma unroll
            for(l = 0; l < B_COLS_PER_WARP; l++)
                rBj[l] = shfl(rB[l], j);
            if(j >= tx){
              //a[0] = CONJG ? conjugate(sA[tx + j * WARP1]) : sA[tx + j * WARP1];
              a[0] = CONJG ? conjugate(sA[j + tx * WARP1]) : sA[j + tx * WARP1];//TODO
              #pragma unroll
              for(l = 0; l < B_COLS_PER_WARP; l++)
                  s[l] = FMA( a[0], rBj[l], s[l]);
            }
          }
        }else{
          /*#pragma unroll
          for(j = WARP-1; j > -1; j-=4){
            #pragma unroll
            for(i = 3; i > -1; i--){
              a[i] = ((j-i) > tx) ? make_zero<T>() : (CONJG ? conjugate(sA[tx + (j-i) * WARP1]) : sA[tx + (j-i) * WARP1]);
            }
            #pragma unroll
            for(l = 0; l < B_COLS_PER_WARP; l++){
              #pragma unroll
              for(i = 3; i > -1; i--)
                b[i] = shfl(rB[l], j-i);
              #pragma unroll
              for(i = 3; i > -1; i--)
                s[l] = FMA( a[i], b[i], s[l]);
            }
          }/*/
          #pragma unroll
          for(j = WARP-1; j > -1; j--){
            #pragma unroll
            for(l = 0; l < B_COLS_PER_WARP; l++)
              rBj[l] = shfl(rB[l], j);
            if(j <= tx){
              a[0] = CONJG ? conjugate(sA[tx + j * WARP1]) : sA[tx + j * WARP1];
              #pragma unroll
              for(l = 0; l < B_COLS_PER_WARP; l++)
                s[l] = FMA( a[0], rBj[l], s[l]);
            }
          }//*/
        }
      }
      __syncthreads();

      for(r = (forward ? c+1 : 0); (forward && (r < mb)) || (!forward && (r < c)); r++){
        #pragma unroll
        for(l = 0; l < A_COL_PER_WARP; l++){
          if(TRANS)//load A(r,c)
            //sA[tyxw + l * WARPS_PER_BLOCK] = A[txyiA + WARP * (r + c * incA) + l * WARPS_PER_BLOCK * incA];
            sA[txyw + l * WARPS_PER_BLOCK * WARP1] = A[txyiA + WARP * (r + c * incA) + l * WARPS_PER_BLOCK * incA];
          else//load A(c,r)
            sA[txyw + l * WARPS_PER_BLOCK * WARP1] = A[txyiA + WARP * (c + r * incA) + l * WARPS_PER_BLOCK * incA];
        }
        //load B(r)
        #pragma unroll
        for(l = 0; l < B_COLS_PER_WARP; l++)
          if(active_col > l)
            rB[l] = B[txyiB + WARP * r + l * WARPS_PER_BLOCK * incB];
        __syncthreads();

        //gemm A(r,c)|A(c,r) & B(r) onto B(c) held at s
        if(active_col > 0){
          if(TRANS)
            sAA = sA + tx*WARP1;
          else
            sAA = sA + tx;
          #pragma unroll
          for(j = 0; j < WARP; j+=4){
            if(TRANS){
              #pragma unroll
              for(i = 0; i < 4; i++)
                //a[i] = CONJG ? conjugate(sAA[(j + i) * WARP1]) : sAA[(j + i) * WARP1];
                a[i] = CONJG ? conjugate(sAA[j + i]) : sAA[j + i];
            }
            else{
              #pragma unroll
              for(i = 0; i < 4; i++)
                a[i] = sAA[(j + i) * WARP1];
            }

            #pragma unroll
            for(l = 0; l < B_COLS_PER_WARP; l++){
                #pragma unroll
                for(i = 0; i < 4; i++)
                  b[i] = shfl(rB[l], j + i);
                #pragma unroll
                for(i = 0; i < 4; i++)
                  s[l] = FMA( a[i], b[i], s[l] );
            }
          }
        }
        __syncthreads();
      }
      //store back B(c) to global mem
      #pragma unroll
      for(l = 0; l < B_COLS_PER_WARP; l++){
        if(active_col > l){
          B[txyiB + WARP * c + l * WARPS_PER_BLOCK * incB] = alpha * s[l];
        }
      }
    }
  }
}
//==============================================================================================
template<typename T, int WARPS_PER_BLOCK, int B_ROWS_PER_WARP, bool LOWER, bool TRANS, bool UNIT, bool CONJG>
__global__ void //__launch_bounds__(256)
trmm_mul32_R(int M, int N, T alpha, const T* A, int incA, T* B, int incB, int nb){

  const int A_COL_PER_WARP = WARP / WARPS_PER_BLOCK;
  const int B_ROWS_PER_BLOCK = WARPS_PER_BLOCK * B_ROWS_PER_WARP;
  
  int txyw = tx + ty*WARP1, tyxw = ty + tx*WARP1, txyiA = tx + ty*incA;
  //int txyiB = tx % B_ROWS_PER_BLOCK + (tx / B_ROWS_PER_BLOCK + ty * WARP / B_ROWS_PER_BLOCK) * incB;
  
  //setup shared memory
  __shared__ T sA[WARP * WARP1];//strided to avoid bank conflict
  T rB[B_ROWS_PER_WARP], rBj[B_ROWS_PER_WARP], s[B_ROWS_PER_WARP], a[4], b[4], *sAA;
  int c, j, r, l, i;
  {
    B += blockIdx.x * B_ROWS_PER_BLOCK;
    const bool forward = (LOWER != TRANS);
    int active_row = 0;//an inactive warp will still contribute to data fetching but not to computation
    
    #pragma unroll
    for(l = 0; l < B_ROWS_PER_WARP; l++)
      active_row += ((blockIdx.x * B_ROWS_PER_BLOCK + ty + l * WARPS_PER_BLOCK) < M);

    for( c = (forward ? 0 : nb-1); (forward && (c < nb)) || (!forward && (c > -1)); c += (forward ? 1 : -1))
    {
      //load B(c) into registers in steps: 1. read coalesced into shared memory. 2. read into registers
      if( (blockIdx.x * B_ROWS_PER_BLOCK + tx % B_ROWS_PER_BLOCK) < M){
        #pragma unroll
        for(l = 0; l < B_ROWS_PER_WARP; l++)
          sA[tx % B_ROWS_PER_BLOCK + (tx / B_ROWS_PER_BLOCK + (ty + l * WARPS_PER_BLOCK) * WARP / B_ROWS_PER_BLOCK) * WARP1] = B[tx % B_ROWS_PER_BLOCK + (tx / B_ROWS_PER_BLOCK + (ty + l * WARPS_PER_BLOCK) * WARP / B_ROWS_PER_BLOCK) * incB + WARP * c * incB];
      }
      __syncthreads();
      #pragma unroll
      for(l = 0; l < B_ROWS_PER_WARP; l++)
        rB[l] = sA[tyxw + l * WARPS_PER_BLOCK];
      __syncthreads();
      
      #pragma unroll
      for(l = 0; l < B_ROWS_PER_WARP; l++)
        s[l] = make_zero<T>();
      //load A(c,c) from global to shared mem
      #pragma unroll
      for(l = 0; l < A_COL_PER_WARP; l++)
        sA[txyw + l * WARPS_PER_BLOCK * WARP1] = A[txyiA + WARP * c * (incA+1) + l * WARPS_PER_BLOCK * incA];
      __syncthreads();
      if(!TRANS){
        #pragma unroll
        for(l = 0; l < A_COL_PER_WARP; l++)
          if(tx < (ty + l * WARPS_PER_BLOCK))
            sA[(ty + l * WARPS_PER_BLOCK) + WARP1 * tx] = sA[tx + WARP1 * (ty + l * WARPS_PER_BLOCK)];
      }
      __syncthreads();

      //perform trmm on shared mem
      if(active_row > 0){
        for(j = (forward ? 0 : WARP - 1); (forward && (j < WARP)) || (!forward && (j > -1)); j+=(forward ? 1 : -1)){
          #pragma unroll
          for(l = 0; l < B_ROWS_PER_WARP; l++)
            rBj[l] = shfl(rB[l], j);
          if( (forward && (j >= tx)) || (!forward && (j <= tx)) ){
            a[0] = CONJG ? conjugate(sA[tx + j * WARP1]) : sA[tx + j * WARP1];
            #pragma unroll
            for(l = 0; l < B_ROWS_PER_WARP; l++)
              s[l] = FMA( a[0], rBj[l], s[l]);
          }
        }
      }
      __syncthreads();

      for(r = (forward ? c+1 : 0); (forward && (r < nb)) || (!forward && (r < c)); r++){
        
        //load B(r) into registers in 2 steps: 1. read coalesced into shared memory. 2. read into registers
        if( (blockIdx.x * B_ROWS_PER_BLOCK + tx % B_ROWS_PER_BLOCK) < M){
          #pragma unroll
          for(l = 0; l < B_ROWS_PER_WARP; l++)
            sA[tx % B_ROWS_PER_BLOCK + (tx / B_ROWS_PER_BLOCK + (ty + l * WARPS_PER_BLOCK) * WARP / B_ROWS_PER_BLOCK) * WARP1] = B[tx % B_ROWS_PER_BLOCK + (tx / B_ROWS_PER_BLOCK + (ty + l * WARPS_PER_BLOCK) * WARP / B_ROWS_PER_BLOCK) * incB + WARP * r * incB];
        }
        __syncthreads();
        #pragma unroll
        for(l = 0; l < B_ROWS_PER_WARP; l++)
          rB[l] = sA[tyxw + l * WARPS_PER_BLOCK];
        __syncthreads();
        
        #pragma unroll
        for(l = 0; l < A_COL_PER_WARP; l++){
          if(!TRANS)//load A(r,c)
            //sA[tyxw + l * WARPS_PER_BLOCK] = A[txyiA + WARP * (r + c * incA) + l * WARPS_PER_BLOCK * incA];
            sA[txyw + l * WARPS_PER_BLOCK * WARP1] = A[txyiA + WARP * (r + c * incA) + l * WARPS_PER_BLOCK * incA];
          else//load A(c,r)
            sA[txyw + l * WARPS_PER_BLOCK * WARP1] = A[txyiA + WARP * (c + r * incA) + l * WARPS_PER_BLOCK * incA];
        }
        __syncthreads();

        //gemm B(r) & A(r,c)|A(c,r) onto B(c) held at s
        if(active_row > 0){
          if(!TRANS)
            sAA = sA + tx*WARP1;
          else
            sAA = sA + tx;
          #pragma unroll
          for(j = 0; j < WARP; j+=4){
            if(!TRANS){
              #pragma unroll
              for(i = 0; i < 4; i++)
                //a[i] = CONJG ? conjugate(sAA[(j + i) * WARP1]) : sAA[(j + i) * WARP1];
                a[i] = CONJG ? conjugate(sAA[j + i]) : sAA[j + i];
            }
            else{
              #pragma unroll
              for(i = 0; i < 4; i++)
                a[i] = sAA[(j + i) * WARP1];
            }

            #pragma unroll
            for(l = 0; l < B_ROWS_PER_WARP; l++){
                #pragma unroll
                for(i = 0; i < 4; i++)
                  b[i] = shfl(rB[l], j + i);
                #pragma unroll
                for(i = 0; i < 4; i++)
                  s[l] = FMA( a[i], b[i], s[l] );
            }
          }
        }
        __syncthreads();
      }
      
      //store back B(c) to global mem in 2 steps: 1. store in shared memory 2. read from shared memory to global memory
      #pragma unroll
      for(l = 0; l < B_ROWS_PER_WARP; l++)
        sA[tyxw + l * WARPS_PER_BLOCK] = alpha * s[l];
      __syncthreads();
      if( (blockIdx.x * B_ROWS_PER_BLOCK + tx % B_ROWS_PER_BLOCK) < M){
        #pragma unroll
        for(l = 0; l < B_ROWS_PER_WARP; l++)
          B[tx % B_ROWS_PER_BLOCK + (tx / B_ROWS_PER_BLOCK + (ty + l * WARPS_PER_BLOCK) * WARP / B_ROWS_PER_BLOCK) * incB + WARP * c * incB] = sA[tx % B_ROWS_PER_BLOCK + (tx / B_ROWS_PER_BLOCK + (ty + l * WARPS_PER_BLOCK) * WARP / B_ROWS_PER_BLOCK) * WARP1];
      }
      __syncthreads();
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
  //handle odd cases with cublas
  if(  (*alpha == make_zero<T>())
    || (!kblas_trmm_use_custom)
    || (side == CUBLAS_SIDE_LEFT && m < WARP)
    || (side == CUBLAS_SIDE_RIGHT && n < WARP)){
    return cublasXtrmm(handle,
                       side, uplo, trans, diag,
                       m, n,
                       alpha, A, incA,
                              B, incB );
  }
  
  typedef void (*trmm_kernels_type)(int M, int N, T alpha, const T* A, int incA, T* B, int incB, int mb);

  #define WARPS_PER_BLOCK 8
  #define B_COLS_PER_WARP 1
  
  trmm_kernels_type trmm_kernels[16] = {// T, WARPS_PER_BLOCK, B_COLS_PER_WARP, LEFT, LOWER, TRANS, UNIT, CONJG
    trmm_mul32_L<T, WARPS_PER_BLOCK, B_COLS_PER_WARP,  true, false, false, false>,
    trmm_mul32_L<T, WARPS_PER_BLOCK, B_COLS_PER_WARP,  true, false,  true, false>,
    trmm_mul32_L<T, WARPS_PER_BLOCK, B_COLS_PER_WARP,  true,  true, false, false>,
    trmm_mul32_L<T, WARPS_PER_BLOCK, B_COLS_PER_WARP,  true,  true,  true, false>,
    trmm_mul32_L<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, false, false, false, false>,
    trmm_mul32_L<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, false, false,  true, false>,
    trmm_mul32_L<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, false,  true, false, false>,
    trmm_mul32_L<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, false,  true,  true, false>,
    trmm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP,  true, false, false, false>,
    trmm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP,  true, false,  true, false>,
    trmm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP,  true,  true, false, false>,
    trmm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP,  true,  true,  true, false>,
    trmm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, false, false, false, false>,
    trmm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, false, false,  true, false>,
    trmm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, false,  true, false, false>,
    trmm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, false,  true,  true, false>
  };
  
  cudaStream_t curStream;
  cublasStatus_t status;

  if((status = cublasGetStream( handle, &curStream )) != CUBLAS_STATUS_SUCCESS ) return status;
  
  /*if(side == CUBLAS_SIDE_RIGHT){
    return cublasXtrmm(handle,
                       side, uplo, trans, diag,
                       m, n,
                       alpha, A, incA,
                              B, incB );
  }else
  if(side == CUBLAS_SIDE_LEFT)*/
  {
    if( ((side == CUBLAS_SIDE_LEFT) && (m % WARP == 0)) || ((side == CUBLAS_SIDE_RIGHT) && (n % WARP == 0)) )
    {
      int func_idx = 8*(side == CUBLAS_SIDE_RIGHT) + 4*(uplo == CUBLAS_FILL_MODE_UPPER) + 2*(trans != CUBLAS_OP_N) + (diag == CUBLAS_DIAG_UNIT);
      dim3 blockDim( WARP, WARPS_PER_BLOCK );
      dim3 gridDim(
        (side == CUBLAS_SIDE_LEFT) * (n / (WARPS_PER_BLOCK * B_COLS_PER_WARP) + (n % (WARPS_PER_BLOCK * B_COLS_PER_WARP) > 0))
        +
        (side == CUBLAS_SIDE_RIGHT) * (m / (WARPS_PER_BLOCK * B_COLS_PER_WARP) + (m % (WARPS_PER_BLOCK * B_COLS_PER_WARP) > 0))
        , 1);
      int mb = (side == CUBLAS_SIDE_LEFT) * m / WARP + (side == CUBLAS_SIDE_RIGHT) * n / WARP;
      trmm_kernels[func_idx]<<< gridDim, blockDim, 0, curStream>>> (m, n, *alpha, A, incA, B, incB, mb);
      if(!_kblas_error( (cudaGetLastError()), __func__, __FILE__, __LINE__ ))
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }else{
      //error: we should not reach this case
      return CUBLAS_STATUS_INTERNAL_ERROR;
    }
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
  return kblasXtrmm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                            B, ldb);
}
cublasStatus_t kblasDtrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const double *alpha,
                          const double *A, int lda,
                                double *B, int ldb){
  return kblasXtrmm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                            B, ldb);
}
cublasStatus_t kblasCtrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const cuComplex *alpha,
                          const cuComplex *A, int lda,
                                cuComplex *B, int ldb){
  return kblasXtrmm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                            B, ldb);
}
cublasStatus_t kblasZtrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const cuDoubleComplex *alpha,
                          const cuDoubleComplex *A, int lda,
                                cuDoubleComplex *B, int ldb){
  return kblasXtrmm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                            B, ldb);
}

}


