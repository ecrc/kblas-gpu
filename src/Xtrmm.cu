/**
  -- (C) Copyright 2013 King Abdullah University of Science and Technology
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
  cublasStatus_t status;
  check_error( status = cublasStrmm(handle,
                                    side, uplo, trans, diag,
                                    m, n,
                                    alpha, A, lda,
                                           B, ldb,
                                           B, ldb ), status);
  check_error( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
  return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasXtrmm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t      diag,
                           int m, int n,
                           const double *alpha,
                           const double *A, int lda,
                                 double *B, int ldb){
  cublasStatus_t status;
  check_error( status = cublasDtrmm(handle,
                                    side, uplo, trans, diag,
                                    m, n,
                                    alpha, A, lda,
                                           B, ldb,
                                           B, ldb ), status);
  check_error( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
  return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasXtrmm (cublasHandle_t handle,
                            cublasSideMode_t side, cublasFillMode_t uplo,
                            cublasOperation_t trans, cublasDiagType_t diag,
                            int m, int n,
                            const cuComplex *alpha,
                            const cuComplex *A, int lda,
                                  cuComplex *B, int ldb){
  cublasStatus_t status;
  check_error( status = cublasCtrmm(handle,
                                    side, uplo, trans, diag,
                                    m, n,
                                    alpha, A, lda,
                                           B, ldb,
                                           B, ldb ), status);
  check_error( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
  return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasXtrmm (cublasHandle_t handle,
                            cublasSideMode_t side, cublasFillMode_t uplo,
                            cublasOperation_t trans, cublasDiagType_t diag,
                            int m, int n,
                            const cuDoubleComplex *alpha,
                            const cuDoubleComplex *A, int lda,
                                  cuDoubleComplex *B, int ldb){
  cublasStatus_t status;
  check_error( status = cublasZtrmm(handle,
                                    side, uplo, trans, diag,
                                    m, n,
                                    alpha, A, lda,
                                           B, ldb,
                                           B, ldb ), status);
  check_error( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
  return CUBLAS_STATUS_SUCCESS;
}


//==============================================================================================
#define WARP 32
#define WARP1 33
#define tx threadIdx.x
#define ty threadIdx.y
//==============================================================================================
int kblas_trmm_ib_custom = 128;
int kblas_trmm_ib_cublas = 128;
int kblas_trmm_ib_data = 512;
bool kblas_trmm_use_custom = 0;
//#define SIMPLE_SIZE_CUSTOM(n) ( ((n)<32) || ((n) % 32 == 0 && (n) <= kblas_trmm_ib_custom) )
#define SIMPLE_SIZE(n) ( ((n) < WARP) || ( ((n) % WARP == 0) && ( (n) <= kblas_trmm_ib_cublas ) ) )
#define SIMPLE_SIZE_DATA(n) ( (n) <= kblas_trmm_ib_data )

//shuffle intrinsic is not supported before KEPLER
#if (SM >= 30)
//==============================================================================================
template<typename T, int WARPS_PER_BLOCK, int B_COLS_PER_WARP, bool LOWER, bool TRANS, bool CONJG>
__global__ void //__launch_bounds__(256)
trmm_mul32_L(int M, int N, T alpha, const T* __restrict__ A, int incA, T* B, int incB, int mb){
  
  const int A_COL_PER_WARP = WARP / WARPS_PER_BLOCK;
  const bool forward = (LOWER == TRANS);
  
  int txyw = tx + ty*WARP1/*, tyxw = ty + tx*WARP1*/, txyiA = tx + ty*incA, txyiB = tx + ty*incB;
  
  //setup shared memory
  __shared__ T sA[WARP * WARP1];//strided to avoid bank conflict
  T rB[B_COLS_PER_WARP], rBj[B_COLS_PER_WARP], s[B_COLS_PER_WARP], a[4], b[4], *sAA, *BB;
  int c, j, r, l, i, startB = 0, active_col;
  
  for(startB = 0; startB < N; startB += gridDim.x * WARPS_PER_BLOCK * B_COLS_PER_WARP)
  {

    if( (startB + blockIdx.x * WARPS_PER_BLOCK * B_COLS_PER_WARP) >= N) return;
    
    BB = B + (startB + blockIdx.x * WARPS_PER_BLOCK * B_COLS_PER_WARP) * incB;
    active_col = 0;//an inactive warp will still contribute to data fetching but not to computation
    
    #pragma unroll
    for(l = 0; l < B_COLS_PER_WARP; l++)
      active_col += ((startB + blockIdx.x * (WARPS_PER_BLOCK * B_COLS_PER_WARP) + ty + l * WARPS_PER_BLOCK) < N);

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
          rB[l] = BB[txyiB + WARP * c + l * WARPS_PER_BLOCK * incB];
      
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
            rB[l] = BB[txyiB + WARP * r + l * WARPS_PER_BLOCK * incB];
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
          BB[txyiB + WARP * c + l * WARPS_PER_BLOCK * incB] = alpha * s[l];
        }
      }
    }
  }
}
//==============================================================================================
template<typename T, int WARPS_PER_BLOCK, int B_ROWS_PER_WARP, bool LOWER, bool TRANS, bool CONJG>
__global__ void //__launch_bounds__(256)
trmm_mul32_R(int M, int N, T alpha, const T* __restrict__ A, int incA, T* B, int incB, int nb){

  const int A_COL_PER_WARP = WARP / WARPS_PER_BLOCK;
  const int B_ROWS_PER_BLOCK = WARPS_PER_BLOCK * B_ROWS_PER_WARP;
  const bool forward = (LOWER != TRANS);
  
  int txyw = tx + ty*WARP1, tyxw = ty + tx*WARP1, txyiA = tx + ty*incA;
  //int txyiB = tx % B_ROWS_PER_BLOCK + (tx / B_ROWS_PER_BLOCK + ty * WARP / B_ROWS_PER_BLOCK) * incB;
  
  //setup shared memory
  __shared__ T sA[WARP * WARP1];//strided to avoid bank conflict
  T rB[B_ROWS_PER_WARP], rBj[B_ROWS_PER_WARP], s[B_ROWS_PER_WARP], a[4], b[4], *sAA, *BB;
  int c, j, r, l, i, startB = 0, active_row = 0;
  
  for(startB = 0; startB < M; startB += gridDim.x * B_ROWS_PER_BLOCK)
  {
    
    if( (startB + blockIdx.x * B_ROWS_PER_BLOCK) >= M) return;
    
    BB = B + (startB + blockIdx.x * B_ROWS_PER_BLOCK);
    active_row = 0;//an inactive warp will still contribute to data fetching but not to computation
    
    #pragma unroll
    for(l = 0; l < B_ROWS_PER_WARP; l++)
      active_row += ( (startB + blockIdx.x * B_ROWS_PER_BLOCK + ty + l * WARPS_PER_BLOCK) < M);

    for( c = (forward ? 0 : nb-1); (forward && (c < nb)) || (!forward && (c > -1)); c += (forward ? 1 : -1))
    {
      //load B(c) into registers in steps: 1. read coalesced into shared memory. 2. read into registers
      if( (blockIdx.x * B_ROWS_PER_BLOCK + tx % B_ROWS_PER_BLOCK) < M){
        #pragma unroll
        for(l = 0; l < B_ROWS_PER_WARP; l++)
          sA[tx % B_ROWS_PER_BLOCK + (tx / B_ROWS_PER_BLOCK + (ty + l * WARPS_PER_BLOCK) * WARP / B_ROWS_PER_BLOCK) * WARP1] = BB[tx % B_ROWS_PER_BLOCK + (tx / B_ROWS_PER_BLOCK + (ty + l * WARPS_PER_BLOCK) * WARP / B_ROWS_PER_BLOCK) * incB + WARP * c * incB];
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
            sA[tx % B_ROWS_PER_BLOCK + (tx / B_ROWS_PER_BLOCK + (ty + l * WARPS_PER_BLOCK) * WARP / B_ROWS_PER_BLOCK) * WARP1] = BB[tx % B_ROWS_PER_BLOCK + (tx / B_ROWS_PER_BLOCK + (ty + l * WARPS_PER_BLOCK) * WARP / B_ROWS_PER_BLOCK) * incB + WARP * r * incB];
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
          BB[tx % B_ROWS_PER_BLOCK + (tx / B_ROWS_PER_BLOCK + (ty + l * WARPS_PER_BLOCK) * WARP / B_ROWS_PER_BLOCK) * incB + WARP * c * incB] = sA[tx % B_ROWS_PER_BLOCK + (tx / B_ROWS_PER_BLOCK + (ty + l * WARPS_PER_BLOCK) * WARP / B_ROWS_PER_BLOCK) * WARP1];
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
                                           T *B, int incB)
{
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
  
  trmm_kernels_type trmm_kernels[8] = {// T, WARPS_PER_BLOCK, B_COLS_PER_WARP, LEFT, LOWER, TRANS, CONJG
    trmm_mul32_L<T, WARPS_PER_BLOCK, B_COLS_PER_WARP,  true, false, false>,
    trmm_mul32_L<T, WARPS_PER_BLOCK, B_COLS_PER_WARP,  true,  true, false>,
    trmm_mul32_L<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, false, false, false>,
    trmm_mul32_L<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, false,  true, false>,
    trmm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP,  true, false, false>,
    trmm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP,  true,  true, false>,
    trmm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, false, false, false>,
    trmm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, false,  true, false>
  };
  
  cudaStream_t curStream;
  cublasStatus_t status;

  check_error( status = cublasGetStream( handle, &curStream ), status);
  
  if( ((side == CUBLAS_SIDE_LEFT) && (m % WARP == 0)) || ((side == CUBLAS_SIDE_RIGHT) && (n % WARP == 0)))
  {
    int func_idx = 4*(side == CUBLAS_SIDE_RIGHT) + 2*(uplo == CUBLAS_FILL_MODE_UPPER) + (trans != CUBLAS_OP_N);// + (diag == CUBLAS_DIAG_UNIT);
    dim3 blockDim( WARP, WARPS_PER_BLOCK );
    dim3 gridDim(
      (side == CUBLAS_SIDE_LEFT) * (n / (WARPS_PER_BLOCK * B_COLS_PER_WARP) + (n % (WARPS_PER_BLOCK * B_COLS_PER_WARP) > 0))
      +
      (side == CUBLAS_SIDE_RIGHT) * (m / (WARPS_PER_BLOCK * B_COLS_PER_WARP) + (m % (WARPS_PER_BLOCK * B_COLS_PER_WARP) > 0))
      , 1);
    int mb = (side == CUBLAS_SIDE_LEFT) * m / WARP + (side == CUBLAS_SIDE_RIGHT) * n / WARP;
    //TODO validate with this run from magma ./testing/testing_dpotri_gpu --dev 1 --range 512:15360:512
    trmm_kernels[func_idx]<<< gridDim, blockDim, 0, curStream>>> (m, n, *alpha, A, incA, B, incB, mb);
    check_error( cudaGetLastError(),  CUBLAS_STATUS_EXECUTION_FAILED );
  }else{
    //error: we should not reach this case
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
  return CUBLAS_STATUS_SUCCESS;
}
#else
template<class T>
cublasStatus_t Xtrmm(cublasHandle_t handle,
                     cublasSideMode_t side, cublasFillMode_t uplo,
                     cublasOperation_t trans, cublasDiagType_t diag,
                     int m, int n,
                     const T *alpha, const T *A, int incA,
                     T *B, int incB)
{
  return cublasXtrmm( handle,
                      side, uplo, trans, diag,
                      m, n,
                      alpha, A, incA,
                             B, incB );
}

#endif //(SM >= 30)

//==============================================================================================
template<class T>
cublasStatus_t kblasXtrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const T *alpha,
                          const T *A, int incA,
                                T *B, int incB)
{
  T one = make_one<T>();
  cublasStatus_t status;
  
  if( (*alpha == make_zero<T>())//TODO
   || ( (side == CUBLAS_SIDE_LEFT) && (SIMPLE_SIZE(m)) )
   || ( (side == CUBLAS_SIDE_RIGHT) && (SIMPLE_SIZE(n)) ) ){
    return Xtrmm(handle,
                 side, uplo, trans, diag,
                 m, n,
                 alpha, A, incA,
                        B, incB );
  }else
  if(side == CUBLAS_SIDE_LEFT){

    int m1, m2;
    if(REG_SIZE(m))
      m1 = m2 = m/2;
    else{
      m1 = CLOSEST_REG_SIZE(m);
      m2 = m-m1;
    }
    cublasOperation_t noTrans = CUBLAS_OP_N;//Trans = CUBLAS_OP_T,

    if(uplo == CUBLAS_FILL_MODE_UPPER){

      //Left / Upper / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, A, incA,
                                       B, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m1, n, m2,
                                 alpha, A+m1*incA, incA,
                                        B+m1, incB,
                                 &one,  B, incB)) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, A+m1+m1*incA, incA,
                                       B+m1, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Upper / [Conj]Trans
      else{
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, A+m1+m1*incA, incA,
                                       B+m1, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m2, n, m1,
                                 alpha, A+m1*incA, incA,
                                        B, incB,
                                 &one,  B+m1, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, A, incA,
                                       B, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }

    }else{//uplo == Lower

      //Left / Lower / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, A+m1+m1*incA, incA,
                                       B+m1, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m2, n, m1,
                                 alpha, A+m1, incA,
                                        B, incB,
                                 &one,  B+m1, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, A, incA,
                                       B, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Lower / [Conj]Trans
      else{//trans == Trans
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, A, incA,
                                       B, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m1, n, m2,
                                 alpha, A+m1, incA,
                                        B+m1, incB,
                                 &one,  B, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, A+m1+m1*incA, incA,
                                       B+m1, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }//trans == Trans
    }//uplo == Lower

  }else{//side == Right
    int n1, n2;

    if(REG_SIZE(n))
      n1 = n2 = n/2;
    else{
      n1 = CLOSEST_REG_SIZE(n);
      n2 = n-n1;
    }

    if(uplo == CUBLAS_FILL_MODE_UPPER){
      //Right / Upper / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, A+n1+n1*incA, incA,
                                       B+n1*incB, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n2, n1,
                                 alpha, B, incB,
                                        A+n1*incA, incA,
                                 &one,  B+n1*incB, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, A, incA,
                                       B, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Upper / [Conj]Trans
      else{
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, A, incA,
                                       B, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n1, n2,
                                 alpha, B+n1*incB, incB,
                                        A+n1*incA, incA,
                                 &one,  B, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, A+n1+n1*incA, incA,
                                       B+n1*incB, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
    }else{
      //Right / Lower / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, A, incA,
                                       B, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n1, n2,
                                 alpha, B+n1*incB, incB,
                                        A+n1, incA,
                                 &one,  B, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, A+n1+n1*incA, incA,
                                       B+n1*incB, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Lower / [Conj]Trans
      else{
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, A+n1+n1*incA, incA,
                                       B+n1*incB, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n2, n1,
                                 alpha, B, incB,
                                        A+n1, incA,
                                 &one,  B+n1*incB, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, A, incA,
                                       B, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
    }

  }//side == Right

  return CUBLAS_STATUS_SUCCESS;
}

//==============================================================================================
template<class T>
cublasStatus_t kblasXtrmm(cublasHandle_t handle, cudaStream_t &strIn, cudaStream_t &strOut,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const T *alpha,
                          const T *h_A, int incA, T* d_A, 
                                T *h_B, int incB, T* d_B,
                          bool BIsIn, bool getBOut, bool AIsIn)
{
  T one = make_one<T>();
  cublasStatus_t status;
  cudaEvent_t eDataIn, eComp;
  check_error( cudaEventCreateWithFlags(&eDataIn, cudaEventDisableTiming), CUBLAS_STATUS_EXECUTION_FAILED);
  check_error( cudaEventCreateWithFlags(&eComp, cudaEventDisableTiming), CUBLAS_STATUS_EXECUTION_FAILED);
  cudaStream_t strComp;
  check_error( cublasGetStream_v2(handle, &strComp), CUBLAS_STATUS_INTERNAL_ERROR);
  
  if( ( *alpha == make_zero<T>() ) //TODO
   || ( (side == CUBLAS_SIDE_LEFT) && (SIMPLE_SIZE(m)) ) 
   || ( (side == CUBLAS_SIDE_RIGHT) && (SIMPLE_SIZE(n)) ) ){

    int Am = (side == CUBLAS_SIDE_LEFT) ? m : n;
    //if B is not already in, copy in B block
    if(!BIsIn)
      check_error( status = cublasSetMatrixAsync( m, n, sizeof(T), h_B, incB, d_B, incB, strIn ), status);
    //copy in A block
    if(!AIsIn)
      check_error( status = cublasSetMatrixAsync( Am, Am, sizeof(T), h_A, incA, d_A, incA, strIn ), status);
    //wait for data to arrive
    if(!AIsIn || !BIsIn){
      check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
    }
    if( (status = Xtrmm( handle,
                         side, uplo, trans, diag,
                         m, n,
                         alpha, d_A, incA,
                                d_B, incB ) ) != CUBLAS_STATUS_SUCCESS ) return status;

    //if stream is done computing and getBOut, copy B back.
    if(getBOut){
      check_error( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error( status = cublasGetMatrixAsync( m, n, sizeof(T), d_B, incB, h_B, incB, strOut), status);
    }
  }else
  if(side == CUBLAS_SIDE_LEFT){

    int m1, m2;
    if(REG_SIZE(m))
      m1 = m2 = m/2;
    else{
      m1 = CLOSEST_REG_SIZE(m);
      m2 = m-m1;
    }
    cublasOperation_t noTrans = CUBLAS_OP_N;//Trans = CUBLAS_OP_T,

    if( (!AIsIn && SIMPLE_SIZE_DATA(m)) || (!BIsIn && SIMPLE_SIZE_DATA(m)) ){
      if( (!AIsIn && SIMPLE_SIZE_DATA(m)) ){
        check_error( status = cublasSetMatrixAsync( m, m, sizeof(T), h_A, incA, d_A, incA, strIn), status);
        AIsIn = true;
      }
      if( (!BIsIn && SIMPLE_SIZE_DATA(m)) ){
        check_error( status = cublasSetMatrixAsync( m, n, sizeof(T), h_B, incB, d_B, incB, strIn), status);
        BIsIn = true;
      }
      //wait for data to arrive
      check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);      
    }
    if(uplo == CUBLAS_FILL_MODE_UPPER){

      //Left / Upper / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, h_A, incA, d_A,
                                       h_B, incB, d_B,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error( status = cublasSetMatrixAsync( m2, n, sizeof(T), h_B + m1, incB, d_B + m1, incB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error( status = cublasSetMatrixAsync( m1, m2, sizeof(T), h_A + m1 * incA, incA, d_A + m1 * incA, incA, strIn), status);
          //wait for data to arrive
          check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m1, n, m2,
                                 alpha, d_A + m1 * incA, incA,
                                        d_B + m1, incB,
                                 &one,  d_B, incB)) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( status = cublasGetMatrixAsync( m1, n, sizeof(T), d_B, incB, h_B, incB, strOut), status);
        }

        //B is already in, no need to copy in
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, h_A + m1 + m1 * incA, incA, d_A + m1 + m1 * incA,
                                       h_B + m1, incB, d_B + m1,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Upper / [Conj]Trans
      else{
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, h_A+m1+m1*incA, incA, d_A+m1+m1*incA,
                                       h_B+m1, incB, d_B+m1,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error( status = cublasSetMatrixAsync( m1, n, sizeof(T), h_B, incB, d_B, incB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error( status = cublasSetMatrixAsync( m1, m2, sizeof(T), h_A + m1 * incA, incA, d_A + m1 * incA, incA, strIn), status);
          //wait for data to arrive
          check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m2, n, m1,
                                 alpha, d_A+m1*incA, incA,
                                        d_B, incB,
                                 &one,  d_B+m1, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( status = cublasGetMatrixAsync( m2, n, sizeof(T), d_B+m1, incB, h_B+m1, incB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, h_A, incA, d_A,
                                       h_B, incB, d_B,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }

    }else{//uplo == Lower

      //Left / Lower / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, h_A+m1+m1*incA, incA, d_A+m1+m1*incA, 
                                       h_B+m1, incB, d_B+m1,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error( status = cublasSetMatrixAsync( m1, n, sizeof(T), h_B, incB, d_B, incB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error( status = cublasSetMatrixAsync( m2, m1, sizeof(T), h_A + m1, incA, d_A + m1, incA, strIn), status);
          //wait for data to arrive
          check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m2, n, m1,
                                 alpha, d_A+m1, incA,
                                        d_B, incB,
                                 &one,  d_B+m1, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( status = cublasGetMatrixAsync( m2, n, sizeof(T), d_B+m1, incB, h_B+m1, incB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, h_A, incA, d_A,
                                       h_B, incB, d_B,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Lower / [Conj]Trans
      else{//trans == Trans
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, h_A, incA, d_A,
                                       h_B, incB, d_B,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error( status = cublasSetMatrixAsync( m2, n, sizeof(T), h_B + m1, incB, d_B + m1, incB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error( status = cublasSetMatrixAsync( m2, m1, sizeof(T), h_A + m1, incA, d_A + m1, incA, strIn), status);
          //wait for data to arrive
          check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }

        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m1, n, m2,
                                 alpha, d_A+m1, incA,
                                        d_B+m1, incB,
                                 &one,  d_B, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( status = cublasGetMatrixAsync( m1, n, sizeof(T), d_B, incB, h_B, incB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, h_A+m1+m1*incA, incA, d_A+m1+m1*incA, 
                                       h_B+m1, incB, d_B+m1,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }//trans == Trans
    }//uplo == Lower

  }else{//side == Right
    int n1, n2;

    if(REG_SIZE(n))
      n1 = n2 = n/2;
    else{
      n1 = CLOSEST_REG_SIZE(n);
      n2 = n-n1;
    }
    
    if( (!AIsIn && SIMPLE_SIZE_DATA(n)) || (!BIsIn && SIMPLE_SIZE_DATA(n)) ){
      if( (!AIsIn && SIMPLE_SIZE_DATA(n)) ){
        check_error( status = cublasSetMatrixAsync( n, n, sizeof(T), h_A, incA, d_A, incA, strIn), status);
        AIsIn = true;
      }
      if( (!BIsIn && SIMPLE_SIZE_DATA(n)) ){
        check_error( status = cublasSetMatrixAsync( m, n, sizeof(T), h_B, incB, d_B, incB, strIn), status);
        BIsIn = true;
      }
      //wait for data to arrive
      check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
    }
    if(uplo == CUBLAS_FILL_MODE_UPPER){
      //Right / Upper / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, h_A+n1+n1*incA, incA, d_A+n1+n1*incA, 
                                       h_B+n1*incB, incB, d_B+n1*incB,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error( status = cublasSetMatrixAsync( m, n1, sizeof(T), h_B, incB, d_B, incB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error( status = cublasSetMatrixAsync( n1, n2, sizeof(T), h_A + n1 * incA, incA, d_A + n1 * incA, incA, strIn), status);
          //wait for data to arrive
          check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n2, n1,
                                 alpha, d_B, incB,
                                        d_A+n1*incA, incA,
                                 &one,  d_B+n1*incB, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( status = cublasGetMatrixAsync( m, n2, sizeof(T), d_B+n1*incB, incB, h_B+n1*incB, incB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, h_A, incA, d_A,
                                       h_B, incB, d_B,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Upper / [Conj]Trans
      else{
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, h_A, incA, d_A,
                                       h_B, incB, d_B,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error( status = cublasSetMatrixAsync( m, n2, sizeof(T), h_B+n1*incB, incB, d_B+n1*incB, incB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error( status = cublasSetMatrixAsync( n1, n2, sizeof(T), h_A+n1*incA, incA, d_A+n1*incA, incA, strIn), status);
          //wait for data to arrive
          check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n1, n2,
                                 alpha, d_B+n1*incB, incB,
                                        d_A+n1*incA, incA,
                                 &one,  d_B, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( status = cublasGetMatrixAsync( m, n1, sizeof(T), d_B, incB, h_B, incB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, h_A+n1+n1*incA, incA, d_A+n1+n1*incA, 
                                       h_B+n1*incB, incB, d_B+n1*incB,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
    }else{
      //Right / Lower / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, h_A, incA, d_A, 
                                       h_B, incB, d_B,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error( status = cublasSetMatrixAsync( m, n2, sizeof(T), h_B+n1*incB, incB, d_B+n1*incB, incB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error( status = cublasSetMatrixAsync( n2, n1, sizeof(T), h_A+n1, incA, d_A+n1, incA, strIn), status);
          //wait for data to arrive
          check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n1, n2,
                                 alpha, d_B+n1*incB, incB,
                                        d_A+n1, incA,
                                 &one,  d_B, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( status = cublasGetMatrixAsync( m, n1, sizeof(T), d_B, incB, h_B, incB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, h_A+n1+n1*incA, incA, d_A+n1+n1*incA, 
                                       h_B+n1*incB, incB, d_B+n1*incB,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Lower / [Conj]Trans
      else{
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, h_A+n1+n1*incA, incA, d_A+n1+n1*incA, 
                                       h_B+n1*incB, incB, d_B+n1*incB,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error( status = cublasSetMatrixAsync( m, n1, sizeof(T), h_B, incB, d_B, incB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error( status = cublasSetMatrixAsync( n1, n2, sizeof(T), h_A+n1, incA, d_A+n1, incA, strIn), status);
          //wait for data to arrive
          check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n2, n1,
                                 alpha, d_B, incB,
                                        d_A+n1, incA,
                                 &one,  d_B+n1*incB, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( status = cublasGetMatrixAsync( m, n2, sizeof(T), d_B+n1*incB, incB, h_B+n1*incB, incB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, h_A, incA, d_A,
                                       h_B, incB, d_B,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
    }

  }//side == Right

  check_error( cudaEventDestroy( eDataIn ), CUBLAS_STATUS_INTERNAL_ERROR);
  check_error( cudaEventDestroy( eComp ), CUBLAS_STATUS_INTERNAL_ERROR);
  check_error( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
  return CUBLAS_STATUS_SUCCESS;
}


/*/==============================================================================================
template<class T>
cublasStatus_t kblasXtrmm(cublasHandle_t handle, cudaStream_t &strIn, cudaStream_t &strOut,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const T *alpha,
                          const T *Ac, int incA, T* Ad, 
                                T *Bc, int incB, T* Bd, bool Bin, bool Bout)//TODO wait on intial event
{
  T one = make_one<T>();
  cublasStatus_t status;
  cudaEvent_t eDataIn, eComp;
  cudaStream_t inStream;
  check_error( cudaStreamCreateWithFlags( &inStream, cudaStreamNonBlocking), CUBLAS_STATUS_EXECUTION_FAILED );
  check_error( cudaEventCreateWithFlags(&eDataIn, cudaEventDisableTiming), CUBLAS_STATUS_EXECUTION_FAILED);
  check_error( cudaEventCreateWithFlags(&eComp, cudaEventDisableTiming), CUBLAS_STATUS_EXECUTION_FAILED);
  cudaStream_t compStream;
  check_error( cublasGetStream_v2(handle, &compStream), CUBLAS_STATUS_INTERNAL_ERROR);
  
  if( ( *alpha == make_zero<T>() ) //TODO
   || ( (side == CUBLAS_SIDE_LEFT) && (SIMPLE_SIZE(m)) ) 
   || ( (side == CUBLAS_SIDE_RIGHT) && (SIMPLE_SIZE(n)) ) ){

    int Am = (side == CUBLAS_SIDE_LEFT) ? m : n;
    //if B is not already in, copy in B block
    if(!Bin)
      check_error( status = cublasSetMatrixAsync( m, n, sizeof(T), Bc, incB, Bd, incB, inStream ), status);
    //copy in A block
    check_error( status = cublasSetMatrixAsync( Am, Am, sizeof(T), Ac, incA, Ad, incA, inStream ), status);
    //wait for data to arrive
    check_error( cudaEventRecord(eDataIn, inStream), CUBLAS_STATUS_INTERNAL_ERROR);
    check_error( cudaStreamWaitEvent(compStream, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
  
    if( (status = Xtrmm( handle,
                         side, uplo, trans, diag,
                         m, n,
                         alpha, Ad, incA,
                                Bd, incB ) ) != CUBLAS_STATUS_SUCCESS ) return status;

    //if stream is done computing and Bout, copy B back.
    if(Bout){
      check_error( cudaEventRecord(eComp, compStream), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error( status = cublasGetMatrixAsync( m, n, sizeof(T), Bd, incB, Bc, incB, strOut), status);
    }
  }else
  if(side == CUBLAS_SIDE_LEFT){

    int m1, m2;
    if(REG_SIZE(m))
      m1 = m2 = m/2;
    else{
      m1 = CLOSEST_REG_SIZE(m);
      m2 = m-m1;
    }
    cublasOperation_t noTrans = CUBLAS_OP_N;//Trans = CUBLAS_OP_T,

    if(uplo == CUBLAS_FILL_MODE_UPPER){

      //Left / Upper / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, Ac, incA, Ad,
                                       Bc, incB, Bd, false, false
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        //if B is not already in, copy B block
        if(!Bin)
          check_error( status = cublasSetMatrixAsync( m2, n, sizeof(T), Bc + m1, incB, Bd + m1, incB, inStream), status);
        //copy in A block
        check_error( status = cublasSetMatrixAsync( m1, m2, sizeof(T), Ac + m1 * incA, incA, Ad + m1 * incA, incA, inStream), status);
        //wait for data to arrive
        check_error( cudaEventRecord(eDataIn, inStream), CUBLAS_STATUS_INTERNAL_ERROR);
        check_error( cudaStreamWaitEvent(compStream, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);

        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m1, n, m2,
                                 alpha, Ad + m1 * incA, incA,
                                        Bd + m1, incB,
                                 &one,  Bd, incB)) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and Bout, copy B back.
        if(Bout){
          check_error( cudaEventRecord(eComp, compStream), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( status = cublasGetMatrixAsync( m1, n, sizeof(T), Bd, incB, Bc, incB, strOut), status);
        }

        //B is already in, no need to copy in
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, Ac + m1 + m1 * incA, incA, Ad + m1 + m1 * incA,
                                       Bc + m1, incB, Bd + m1, true, Bout
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Upper / [Conj]Trans
      else{
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, Ac+m1+m1*incA, incA,
                                       Bc+m1, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m2, n, m1,
                                 alpha, Ac+m1*incA, incA,
                                        Bc, incB,
                                 &one,  Bc+m1, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, Ac, incA,
                                       Bc, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }

    }else{//uplo == Lower

      //Left / Lower / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, Ac+m1+m1*incA, incA,
                                       Bc+m1, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m2, n, m1,
                                 alpha, Ac+m1, incA,
                                        Bc, incB,
                                 &one,  Bc+m1, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, Ac, incA,
                                       Bc, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Lower / [Conj]Trans
      else{//trans == Trans
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, Ac, incA,
                                       Bc, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m1, n, m2,
                                 alpha, Ac+m1, incA,
                                        Bc+m1, incB,
                                 &one,  Bc, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, Ac+m1+m1*incA, incA,
                                       Bc+m1, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }//trans == Trans
    }//uplo == Lower

  }else{//side == Right
    int n1, n2;

    if(REG_SIZE(n))
      n1 = n2 = n/2;
    else{
      n1 = CLOSEST_REG_SIZE(n);
      n2 = n-n1;
    }

    if(uplo == CUBLAS_FILL_MODE_UPPER){
      //Right / Upper / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, Ac+n1+n1*incA, incA,
                                       Bc+n1*incB, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n2, n1,
                                 alpha, Bc, incB,
                                        Ac+n1*incA, incA,
                                 &one,  Bc+n1*incB, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, Ac, incA,
                                       Bc, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Upper / [Conj]Trans
      else{
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, Ac, incA,
                                       Bc, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n1, n2,
                                 alpha, Bc+n1*incB, incB,
                                        Ac+n1*incA, incA,
                                 &one,  Bc, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, Ac+n1+n1*incA, incA,
                                       Bc+n1*incB, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
    }else{
      //Right / Lower / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, Ac, incA,
                                       Bc, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n1, n2,
                                 alpha, Bc+n1*incB, incB,
                                        Ac+n1, incA,
                                 &one,  Bc, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, Ac+n1+n1*incA, incA,
                                       Bc+n1*incB, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Lower / [Conj]Trans
      else{
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, Ac+n1+n1*incA, incA,
                                       Bc+n1*incB, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n2, n1,
                                 alpha, Bc, incB,
                                        Ac+n1, incA,
                                 &one,  Bc+n1*incB, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, Ac, incA,
                                       Bc, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
    }

  }//side == Right

  check_error( cudaStreamDestroy( inStream ), CUBLAS_STATUS_INTERNAL_ERROR );
  check_error( cudaEventDestroy( eDataIn ), CUBLAS_STATUS_INTERNAL_ERROR);
  check_error( cudaEventDestroy( eComp ), CUBLAS_STATUS_INTERNAL_ERROR);
  return CUBLAS_STATUS_SUCCESS;
}
*/
template<class T>
cublasStatus_t kblasXtrmm_cpu(cublasHandle_t handle, 
                              cublasSideMode_t side, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int m, int n,
                              const T *alpha,
                              const T *h_A, int incA,
                                    T *h_B, int incB){
  //allocate memory on device
  T *d_A, *d_B;
  int Am, An, Bm, Bn;
  if ( side == CUBLAS_SIDE_LEFT ) {
    Am = An = m;
  } else {
    Am = An = n;
  }
  Bm = m;
  Bn = n;

  /*check_error( cudaHostRegister((void*)h_A, Am * An * sizeof(T), cudaHostRegisterDefault), CUBLAS_STATUS_INTERNAL_ERROR);
  check_error( cudaHostRegister((void*)h_B, Bm * Bn * sizeof(T), cudaHostRegisterDefault), CUBLAS_STATUS_INTERNAL_ERROR);*/

  cublasStatus_t status;
  int AsyncEngineCount, devID;
  check_error( cudaGetDevice(&devID), CUBLAS_STATUS_INTERNAL_ERROR);
  check_error( cudaDeviceGetAttribute(&AsyncEngineCount, cudaDevAttrAsyncEngineCount, devID), CUBLAS_STATUS_INTERNAL_ERROR);
  bool DO_INLINE_BOUT = AsyncEngineCount > 1;
  
  check_error( cudaMalloc( (void**)&d_A, (Am*An)*sizeof(T) ), CUBLAS_STATUS_ALLOC_FAILED);
  if(d_A == NULL) return CUBLAS_STATUS_ALLOC_FAILED;
  check_error( cudaMalloc( (void**)&d_B, (Bm*Bn)*sizeof(T) ), CUBLAS_STATUS_ALLOC_FAILED);
  if(d_B == NULL) return CUBLAS_STATUS_ALLOC_FAILED;
  
  //setup streams
  cudaStream_t inStream, outStream;
  check_error( cudaStreamCreateWithFlags( &inStream, cudaStreamNonBlocking), CUBLAS_STATUS_INTERNAL_ERROR );
  if(DO_INLINE_BOUT)
    check_error( cudaStreamCreateWithFlags( &outStream, cudaStreamNonBlocking), CUBLAS_STATUS_INTERNAL_ERROR );
  
  //call cpu API trmm
  check_error( 
    (status = kblasXtrmm(handle, inStream, outStream,
                         side, uplo, trans,diag,
                         m, n,
                         alpha, h_A, incA, d_A,
                                h_B, incB, d_B,
                         false, DO_INLINE_BOUT, false)
    ), status);
  //sync streams
  if(DO_INLINE_BOUT){
    check_error( cudaStreamSynchronize( outStream ), CUBLAS_STATUS_INTERNAL_ERROR);
  }else{
    cudaStream_t compStream;
    check_error( cublasGetStream_v2(handle, &compStream), CUBLAS_STATUS_INTERNAL_ERROR);
    check_error( cudaStreamSynchronize( compStream ), CUBLAS_STATUS_INTERNAL_ERROR);
    check_error( status = cublasGetMatrixAsync( m, n, sizeof(T), d_B, incB, h_B, incB, inStream), status);
  }
  //revoke streams
  check_error( cudaStreamDestroy( inStream ), CUBLAS_STATUS_INTERNAL_ERROR);
  if(DO_INLINE_BOUT)
    check_error( cudaStreamDestroy( outStream ), CUBLAS_STATUS_INTERNAL_ERROR);

  /*check_error( cudaHostUnregister( (void*)h_A ), CUBLAS_STATUS_INTERNAL_ERROR );
  check_error( cudaHostUnregister( (void*)h_B ), CUBLAS_STATUS_INTERNAL_ERROR );*/

  //free device memory
  //TODO should free aslo in case some other funtions above failed (don't just return)
  check_error( cudaFree( d_A ), CUBLAS_STATUS_INTERNAL_ERROR );
  check_error( cudaFree( d_B ), CUBLAS_STATUS_INTERNAL_ERROR );  
  check_error( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
  return CUBLAS_STATUS_SUCCESS;
}


//==============================================================================================

  /*
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
}*/
  //==============================================================================================

#define kblasXtrmm_async_BODY {                                                                           \
  cublasHandle_t cublas_handle;                                                                           \
  check_error( cublasCreate(&cublas_handle), void() );                                                      \
  if( cublasSetStream_v2(cublas_handle, stream) != CUBLAS_STATUS_SUCCESS ){                               \
    check_error( cublasDestroy_v2(cublas_handle), void());                                                  \
    return;                                                                                               \
  }                                                                                                       \
  cublasSideMode_t  side_v2  = (side  == KBLAS_Left  ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);             \
  cublasFillMode_t  uplo_v2  = (uplo  == KBLAS_Lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER);  \
  cublasOperation_t trans_v2 = (trans == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);                        \
  cublasDiagType_t  diag_v2  = (diag  == KBLAS_Unit  ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT);          \
                                                                                                          \
  check_error( kblasXtrmm(cublas_handle,                                                                  \
                          side_v2, uplo_v2, trans_v2, diag_v2,                                            \
                          m, n,                                                                           \
                          &alpha, A, lda,                                                                 \
                                  B, ldb), void());                                                         \
                                                                                                          \
  check_error( cublasDestroy_v2(cublas_handle), void());                                                    \
}

extern "C"{
void kblasStrmm_async(char side, char uplo, char trans, char diag,
                      int m, int n,
                      float alpha, const float *A, int lda,
                                         float *B, int ldb,
                      cudaStream_t stream){
  kblasXtrmm_async_BODY
}

void kblasDtrmm_async(char side, char uplo, char trans, char diag,
                      int m, int n,
                      double alpha, const double *A, int lda,
                                          double *B, int ldb,
                      cudaStream_t stream){
  kblasXtrmm_async_BODY
}
void kblasCtrmm_async(char side, char uplo, char trans, char diag,
                      int m, int n,
                      cuComplex alpha, const cuComplex *A, int lda,
                                             cuComplex *B, int ldb,
                      cudaStream_t stream){
  kblasXtrmm_async_BODY
}
void kblasZtrmm_async(char side, char uplo, char trans, char diag,
                      int m, int n,
                      cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                                                   cuDoubleComplex *B, int ldb,
                      cudaStream_t stream){
  kblasXtrmm_async_BODY
}
}
//==============================================================================================

void kblasStrmm(char side, char uplo, char trans, char diag,
                int m, int n,
                float alpha, const float *A, int lda,
                                   float *B, int ldb){

  kblasStrmm_async(side, uplo, trans, diag,
                   m, n,
                   alpha, A, lda,
                          B, ldb,
                   0);
}

void kblasDtrmm(char side, char uplo, char trans, char diag,
                int m, int n,
                double alpha, const double *A, int lda,
                                    double *B, int ldb){

  kblasDtrmm_async(side, uplo, trans, diag,
                   m, n,
                   alpha, A, lda,
                          B, ldb,
                   0);
}
void kblasCtrmm(char side, char uplo, char trans, char diag,
                int m, int n,
                cuComplex alpha, const cuComplex *A, int lda,
                                       cuComplex *B, int ldb){

  kblasCtrmm_async(side, uplo, trans, diag,
                   m, n,
                   alpha, A, lda,
                          B, ldb,
                   0);

}
void kblasZtrmm(char side, char uplo, char trans, char diag,
                int m, int n,
                cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                                             cuDoubleComplex *B, int ldb){

  kblasZtrmm_async(side, uplo, trans, diag,
                   m, n,
                   alpha, A, lda,
                          B, ldb,
                   0);
}
//==============================================================================================

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


//==============================================================================================
cublasStatus_t kblas_strmm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const float *alpha,
                           const float *A, int lda,
                                 float *B, int ldb){
  return kblasXtrmm_cpu(handle,
                        side, uplo, trans, diag,
                        m, n,
                        alpha, A, lda,
                               B, ldb);
}

cublasStatus_t kblas_dtrmm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const double *alpha,
                           const double *A, int lda,
                                 double *B, int ldb){
  return kblasXtrmm_cpu(handle,
                        side, uplo, trans, diag,
                        m, n,
                        alpha, A, lda,
                               B, ldb);
}
cublasStatus_t kblas_ctrmm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const cuComplex *alpha,
                           const cuComplex *A, int lda,
                                 cuComplex *B, int ldb){
  return kblasXtrmm_cpu(handle,
                        side, uplo, trans, diag,
                        m, n,
                        alpha, A, lda,
                               B, ldb);
}
cublasStatus_t kblas_ztrmm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda,
                                 cuDoubleComplex *B, int ldb){
  return kblasXtrmm_cpu(handle,
                        side, uplo, trans, diag,
                        m, n,
                        alpha, A, lda,
                               B, ldb);
}

//}


