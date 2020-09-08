/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l3/Xtrmm.cu

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
#include "kblas_common.h"
#include "kblas_operators.h"
#include "omp.h"

//==============================================================================================

cublasStatus_t cublasXtrmm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const float *alpha,
                           const float *A, int lda,
                                 float *B, int ldb){
  cublasStatus_t status;
  check_error_ret( status = cublasStrmm(handle,
                                    side, uplo, trans, diag,
                                    m, n,
                                    alpha, A, lda,
                                           B, ldb,
                                           B, ldb ), status);
  check_error_ret( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
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
  check_error_ret( status = cublasDtrmm(handle,
                                    side, uplo, trans, diag,
                                    m, n,
                                    alpha, A, lda,
                                           B, ldb,
                                           B, ldb ), status);
  check_error_ret( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
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
  check_error_ret( status = cublasCtrmm(handle,
                                    side, uplo, trans, diag,
                                    m, n,
                                    alpha, A, lda,
                                           B, ldb,
                                           B, ldb ), status);
  check_error_ret( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
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
  check_error_ret( status = cublasZtrmm(handle,
                                    side, uplo, trans, diag,
                                    m, n,
                                    alpha, A, lda,
                                           B, ldb,
                                           B, ldb ), status);
  check_error_ret( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
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
#if (TARGET_SM >= 30)
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

  check_error_ret( status = cublasGetStream( handle, &curStream ), status);

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
    check_error_ret( cudaGetLastError(),  CUBLAS_STATUS_EXECUTION_FAILED );
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

#endif //(TARGET_SM >= 30)

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
                          const T *h_A, int ldA, T* d_A, int lddA,
                                T *h_B, int ldB, T* d_B, int lddB,
                          bool BIsIn, bool getBOut, bool AIsIn)
{
  T one = make_one<T>();
  cublasStatus_t status;
  cudaEvent_t eDataIn, eComp;
  check_error_ret( cudaEventCreateWithFlags(&eDataIn, cudaEventDisableTiming), CUBLAS_STATUS_EXECUTION_FAILED);
  check_error_ret( cudaEventCreateWithFlags(&eComp, cudaEventDisableTiming), CUBLAS_STATUS_EXECUTION_FAILED);
  cudaStream_t strComp;
  check_error_ret( cublasGetStream_v2(handle, &strComp), CUBLAS_STATUS_INTERNAL_ERROR);

  if( ( *alpha == make_zero<T>() ) //TODO
   || ( (side == CUBLAS_SIDE_LEFT) && (SIMPLE_SIZE(m)) )
   || ( (side == CUBLAS_SIDE_RIGHT) && (SIMPLE_SIZE(n)) ) ){

    int Am = (side == CUBLAS_SIDE_LEFT) ? m : n;
    //if B is not already in, copy in B block
    if(!BIsIn)
      check_error_ret( status = cublasSetMatrixAsync( m, n, sizeof(T), h_B, ldB, d_B, lddB, strIn ), status);
    //copy in A block
    if(!AIsIn)
      check_error_ret( status = cublasSetMatrixAsync( Am, Am, sizeof(T), h_A, ldA, d_A, lddA, strIn ), status);
    //wait for data to arrive
    if(!AIsIn || !BIsIn){
      check_error_ret( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error_ret( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
    }
    if( (status = Xtrmm( handle,
                         side, uplo, trans, diag,
                         m, n,
                         alpha, d_A, lddA,
                                d_B, lddB ) ) != CUBLAS_STATUS_SUCCESS ) return status;

    //if stream is done computing and getBOut, copy B back.
    if(getBOut){
      check_error_ret( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error_ret( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error_ret( status = cublasGetMatrixAsync( m, n, sizeof(T), d_B, lddB, h_B, ldB, strOut), status);
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
        check_error_ret( status = cublasSetMatrixAsync( m, m, sizeof(T), h_A, ldA, d_A, lddA, strIn), status);
        AIsIn = true;
      }
      if( (!BIsIn && SIMPLE_SIZE_DATA(m)) ){
        check_error_ret( status = cublasSetMatrixAsync( m, n, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
        BIsIn = true;
      }
      //wait for data to arrive
      check_error_ret( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error_ret( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
    }
    if(uplo == CUBLAS_FILL_MODE_UPPER){

      //Left / Upper / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = cublasSetMatrixAsync( m2, n, sizeof(T), h_B + m1, ldB, d_B + m1, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = cublasSetMatrixAsync( m1, m2, sizeof(T), h_A + m1 * ldA, ldA, d_A + m1 * lddA, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m1, n, m2,
                                 alpha, d_A + m1 * lddA, lddA,
                                        d_B + m1, lddB,
                                 &one,  d_B, lddB)) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = cublasGetMatrixAsync( m1, n, sizeof(T), d_B, lddB, h_B, ldB, strOut), status);
        }

        //B is already in, no need to copy in
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, h_A + m1 + m1 * ldA, ldA, d_A + m1 + m1 * lddA, lddA,
                                       h_B + m1, ldB, d_B + m1, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Upper / [Conj]Trans
      else{
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, h_A+m1+m1*ldA, ldA, d_A+m1+m1*lddA, lddA,
                                       h_B+m1, ldB, d_B+m1, lddB,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = cublasSetMatrixAsync( m1, n, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = cublasSetMatrixAsync( m1, m2, sizeof(T), h_A + m1 * ldA, ldA, d_A + m1 * lddA, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m2, n, m1,
                                 alpha, d_A+m1*lddA, lddA,
                                        d_B, lddB,
                                 &one,  d_B+m1, lddB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = cublasGetMatrixAsync( m2, n, sizeof(T), d_B+m1, lddB, h_B+m1, ldB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }

    }else{//uplo == Lower

      //Left / Lower / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, h_A+m1+m1*ldA, ldA, d_A+m1+m1*lddA, lddA,
                                       h_B+m1, ldB, d_B+m1, lddB,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = cublasSetMatrixAsync( m1, n, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = cublasSetMatrixAsync( m2, m1, sizeof(T), h_A + m1, ldA, d_A + m1, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m2, n, m1,
                                 alpha, d_A+m1, lddA,
                                        d_B, lddB,
                                 &one,  d_B+m1, lddB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = cublasGetMatrixAsync( m2, n, sizeof(T), d_B+m1, lddB, h_B+m1, ldB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Lower / [Conj]Trans
      else{//trans == Trans
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = cublasSetMatrixAsync( m2, n, sizeof(T), h_B + m1, ldB, d_B + m1, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = cublasSetMatrixAsync( m2, m1, sizeof(T), h_A + m1, ldA, d_A + m1, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }

        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m1, n, m2,
                                 alpha, d_A+m1, lddA,
                                        d_B+m1, lddB,
                                 &one,  d_B, lddB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = cublasGetMatrixAsync( m1, n, sizeof(T), d_B, lddB, h_B, ldB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, h_A+m1+m1*ldA, ldA, d_A+m1+m1*lddA, lddA,
                                       h_B+m1, ldB, d_B+m1, lddB,
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
        check_error_ret( status = cublasSetMatrixAsync( n, n, sizeof(T), h_A, ldA, d_A, lddA, strIn), status);
        AIsIn = true;
      }
      if( (!BIsIn && SIMPLE_SIZE_DATA(n)) ){
        check_error_ret( status = cublasSetMatrixAsync( m, n, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
        BIsIn = true;
      }
      //wait for data to arrive
      check_error_ret( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error_ret( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
    }
    if(uplo == CUBLAS_FILL_MODE_UPPER){
      //Right / Upper / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, h_A+n1+n1*ldA, ldA, d_A+n1+n1*lddA, lddA,
                                       h_B+n1*ldB, ldB, d_B+n1*lddB, lddB,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = cublasSetMatrixAsync( m, n1, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = cublasSetMatrixAsync( n1, n2, sizeof(T), h_A + n1 * ldA, ldA, d_A + n1 * lddA, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n2, n1,
                                 alpha, d_B, lddB,
                                        d_A+n1*lddA, lddA,
                                 &one,  d_B+n1*lddB, lddB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = cublasGetMatrixAsync( m, n2, sizeof(T), d_B+n1*lddB, lddB, h_B+n1*ldB, ldB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Upper / [Conj]Trans
      else{
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = cublasSetMatrixAsync( m, n2, sizeof(T), h_B+n1*ldB, ldB, d_B+n1*lddB, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = cublasSetMatrixAsync( n1, n2, sizeof(T), h_A+n1*ldA, ldA, d_A+n1*lddA, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n1, n2,
                                 alpha, d_B+n1*lddB, lddB,
                                        d_A+n1*lddA, lddA,
                                 &one,  d_B, lddB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = cublasGetMatrixAsync( m, n1, sizeof(T), d_B, lddB, h_B, ldB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, h_A+n1+n1*ldA, ldA, d_A+n1+n1*lddA, lddA,
                                       h_B+n1*ldB, ldB, d_B+n1*lddB, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
    }else{
      //Right / Lower / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = cublasSetMatrixAsync( m, n2, sizeof(T), h_B+n1*ldB, ldB, d_B+n1*lddB, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = cublasSetMatrixAsync( n2, n1, sizeof(T), h_A+n1, ldA, d_A+n1, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n1, n2,
                                 alpha, d_B+n1*lddB, lddB,
                                        d_A+n1, lddA,
                                 &one,  d_B, lddB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = cublasGetMatrixAsync( m, n1, sizeof(T), d_B, lddB, h_B, ldB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, h_A+n1+n1*ldA, ldA, d_A+n1+n1*lddA, lddA,
                                       h_B+n1*ldB, ldB, d_B+n1*lddB, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Lower / [Conj]Trans
      else{
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, h_A+n1+n1*ldA, ldA, d_A+n1+n1*lddA, lddA,
                                       h_B+n1*ldB, ldB, d_B+n1*lddB, lddB,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = cublasSetMatrixAsync( m, n1, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = cublasSetMatrixAsync( n1, n2, sizeof(T), h_A+n1, ldA, d_A+n1, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n2, n1,
                                 alpha, d_B, lddB,
                                        d_A+n1, lddA,
                                 &one,  d_B+n1*lddB, lddB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = cublasGetMatrixAsync( m, n2, sizeof(T), d_B+n1*lddB, lddB, h_B+n1*ldB, ldB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
    }

  }//side == Right

  check_error_ret( cudaEventDestroy( eDataIn ), CUBLAS_STATUS_INTERNAL_ERROR);
  check_error_ret( cudaEventDestroy( eComp ), CUBLAS_STATUS_INTERNAL_ERROR);
  check_error_ret( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
  return CUBLAS_STATUS_SUCCESS;
}

//==============================================================================================
template<class T>
cublasStatus_t kblasXtrmm_cpu(cublasHandle_t handle,
                              cublasSideMode_t side, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int m, int n,
                              const T *alpha,
                              const T *h_A, int ldA,
                                    T *h_B, int ldB){
  //allocate memory on device
  T *d_A, *d_B;
  int Am, An, Bm, Bn, lddA, lddB;
  if ( side == CUBLAS_SIDE_LEFT ) {
    Am = An = m;
  } else {
    Am = An = n;
  }
  Bm = m;
  Bn = n;
  lddA = ((Am+31)/32)*32;
  lddB = ((Bm+31)/32)*32;

  /*check_error_ret( cudaHostRegister((void*)h_A, Am * An * sizeof(T), cudaHostRegisterDefault), CUBLAS_STATUS_INTERNAL_ERROR);
  check_error_ret( cudaHostRegister((void*)h_B, Bm * Bn * sizeof(T), cudaHostRegisterDefault), CUBLAS_STATUS_INTERNAL_ERROR);*/

  cublasStatus_t status;
  int AsyncEngineCount, devID;
  check_error_ret( cudaGetDevice(&devID), CUBLAS_STATUS_INTERNAL_ERROR);
  check_error_ret( cudaDeviceGetAttribute(&AsyncEngineCount, cudaDevAttrAsyncEngineCount, devID), CUBLAS_STATUS_INTERNAL_ERROR);
  bool DO_INLINE_BOUT = AsyncEngineCount > 1;

  check_error_ret( cudaMalloc( (void**)&d_A, (lddA*An)*sizeof(T) ), CUBLAS_STATUS_INTERNAL_ERROR);
  if(d_A == NULL) return CUBLAS_STATUS_ALLOC_FAILED;
  check_error_ret( cudaMalloc( (void**)&d_B, (lddB*Bn)*sizeof(T) ), CUBLAS_STATUS_INTERNAL_ERROR);
  if(d_B == NULL) return CUBLAS_STATUS_ALLOC_FAILED;

  //setup streams
  cudaStream_t inStream, outStream;
  check_error_ret( cudaStreamCreateWithFlags( &inStream, cudaStreamNonBlocking), CUBLAS_STATUS_INTERNAL_ERROR );
  if(DO_INLINE_BOUT)
    check_error_ret( cudaStreamCreateWithFlags( &outStream, cudaStreamNonBlocking), CUBLAS_STATUS_INTERNAL_ERROR );

  //call cpu API trmm
  check_error_ret(
    (status = kblasXtrmm(handle, inStream, outStream,
                         side, uplo, trans,diag,
                         m, n,
                         alpha, h_A, ldA, d_A, lddA,
                                h_B, ldB, d_B, lddB,
                         false, DO_INLINE_BOUT, false)
    ), status);
  //sync streams
  if(DO_INLINE_BOUT){
    check_error_ret( cudaStreamSynchronize( outStream ), CUBLAS_STATUS_INTERNAL_ERROR);
  }else{
    cudaStream_t compStream;
    check_error_ret( cublasGetStream_v2(handle, &compStream), CUBLAS_STATUS_INTERNAL_ERROR);
    check_error_ret( cudaStreamSynchronize( compStream ), CUBLAS_STATUS_INTERNAL_ERROR);
    check_error_ret( status = cublasGetMatrixAsync( m, n, sizeof(T), d_B, lddB, h_B, ldB, inStream), status);
  }
  //revoke streams
  check_error_ret( cudaStreamDestroy( inStream ), CUBLAS_STATUS_INTERNAL_ERROR);
  if(DO_INLINE_BOUT)
    check_error_ret( cudaStreamDestroy( outStream ), CUBLAS_STATUS_INTERNAL_ERROR);

  /*check_error_ret( cudaHostUnregister( (void*)h_A ), CUBLAS_STATUS_INTERNAL_ERROR );
  check_error_ret( cudaHostUnregister( (void*)h_B ), CUBLAS_STATUS_INTERNAL_ERROR );*/

  //free device memory
  //TODO should free aslo in case some other funtions above failed (don't just return)
  check_error_ret( cudaFree( d_A ), CUBLAS_STATUS_INTERNAL_ERROR );
  check_error_ret( cudaFree( d_B ), CUBLAS_STATUS_INTERNAL_ERROR );
  check_error_ret( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
  return CUBLAS_STATUS_SUCCESS;
}
//==============================================================================================
template<class T>
cublasStatus_t kblasXtrmm_cpu_m(cublasHandle_t handle,
                              cublasSideMode_t side, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int m, int n,
                              const T *alpha,
                              const T *h_A, int ldA,
                                    T *h_B, int ldB,
                              //TODO should accept an array of device IDs or a set of cublas handles intead
                              int ngpu){
  //allocate memory on device
  T *d_A[ngpu], *d_B[ngpu];
  int Am, An, Bm, Bn, lddA, lddB;
  if ( side == CUBLAS_SIDE_LEFT ) {
    Am = An = m;
  } else {
    Am = An = n;
  }
  Bm = m;
  Bn = n;

  /*check_error_ret( cudaHostRegister((void*)h_A, Am * An * sizeof(T), cudaHostRegisterDefault), CUBLAS_STATUS_INTERNAL_ERROR);
  check_error_ret( cudaHostRegister((void*)h_B, Bm * Bn * sizeof(T), cudaHostRegisterDefault), CUBLAS_STATUS_INTERNAL_ERROR);*/

  cudaStream_t inStream[ngpu], outStream[ngpu];
  cublasStatus_t status[ngpu];
  cublasHandle_t cub_handle[ngpu];
  cub_handle[0] = handle;
  //*
  bool DO_INLINE_BOUT[ngpu];
  for(int g = 0; g < ngpu; g++){
    check_error_ret( cudaSetDevice(g), CUBLAS_STATUS_INTERNAL_ERROR);
    int AsyncEngineCount;
    check_error_ret( cudaDeviceGetAttribute(&AsyncEngineCount, cudaDevAttrAsyncEngineCount, g), CUBLAS_STATUS_INTERNAL_ERROR);
    DO_INLINE_BOUT[g] = AsyncEngineCount > 1;

    if(g > 0){
      check_error_ret( cublasCreate(&cub_handle[g]), CUBLAS_STATUS_INTERNAL_ERROR);
    }

    //setup streams
    check_error_ret( cudaStreamCreateWithFlags( &inStream[g], cudaStreamNonBlocking), CUBLAS_STATUS_INTERNAL_ERROR);
    if(DO_INLINE_BOUT[g])
      check_error_ret( cudaStreamCreateWithFlags( &outStream[g], cudaStreamNonBlocking), CUBLAS_STATUS_INTERNAL_ERROR);
  }
  //*/
  //TODO IMPORTANT: handle when data does not fit on all gpus
  int Bn_gpu, Bm_gpu;
  bool left = (side == CUBLAS_SIDE_LEFT);
  if(left){
    Bn_gpu = Bn / ngpu;//TODO handle odd cases
    Bm_gpu = Bm;
  }else{
    Bm_gpu = Bm / ngpu;
    Bn_gpu = Bn;
  }
  lddA = ((Am+31)/32)*32;
  lddB = ((Bm_gpu+31)/32)*32;

  omp_set_num_threads(ngpu);
  #pragma omp parallel
  {
    #pragma omp for
    for(int g = 0; g < ngpu; g++){
      //TODO check status
      cudaSetDevice(g);
      cudaMalloc( (void**)&d_A[g], (lddA*An)*sizeof(T) );
      cudaMalloc( (void**)&d_B[g], (lddB*Bn_gpu)*sizeof(T) );

      //call cpu API trsm
      status[g] = kblasXtrmm(cub_handle[g], inStream[g], outStream[g],
                            side, uplo, trans,diag,
                            Bm_gpu, Bn_gpu,
                            alpha, h_A, ldA, d_A[g], lddA,
                                   h_B+g*(left ? Bn_gpu*ldB : Bm_gpu), ldB, d_B[g], lddB,
                            false, DO_INLINE_BOUT[g], false);
      //TODO check this status for error

      //sync streams
      if(DO_INLINE_BOUT[g]){
        cudaStreamSynchronize( outStream[g] );
      }else{
        cudaStream_t compStream;
        cublasGetStream_v2(cub_handle[g], &compStream);
        cudaStreamSynchronize( compStream );
        status[g] = cublasGetMatrixAsync( Bm_gpu, Bn_gpu, sizeof(T), d_B[g], lddB, h_B+g*(left ? Bn_gpu*ldB : Bm_gpu), ldB, inStream[g]);
        cudaStreamSynchronize( inStream[g] );
      }
    }
  }
  /*/for(int g = 0; g < ngpu; g++){
    check_error_ret( cudaSetDevice(g), CUBLAS_STATUS_INTERNAL_ERROR);
    check_error_ret( cudaMalloc( (void**)&d_A[g], (Am*An)*sizeof(T) ), CUBLAS_STATUS_INTERNAL_ERROR);
    check_error_ret( cudaMalloc( (void**)&d_B[g], (Bm_gpu*Bn_gpu)*sizeof(T) ), CUBLAS_STATUS_INTERNAL_ERROR);

    //setup streams
    check_error_ret( cudaStreamCreateWithFlags( &inStream[g], cudaStreamNonBlocking), CUBLAS_STATUS_INTERNAL_ERROR );
    if(DO_INLINE_BOUT[g])
      check_error_ret( cudaStreamCreateWithFlags( &outStream[g], cudaStreamNonBlocking), CUBLAS_STATUS_INTERNAL_ERROR );

    //call cpu API trsm
    check_error_ret(
      (status[g] = kblasXtrmm(cub_handle[g], inStream[g], outStream[g],
                           side, uplo, trans,diag,
                           Bm_gpu, Bn_gpu,
                           alpha, h_A, incA, d_A[g],
                                  h_B+g*(left ? Bn_gpu*incB : Bm_gpu), incB, d_B[g],
                           false, DO_INLINE_BOUT[g], false)
      ), status[g]);
  }
  for(int g = 0; g < ngpu; g++){
    check_error_ret( cudaSetDevice(g), CUBLAS_STATUS_INTERNAL_ERROR );

    //sync streams
    if(DO_INLINE_BOUT[g]){
      check_error_ret( cudaStreamSynchronize( outStream[g] ), CUBLAS_STATUS_INTERNAL_ERROR);
    }else{
      cudaStream_t compStream;
      check_error_ret( cublasGetStream_v2(cub_handle[g], &compStream), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error_ret( cudaStreamSynchronize( compStream ), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error_ret( status[g] = cublasGetMatrixAsync( Bm_gpu, Bn_gpu, sizeof(T), d_B[g], incB, h_B+g*(left ? Bn_gpu*incB : Bm_gpu), incB, inStream[g]), status[g]);
    }
  }//*/


  for(int g = 0; g < ngpu; g++){
    check_error_ret( cudaSetDevice(g), CUBLAS_STATUS_INTERNAL_ERROR);
    check_error_ret( cudaDeviceSynchronize(), CUBLAS_STATUS_INTERNAL_ERROR );
    //revoke streams
    check_error_ret( cudaStreamDestroy( inStream[g] ), CUBLAS_STATUS_INTERNAL_ERROR);
    if(DO_INLINE_BOUT[g])
      check_error_ret( cudaStreamDestroy( outStream[g] ), CUBLAS_STATUS_INTERNAL_ERROR);
  /*check_error_ret( cudaHostUnregister( (void*)h_A ), CUBLAS_STATUS_INTERNAL_ERROR );
  check_error_ret( cudaHostUnregister( (void*)h_B ), CUBLAS_STATUS_INTERNAL_ERROR );*/

    //free device memory
    check_error_ret( cudaFree( d_A[g] ), CUBLAS_STATUS_INTERNAL_ERROR );
    check_error_ret( cudaFree( d_B[g] ), CUBLAS_STATUS_INTERNAL_ERROR );
    if(g > 0){
      check_error_ret( cublasDestroy(cub_handle[g]), CUBLAS_STATUS_INTERNAL_ERROR);
    }

  }
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
  check_error_ret( cublasCreate(&cublas_handle), void() );                                                      \
  if( cublasSetStream_v2(cublas_handle, stream) != CUBLAS_STATUS_SUCCESS ){                               \
    check_error_ret( cublasDestroy_v2(cublas_handle), void());                                                  \
    return;                                                                                               \
  }                                                                                                       \
  cublasSideMode_t  side_v2  = (side  == KBLAS_Left  ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);             \
  cublasFillMode_t  uplo_v2  = (uplo  == KBLAS_Lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER);  \
  cublasOperation_t trans_v2 = (trans == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);                        \
  cublasDiagType_t  diag_v2  = (diag  == KBLAS_Unit  ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT);          \
                                                                                                          \
  check_error_ret( kblasXtrmm(cublas_handle,                                                                  \
                          side_v2, uplo_v2, trans_v2, diag_v2,                                            \
                          m, n,                                                                           \
                          &alpha, A, lda,                                                                 \
                                  B, ldb), void());                                                         \
                                                                                                          \
  check_error_ret( cublasDestroy_v2(cublas_handle), void());                                                    \
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

//==============================================================================================
cublasStatus_t kblas_strmm_mgpu(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const float *alpha,
                                const float *A, int lda,
                                      float *B, int ldb,
                                int ngpu){
  return kblasXtrmm_cpu_m(handle,
                        side, uplo, trans, diag,
                        m, n,
                        alpha, A, lda,
                              B, ldb,
                        ngpu);
}
cublasStatus_t kblas_dtrmm_mgpu(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const double *alpha,
                                const double *A, int lda,
                                      double *B, int ldb,
                                int ngpu){
  return kblasXtrmm_cpu_m(handle,
                        side, uplo, trans, diag,
                        m, n,
                        alpha, A, lda,
                              B, ldb,
                        ngpu);
}

cublasStatus_t kblas_ctrmm_mgpu(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuComplex *alpha,
                                const cuComplex *A, int lda,
                                      cuComplex *B, int ldb,
                                int ngpu){
  return kblasXtrmm_cpu_m(handle,
                        side, uplo, trans, diag,
                        m, n,
                        alpha, A, lda,
                              B, ldb,
                        ngpu);
}
cublasStatus_t kblas_ztrmm_mgpu(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, int lda,
                                      cuDoubleComplex *B, int ldb,
                                int ngpu){
  return kblasXtrmm_cpu_m(handle,
                        side, uplo, trans, diag,
                        m, n,
                        alpha, A, lda,
                              B, ldb,
                        ngpu);
}

//}


