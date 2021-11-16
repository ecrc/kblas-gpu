#include "hip/hip_runtime.h"
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
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "hipblas.h"
#include "kblas.h"
#include "kblas_common.h"
#include "kblas_operators.h"
#include "omp.h"

//==============================================================================================

hipblasStatus_t cublasXtrmm(hipblasHandle_t handle,
                           hipblasSideMode_t side, hipblasFillMode_t uplo,
                           hipblasOperation_t trans, hipblasDiagType_t diag,
                           int m, int n,
                           const float *alpha,
                           const float *A, int lda,
                                 float *B, int ldb){
  hipblasStatus_t status;
  check_error_ret( status = hipblasStrmm(handle,
                                    side, uplo, trans, diag,
                                    m, n,
                                    alpha, A, lda,
                                           B, ldb), status);
  check_error_ret( hipGetLastError(), HIPBLAS_STATUS_EXECUTION_FAILED );
  return HIPBLAS_STATUS_SUCCESS;
}
hipblasStatus_t cublasXtrmm(hipblasHandle_t handle,
                           hipblasSideMode_t side, hipblasFillMode_t uplo,
                           hipblasOperation_t trans, hipblasDiagType_t      diag,
                           int m, int n,
                           const double *alpha,
                           const double *A, int lda,
                                 double *B, int ldb){
  hipblasStatus_t status;
  check_error_ret( status = hipblasDtrmm(handle,
                                    side, uplo, trans, diag,
                                    m, n,
                                    alpha, A, lda,
                                           B, ldb), status);
  check_error_ret( hipGetLastError(), HIPBLAS_STATUS_EXECUTION_FAILED );
  return HIPBLAS_STATUS_SUCCESS;
}
// hipblasStatus_t cublasXtrmm (hipblasHandle_t handle,
//                             hipblasSideMode_t side, hipblasFillMode_t uplo,
//                             hipblasOperation_t trans, hipblasDiagType_t diag,
//                             int m, int n,
//                             const hipComplex *alpha,
//                             const hipComplex *A, int lda,
//                                   hipComplex *B, int ldb){
//   hipblasStatus_t status;
//   check_error_ret( status = cublasCtrmm(handle,
//                                     side, uplo, trans, diag,
//                                     m, n,
//                                     alpha, A, lda,
//                                            B, ldb,
//                                            B, ldb ), status);
//   check_error_ret( hipGetLastError(), HIPBLAS_STATUS_EXECUTION_FAILED );
//   return HIPBLAS_STATUS_SUCCESS;
// }
// hipblasStatus_t cublasXtrmm (hipblasHandle_t handle,
//                             hipblasSideMode_t side, hipblasFillMode_t uplo,
//                             hipblasOperation_t trans, hipblasDiagType_t diag,
//                             int m, int n,
//                             const hipDoubleComplex *alpha,
//                             const hipDoubleComplex *A, int lda,
//                                   hipDoubleComplex *B, int ldb){
//   hipblasStatus_t status;
//   check_error_ret( status = cublasZtrmm(handle,
//                                     side, uplo, trans, diag,
//                                     m, n,
//                                     alpha, A, lda,
//                                            B, ldb,
//                                            B, ldb ), status);
//   check_error_ret( hipGetLastError(), HIPBLAS_STATUS_EXECUTION_FAILED );
//   return HIPBLAS_STATUS_SUCCESS;
// }


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
hipblasStatus_t Xtrmm(hipblasHandle_t handle,
                     hipblasSideMode_t side, hipblasFillMode_t uplo,
                     hipblasOperation_t trans, hipblasDiagType_t diag,
                     int m, int n,
                     const T *alpha, const T *A, int incA,
                                           T *B, int incB)
{
  //handle odd cases with cublas
  if(  (*alpha == make_zero<T>())
    || (!kblas_trmm_use_custom)
    || (side == HIPBLAS_SIDE_LEFT && m < WARP)
    || (side == HIPBLAS_SIDE_RIGHT && n < WARP)){
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

  hipStream_t curStream;
  hipblasStatus_t status;

  check_error_ret( status = hipblasGetStream( handle, &curStream ), status);

  if( ((side == HIPBLAS_SIDE_LEFT) && (m % WARP == 0)) || ((side == HIPBLAS_SIDE_RIGHT) && (n % WARP == 0)))
  {
    int func_idx = 4*(side == HIPBLAS_SIDE_RIGHT) + 2*(uplo == HIPBLAS_FILL_MODE_UPPER) + (trans != HIPBLAS_OP_N);// + (diag == HIPBLAS_DIAG_UNIT);
    dim3 blockDim( WARP, WARPS_PER_BLOCK );
    dim3 gridDim(
      (side == HIPBLAS_SIDE_LEFT) * (n / (WARPS_PER_BLOCK * B_COLS_PER_WARP) + (n % (WARPS_PER_BLOCK * B_COLS_PER_WARP) > 0))
      +
      (side == HIPBLAS_SIDE_RIGHT) * (m / (WARPS_PER_BLOCK * B_COLS_PER_WARP) + (m % (WARPS_PER_BLOCK * B_COLS_PER_WARP) > 0))
      , 1);
    int mb = (side == HIPBLAS_SIDE_LEFT) * m / WARP + (side == HIPBLAS_SIDE_RIGHT) * n / WARP;
    //TODO validate with this run from magma ./testing/testing_dpotri_gpu --dev 1 --range 512:15360:512
    trmm_kernels[func_idx]<<< gridDim, blockDim, 0, curStream>>> (m, n, *alpha, A, incA, B, incB, mb);
    check_error_ret( hipGetLastError(),  HIPBLAS_STATUS_EXECUTION_FAILED );
  }else{
    //error: we should not reach this case
    return HIPBLAS_STATUS_INTERNAL_ERROR;
  }
  return HIPBLAS_STATUS_SUCCESS;
}
#else
template<class T>
hipblasStatus_t Xtrmm(hipblasHandle_t handle,
                     hipblasSideMode_t side, hipblasFillMode_t uplo,
                     hipblasOperation_t trans, hipblasDiagType_t diag,
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
hipblasStatus_t kblasXtrmm(hipblasHandle_t handle,
                          hipblasSideMode_t side, hipblasFillMode_t uplo,
                          hipblasOperation_t trans, hipblasDiagType_t diag,
                          int m, int n,
                          const T *alpha,
                          const T *A, int incA,
                                T *B, int incB)
{
  T one = make_one<T>();
  hipblasStatus_t status;

  if( (*alpha == make_zero<T>())//TODO
   || ( (side == HIPBLAS_SIDE_LEFT) && (SIMPLE_SIZE(m)) )
   || ( (side == HIPBLAS_SIDE_RIGHT) && (SIMPLE_SIZE(n)) ) ){
    return Xtrmm(handle,
                 side, uplo, trans, diag,
                 m, n,
                 alpha, A, incA,
                        B, incB );
  }else
  if(side == HIPBLAS_SIDE_LEFT){

    int m1, m2;
    if(REG_SIZE(m))
      m1 = m2 = m/2;
    else{
      m1 = CLOSEST_REG_SIZE(m);
      m2 = m-m1;
    }
    hipblasOperation_t noTrans = HIPBLAS_OP_N;//Trans = HIPBLAS_OP_T,

    if(uplo == HIPBLAS_FILL_MODE_UPPER){

      //Left / Upper / NoTrans
      if(trans == HIPBLAS_OP_N){
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, A, incA,
                                       B, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m1, n, m2,
                                 alpha, A+m1*incA, incA,
                                        B+m1, incB,
                                 &one,  B, incB)) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, A+m1+m1*incA, incA,
                                       B+m1, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Upper / [Conj]Trans
      else{
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, A+m1+m1*incA, incA,
                                       B+m1, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m2, n, m1,
                                 alpha, A+m1*incA, incA,
                                        B, incB,
                                 &one,  B+m1, incB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, A, incA,
                                       B, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }

    }else{//uplo == Lower

      //Left / Lower / NoTrans
      if(trans == HIPBLAS_OP_N){
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, A+m1+m1*incA, incA,
                                       B+m1, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m2, n, m1,
                                 alpha, A+m1, incA,
                                        B, incB,
                                 &one,  B+m1, incB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, A, incA,
                                       B, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Lower / [Conj]Trans
      else{//trans == Trans
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, A, incA,
                                       B, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m1, n, m2,
                                 alpha, A+m1, incA,
                                        B+m1, incB,
                                 &one,  B, incB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, A+m1+m1*incA, incA,
                                       B+m1, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
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

    if(uplo == HIPBLAS_FILL_MODE_UPPER){
      //Right / Upper / NoTrans
      if(trans == HIPBLAS_OP_N){
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, A+n1+n1*incA, incA,
                                       B+n1*incB, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 HIPBLAS_OP_N, trans,
                                 m, n2, n1,
                                 alpha, B, incB,
                                        A+n1*incA, incA,
                                 &one,  B+n1*incB, incB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, A, incA,
                                       B, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Upper / [Conj]Trans
      else{
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, A, incA,
                                       B, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 HIPBLAS_OP_N, trans,
                                 m, n1, n2,
                                 alpha, B+n1*incB, incB,
                                        A+n1*incA, incA,
                                 &one,  B, incB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, A+n1+n1*incA, incA,
                                       B+n1*incB, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
    }else{
      //Right / Lower / NoTrans
      if(trans == HIPBLAS_OP_N){
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, A, incA,
                                       B, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 HIPBLAS_OP_N, trans,
                                 m, n1, n2,
                                 alpha, B+n1*incB, incB,
                                        A+n1, incA,
                                 &one,  B, incB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, A+n1+n1*incA, incA,
                                       B+n1*incB, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Lower / [Conj]Trans
      else{
        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, A+n1+n1*incA, incA,
                                       B+n1*incB, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 HIPBLAS_OP_N, trans,
                                 m, n2, n1,
                                 alpha, B, incB,
                                        A+n1, incA,
                                 &one,  B+n1*incB, incB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrmm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, A, incA,
                                       B, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
    }

  }//side == Right

  return HIPBLAS_STATUS_SUCCESS;
}

//==============================================================================================
template<class T>
hipblasStatus_t kblasXtrmm(hipblasHandle_t handle, hipStream_t &strIn, hipStream_t &strOut,
                          hipblasSideMode_t side, hipblasFillMode_t uplo,
                          hipblasOperation_t trans, hipblasDiagType_t diag,
                          int m, int n,
                          const T *alpha,
                          const T *h_A, int ldA, T* d_A, int lddA,
                                T *h_B, int ldB, T* d_B, int lddB,
                          bool BIsIn, bool getBOut, bool AIsIn)
{
  T one = make_one<T>();
  hipblasStatus_t status;
  hipEvent_t eDataIn, eComp;
  check_error_ret( hipEventCreateWithFlags(&eDataIn, hipEventDisableTiming), HIPBLAS_STATUS_EXECUTION_FAILED);
  check_error_ret( hipEventCreateWithFlags(&eComp, hipEventDisableTiming), HIPBLAS_STATUS_EXECUTION_FAILED);
  hipStream_t strComp;
  check_error_ret( hipblasGetStream(handle, &strComp), HIPBLAS_STATUS_INTERNAL_ERROR);

  if( ( *alpha == make_zero<T>() ) //TODO
   || ( (side == HIPBLAS_SIDE_LEFT) && (SIMPLE_SIZE(m)) )
   || ( (side == HIPBLAS_SIDE_RIGHT) && (SIMPLE_SIZE(n)) ) ){

    int Am = (side == HIPBLAS_SIDE_LEFT) ? m : n;
    //if B is not already in, copy in B block
    if(!BIsIn)
      check_error_ret( status = hipblasSetMatrixAsync( m, n, sizeof(T), h_B, ldB, d_B, lddB, strIn ), status);
    //copy in A block
    if(!AIsIn)
      check_error_ret( status = hipblasSetMatrixAsync( Am, Am, sizeof(T), h_A, ldA, d_A, lddA, strIn ), status);
    //wait for data to arrive
    if(!AIsIn || !BIsIn){
      check_error_ret( hipEventRecord(eDataIn, strIn), HIPBLAS_STATUS_INTERNAL_ERROR);
      check_error_ret( hipStreamWaitEvent(strComp, eDataIn, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
    }
    if( (status = Xtrmm( handle,
                         side, uplo, trans, diag,
                         m, n,
                         alpha, d_A, lddA,
                                d_B, lddB ) ) != HIPBLAS_STATUS_SUCCESS ) return status;

    //if stream is done computing and getBOut, copy B back.
    if(getBOut){
      check_error_ret( hipEventRecord(eComp, strComp), HIPBLAS_STATUS_INTERNAL_ERROR);
      check_error_ret( hipStreamWaitEvent(strOut, eComp, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
      check_error_ret( status = hipblasGetMatrixAsync( m, n, sizeof(T), d_B, lddB, h_B, ldB, strOut), status);
    }
  }else
  if(side == HIPBLAS_SIDE_LEFT){

    int m1, m2;
    if(REG_SIZE(m))
      m1 = m2 = m/2;
    else{
      m1 = CLOSEST_REG_SIZE(m);
      m2 = m-m1;
    }
    hipblasOperation_t noTrans = HIPBLAS_OP_N;//Trans = HIPBLAS_OP_T,

    if( (!AIsIn && SIMPLE_SIZE_DATA(m)) || (!BIsIn && SIMPLE_SIZE_DATA(m)) ){
      if( (!AIsIn && SIMPLE_SIZE_DATA(m)) ){
        check_error_ret( status = hipblasSetMatrixAsync( m, m, sizeof(T), h_A, ldA, d_A, lddA, strIn), status);
        AIsIn = true;
      }
      if( (!BIsIn && SIMPLE_SIZE_DATA(m)) ){
        check_error_ret( status = hipblasSetMatrixAsync( m, n, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
        BIsIn = true;
      }
      //wait for data to arrive
      check_error_ret( hipEventRecord(eDataIn, strIn), HIPBLAS_STATUS_INTERNAL_ERROR);
      check_error_ret( hipStreamWaitEvent(strComp, eDataIn, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
    }
    if(uplo == HIPBLAS_FILL_MODE_UPPER){

      //Left / Upper / NoTrans
      if(trans == HIPBLAS_OP_N){
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, false, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = hipblasSetMatrixAsync( m2, n, sizeof(T), h_B + m1, ldB, d_B + m1, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = hipblasSetMatrixAsync( m1, m2, sizeof(T), h_A + m1 * ldA, ldA, d_A + m1 * lddA, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( hipEventRecord(eDataIn, strIn), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strComp, eDataIn, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m1, n, m2,
                                 alpha, d_A + m1 * lddA, lddA,
                                        d_B + m1, lddB,
                                 &one,  d_B, lddB)) != HIPBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( hipEventRecord(eComp, strComp), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strOut, eComp, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = hipblasGetMatrixAsync( m1, n, sizeof(T), d_B, lddB, h_B, ldB, strOut), status);
        }

        //B is already in, no need to copy in
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, h_A + m1 + m1 * ldA, ldA, d_A + m1 + m1 * lddA, lddA,
                                       h_B + m1, ldB, d_B + m1, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Upper / [Conj]Trans
      else{
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, h_A+m1+m1*ldA, ldA, d_A+m1+m1*lddA, lddA,
                                       h_B+m1, ldB, d_B+m1, lddB,
                                BIsIn, false, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = hipblasSetMatrixAsync( m1, n, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = hipblasSetMatrixAsync( m1, m2, sizeof(T), h_A + m1 * ldA, ldA, d_A + m1 * lddA, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( hipEventRecord(eDataIn, strIn), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strComp, eDataIn, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m2, n, m1,
                                 alpha, d_A+m1*lddA, lddA,
                                        d_B, lddB,
                                 &one,  d_B+m1, lddB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( hipEventRecord(eComp, strComp), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strOut, eComp, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = hipblasGetMatrixAsync( m2, n, sizeof(T), d_B+m1, lddB, h_B+m1, ldB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }

    }else{//uplo == Lower

      //Left / Lower / NoTrans
      if(trans == HIPBLAS_OP_N){
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, h_A+m1+m1*ldA, ldA, d_A+m1+m1*lddA, lddA,
                                       h_B+m1, ldB, d_B+m1, lddB,
                                BIsIn, false, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = hipblasSetMatrixAsync( m1, n, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = hipblasSetMatrixAsync( m2, m1, sizeof(T), h_A + m1, ldA, d_A + m1, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( hipEventRecord(eDataIn, strIn), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strComp, eDataIn, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m2, n, m1,
                                 alpha, d_A+m1, lddA,
                                        d_B, lddB,
                                 &one,  d_B+m1, lddB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( hipEventRecord(eComp, strComp), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strOut, eComp, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = hipblasGetMatrixAsync( m2, n, sizeof(T), d_B+m1, lddB, h_B+m1, ldB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Lower / [Conj]Trans
      else{//trans == Trans
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, false, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = hipblasSetMatrixAsync( m2, n, sizeof(T), h_B + m1, ldB, d_B + m1, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = hipblasSetMatrixAsync( m2, m1, sizeof(T), h_A + m1, ldA, d_A + m1, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( hipEventRecord(eDataIn, strIn), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strComp, eDataIn, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
        }

        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m1, n, m2,
                                 alpha, d_A+m1, lddA,
                                        d_B+m1, lddB,
                                 &one,  d_B, lddB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( hipEventRecord(eComp, strComp), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strOut, eComp, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = hipblasGetMatrixAsync( m1, n, sizeof(T), d_B, lddB, h_B, ldB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, h_A+m1+m1*ldA, ldA, d_A+m1+m1*lddA, lddA,
                                       h_B+m1, ldB, d_B+m1, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
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
        check_error_ret( status = hipblasSetMatrixAsync( n, n, sizeof(T), h_A, ldA, d_A, lddA, strIn), status);
        AIsIn = true;
      }
      if( (!BIsIn && SIMPLE_SIZE_DATA(n)) ){
        check_error_ret( status = hipblasSetMatrixAsync( m, n, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
        BIsIn = true;
      }
      //wait for data to arrive
      check_error_ret( hipEventRecord(eDataIn, strIn), HIPBLAS_STATUS_INTERNAL_ERROR);
      check_error_ret( hipStreamWaitEvent(strComp, eDataIn, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
    }
    if(uplo == HIPBLAS_FILL_MODE_UPPER){
      //Right / Upper / NoTrans
      if(trans == HIPBLAS_OP_N){
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, h_A+n1+n1*ldA, ldA, d_A+n1+n1*lddA, lddA,
                                       h_B+n1*ldB, ldB, d_B+n1*lddB, lddB,
                                BIsIn, false, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = hipblasSetMatrixAsync( m, n1, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = hipblasSetMatrixAsync( n1, n2, sizeof(T), h_A + n1 * ldA, ldA, d_A + n1 * lddA, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( hipEventRecord(eDataIn, strIn), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strComp, eDataIn, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 HIPBLAS_OP_N, trans,
                                 m, n2, n1,
                                 alpha, d_B, lddB,
                                        d_A+n1*lddA, lddA,
                                 &one,  d_B+n1*lddB, lddB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( hipEventRecord(eComp, strComp), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strOut, eComp, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = hipblasGetMatrixAsync( m, n2, sizeof(T), d_B+n1*lddB, lddB, h_B+n1*ldB, ldB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Upper / [Conj]Trans
      else{
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, false, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = hipblasSetMatrixAsync( m, n2, sizeof(T), h_B+n1*ldB, ldB, d_B+n1*lddB, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = hipblasSetMatrixAsync( n1, n2, sizeof(T), h_A+n1*ldA, ldA, d_A+n1*lddA, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( hipEventRecord(eDataIn, strIn), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strComp, eDataIn, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 HIPBLAS_OP_N, trans,
                                 m, n1, n2,
                                 alpha, d_B+n1*lddB, lddB,
                                        d_A+n1*lddA, lddA,
                                 &one,  d_B, lddB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( hipEventRecord(eComp, strComp), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strOut, eComp, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = hipblasGetMatrixAsync( m, n1, sizeof(T), d_B, lddB, h_B, ldB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, h_A+n1+n1*ldA, ldA, d_A+n1+n1*lddA, lddA,
                                       h_B+n1*ldB, ldB, d_B+n1*lddB, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
    }else{
      //Right / Lower / NoTrans
      if(trans == HIPBLAS_OP_N){
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, false, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = hipblasSetMatrixAsync( m, n2, sizeof(T), h_B+n1*ldB, ldB, d_B+n1*lddB, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = hipblasSetMatrixAsync( n2, n1, sizeof(T), h_A+n1, ldA, d_A+n1, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( hipEventRecord(eDataIn, strIn), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strComp, eDataIn, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 HIPBLAS_OP_N, trans,
                                 m, n1, n2,
                                 alpha, d_B+n1*lddB, lddB,
                                        d_A+n1, lddA,
                                 &one,  d_B, lddB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( hipEventRecord(eComp, strComp), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strOut, eComp, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = hipblasGetMatrixAsync( m, n1, sizeof(T), d_B, lddB, h_B, ldB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, h_A+n1+n1*ldA, ldA, d_A+n1+n1*lddA, lddA,
                                       h_B+n1*ldB, ldB, d_B+n1*lddB, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Lower / [Conj]Trans
      else{
        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, h_A+n1+n1*ldA, ldA, d_A+n1+n1*lddA, lddA,
                                       h_B+n1*ldB, ldB, d_B+n1*lddB, lddB,
                                BIsIn, false, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = hipblasSetMatrixAsync( m, n1, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = hipblasSetMatrixAsync( n1, n2, sizeof(T), h_A+n1, ldA, d_A+n1, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( hipEventRecord(eDataIn, strIn), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strComp, eDataIn, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 HIPBLAS_OP_N, trans,
                                 m, n2, n1,
                                 alpha, d_B, lddB,
                                        d_A+n1, lddA,
                                 &one,  d_B+n1*lddB, lddB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( hipEventRecord(eComp, strComp), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strOut, eComp, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = hipblasGetMatrixAsync( m, n2, sizeof(T), d_B+n1*lddB, lddB, h_B+n1*ldB, ldB, strOut), status);
        }

        if((status = kblasXtrmm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
    }

  }//side == Right

  check_error_ret( hipEventDestroy( eDataIn ), HIPBLAS_STATUS_INTERNAL_ERROR);
  check_error_ret( hipEventDestroy( eComp ), HIPBLAS_STATUS_INTERNAL_ERROR);
  check_error_ret( hipGetLastError(), HIPBLAS_STATUS_EXECUTION_FAILED );
  return HIPBLAS_STATUS_SUCCESS;
}

//==============================================================================================
template<class T>
hipblasStatus_t kblasXtrmm_cpu(hipblasHandle_t handle,
                              hipblasSideMode_t side, hipblasFillMode_t uplo,
                              hipblasOperation_t trans, hipblasDiagType_t diag,
                              int m, int n,
                              const T *alpha,
                              const T *h_A, int ldA,
                                    T *h_B, int ldB){
  //allocate memory on device
  T *d_A, *d_B;
  int Am, An, Bm, Bn, lddA, lddB;
  if ( side == HIPBLAS_SIDE_LEFT ) {
    Am = An = m;
  } else {
    Am = An = n;
  }
  Bm = m;
  Bn = n;
  lddA = ((Am+31)/32)*32;
  lddB = ((Bm+31)/32)*32;

  /*check_error_ret( hipHostRegister((void*)h_A, Am * An * sizeof(T), hipHostRegisterDefault), HIPBLAS_STATUS_INTERNAL_ERROR);
  check_error_ret( hipHostRegister((void*)h_B, Bm * Bn * sizeof(T), hipHostRegisterDefault), HIPBLAS_STATUS_INTERNAL_ERROR);*/

  hipblasStatus_t status;
  int AsyncEngineCount, devID;
  check_error_ret( hipGetDevice(&devID), HIPBLAS_STATUS_INTERNAL_ERROR);
//   check_error_ret( hipDeviceGetAttribute(&AsyncEngineCount, cudaDevAttrAsyncEngineCount, devID), HIPBLAS_STATUS_INTERNAL_ERROR);
//   bool DO_INLINE_BOUT = AsyncEngineCount > 1;
  bool DO_INLINE_BOUT = false;
  check_error_ret( hipMalloc( (void**)&d_A, (lddA*An)*sizeof(T) ), HIPBLAS_STATUS_INTERNAL_ERROR);
  if(d_A == NULL) return HIPBLAS_STATUS_ALLOC_FAILED;
  check_error_ret( hipMalloc( (void**)&d_B, (lddB*Bn)*sizeof(T) ), HIPBLAS_STATUS_INTERNAL_ERROR);
  if(d_B == NULL) return HIPBLAS_STATUS_ALLOC_FAILED;

  //setup streams
  hipStream_t inStream, outStream;
  check_error_ret( hipStreamCreateWithFlags( &inStream, hipStreamNonBlocking), HIPBLAS_STATUS_INTERNAL_ERROR );
  if(DO_INLINE_BOUT)
    check_error_ret( hipStreamCreateWithFlags( &outStream, hipStreamNonBlocking), HIPBLAS_STATUS_INTERNAL_ERROR );

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
    check_error_ret( hipStreamSynchronize( outStream ), HIPBLAS_STATUS_INTERNAL_ERROR);
  }else{
    hipStream_t compStream;
    check_error_ret( hipblasGetStream(handle, &compStream), HIPBLAS_STATUS_INTERNAL_ERROR);
    check_error_ret( hipStreamSynchronize( compStream ), HIPBLAS_STATUS_INTERNAL_ERROR);
    check_error_ret( status = hipblasGetMatrixAsync( m, n, sizeof(T), d_B, lddB, h_B, ldB, inStream), status);
  }
  //revoke streams
  check_error_ret( hipStreamDestroy( inStream ), HIPBLAS_STATUS_INTERNAL_ERROR);
  if(DO_INLINE_BOUT)
    check_error_ret( hipStreamDestroy( outStream ), HIPBLAS_STATUS_INTERNAL_ERROR);

  /*check_error_ret( hipHostUnregister( (void*)h_A ), HIPBLAS_STATUS_INTERNAL_ERROR );
  check_error_ret( hipHostUnregister( (void*)h_B ), HIPBLAS_STATUS_INTERNAL_ERROR );*/

  //free device memory
  //TODO should free aslo in case some other funtions above failed (don't just return)
  check_error_ret( hipFree( d_A ), HIPBLAS_STATUS_INTERNAL_ERROR );
  check_error_ret( hipFree( d_B ), HIPBLAS_STATUS_INTERNAL_ERROR );
  check_error_ret( hipGetLastError(), HIPBLAS_STATUS_EXECUTION_FAILED );
  return HIPBLAS_STATUS_SUCCESS;
}
//==============================================================================================
template<class T>
hipblasStatus_t kblasXtrmm_cpu_m(hipblasHandle_t handle,
                              hipblasSideMode_t side, hipblasFillMode_t uplo,
                              hipblasOperation_t trans, hipblasDiagType_t diag,
                              int m, int n,
                              const T *alpha,
                              const T *h_A, int ldA,
                                    T *h_B, int ldB,
                              //TODO should accept an array of device IDs or a set of cublas handles intead
                              int ngpu){
  //allocate memory on device
  T *d_A[ngpu], *d_B[ngpu];
  int Am, An, Bm, Bn, lddA, lddB;
  if ( side == HIPBLAS_SIDE_LEFT ) {
    Am = An = m;
  } else {
    Am = An = n;
  }
  Bm = m;
  Bn = n;

  /*check_error_ret( hipHostRegister((void*)h_A, Am * An * sizeof(T), hipHostRegisterDefault), HIPBLAS_STATUS_INTERNAL_ERROR);
  check_error_ret( hipHostRegister((void*)h_B, Bm * Bn * sizeof(T), hipHostRegisterDefault), HIPBLAS_STATUS_INTERNAL_ERROR);*/

  hipStream_t inStream[ngpu], outStream[ngpu];
  hipblasStatus_t status[ngpu];
  hipblasHandle_t cub_handle[ngpu];
  cub_handle[0] = handle;
  //*
  bool DO_INLINE_BOUT[ngpu];
  for(int g = 0; g < ngpu; g++){
    check_error_ret( hipSetDevice(g), HIPBLAS_STATUS_INTERNAL_ERROR);
    // int AsyncEngineCount;
    // check_error_ret( hipDeviceGetAttribute(&AsyncEngineCount, cudaDevAttrAsyncEngineCount, g), HIPBLAS_STATUS_INTERNAL_ERROR);
    // DO_INLINE_BOUT[g] = AsyncEngineCount > 1;
    DO_INLINE_BOUT[g] = false;
    if(g > 0){
      check_error_ret( hipblasCreate(&cub_handle[g]), HIPBLAS_STATUS_INTERNAL_ERROR);
    }

    //setup streams
    check_error_ret( hipStreamCreateWithFlags( &inStream[g], hipStreamNonBlocking), HIPBLAS_STATUS_INTERNAL_ERROR);
    if(DO_INLINE_BOUT[g])
      check_error_ret( hipStreamCreateWithFlags( &outStream[g], hipStreamNonBlocking), HIPBLAS_STATUS_INTERNAL_ERROR);
  }
  //*/
  //TODO IMPORTANT: handle when data does not fit on all gpus
  int Bn_gpu, Bm_gpu;
  bool left = (side == HIPBLAS_SIDE_LEFT);
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
      hipSetDevice(g);
      hipMalloc( (void**)&d_A[g], (lddA*An)*sizeof(T) );
      hipMalloc( (void**)&d_B[g], (lddB*Bn_gpu)*sizeof(T) );

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
        hipStreamSynchronize( outStream[g] );
      }else{
        hipStream_t compStream;
        hipblasGetStream(cub_handle[g], &compStream);
        hipStreamSynchronize( compStream );
        status[g] = hipblasGetMatrixAsync( Bm_gpu, Bn_gpu, sizeof(T), d_B[g], lddB, h_B+g*(left ? Bn_gpu*ldB : Bm_gpu), ldB, inStream[g]);
        hipStreamSynchronize( inStream[g] );
      }
    }
  }
  /*/for(int g = 0; g < ngpu; g++){
    check_error_ret( hipSetDevice(g), HIPBLAS_STATUS_INTERNAL_ERROR);
    check_error_ret( hipMalloc( (void**)&d_A[g], (Am*An)*sizeof(T) ), HIPBLAS_STATUS_INTERNAL_ERROR);
    check_error_ret( hipMalloc( (void**)&d_B[g], (Bm_gpu*Bn_gpu)*sizeof(T) ), HIPBLAS_STATUS_INTERNAL_ERROR);

    //setup streams
    check_error_ret( hipStreamCreateWithFlags( &inStream[g], hipStreamNonBlocking), HIPBLAS_STATUS_INTERNAL_ERROR );
    if(DO_INLINE_BOUT[g])
      check_error_ret( hipStreamCreateWithFlags( &outStream[g], hipStreamNonBlocking), HIPBLAS_STATUS_INTERNAL_ERROR );

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
    check_error_ret( hipSetDevice(g), HIPBLAS_STATUS_INTERNAL_ERROR );

    //sync streams
    if(DO_INLINE_BOUT[g]){
      check_error_ret( hipStreamSynchronize( outStream[g] ), HIPBLAS_STATUS_INTERNAL_ERROR);
    }else{
      hipStream_t compStream;
      check_error_ret( hipblasGetStream(cub_handle[g], &compStream), HIPBLAS_STATUS_INTERNAL_ERROR);
      check_error_ret( hipStreamSynchronize( compStream ), HIPBLAS_STATUS_INTERNAL_ERROR);
      check_error_ret( status[g] = hipblasGetMatrixAsync( Bm_gpu, Bn_gpu, sizeof(T), d_B[g], incB, h_B+g*(left ? Bn_gpu*incB : Bm_gpu), incB, inStream[g]), status[g]);
    }
  }//*/


  for(int g = 0; g < ngpu; g++){
    check_error_ret( hipSetDevice(g), HIPBLAS_STATUS_INTERNAL_ERROR);
    check_error_ret( hipDeviceSynchronize(), HIPBLAS_STATUS_INTERNAL_ERROR );
    //revoke streams
    check_error_ret( hipStreamDestroy( inStream[g] ), HIPBLAS_STATUS_INTERNAL_ERROR);
    if(DO_INLINE_BOUT[g])
      check_error_ret( hipStreamDestroy( outStream[g] ), HIPBLAS_STATUS_INTERNAL_ERROR);
  /*check_error_ret( hipHostUnregister( (void*)h_A ), HIPBLAS_STATUS_INTERNAL_ERROR );
  check_error_ret( hipHostUnregister( (void*)h_B ), HIPBLAS_STATUS_INTERNAL_ERROR );*/

    //free device memory
    check_error_ret( hipFree( d_A[g] ), HIPBLAS_STATUS_INTERNAL_ERROR );
    check_error_ret( hipFree( d_B[g] ), HIPBLAS_STATUS_INTERNAL_ERROR );
    if(g > 0){
      check_error_ret( hipblasDestroy(cub_handle[g]), HIPBLAS_STATUS_INTERNAL_ERROR);
    }

  }
  return HIPBLAS_STATUS_SUCCESS;
}


//==============================================================================================

  /*
extern "C" {
int kblas_strmm_async(
  char side, char uplo, char trans, char diag,
  int m, int n,
  float alpha, const float *A, int incA,
                       float *B, int incB,
  hipStream_t    stream){

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
  hipStream_t    stream){

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
  hipComplex alpha, const hipComplex *A, int incA,
                          hipComplex *B, int incB,
  hipStream_t    stream){

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
  hipDoubleComplex alpha, const hipDoubleComplex *A, int incA,
                                hipDoubleComplex *B, int incB,
  hipStream_t    stream){

  return kblasXtrmm(
    side, uplo, trans, diag,
    m, n,
    alpha, A, incA,
           B, incB,
    stream);
}*/
  //==============================================================================================

#define kblasXtrmm_async_BODY {                                                                           \
  hipblasHandle_t cublas_handle;                                                                           \
  check_error_ret( hipblasCreate(&cublas_handle), void() );                                                      \
  if( hipblasSetStream(cublas_handle, stream) != HIPBLAS_STATUS_SUCCESS ){                               \
    check_error_ret( hipblasDestroy(cublas_handle), void());                                                  \
    return;                                                                                               \
  }                                                                                                       \
  hipblasSideMode_t  side_v2  = (side  == KBLAS_Left  ? HIPBLAS_SIDE_LEFT : HIPBLAS_SIDE_RIGHT);             \
  hipblasFillMode_t  uplo_v2  = (uplo  == KBLAS_Lower ? HIPBLAS_FILL_MODE_LOWER : HIPBLAS_FILL_MODE_UPPER);  \
  hipblasOperation_t trans_v2 = (trans == KBLAS_Trans ? HIPBLAS_OP_T : HIPBLAS_OP_N);                        \
  hipblasDiagType_t  diag_v2  = (diag  == KBLAS_Unit  ? HIPBLAS_DIAG_UNIT : HIPBLAS_DIAG_NON_UNIT);          \
                                                                                                          \
  check_error_ret( kblasXtrmm(cublas_handle,                                                                  \
                          side_v2, uplo_v2, trans_v2, diag_v2,                                            \
                          m, n,                                                                           \
                          &alpha, A, lda,                                                                 \
                                  B, ldb), void());                                                         \
                                                                                                          \
  check_error_ret( hipblasDestroy(cublas_handle), void());                                                    \
}

extern "C"{
void kblasStrmm_async(char side, char uplo, char trans, char diag,
                      int m, int n,
                      float alpha, const float *A, int lda,
                                         float *B, int ldb,
                      hipStream_t stream){
  kblasXtrmm_async_BODY
}

void kblasDtrmm_async(char side, char uplo, char trans, char diag,
                      int m, int n,
                      double alpha, const double *A, int lda,
                                          double *B, int ldb,
                      hipStream_t stream){
  kblasXtrmm_async_BODY
}
// void kblasCtrmm_async(char side, char uplo, char trans, char diag,
//                       int m, int n,
//                       hipComplex alpha, const hipComplex *A, int lda,
//                                              hipComplex *B, int ldb,
//                       hipStream_t stream){
//   kblasXtrmm_async_BODY
// }
// void kblasZtrmm_async(char side, char uplo, char trans, char diag,
//                       int m, int n,
//                       hipDoubleComplex alpha, const hipDoubleComplex *A, int lda,
//                                                    hipDoubleComplex *B, int ldb,
//                       hipStream_t stream){
//   kblasXtrmm_async_BODY
// }
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
// void kblasCtrmm(char side, char uplo, char trans, char diag,
//                 int m, int n,
//                 hipComplex alpha, const hipComplex *A, int lda,
//                                        hipComplex *B, int ldb){

//   kblasCtrmm_async(side, uplo, trans, diag,
//                    m, n,
//                    alpha, A, lda,
//                           B, ldb,
//                    0);

// }
// void kblasZtrmm(char side, char uplo, char trans, char diag,
//                 int m, int n,
//                 hipDoubleComplex alpha, const hipDoubleComplex *A, int lda,
//                                              hipDoubleComplex *B, int ldb){

//   kblasZtrmm_async(side, uplo, trans, diag,
//                    m, n,
//                    alpha, A, lda,
//                           B, ldb,
//                    0);
// }
//==============================================================================================

hipblasStatus_t kblasStrmm(hipblasHandle_t handle,
                          hipblasSideMode_t side, hipblasFillMode_t uplo,
                          hipblasOperation_t trans, hipblasDiagType_t diag,
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
hipblasStatus_t kblasDtrmm(hipblasHandle_t handle,
                          hipblasSideMode_t side, hipblasFillMode_t uplo,
                          hipblasOperation_t trans, hipblasDiagType_t diag,
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
// hipblasStatus_t kblasCtrmm(hipblasHandle_t handle,
//                           hipblasSideMode_t side, hipblasFillMode_t uplo,
//                           hipblasOperation_t trans, hipblasDiagType_t diag,
//                           int m, int n,
//                           const hipComplex *alpha,
//                           const hipComplex *A, int lda,
//                                 hipComplex *B, int ldb){
//   return kblasXtrmm(handle,
//                     side, uplo, trans, diag,
//                     m, n,
//                     alpha, A, lda,
//                            B, ldb);
// }
// hipblasStatus_t kblasZtrmm(hipblasHandle_t handle,
//                           hipblasSideMode_t side, hipblasFillMode_t uplo,
//                           hipblasOperation_t trans, hipblasDiagType_t diag,
//                           int m, int n,
//                           const hipDoubleComplex *alpha,
//                           const hipDoubleComplex *A, int lda,
//                                 hipDoubleComplex *B, int ldb){
//   return kblasXtrmm(handle,
//                     side, uplo, trans, diag,
//                     m, n,
//                     alpha, A, lda,
//                            B, ldb);
// }


//==============================================================================================
hipblasStatus_t kblas_strmm(hipblasHandle_t handle,
                           hipblasSideMode_t side, hipblasFillMode_t uplo,
                           hipblasOperation_t trans, hipblasDiagType_t diag,
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

hipblasStatus_t kblas_dtrmm(hipblasHandle_t handle,
                           hipblasSideMode_t side, hipblasFillMode_t uplo,
                           hipblasOperation_t trans, hipblasDiagType_t diag,
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
// hipblasStatus_t kblas_ctrmm(hipblasHandle_t handle,
//                            hipblasSideMode_t side, hipblasFillMode_t uplo,
//                            hipblasOperation_t trans, hipblasDiagType_t diag,
//                            int m, int n,
//                            const hipComplex *alpha,
//                            const hipComplex *A, int lda,
//                                  hipComplex *B, int ldb){
//   return kblasXtrmm_cpu(handle,
//                         side, uplo, trans, diag,
//                         m, n,
//                         alpha, A, lda,
//                                B, ldb);
// }
// hipblasStatus_t kblas_ztrmm(hipblasHandle_t handle,
//                            hipblasSideMode_t side, hipblasFillMode_t uplo,
//                            hipblasOperation_t trans, hipblasDiagType_t diag,
//                            int m, int n,
//                            const hipDoubleComplex *alpha,
//                            const hipDoubleComplex *A, int lda,
//                                  hipDoubleComplex *B, int ldb){
//   return kblasXtrmm_cpu(handle,
//                         side, uplo, trans, diag,
//                         m, n,
//                         alpha, A, lda,
//                                B, ldb);
// }

//==============================================================================================
hipblasStatus_t kblas_strmm_mgpu(hipblasHandle_t handle,
                                hipblasSideMode_t side, hipblasFillMode_t uplo,
                                hipblasOperation_t trans, hipblasDiagType_t diag,
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
hipblasStatus_t kblas_dtrmm_mgpu(hipblasHandle_t handle,
                                hipblasSideMode_t side, hipblasFillMode_t uplo,
                                hipblasOperation_t trans, hipblasDiagType_t diag,
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

// hipblasStatus_t kblas_ctrmm_mgpu(hipblasHandle_t handle,
//                                 hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                 hipblasOperation_t trans, hipblasDiagType_t diag,
//                                 int m, int n,
//                                 const hipComplex *alpha,
//                                 const hipComplex *A, int lda,
//                                       hipComplex *B, int ldb,
//                                 int ngpu){
//   return kblasXtrmm_cpu_m(handle,
//                         side, uplo, trans, diag,
//                         m, n,
//                         alpha, A, lda,
//                               B, ldb,
//                         ngpu);
// }
// hipblasStatus_t kblas_ztrmm_mgpu(hipblasHandle_t handle,
//                                 hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                 hipblasOperation_t trans, hipblasDiagType_t diag,
//                                 int m, int n,
//                                 const hipDoubleComplex *alpha,
//                                 const hipDoubleComplex *A, int lda,
//                                       hipDoubleComplex *B, int ldb,
//                                 int ngpu){
//   return kblasXtrmm_cpu_m(handle,
//                         side, uplo, trans, diag,
//                         m, n,
//                         alpha, A, lda,
//                               B, ldb,
//                         ngpu);
// }

//}


