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
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include "kblas.h"
#include "kblas_common.h"
#include "operators.h"
#include "omp.h"

//==============================================================================================

cublasStatus_t cublasXtrsm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const float *alpha,
                           const float *A, int lda,
                           float *B, int ldb){
  cublasStatus_t status;
  check_error( status = cublasStrsm(handle,
                                    side, uplo, trans, diag,
                                    m, n,
                                    alpha, A, lda,
                                           B, ldb ), status);
  check_error( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
  return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasXtrsm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t      diag,
                           int m, int n,
                           const double *alpha,
                           const double *A, int lda,
                                 double *B, int ldb){
  cublasStatus_t status;
  check_error( status = cublasDtrsm(handle,
                                    side, uplo, trans, diag,
                                    m, n,
                                    alpha, A, lda,
                                           B, ldb ), status);
  check_error( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
  return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasXtrsm (cublasHandle_t handle,
                            cublasSideMode_t side, cublasFillMode_t uplo,
                            cublasOperation_t trans, cublasDiagType_t diag,
                            int m, int n,
                            const cuComplex *alpha,
                            const cuComplex *A, int lda,
                                  cuComplex *B, int ldb){
  cublasStatus_t status;
  check_error( status = cublasCtrsm(handle,
                                    side, uplo, trans, diag,
                                    m, n,
                                    alpha, A, lda,
                                           B, ldb ), status);
  check_error( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
  return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasXtrsm (cublasHandle_t handle,
                            cublasSideMode_t side, cublasFillMode_t uplo,
                            cublasOperation_t trans, cublasDiagType_t diag,
                            int m, int n,
                            const cuDoubleComplex *alpha,
                            const cuDoubleComplex *A, int lda,
                                  cuDoubleComplex *B, int ldb){
  cublasStatus_t status;
  check_error( status = cublasZtrsm(handle,
                                    side, uplo, trans, diag,
                                    m, n,
                                    alpha, A, lda,
                                           B, ldb ), status);
  check_error( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
  return CUBLAS_STATUS_SUCCESS;
}

//==============================================================================================
#define WARP 32
//#define WARP1 33
#define WARP2 34
#define tx threadIdx.x
#define ty threadIdx.y
//==============================================================================================
int kblas_trsm_ib_cublas = 128;
bool kblas_trsm_use_custom = 0;
int kblas_trsm_ib_data = 512;
#define SIMPLE_SIZE(n) ( ((n) < WARP) || ( ((n) % WARP == 0) && ( (n) <= kblas_trsm_ib_cublas ) ) )
#define SIMPLE_SIZE_DATA(n) ( (n) <= kblas_trsm_ib_data )
//==============================================================================================

//shuffle intrinsic is not supported before KEPLER
#if (SM >= 30)
template<typename T, int WARPS_PER_BLOCK, bool LOWER, bool TRANS, bool CONJG, bool UNIT>
__global__ void //__launch_bounds__(WARP * WARPS_PER_BLOCK)
trsm_mul32_L(int M, int N, T alpha, const T* /*__restrict__*/ A, int incA, T* B, int incB, int mb)
{
  const int A_COLS_PER_WARP = WARP / WARPS_PER_BLOCK;
  const bool forward = (LOWER != TRANS);
  const short WARP1 = (TRANS ? 33 : 32);

  //setup shared memory
  __shared__ T sA[WARP * WARP1];//strided to avoid bank conflicts

  int txyw = tx + ty * WARP1, txyiA = tx + ty * incA, txyiB = tx + ty * incB, jtxw;
  int l, c, r, startB = 0, i;
  T rB, s, rBj, a[4], b[4], *sAA, *BB;

  for(startB = 0; startB < N; startB += gridDim.x * WARPS_PER_BLOCK)
  {

    if( (blockIdx.x * WARPS_PER_BLOCK + startB) >= N)
      return;

    BB = B + (blockIdx.x * WARPS_PER_BLOCK + startB) * incB;

    //checking boundary case, the column indices of B this warp is computing
    //if not active, this warp will only participate in fetching A sub-matrices, will not compute
    bool active = ( (blockIdx.x * WARPS_PER_BLOCK + startB + ty) < N );

    for(c = (forward ? 0 : mb-1); (forward && c < mb) || (!forward && c >= 0); c += (forward ? 1 : -1))
    {
      s = make_zero<T>();

      for(r = (forward ? 0 : mb-1); (forward && r < c) || (!forward && r > c); r += (forward ? 1 : -1))
      {
        #pragma unroll
        for(l = 0; l < A_COLS_PER_WARP; l++){
          if(TRANS)
            //load A(r,c)
            sA[txyw + l * WARPS_PER_BLOCK * WARP1] = A[txyiA + WARP * (r + c * incA) + l * WARPS_PER_BLOCK * incA];
          else
            //load A(c,r)
            sA[txyw + l * WARPS_PER_BLOCK * WARP1] = __ldg(&(A[txyiA + WARP * (c + r * incA) + l * WARPS_PER_BLOCK * incA]));
        }
        //load B(r)
        if(active)
          rB = __ldg(&(BB[txyiB + WARP * r]));

        __syncthreads();
        if(active){
          //gemm A(r,c)/A(c,r) & B(r) onto B(c) held at s
          if(TRANS)
            sAA = sA + tx*WARP1;
          else
            sAA = sA + tx;
            #pragma unroll
            for(int j = 0; j < WARP; j+=4){
              if(TRANS){
                #pragma unroll
                for(i = 0; i < 4; i++)
                  a[i] = CONJG ? conjugate(sAA[j + i]) : sAA[j + i];
              }else{
                #pragma unroll
                for(i = 0; i < 4; i++)
                  a[i] = sAA[(j + i)*WARP1];
              }
              #pragma unroll
              for(i = 0; i < 4; i++)
                b[i] = shfl(rB, j + i);
              #pragma unroll
              for(i = 0; i < 4; i++)
                s = FMA( a[i], b[i], s );
            }
        }
        __syncthreads();
      }

      //load A(c,c) from global to shared mem
      #pragma unroll
      for(l = 0; l < A_COLS_PER_WARP; l++){
        sA[txyw + l * WARPS_PER_BLOCK * WARP1] = __ldg(&(A[txyiA + WARP * c * (incA + 1) + l * WARPS_PER_BLOCK * incA]));
      }

      //load B(c) into registers
      if(active){
        rB = __ldg(&(BB[txyiB + WARP * c]));
      }
      __syncthreads();
      if(active)
      {
        //perform trsm on shared mem
        if(!LOWER && TRANS)
          jtxw = tx * WARP1;
        else
        if(!LOWER && !TRANS)
          jtxw = tx         + (WARP - 1) * WARP1;
        else
        if(LOWER && TRANS)
          jtxw = tx * WARP1 + (WARP - 1);
        else
        if(LOWER && !TRANS)
          jtxw = tx;

        #pragma unroll
        for(int j = (forward ? 0 : WARP-1); (forward && (j < WARP)) || (!forward && (j >= 0)); j += (forward ? 1 : -1)){
          if(j == tx){
            rB = FMA(alpha, rB, -s);//TODO
            if(!UNIT){
              a[0] = (TRANS && CONJG) ? conjugate(sA[tx * (WARP1+1)]) : sA[tx * (WARP1+1)];//diagonal element
              rB = rB / a[0];//TODO
            }
          }
          rBj = shfl(rB, j);

          if( (forward && (j < tx)) || (!forward && (j > tx)) ){
            a[0] = (TRANS && CONJG) ? conjugate(sA[jtxw]) : sA[jtxw];
            s = FMA(a[0], rBj, s);
          }
          jtxw += (TRANS ? 1 : WARP1) * (forward ? 1 : -1);
        }

        //store back B(c) to global mem
        BB[txyiB + WARP * c] = rB;
      }
      __syncthreads();
    }
  }
}


//==============================================================================================
#define TRSM_NUM_VARIANTS 4
#define TRSM_kernel_variants(__WPB)                  \
        trsm_mul32_L<T, __WPB,  true, false, false, false>, \
        trsm_mul32_L<T, __WPB,  true,  true, false, false>, \
        trsm_mul32_L<T, __WPB, false, false, false, false>, \
        trsm_mul32_L<T, __WPB, false,  true, false, false>
        /*,TODO
        trsm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP,  true, false, false>,
        trsm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP,  true,  true, false>,
        trsm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, false, false, false>,
        trsm_mul32_R<T, WARPS_PER_BLOCK, B_COLS_PER_WARP, false,  true, false>*/
template<class T>
cublasStatus_t Xtrsm(cublasHandle_t handle,
                     cublasSideMode_t side, cublasFillMode_t uplo,
                     cublasOperation_t trans, cublasDiagType_t diag,
                     int m, int n,
                     const T *alpha,
                     const T *A, int incA,
                           T *B, int incB){

  //handle odd cases with cublas
  if(  (*alpha == make_zero<T>())
    || (!kblas_trsm_use_custom)
    || (side == CUBLAS_SIDE_LEFT && m < WARP)
    || (side == CUBLAS_SIDE_RIGHT/* && n < WARP*/))//TODO
  {
    return cublasXtrsm(handle,
                       side, uplo, trans, diag,
                       m, n,
                       alpha, A, incA,
                              B, incB );
  }

  typedef void (*trsm_kernels_type)(int M, int N, T alpha, const T* A, int incA, T* B, int incB, int mb);

  #define WARPS_PER_BLOCK 8
  #define B_COLS_PER_WARP 1

  trsm_kernels_type trsm_kernels[TRSM_NUM_VARIANTS] = {// T, WARPS_PER_BLOCK, LOWER, TRANS, CONJG, UNIT
    TRSM_kernel_variants(WARPS_PER_BLOCK)
  };

  cudaStream_t curStream;
  cublasStatus_t status;

  check_error( status = cublasGetStream( handle, &curStream ), status);

  if( ((side == CUBLAS_SIDE_LEFT) && (m % WARP == 0)) /*|| ((side == CUBLAS_SIDE_RIGHT) && (n % WARP == 0))*/ )//TODO
  {
    int func_idx = /*4*(side == CUBLAS_SIDE_RIGHT) + */2*(uplo == CUBLAS_FILL_MODE_UPPER) + (trans != CUBLAS_OP_N);// + (diag == CUBLAS_DIAG_UNIT);TODO
    dim3 blockDim( WARP, WARPS_PER_BLOCK );
    dim3 gridDim(
      (side == CUBLAS_SIDE_LEFT) * (n / (WARPS_PER_BLOCK * B_COLS_PER_WARP) + (n % (WARPS_PER_BLOCK * B_COLS_PER_WARP) > 0))
      /*+TODO
      (side == CUBLAS_SIDE_RIGHT) * (m / (WARPS_PER_BLOCK * B_COLS_PER_WARP) + (m % (WARPS_PER_BLOCK * B_COLS_PER_WARP) > 0))*/
      , 1);
    int mb = (side == CUBLAS_SIDE_LEFT) * m / WARP /*+ (side == CUBLAS_SIDE_RIGHT) * n / WARP*/;//TODO
    trsm_kernels[func_idx]<<< gridDim, blockDim, 0, curStream>>> (m, n, *alpha, A, incA, B, incB, mb);
    check_error( cudaGetLastError(),  CUBLAS_STATUS_EXECUTION_FAILED );
  }else{
    //error: we should not reach this case
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
  return CUBLAS_STATUS_SUCCESS;
}

#else

template<class T>
cublasStatus_t Xtrsm(cublasHandle_t handle,
                     cublasSideMode_t side, cublasFillMode_t uplo,
                     cublasOperation_t trans, cublasDiagType_t diag,
                     int m, int n,
                     const T *alpha,
                     const T *A, int incA,
                     T *B, int incB){
  return cublasXtrsm( handle,
                      side, uplo, trans, diag,
                      m, n,
                      alpha, A, incA,
                             B, incB );
}

#endif
//==============================================================================================
template<typename T>
cublasStatus_t kblasXtrsm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const T *alpha,
                          const T *A, int incA,
                                T *B, int incB)
{
  T one = make_one<T>();
  T mone = make_zero<T>() - one;
  T mInvAlpha = mone / *alpha;
  cublasStatus_t status;

  if( (*alpha == make_zero<T>())//TODO
   || ( (side == CUBLAS_SIDE_LEFT) && (SIMPLE_SIZE(m)) )
   || ( (side == CUBLAS_SIDE_RIGHT) && (SIMPLE_SIZE(n)) ) ){
    return Xtrsm(handle,
                 side, uplo, trans, diag,
                 m, n,
                 alpha, A, incA,
                        B, incB );
  }
  else
  if(side == CUBLAS_SIDE_LEFT){

    int m1, m2;
    if(REG_SIZE(m))
      m1 = m2 = m/2;
    else{
      m1 = CLOSEST_REG_SIZE(m);
      m2 = m-m1;
    }

    if(uplo == CUBLAS_FILL_MODE_UPPER){

      //Left / Upper / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, A+m1+m1*incA, incA,
                                       B+m1, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 trans, CUBLAS_OP_N,
                                 m1, n, m2,
                                 &mone, A+m1*incA, incA,
                                        B+m1, incB,
                                 alpha, B, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                &one, A, incA,
                                      B, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Upper / [Conj]Trans
      else{
        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, A, incA,
                                       B, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 trans, CUBLAS_OP_N,
                                 m2, n, m1,
                                 &mone, A+m1*incA, incA,
                                        B, incB,
                                 alpha, B+m1, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m2, n,
                                &one, A+m1+m1*incA, incA,
                                      B+m1, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
    }else{//uplo == KBLAS_Lower

      //Left / Lower / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, A, incA,
                                       B, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 trans, CUBLAS_OP_N,
                                 m2, n, m1,
                                 &mone, A+m1, incA,
                                        B, incB,
                                 alpha, B+m1, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m2, n,
                                &one, A+m1+m1*incA, incA,
                                      B+m1, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Lower / [Conj]Trans
      else{//transa == KBLAS_Trans

        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, A+m1+m1*incA, incA,
                                       B+m1, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 trans, CUBLAS_OP_N,
                                 m1, n, m2,
                                 &mone, A+m1, incA,
                                        B+m1, incB,
                                 alpha, B, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                &one, A, incA,
                                      B, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }//transa == KBLAS_Trans
    }
  }
  else{//side == KBLAS_Right
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
        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, A, incA,
                                       B, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n2, n1,
                                 &mone, B, incB,
                                        A+n1*incA, incA,
                                 alpha, B+n1*incB, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                &one, A+n1+n1*incA, incA,
                                      B+n1*incB, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Upper / [Conj]Trans
      else{
        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, A+n1+n1*incA, incA,
                                       B+n1*incB, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n1, n2,
                                 &mInvAlpha, B+n1*incB, incB,
                                             A+n1*incA, incA,
                                 &one,       B, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, A, incA,
                                       B, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
    }
    else{
      //Right / Lower / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, A+n1+n1*incA, incA,
                                       B+n1*incB, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n1, n2,
                                 &mone, B+n1*incB, incB,
                                        A+n1, incA,
                                 alpha, B, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                &one, A, incA,
                                      B, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Lower / [Conj]Trans
      else{
        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, A, incA,
                                       B, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 CUBLAS_OP_N, trans,
                                 m, n2, n1,
                                 &mInvAlpha, B, incB,
                                             A+n1, incA,
                                 &one,       B+n1*incB, incB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, A+n1+n1*incA, incA,
                                       B+n1*incB, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
    }

  }//side == Right

  return CUBLAS_STATUS_SUCCESS;
}

//==============================================================================================
template<typename T>
cublasStatus_t kblasXtrsm(cublasHandle_t handle, cudaStream_t &strIn, cudaStream_t &strOut,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const T *alpha,
                          const T *h_A, int ldA, T* d_A, int lddA,
                                T *h_B, int ldB, T* d_B, int lddB,
                          bool BIsIn, bool getBOut, bool AIsIn)
{
  T one = make_one<T>();
  T mone = make_zero<T>() - one;
  T mInvAlpha = mone / *alpha;
  cublasStatus_t status;
  cublasOperation_t noTrans = CUBLAS_OP_N;//Trans = CUBLAS_OP_T,

  cudaEvent_t eDataIn, eComp;
  check_error( cudaEventCreateWithFlags(&eDataIn, cudaEventDisableTiming), CUBLAS_STATUS_EXECUTION_FAILED);
  check_error( cudaEventCreateWithFlags(&eComp, cudaEventDisableTiming), CUBLAS_STATUS_EXECUTION_FAILED);
  cudaStream_t strComp;
  check_error( cublasGetStream_v2(handle, &strComp), CUBLAS_STATUS_INTERNAL_ERROR);

  if( (*alpha == make_zero<T>())//TODO
   || ( (side == CUBLAS_SIDE_LEFT) && (SIMPLE_SIZE(m)) )
   || ( (side == CUBLAS_SIDE_RIGHT) && (SIMPLE_SIZE(n)) ) ){

    int Am = (side == CUBLAS_SIDE_LEFT) ? m : n;
    //if B is not already in, copy in B block
    if(!BIsIn)
      check_error( status = cublasSetMatrixAsync( m, n, sizeof(T), h_B, ldB, d_B, lddB, strIn ), status);
    //copy in A block
    if(!AIsIn)
      check_error( status = cublasSetMatrixAsync( Am, Am, sizeof(T), h_A, ldA, d_A, lddA, strIn ), status);
    //wait for data to arrive
    if(!AIsIn || !BIsIn){
      check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
    }
    if( (status = Xtrsm(handle,
                        side, uplo, trans, diag,
                        m, n,
                        alpha, d_A, lddA,
                               d_B, lddB ) ) != CUBLAS_STATUS_SUCCESS ) return status;

    //if stream is done computing and getBOut, copy B back.
    if(getBOut){
      check_error( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error( status = cublasGetMatrixAsync( m, n, sizeof(T), d_B, lddB, h_B, ldB, strOut), status);
    }
  }
  else
  if(side == CUBLAS_SIDE_LEFT){

    int m1, m2;
    if(REG_SIZE(m))
      m1 = m2 = m/2;
    else{
      m1 = CLOSEST_REG_SIZE(m);
      m2 = m-m1;
    }

    if( (!AIsIn && SIMPLE_SIZE_DATA(m)) || (!BIsIn && SIMPLE_SIZE_DATA(m)) ){
      if( (!AIsIn && SIMPLE_SIZE_DATA(m)) ){
        check_error( status = cublasSetMatrixAsync( m, m, sizeof(T), h_A, ldA, d_A, lddA, strIn), status);
        AIsIn = true;
      }
      if( (!BIsIn && SIMPLE_SIZE_DATA(m)) ){
        check_error( status = cublasSetMatrixAsync( m, n, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
        BIsIn = true;
      }
      //wait for data to arrive
      check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
    }

    if(uplo == CUBLAS_FILL_MODE_UPPER){

      //Left / Upper / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, h_A+m1+m1*ldA, ldA, d_A+m1+m1*lddA, lddA,
                                       h_B+m1, ldB, d_B+m1, lddB,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( status = cublasGetMatrixAsync( m2, n, sizeof(T), d_B+m1, lddB, h_B+m1, ldB, strOut), status);
        }

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error( status = cublasSetMatrixAsync( m1, n, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error( status = cublasSetMatrixAsync( m1, m2, sizeof(T), h_A+m1*ldA, ldA, d_A+m1*lddA, lddA, strIn), status);
          //wait for data to arrive
          check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m1, n, m2,
                                 &mone, d_A+m1*lddA, lddA,
                                        d_B+m1, lddB,
                                 alpha, d_B, lddB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                &one, h_A, ldA, d_A, lddA,
                                      h_B, ldB, d_B, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Upper / [Conj]Trans
      else{
        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( status = cublasGetMatrixAsync( m1, n, sizeof(T), d_B, lddB, h_B, ldB, strOut), status);
        }

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error( status = cublasSetMatrixAsync( m2, n, sizeof(T), h_B+m1, ldB, d_B+m1, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error( status = cublasSetMatrixAsync( m1, m2, sizeof(T), h_A+m1*ldA, ldA, d_A+m1*lddA, lddA, strIn), status);
          //wait for data to arrive
          check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m2, n, m1,
                                 &mone, d_A+m1*lddA, lddA,
                                        d_B, lddB,
                                 alpha, d_B+m1, lddB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                &one, h_A+m1+m1*ldA, ldA, d_A+m1+m1*lddA, lddA,
                                      h_B+m1, ldB, d_B+m1, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
    }else{//uplo == KBLAS_Lower

      //Left / Lower / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( status = cublasGetMatrixAsync( m1, n, sizeof(T), d_B, lddB, h_B, ldB, strOut), status);
        }

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error( status = cublasSetMatrixAsync( m2, n, sizeof(T), h_B+m1, ldB, d_B+m1, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error( status = cublasSetMatrixAsync( m2, m1, sizeof(T), h_A+m1, ldA, d_A+m1, lddA, strIn), status);
          //wait for data to arrive
          check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }

        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m2, n, m1,
                                 &mone, d_A+m1, lddA,
                                        d_B, lddB,
                                 alpha, d_B+m1, lddB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                &one, h_A+m1+m1*ldA, ldA, d_A+m1+m1*lddA, lddA,
                                      h_B+m1, ldB, d_B+m1, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Lower / [Conj]Trans
      else{//transa == KBLAS_Trans

        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, h_A+m1+m1*ldA, ldA, d_A+m1+m1*lddA, lddA,
                                       h_B+m1, ldB, d_B+m1, lddB,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( status = cublasGetMatrixAsync( m2, n, sizeof(T), d_B+m1, lddB, h_B+m1, ldB, strOut), status);
        }

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error( status = cublasSetMatrixAsync( m1, n, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error( status = cublasSetMatrixAsync( m1, m2, sizeof(T), h_A+m1, ldA, d_A+m1, lddA, strIn), status);
          //wait for data to arrive
          check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }

        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m1, n, m2,
                                 &mone, d_A+m1, lddA,
                                        d_B+m1, lddB,
                                 alpha, d_B, lddB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                &one, h_A, ldA, d_A, lddA,
                                      h_B, ldB, d_B, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }//transa == KBLAS_Trans
    }
  }
  else{//side == KBLAS_Right
    int n1, n2;

    if(REG_SIZE(n))
      n1 = n2 = n/2;
    else{
      n1 = CLOSEST_REG_SIZE(n);
      n2 = n-n1;
    }

    if(uplo == CUBLAS_FILL_MODE_UPPER){
      //Right / Upper / NoTrans
      if(trans == noTrans){
        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( status = cublasGetMatrixAsync( m, n1, sizeof(T), d_B, lddB, h_B, ldB, strOut), status);
        }

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error( status = cublasSetMatrixAsync( m, n2, sizeof(T), h_B+n1*ldB, ldB, d_B+n1*lddB, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error( status = cublasSetMatrixAsync( n1, n2, sizeof(T), h_A+n1*ldA, ldA, d_A+n1*lddA, lddA, strIn), status);
          //wait for data to arrive
          check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }

        if((status = cublasXgemm(handle,
                                 noTrans, trans,
                                 m, n2, n1,
                                 &mone, d_B, lddB,
                                        d_A+n1*lddA, lddA,
                                 alpha, d_B+n1*lddB, lddB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                &one, h_A+n1+n1*ldA, ldA, d_A+n1+n1*lddA, lddA,
                                      h_B+n1*ldB, ldB, d_B+n1*lddB, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Upper / [Conj]Trans
      else{
        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, h_A+n1+n1*ldA, ldA, d_A+n1+n1*lddA, lddA,
                                       h_B+n1*ldB, ldB, d_B+n1*lddB, lddB,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( status = cublasGetMatrixAsync( m, n2, sizeof(T), d_B+n1*lddB, lddB, h_B+n1*ldB, ldB, strOut), status);
        }

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error( status = cublasSetMatrixAsync( m, n1, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error( status = cublasSetMatrixAsync( n1, n2, sizeof(T), h_A+n1*ldA, ldA, d_A+n1*lddA, lddA, strIn), status);
          //wait for data to arrive
          check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }

        if((status = cublasXgemm(handle,
                                 noTrans, trans,
                                 m, n1, n2,
                                 &mInvAlpha, d_B+n1*lddB, lddB,
                                             d_A+n1*lddA, lddA,
                                 &one,       d_B, lddB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
    }
    else{
      //Right / Lower / NoTrans
      if(trans == CUBLAS_OP_N){
        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, h_A+n1+n1*ldA, ldA, d_A+n1+n1*lddA, lddA,
                                       h_B+n1*ldB, ldB, d_B+n1*lddB, lddB,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( status = cublasGetMatrixAsync( m, n2, sizeof(T), d_B+n1*lddB, lddB, h_B+n1*ldB, ldB, strOut), status);
        }

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error( status = cublasSetMatrixAsync( m, n1, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error( status = cublasSetMatrixAsync( n2, n1, sizeof(T), h_A+n1, ldA, d_A+n1, lddA, strIn), status);
          //wait for data to arrive
          check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }

        if((status = cublasXgemm(handle,
                                 noTrans, trans,
                                 m, n1, n2,
                                 &mone, d_B+n1*lddB, lddB,
                                        d_A+n1, lddA,
                                 alpha, d_B, lddB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                &one, h_A, ldA, d_A, lddA,
                                      h_B, ldB, d_B, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Lower / [Conj]Trans
      else{
        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, false, AIsIn
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error( cudaEventRecord(eComp, strComp), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strOut, eComp, 0), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( status = cublasGetMatrixAsync( m, n1, sizeof(T), d_B, lddB, h_B, ldB, strOut), status);
        }

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error( status = cublasSetMatrixAsync( m, n2, sizeof(T), h_B+n1*ldB, ldB, d_B+n1*lddB, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error( status = cublasSetMatrixAsync( n1, n2, sizeof(T), h_A+n1, ldA, d_A+n1, lddA, strIn), status);
          //wait for data to arrive
          check_error( cudaEventRecord(eDataIn, strIn), CUBLAS_STATUS_INTERNAL_ERROR);
          check_error( cudaStreamWaitEvent(strComp, eDataIn, 0), CUBLAS_STATUS_INTERNAL_ERROR);
        }

        if((status = cublasXgemm(handle,
                                 noTrans, trans,
                                 m, n2, n1,
                                 &mInvAlpha, d_B, lddB,
                                             d_A+n1, lddA,
                                 &one,       d_B+n1*lddB, lddB
                                 )) != CUBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, h_A+n1+n1*ldA, ldA, d_A+n1+n1*lddA, lddA,
                                       h_B+n1*ldB, ldB, d_B+n1*lddB, lddB,
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

//==============================================================================================
//#define DO_INLINE_BOUT 0
template<class T>
cublasStatus_t kblasXtrsm_cpu(cublasHandle_t handle,
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

  /*check_error( cudaHostRegister((void*)h_A, Am * An * sizeof(T), cudaHostRegisterDefault), CUBLAS_STATUS_INTERNAL_ERROR);
  check_error( cudaHostRegister((void*)h_B, Bm * Bn * sizeof(T), cudaHostRegisterDefault), CUBLAS_STATUS_INTERNAL_ERROR);*/

  cublasStatus_t status;
  //*
  int AsyncEngineCount, devID;
  check_error( cudaGetDevice(&devID), CUBLAS_STATUS_INTERNAL_ERROR);
  check_error( cudaDeviceGetAttribute(&AsyncEngineCount, cudaDevAttrAsyncEngineCount, devID), CUBLAS_STATUS_INTERNAL_ERROR);
  bool DO_INLINE_BOUT = AsyncEngineCount > 1;
  //*/

  check_error( cudaMalloc( (void**)&d_A, (lddA*An)*sizeof(T) ), CUBLAS_STATUS_INTERNAL_ERROR);
  check_error( cudaMalloc( (void**)&d_B, (lddB*Bn)*sizeof(T) ), CUBLAS_STATUS_INTERNAL_ERROR);

  //setup streams
  cudaStream_t inStream, outStream;
  check_error( cudaStreamCreateWithFlags( &inStream, cudaStreamNonBlocking), CUBLAS_STATUS_INTERNAL_ERROR );
  if(DO_INLINE_BOUT)
    check_error( cudaStreamCreateWithFlags( &outStream, cudaStreamNonBlocking), CUBLAS_STATUS_INTERNAL_ERROR );

  //call cpu API trsm
  check_error(
    (status = kblasXtrsm(handle, inStream, outStream,
                         side, uplo, trans,diag,
                         m, n,
                         alpha, h_A, ldA, d_A, lddA,
                                h_B, ldB, d_B, lddB,
                         false, DO_INLINE_BOUT, false)
    ), status);
  //sync streams
  if(DO_INLINE_BOUT){
    check_error( cudaStreamSynchronize( outStream ), CUBLAS_STATUS_INTERNAL_ERROR);
  }else{
    cudaStream_t compStream;
    check_error( cublasGetStream_v2(handle, &compStream), CUBLAS_STATUS_INTERNAL_ERROR);
    check_error( cudaStreamSynchronize( compStream ), CUBLAS_STATUS_INTERNAL_ERROR);
    check_error( status = cublasGetMatrixAsync( m, n, sizeof(T), d_B, lddB, h_B, ldB, inStream), status);
  }
  //revoke streams
  check_error( cudaStreamDestroy( inStream ), CUBLAS_STATUS_INTERNAL_ERROR);
  if(DO_INLINE_BOUT)
    check_error( cudaStreamDestroy( outStream ), CUBLAS_STATUS_INTERNAL_ERROR);

  /*check_error( cudaHostUnregister( (void*)h_A ), CUBLAS_STATUS_INTERNAL_ERROR );
  check_error( cudaHostUnregister( (void*)h_B ), CUBLAS_STATUS_INTERNAL_ERROR );*/

  //free device memory
  check_error( cudaFree( d_A ), CUBLAS_STATUS_INTERNAL_ERROR );
  check_error( cudaFree( d_B ), CUBLAS_STATUS_INTERNAL_ERROR );
  check_error( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
  return CUBLAS_STATUS_SUCCESS;
}
//==============================================================================================
template<class T>
cublasStatus_t kblasXtrsm_cpu_m(cublasHandle_t handle,
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

  /*check_error( cudaHostRegister((void*)h_A, Am * An * sizeof(T), cudaHostRegisterDefault), CUBLAS_STATUS_INTERNAL_ERROR);
  check_error( cudaHostRegister((void*)h_B, Bm * Bn * sizeof(T), cudaHostRegisterDefault), CUBLAS_STATUS_INTERNAL_ERROR);*/

  cudaStream_t inStream[ngpu], outStream[ngpu];
  cublasStatus_t status[ngpu];
  cublasHandle_t cub_handle[ngpu];
  cub_handle[0] = handle;
  //*
  bool DO_INLINE_BOUT[ngpu];
  for(int g = 0; g < ngpu; g++){
    check_error( cudaSetDevice(g), CUBLAS_STATUS_INTERNAL_ERROR);
    int AsyncEngineCount;
    check_error( cudaDeviceGetAttribute(&AsyncEngineCount, cudaDevAttrAsyncEngineCount, g), CUBLAS_STATUS_INTERNAL_ERROR);
    DO_INLINE_BOUT[g] = AsyncEngineCount > 1;

    if(g > 0)
    {
      check_error( cublasCreate(&cub_handle[g]), CUBLAS_STATUS_INTERNAL_ERROR);
    }
    //setup streams
    check_error( cudaStreamCreateWithFlags( &inStream[g], cudaStreamNonBlocking), CUBLAS_STATUS_INTERNAL_ERROR );
    if(DO_INLINE_BOUT[g])
      check_error( cudaStreamCreateWithFlags( &outStream[g], cudaStreamNonBlocking), CUBLAS_STATUS_INTERNAL_ERROR );
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

  //
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
      status[g] = kblasXtrsm(cub_handle[g], inStream[g], outStream[g],
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
      //cudaDeviceSynchronize();
    }
    //#pragma omp barrier
  }
  /*/
  for(int g = 0; g < ngpu; g++){
    check_error( cudaSetDevice(g), CUBLAS_STATUS_INTERNAL_ERROR);
    check_error( cudaMalloc( (void**)&d_A[g], (Am*An)*sizeof(T) ), CUBLAS_STATUS_INTERNAL_ERROR);
    check_error( cudaMalloc( (void**)&d_B[g], (Bm_gpu*Bn_gpu)*sizeof(T) ), CUBLAS_STATUS_INTERNAL_ERROR);


    //call cpu API trsm
    check_error(
      (status[g] = kblasXtrsm(cub_handle[g], inStream[g], outStream[g],
                           side, uplo, trans,diag,
                           Bm_gpu, Bn_gpu,
                           alpha, h_A, incA, d_A[g],
                                  h_B+g*(left ? Bn_gpu*incB : Bm_gpu), incB, d_B[g],
                           false, DO_INLINE_BOUT[g], false)
      ), status[g]);
  }

  for(int g = 0; g < ngpu; g++){
    check_error( cudaSetDevice(g), CUBLAS_STATUS_INTERNAL_ERROR );

    //sync streams
    if(DO_INLINE_BOUT[g]){
      check_error( cudaStreamSynchronize( outStream[g] ), CUBLAS_STATUS_INTERNAL_ERROR);
    }else{
      cudaStream_t compStream;
      check_error( cublasGetStream_v2(cub_handle[g], &compStream), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error( cudaStreamSynchronize( compStream ), CUBLAS_STATUS_INTERNAL_ERROR);
      check_error( status[g] = cublasGetMatrixAsync( Bm_gpu, Bn_gpu, sizeof(T), d_B[g], incB, h_B+g*(left ? Bn_gpu*incB : Bm_gpu), incB, inStream[g]), status[g]);
      cudaStreamSynchronize( inStream[g] );
    }
  }//*/

  for(int g = 0; g < ngpu; g++){
    check_error( cudaSetDevice(g), CUBLAS_STATUS_INTERNAL_ERROR);
    check_error( cudaDeviceSynchronize(), CUBLAS_STATUS_INTERNAL_ERROR );
    //revoke streams
    check_error( cudaStreamDestroy( inStream[g] ), CUBLAS_STATUS_INTERNAL_ERROR);
    if(DO_INLINE_BOUT[g])
      check_error( cudaStreamDestroy( outStream[g] ), CUBLAS_STATUS_INTERNAL_ERROR);
  /*check_error( cudaHostUnregister( (void*)h_A ), CUBLAS_STATUS_INTERNAL_ERROR );
  check_error( cudaHostUnregister( (void*)h_B ), CUBLAS_STATUS_INTERNAL_ERROR );*/

    //free device memory
    check_error( cudaFree( d_A[g] ), CUBLAS_STATUS_INTERNAL_ERROR );
    check_error( cudaFree( d_B[g] ), CUBLAS_STATUS_INTERNAL_ERROR );
    if(g > 0)
    {
      check_error( cublasDestroy(cub_handle[g]), CUBLAS_STATUS_INTERNAL_ERROR  );
    }

  }
  return CUBLAS_STATUS_SUCCESS;
}
//==============================================================================================
/*extern "C" {
  int kblas_strsm_async(
                        char side, char uplo, char trans, char diag,
                        int m, int n,
                        float alpha, const float *A, int incA,
                        float *B, int incB,
                        cudaStream_t    stream){

    check_error(cublasSetKernelStream(stream));
    return kblasXtrsm(
                      side, uplo, trans, diag,
                      m, n,
                      alpha, A, incA,
                      B, incB);
  }
  int kblas_dtrsm_async(
                        char side, char uplo, char trans, char diag,
                        int m, int n,
                        double alpha, const double *A, int incA,
                        double *B, int incB,
                        cudaStream_t    stream){

    check_error(cublasSetKernelStream(stream));
    return kblasXtrsm(
                      side, uplo, trans, diag,
                      m, n,
                      alpha, A, incA,
                      B, incB);
  }
  int kblas_ctrsm_async(
                        char side, char uplo, char trans, char diag,
                        int m, int n,
                        cuComplex alpha, const cuComplex *A, int incA,
                        cuComplex *B, int incB,
                        cudaStream_t    stream){

    check_error(cublasSetKernelStream(stream));
    return kblasXtrsm(
                      side, uplo, trans, diag,
                      m, n,
                      alpha, A, incA,
                      B, incB);
  }
  int kblas_ztrsm_async(
                        char side, char uplo, char trans, char diag,
                        int m, int n,
                        cuDoubleComplex alpha, const cuDoubleComplex *A, int incA,
                        cuDoubleComplex *B, int incB,
                        cudaStream_t    stream){

    check_error(cublasSetKernelStream(stream));
    return kblasXtrsm(
                      side, uplo, trans, diag,
                      m, n,
                      alpha, A, incA,
                      B, incB);
  }

  int kblas_strsm(
                  char side, char uplo, char trans, char diag,
                  int m, int n,
                  float alpha, const float *A, int incA,
                  float *B, int incB){
    return kblasXtrsm(
                      side, uplo, trans, diag,
                      m, n,
                      alpha, A, incA,
                      B, incB);
  }
  int kblas_dtrsm(
                  char side, char uplo, char trans, char diag,
                  int m, int n,
                  double alpha, const double *A, int incA,
                  double *B, int incB){
    return kblasXtrsm(
                      side, uplo, trans, diag,
                      m, n,
                      alpha, A, incA,
                      B, incB);
  }
  int kblas_ctrsm(
                  char side, char uplo, char trans, char diag,
                  int m, int n,
                  cuComplex alpha, const cuComplex *A, int incA,
                  cuComplex *B, int incB){
    return kblasXtrsm(
                      side, uplo, trans, diag,
                      m, n,
                      alpha, A, incA,
                      B, incB);
  }
  int kblas_ztrsm(
                  char side, char uplo, char trans, char diag,
                  int m, int n,
                  cuDoubleComplex alpha, const cuDoubleComplex *A, int incA,
                  cuDoubleComplex *B, int incB){
    return kblasXtrsm(
                      side, uplo, trans, diag,
                      m, n,
                      alpha, A, incA,
                      B, incB);
  }

}*/

//==============================================================================================

#define kblasXtrsm_async_BODY {                                                                          \
                                                                                                         \
  cublasHandle_t cublas_handle;                                                                          \
  check_error( cublasCreate(&cublas_handle), void() );                                                      \
  if( cublasSetStream_v2(cublas_handle, stream) != CUBLAS_STATUS_SUCCESS ){                              \
    check_error( cublasDestroy_v2(cublas_handle), void());                                                  \
    return;                                                                                              \
  }                                                                                                      \
  cublasSideMode_t  side_v2  = (side  == KBLAS_Left  ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT);            \
  cublasFillMode_t  uplo_v2  = (uplo  == KBLAS_Lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER); \
  cublasOperation_t trans_v2 = (trans == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N);                       \
  cublasDiagType_t  diag_v2  = (diag  == KBLAS_Unit  ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT);         \
                                                                                                         \
  check_error( kblasXtrsm(cublas_handle,                                                                              \
                          side_v2, uplo_v2, trans_v2, diag_v2,                                                        \
                          m, n,                                                                                       \
                          &alpha, A, lda,                                                                             \
                                  B, ldb), void());                                                                             \
                                                                                                         \
  check_error( cublasDestroy_v2(cublas_handle), void());                                                    \
}
extern "C"{
void kblasStrsm_async(char side, char uplo, char trans, char diag,
                      int m, int n,
                      float alpha, const float *A, int lda,
                                         float *B, int ldb,
                      cudaStream_t stream){
  kblasXtrsm_async_BODY
}

void kblasDtrsm_async(char side, char uplo, char trans, char diag,
                      int m, int n,
                      double alpha, const double *A, int lda,
                                          double *B, int ldb,
                      cudaStream_t stream){
  kblasXtrsm_async_BODY
}
void kblasCtrsm_async(char side, char uplo, char trans, char diag,
                      int m, int n,
                      cuComplex alpha, const cuComplex *A, int lda,
                                             cuComplex *B, int ldb,
                      cudaStream_t stream){
  kblasXtrsm_async_BODY
}
void kblasZtrsm_async(char side, char uplo, char trans, char diag,
                      int m, int n,
                      cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                                                   cuDoubleComplex *B, int ldb,
                      cudaStream_t stream){
  kblasXtrsm_async_BODY
}
}
//==============================================================================================

void kblasStrsm(char side, char uplo, char trans, char diag,
                int m, int n,
                float alpha, const float *A, int lda,
                                   float *B, int ldb){

  kblasStrsm_async(side, uplo, trans, diag,
                   m, n,
                   alpha, A, lda,
                          B, ldb,
                   0);
}

void kblasDtrsm(char side, char uplo, char trans, char diag,
                int m, int n,
                double alpha, const double *A, int lda,
                                    double *B, int ldb){

  kblasDtrsm_async(side, uplo, trans, diag,
                   m, n,
                   alpha, A, lda,
                          B, ldb,
                   0);
}
void kblasCtrsm(char side, char uplo, char trans, char diag,
                int m, int n,
                cuComplex alpha, const cuComplex *A, int lda,
                                       cuComplex *B, int ldb){

  kblasCtrsm_async(side, uplo, trans, diag,
                   m, n,
                   alpha, A, lda,
                          B, ldb,
                   0);

}
void kblasZtrsm(char side, char uplo, char trans, char diag,
                int m, int n,
                cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                                             cuDoubleComplex *B, int ldb){

  kblasZtrsm_async(side, uplo, trans, diag,
                   m, n,
                   alpha, A, lda,
                          B, ldb,
                   0);
}

//==============================================================================================

cublasStatus_t kblasStrsm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const float *alpha,
                          const float *A, int lda,
                                float *B, int ldb){
  return kblasXtrsm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
cublasStatus_t kblasDtrsm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const double *alpha,
                          const double *A, int lda,
                                double *B, int ldb){
  return kblasXtrsm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
cublasStatus_t kblasCtrsm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const cuComplex *alpha,
                          const cuComplex *A, int lda,
                                cuComplex *B, int ldb){
  return kblasXtrsm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
cublasStatus_t kblasZtrsm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const cuDoubleComplex *alpha,
                          const cuDoubleComplex *A, int lda,
                                cuDoubleComplex *B, int ldb){
  return kblasXtrsm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
//==============================================================================================

cublasStatus_t kblas_strsm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const float *alpha,
                           const float *A, int lda,
                                 float *B, int ldb){
  return kblasXtrsm_cpu(handle,
                        side, uplo, trans, diag,
                        m, n,
                        alpha, A, lda,
                              B, ldb);
}
cublasStatus_t kblas_dtrsm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const double *alpha,
                           const double *A, int lda,
                                 double *B, int ldb){
  return kblasXtrsm_cpu(handle,
                        side, uplo, trans, diag,
                        m, n,
                        alpha, A, lda,
                              B, ldb);
}
cublasStatus_t kblas_ctrsm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const cuComplex *alpha,
                           const cuComplex *A, int lda,
                                 cuComplex *B, int ldb){
  return kblasXtrsm_cpu(handle,
                        side, uplo, trans, diag,
                        m, n,
                        alpha, A, lda,
                              B, ldb);
}
cublasStatus_t kblas_ztrsm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda,
                                 cuDoubleComplex *B, int ldb){
  return kblasXtrsm_cpu(handle,
                        side, uplo, trans, diag,
                        m, n,
                        alpha, A, lda,
                              B, ldb);
}

//==============================================================================================
cublasStatus_t kblas_strsm_mgpu(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const float *alpha,
                                const float *A, int lda,
                                      float *B, int ldb,
                                int ngpu){
  return kblasXtrsm_cpu_m(handle,
                        side, uplo, trans, diag,
                        m, n,
                        alpha, A, lda,
                              B, ldb,
                        ngpu);
}
cublasStatus_t kblas_dtrsm_mgpu(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const double *alpha,
                                const double *A, int lda,
                                      double *B, int ldb,
                                int ngpu){
  return kblasXtrsm_cpu_m(handle,
                        side, uplo, trans, diag,
                        m, n,
                        alpha, A, lda,
                              B, ldb,
                        ngpu);
}

cublasStatus_t kblas_ctrsm_mgpu(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuComplex *alpha,
                                const cuComplex *A, int lda,
                                      cuComplex *B, int ldb,
                                int ngpu){
  return kblasXtrsm_cpu_m(handle,
                        side, uplo, trans, diag,
                        m, n,
                        alpha, A, lda,
                              B, ldb,
                        ngpu);
}
cublasStatus_t kblas_ztrsm_mgpu(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, int lda,
                                      cuDoubleComplex *B, int ldb,
                                int ngpu){
  return kblasXtrsm_cpu_m(handle,
                        side, uplo, trans, diag,
                        m, n,
                        alpha, A, lda,
                              B, ldb,
                        ngpu);
}



