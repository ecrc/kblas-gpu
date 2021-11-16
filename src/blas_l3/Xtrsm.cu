#include "hip/hip_runtime.h"
/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l3/Xtrsm.cu

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

hipblasStatus_t cublasXtrsm(hipblasHandle_t handle,
                           hipblasSideMode_t side, hipblasFillMode_t uplo,
                           hipblasOperation_t trans, hipblasDiagType_t diag,
                           int m, int n,
                           const float *alpha,
                           const float *A, int lda,
                           float *B, int ldb){
  hipblasStatus_t status;
  check_error_ret( status = hipblasStrsm(handle,
                                    side, uplo, trans, diag,
                                    m, n,
                                    alpha, (float*)A, lda,
                                           B, ldb ), status);
  check_error_ret( hipGetLastError(), HIPBLAS_STATUS_EXECUTION_FAILED );
  return HIPBLAS_STATUS_SUCCESS;
}
hipblasStatus_t cublasXtrsm(hipblasHandle_t handle,
                           hipblasSideMode_t side, hipblasFillMode_t uplo,
                           hipblasOperation_t trans, hipblasDiagType_t      diag,
                           int m, int n,
                           const double *alpha,
                           const double *A, int lda,
                                 double *B, int ldb){
  hipblasStatus_t status;
  check_error_ret( status = hipblasDtrsm(handle,
                                    side, uplo, trans, diag,
                                    m, n,
                                    alpha, (double*)A, lda,
                                           B, ldb ), status);
  check_error_ret( hipGetLastError(), HIPBLAS_STATUS_EXECUTION_FAILED );
  return HIPBLAS_STATUS_SUCCESS;
}
// hipblasStatus_t cublasXtrsm (hipblasHandle_t handle,
//                             hipblasSideMode_t side, hipblasFillMode_t uplo,
//                             hipblasOperation_t trans, hipblasDiagType_t diag,
//                             int m, int n,
//                             const hipComplex *alpha,
//                             const hipComplex *A, int lda,
//                                   hipComplex *B, int ldb){
//   hipblasStatus_t status;
//   check_error_ret( status = cublasCtrsm(handle,
//                                     side, uplo, trans, diag,
//                                     m, n,
//                                     alpha, A, lda,
//                                            B, ldb ), status);
//   check_error_ret( hipGetLastError(), HIPBLAS_STATUS_EXECUTION_FAILED );
//   return HIPBLAS_STATUS_SUCCESS;
// }
// hipblasStatus_t cublasXtrsm (hipblasHandle_t handle,
//                             hipblasSideMode_t side, hipblasFillMode_t uplo,
//                             hipblasOperation_t trans, hipblasDiagType_t diag,
//                             int m, int n,
//                             const hipDoubleComplex *alpha,
//                             const hipDoubleComplex *A, int lda,
//                                   hipDoubleComplex *B, int ldb){
//   hipblasStatus_t status;
//   check_error_ret( status = cublasZtrsm(handle,
//                                     side, uplo, trans, diag,
//                                     m, n,
//                                     alpha, A, lda,
//                                            B, ldb ), status);
//   check_error_ret( hipGetLastError(), HIPBLAS_STATUS_EXECUTION_FAILED );
//   return HIPBLAS_STATUS_SUCCESS;
// }

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
#if (TARGET_SM >= 30)
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
hipblasStatus_t Xtrsm(hipblasHandle_t handle,
                     hipblasSideMode_t side, hipblasFillMode_t uplo,
                     hipblasOperation_t trans, hipblasDiagType_t diag,
                     int m, int n,
                     const T *alpha,
                     const T *A, int incA,
                           T *B, int incB)
{

  //handle odd cases with cublas
  if(  (*alpha == make_zero<T>())
    || (!kblas_trsm_use_custom)
    || (side == HIPBLAS_SIDE_LEFT && m < WARP)
    || (side == HIPBLAS_SIDE_RIGHT/* && n < WARP*/))//TODO
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

  hipStream_t curStream;
  hipblasStatus_t status;

  check_error_ret( status = hipblasGetStream( handle, &curStream ), status);

  if( ((side == HIPBLAS_SIDE_LEFT) && (m % WARP == 0)) /*|| ((side == HIPBLAS_SIDE_RIGHT) && (n % WARP == 0))*/ )//TODO
  {
    int func_idx = /*4*(side == HIPBLAS_SIDE_RIGHT) + */2*(uplo == HIPBLAS_FILL_MODE_UPPER) + (trans != HIPBLAS_OP_N);// + (diag == HIPBLAS_DIAG_UNIT);TODO
    dim3 blockDim( WARP, WARPS_PER_BLOCK );
    dim3 gridDim(
      (side == HIPBLAS_SIDE_LEFT) * (n / (WARPS_PER_BLOCK * B_COLS_PER_WARP) + (n % (WARPS_PER_BLOCK * B_COLS_PER_WARP) > 0))
      /*+TODO
      (side == HIPBLAS_SIDE_RIGHT) * (m / (WARPS_PER_BLOCK * B_COLS_PER_WARP) + (m % (WARPS_PER_BLOCK * B_COLS_PER_WARP) > 0))*/
      , 1);
    int mb = (side == HIPBLAS_SIDE_LEFT) * m / WARP /*+ (side == HIPBLAS_SIDE_RIGHT) * n / WARP*/;//TODO
    trsm_kernels[func_idx]<<< gridDim, blockDim, 0, curStream>>> (m, n, *alpha, A, incA, B, incB, mb);
    check_error_ret( hipGetLastError(),  HIPBLAS_STATUS_EXECUTION_FAILED );
  }else{
    //error: we should not reach this case
    return HIPBLAS_STATUS_INTERNAL_ERROR;
  }
  return HIPBLAS_STATUS_SUCCESS;
}

#else

template<class T>
hipblasStatus_t Xtrsm(hipblasHandle_t handle,
                     hipblasSideMode_t side, hipblasFillMode_t uplo,
                     hipblasOperation_t trans, hipblasDiagType_t diag,
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
hipblasStatus_t kblasXtrsm(hipblasHandle_t handle,
                          hipblasSideMode_t side, hipblasFillMode_t uplo,
                          hipblasOperation_t trans, hipblasDiagType_t diag,
                          int m, int n,
                          const T *alpha,
                          const T *A, int incA,
                                T *B, int incB)
{
  T one = make_one<T>();
  T mone = make_zero<T>() - one;
  T mInvAlpha = mone / *alpha;
  hipblasStatus_t status;

  if( (*alpha == make_zero<T>())//TODO
   || ( (side == HIPBLAS_SIDE_LEFT) && (SIMPLE_SIZE(m)) )
   || ( (side == HIPBLAS_SIDE_RIGHT) && (SIMPLE_SIZE(n)) ) ){
    return Xtrsm(handle,
                 side, uplo, trans, diag,
                 m, n,
                 alpha, A, incA,
                        B, incB );
  }
  else
  if(side == HIPBLAS_SIDE_LEFT){

    int m1, m2;
    if(REG_SIZE(m))
      m1 = m2 = m/2;
    else{
      m1 = CLOSEST_REG_SIZE(m);
      m2 = m-m1;
    }

    if(uplo == HIPBLAS_FILL_MODE_UPPER){

      //Left / Upper / NoTrans
      if(trans == HIPBLAS_OP_N){
        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, A+m1+m1*incA, incA,
                                       B+m1, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 trans, HIPBLAS_OP_N,
                                 m1, n, m2,
                                 &mone, A+m1*incA, incA,
                                        B+m1, incB,
                                 alpha, B, incB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                &one, A, incA,
                                      B, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Upper / [Conj]Trans
      else{
        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, A, incA,
                                       B, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 trans, HIPBLAS_OP_N,
                                 m2, n, m1,
                                 &mone, A+m1*incA, incA,
                                        B, incB,
                                 alpha, B+m1, incB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m2, n,
                                &one, A+m1+m1*incA, incA,
                                      B+m1, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
    }else{//uplo == KBLAS_Lower

      //Left / Lower / NoTrans
      if(trans == HIPBLAS_OP_N){
        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, A, incA,
                                       B, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 trans, HIPBLAS_OP_N,
                                 m2, n, m1,
                                 &mone, A+m1, incA,
                                        B, incB,
                                 alpha, B+m1, incB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m2, n,
                                &one, A+m1+m1*incA, incA,
                                      B+m1, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Lower / [Conj]Trans
      else{//transa == KBLAS_Trans

        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, A+m1+m1*incA, incA,
                                       B+m1, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 trans, HIPBLAS_OP_N,
                                 m1, n, m2,
                                 &mone, A+m1, incA,
                                        B+m1, incB,
                                 alpha, B, incB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                &one, A, incA,
                                      B, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
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

    if(uplo == HIPBLAS_FILL_MODE_UPPER){
      //Right / Upper / NoTrans
      if(trans == HIPBLAS_OP_N){
        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, A, incA,
                                       B, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 HIPBLAS_OP_N, trans,
                                 m, n2, n1,
                                 &mone, B, incB,
                                        A+n1*incA, incA,
                                 alpha, B+n1*incB, incB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                &one, A+n1+n1*incA, incA,
                                      B+n1*incB, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Upper / [Conj]Trans
      else{
        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, A+n1+n1*incA, incA,
                                       B+n1*incB, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 HIPBLAS_OP_N, trans,
                                 m, n1, n2,
                                 &mInvAlpha, B+n1*incB, incB,
                                             A+n1*incA, incA,
                                 &one,       B, incB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, A, incA,
                                       B, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
    }
    else{
      //Right / Lower / NoTrans
      if(trans == HIPBLAS_OP_N){
        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, A+n1+n1*incA, incA,
                                       B+n1*incB, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 HIPBLAS_OP_N, trans,
                                 m, n1, n2,
                                 &mone, B+n1*incB, incB,
                                        A+n1, incA,
                                 alpha, B, incB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                &one, A, incA,
                                      B, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Lower / [Conj]Trans
      else{
        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, A, incA,
                                       B, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = cublasXgemm(handle,
                                 HIPBLAS_OP_N, trans,
                                 m, n2, n1,
                                 &mInvAlpha, B, incB,
                                             A+n1, incA,
                                 &one,       B+n1*incB, incB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, A+n1+n1*incA, incA,
                                       B+n1*incB, incB
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
    }

  }//side == Right

  return HIPBLAS_STATUS_SUCCESS;
}

//==============================================================================================
template<typename T>
hipblasStatus_t kblasXtrsm(hipblasHandle_t handle, hipStream_t &strIn, hipStream_t &strOut,
                          hipblasSideMode_t side, hipblasFillMode_t uplo,
                          hipblasOperation_t trans, hipblasDiagType_t diag,
                          int m, int n,
                          const T *alpha,
                          const T *h_A, int ldA, T* d_A, int lddA,
                                T *h_B, int ldB, T* d_B, int lddB,
                          bool BIsIn, bool getBOut, bool AIsIn)
{
  T one = make_one<T>();
  T mone = make_zero<T>() - one;
  T mInvAlpha = mone / *alpha;
  hipblasStatus_t status;
  hipblasOperation_t noTrans = HIPBLAS_OP_N;//Trans = HIPBLAS_OP_T,

  hipEvent_t eDataIn, eComp;
  check_error_ret( hipEventCreateWithFlags(&eDataIn, hipEventDisableTiming), HIPBLAS_STATUS_EXECUTION_FAILED);
  check_error_ret( hipEventCreateWithFlags(&eComp, hipEventDisableTiming), HIPBLAS_STATUS_EXECUTION_FAILED);
  hipStream_t strComp;
  check_error_ret( hipblasGetStream(handle, &strComp), HIPBLAS_STATUS_INTERNAL_ERROR);

  if( (*alpha == make_zero<T>())//TODO
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
    if( (status = Xtrsm(handle,
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
  }
  else
  if(side == HIPBLAS_SIDE_LEFT){

    int m1, m2;
    if(REG_SIZE(m))
      m1 = m2 = m/2;
    else{
      m1 = CLOSEST_REG_SIZE(m);
      m2 = m-m1;
    }

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
        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, h_A+m1+m1*ldA, ldA, d_A+m1+m1*lddA, lddA,
                                       h_B+m1, ldB, d_B+m1, lddB,
                                BIsIn, false, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( hipEventRecord(eComp, strComp), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strOut, eComp, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = hipblasGetMatrixAsync( m2, n, sizeof(T), d_B+m1, lddB, h_B+m1, ldB, strOut), status);
        }

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = hipblasSetMatrixAsync( m1, n, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = hipblasSetMatrixAsync( m1, m2, sizeof(T), h_A+m1*ldA, ldA, d_A+m1*lddA, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( hipEventRecord(eDataIn, strIn), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strComp, eDataIn, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m1, n, m2,
                                 &mone, d_A+m1*lddA, lddA,
                                        d_B+m1, lddB,
                                 alpha, d_B, lddB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                &one, h_A, ldA, d_A, lddA,
                                      h_B, ldB, d_B, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Upper / [Conj]Trans
      else{
        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, false, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( hipEventRecord(eComp, strComp), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strOut, eComp, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = hipblasGetMatrixAsync( m1, n, sizeof(T), d_B, lddB, h_B, ldB, strOut), status);
        }

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = hipblasSetMatrixAsync( m2, n, sizeof(T), h_B+m1, ldB, d_B+m1, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = hipblasSetMatrixAsync( m1, m2, sizeof(T), h_A+m1*ldA, ldA, d_A+m1*lddA, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( hipEventRecord(eDataIn, strIn), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strComp, eDataIn, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
        }
        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m2, n, m1,
                                 &mone, d_A+m1*lddA, lddA,
                                        d_B, lddB,
                                 alpha, d_B+m1, lddB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                &one, h_A+m1+m1*ldA, ldA, d_A+m1+m1*lddA, lddA,
                                      h_B+m1, ldB, d_B+m1, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
    }else{//uplo == KBLAS_Lower

      //Left / Lower / NoTrans
      if(trans == HIPBLAS_OP_N){
        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, false, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( hipEventRecord(eComp, strComp), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strOut, eComp, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = hipblasGetMatrixAsync( m1, n, sizeof(T), d_B, lddB, h_B, ldB, strOut), status);
        }

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = hipblasSetMatrixAsync( m2, n, sizeof(T), h_B+m1, ldB, d_B+m1, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = hipblasSetMatrixAsync( m2, m1, sizeof(T), h_A+m1, ldA, d_A+m1, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( hipEventRecord(eDataIn, strIn), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strComp, eDataIn, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
        }

        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m2, n, m1,
                                 &mone, d_A+m1, lddA,
                                        d_B, lddB,
                                 alpha, d_B+m1, lddB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                &one, h_A+m1+m1*ldA, ldA, d_A+m1+m1*lddA, lddA,
                                      h_B+m1, ldB, d_B+m1, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
      //Left / Lower / [Conj]Trans
      else{//transa == KBLAS_Trans

        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m2, n,
                                alpha, h_A+m1+m1*ldA, ldA, d_A+m1+m1*lddA, lddA,
                                       h_B+m1, ldB, d_B+m1, lddB,
                                BIsIn, false, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( hipEventRecord(eComp, strComp), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strOut, eComp, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = hipblasGetMatrixAsync( m2, n, sizeof(T), d_B+m1, lddB, h_B+m1, ldB, strOut), status);
        }

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = hipblasSetMatrixAsync( m1, n, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
            BIsIn = true;
          }
          //copy in A block
          if(!AIsIn)
            check_error_ret( status = hipblasSetMatrixAsync( m1, m2, sizeof(T), h_A+m1, ldA, d_A+m1, lddA, strIn), status);
          //wait for data to arrive
          check_error_ret( hipEventRecord(eDataIn, strIn), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strComp, eDataIn, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
        }

        if((status = cublasXgemm(handle,
                                 trans, noTrans,
                                 m1, n, m2,
                                 &mone, d_A+m1, lddA,
                                        d_B+m1, lddB,
                                 alpha, d_B, lddB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m1, n,
                                &one, h_A, ldA, d_A, lddA,
                                      h_B, ldB, d_B, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
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

    if(uplo == HIPBLAS_FILL_MODE_UPPER){
      //Right / Upper / NoTrans
      if(trans == noTrans){
        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, false, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( hipEventRecord(eComp, strComp), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strOut, eComp, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = hipblasGetMatrixAsync( m, n1, sizeof(T), d_B, lddB, h_B, ldB, strOut), status);
        }

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
                                 noTrans, trans,
                                 m, n2, n1,
                                 &mone, d_B, lddB,
                                        d_A+n1*lddA, lddA,
                                 alpha, d_B+n1*lddB, lddB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                &one, h_A+n1+n1*ldA, ldA, d_A+n1+n1*lddA, lddA,
                                      h_B+n1*ldB, ldB, d_B+n1*lddB, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Upper / [Conj]Trans
      else{
        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, h_A+n1+n1*ldA, ldA, d_A+n1+n1*lddA, lddA,
                                       h_B+n1*ldB, ldB, d_B+n1*lddB, lddB,
                                BIsIn, false, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( hipEventRecord(eComp, strComp), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strOut, eComp, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = hipblasGetMatrixAsync( m, n2, sizeof(T), d_B+n1*lddB, lddB, h_B+n1*ldB, ldB, strOut), status);
        }

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = hipblasSetMatrixAsync( m, n1, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
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
                                 noTrans, trans,
                                 m, n1, n2,
                                 &mInvAlpha, d_B+n1*lddB, lddB,
                                             d_A+n1*lddA, lddA,
                                 &one,       d_B, lddB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
    }
    else{
      //Right / Lower / NoTrans
      if(trans == HIPBLAS_OP_N){
        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, h_A+n1+n1*ldA, ldA, d_A+n1+n1*lddA, lddA,
                                       h_B+n1*ldB, ldB, d_B+n1*lddB, lddB,
                                BIsIn, false, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( hipEventRecord(eComp, strComp), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strOut, eComp, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = hipblasGetMatrixAsync( m, n2, sizeof(T), d_B+n1*lddB, lddB, h_B+n1*ldB, ldB, strOut), status);
        }

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = hipblasSetMatrixAsync( m, n1, sizeof(T), h_B, ldB, d_B, lddB, strIn), status);
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
                                 noTrans, trans,
                                 m, n1, n2,
                                 &mone, d_B+n1*lddB, lddB,
                                        d_A+n1, lddA,
                                 alpha, d_B, lddB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                &one, h_A, ldA, d_A, lddA,
                                      h_B, ldB, d_B, lddB,
                                BIsIn, getBOut, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Lower / [Conj]Trans
      else{
        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n1,
                                alpha, h_A, ldA, d_A, lddA,
                                       h_B, ldB, d_B, lddB,
                                BIsIn, false, AIsIn
                                )) != HIPBLAS_STATUS_SUCCESS) return status;
        //if stream is done computing and getBOut, copy B back.
        if(getBOut){
          check_error_ret( hipEventRecord(eComp, strComp), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( hipStreamWaitEvent(strOut, eComp, 0), HIPBLAS_STATUS_INTERNAL_ERROR);
          check_error_ret( status = hipblasGetMatrixAsync( m, n1, sizeof(T), d_B, lddB, h_B, ldB, strOut), status);
        }

        //prepare needed data
        if(!AIsIn || !BIsIn){
          //if B is not already in, copy B block
          if(!BIsIn){
            check_error_ret( status = hipblasSetMatrixAsync( m, n2, sizeof(T), h_B+n1*ldB, ldB, d_B+n1*lddB, lddB, strIn), status);
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
                                 noTrans, trans,
                                 m, n2, n1,
                                 &mInvAlpha, d_B, lddB,
                                             d_A+n1, lddA,
                                 &one,       d_B+n1*lddB, lddB
                                 )) != HIPBLAS_STATUS_SUCCESS) return status;

        if((status = kblasXtrsm(handle, strIn, strOut,
                                side, uplo, trans, diag,
                                m, n2,
                                alpha, h_A+n1+n1*ldA, ldA, d_A+n1+n1*lddA, lddA,
                                       h_B+n1*ldB, ldB, d_B+n1*lddB, lddB,
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
//#define DO_INLINE_BOUT 0
template<class T>
hipblasStatus_t kblasXtrsm_cpu(hipblasHandle_t handle,
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
  //*
  int AsyncEngineCount, devID;
  // check_error_ret( hipGetDevice(&devID), HIPBLAS_STATUS_INTERNAL_ERROR);
  // check_error_ret( hipDeviceGetAttribute(&AsyncEngineCount, cudaDevAttrAsyncEngineCount, devID), HIPBLAS_STATUS_INTERNAL_ERROR);
  // bool DO_INLINE_BOUT = AsyncEngineCount > 1;
  bool DO_INLINE_BOUT = false;
  //*/

  check_error_ret( hipMalloc( (void**)&d_A, (lddA*An)*sizeof(T) ), HIPBLAS_STATUS_INTERNAL_ERROR);
  check_error_ret( hipMalloc( (void**)&d_B, (lddB*Bn)*sizeof(T) ), HIPBLAS_STATUS_INTERNAL_ERROR);

  //setup streams
  hipStream_t inStream, outStream;
  check_error_ret( hipStreamCreateWithFlags( &inStream, hipStreamNonBlocking), HIPBLAS_STATUS_INTERNAL_ERROR );
  if(DO_INLINE_BOUT)
    check_error_ret( hipStreamCreateWithFlags( &outStream, hipStreamNonBlocking), HIPBLAS_STATUS_INTERNAL_ERROR );

  //call cpu API trsm
  check_error_ret(
    (status = kblasXtrsm(handle, inStream, outStream,
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
  check_error_ret( hipFree( d_A ), HIPBLAS_STATUS_INTERNAL_ERROR );
  check_error_ret( hipFree( d_B ), HIPBLAS_STATUS_INTERNAL_ERROR );
  check_error_ret( hipGetLastError(), HIPBLAS_STATUS_EXECUTION_FAILED );
  return HIPBLAS_STATUS_SUCCESS;
}
//==============================================================================================
template<class T>
hipblasStatus_t kblasXtrsm_cpu_m(hipblasHandle_t handle,
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
    int AsyncEngineCount;
    // check_error_ret( hipDeviceGetAttribute(&AsyncEngineCount, cudaDevAttrAsyncEngineCount, g), HIPBLAS_STATUS_INTERNAL_ERROR);
    // DO_INLINE_BOUT[g] = AsyncEngineCount > 1;
    DO_INLINE_BOUT[g] = false;

    if(g > 0)
    {
      check_error_ret( hipblasCreate(&cub_handle[g]), HIPBLAS_STATUS_INTERNAL_ERROR);
    }
    //setup streams
    check_error_ret( hipStreamCreateWithFlags( &inStream[g], hipStreamNonBlocking), HIPBLAS_STATUS_INTERNAL_ERROR );
    if(DO_INLINE_BOUT[g])
      check_error_ret( hipStreamCreateWithFlags( &outStream[g], hipStreamNonBlocking), HIPBLAS_STATUS_INTERNAL_ERROR );
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

  //
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
      status[g] = kblasXtrsm(cub_handle[g], inStream[g], outStream[g],
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
      //hipDeviceSynchronize();
    }
    //#pragma omp barrier
  }
  /*/
  for(int g = 0; g < ngpu; g++){
    check_error_ret( hipSetDevice(g), HIPBLAS_STATUS_INTERNAL_ERROR);
    check_error_ret( hipMalloc( (void**)&d_A[g], (Am*An)*sizeof(T) ), HIPBLAS_STATUS_INTERNAL_ERROR);
    check_error_ret( hipMalloc( (void**)&d_B[g], (Bm_gpu*Bn_gpu)*sizeof(T) ), HIPBLAS_STATUS_INTERNAL_ERROR);


    //call cpu API trsm
    check_error_ret(
      (status[g] = kblasXtrsm(cub_handle[g], inStream[g], outStream[g],
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
      hipStreamSynchronize( inStream[g] );
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
    if(g > 0)
    {
      check_error_ret( hipblasDestroy(cub_handle[g]), HIPBLAS_STATUS_INTERNAL_ERROR  );
    }

  }
  return HIPBLAS_STATUS_SUCCESS;
}
//==============================================================================================
/*extern "C" {
  int kblas_strsm_async(
                        char side, char uplo, char trans, char diag,
                        int m, int n,
                        float alpha, const float *A, int incA,
                        float *B, int incB,
                        hipStream_t    stream){

    check_error_ret(cublasSetKernelStream(stream));
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
                        hipStream_t    stream){

    check_error_ret(cublasSetKernelStream(stream));
    return kblasXtrsm(
                      side, uplo, trans, diag,
                      m, n,
                      alpha, A, incA,
                      B, incB);
  }
  int kblas_ctrsm_async(
                        char side, char uplo, char trans, char diag,
                        int m, int n,
                        hipComplex alpha, const hipComplex *A, int incA,
                        hipComplex *B, int incB,
                        hipStream_t    stream){

    check_error_ret(cublasSetKernelStream(stream));
    return kblasXtrsm(
                      side, uplo, trans, diag,
                      m, n,
                      alpha, A, incA,
                      B, incB);
  }
  int kblas_ztrsm_async(
                        char side, char uplo, char trans, char diag,
                        int m, int n,
                        hipDoubleComplex alpha, const hipDoubleComplex *A, int incA,
                        hipDoubleComplex *B, int incB,
                        hipStream_t    stream){

    check_error_ret(cublasSetKernelStream(stream));
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
                  hipComplex alpha, const hipComplex *A, int incA,
                  hipComplex *B, int incB){
    return kblasXtrsm(
                      side, uplo, trans, diag,
                      m, n,
                      alpha, A, incA,
                      B, incB);
  }
  int kblas_ztrsm(
                  char side, char uplo, char trans, char diag,
                  int m, int n,
                  hipDoubleComplex alpha, const hipDoubleComplex *A, int incA,
                  hipDoubleComplex *B, int incB){
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
  hipblasHandle_t cublas_handle;                                                                          \
  check_error_ret( hipblasCreate(&cublas_handle), void() );                                                      \
  if( hipblasSetStream(cublas_handle, stream) != HIPBLAS_STATUS_SUCCESS ){                              \
    check_error_ret( hipblasDestroy(cublas_handle), void());                                                  \
    return;                                                                                              \
  }                                                                                                      \
  hipblasSideMode_t  side_v2  = (side  == KBLAS_Left  ? HIPBLAS_SIDE_LEFT : HIPBLAS_SIDE_RIGHT);            \
  hipblasFillMode_t  uplo_v2  = (uplo  == KBLAS_Lower ? HIPBLAS_FILL_MODE_LOWER : HIPBLAS_FILL_MODE_UPPER); \
  hipblasOperation_t trans_v2 = (trans == KBLAS_Trans ? HIPBLAS_OP_T : HIPBLAS_OP_N);                       \
  hipblasDiagType_t  diag_v2  = (diag  == KBLAS_Unit  ? HIPBLAS_DIAG_UNIT : HIPBLAS_DIAG_NON_UNIT);         \
                                                                                                         \
  check_error_ret( kblasXtrsm(cublas_handle,                                                                              \
                          side_v2, uplo_v2, trans_v2, diag_v2,                                                        \
                          m, n,                                                                                       \
                          &alpha, A, lda,                                                                             \
                                  B, ldb), void());                                                                             \
                                                                                                         \
  check_error_ret( hipblasDestroy(cublas_handle), void());                                                    \
}
extern "C"{
void kblasStrsm_async(char side, char uplo, char trans, char diag,
                      int m, int n,
                      float alpha, const float *A, int lda,
                                         float *B, int ldb,
                      hipStream_t stream){
  kblasXtrsm_async_BODY
}

void kblasDtrsm_async(char side, char uplo, char trans, char diag,
                      int m, int n,
                      double alpha, const double *A, int lda,
                                          double *B, int ldb,
                      hipStream_t stream){
  kblasXtrsm_async_BODY
}
// void kblasCtrsm_async(char side, char uplo, char trans, char diag,
//                       int m, int n,
//                       hipComplex alpha, const hipComplex *A, int lda,
//                                              hipComplex *B, int ldb,
//                       hipStream_t stream){
//   kblasXtrsm_async_BODY
// }
// void kblasZtrsm_async(char side, char uplo, char trans, char diag,
//                       int m, int n,
//                       hipDoubleComplex alpha, const hipDoubleComplex *A, int lda,
//                                                    hipDoubleComplex *B, int ldb,
//                       hipStream_t stream){
//   kblasXtrsm_async_BODY
// }
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
// void kblasCtrsm(char side, char uplo, char trans, char diag,
//                 int m, int n,
//                 hipComplex alpha, const hipComplex *A, int lda,
//                                        hipComplex *B, int ldb){

//   kblasCtrsm_async(side, uplo, trans, diag,
//                    m, n,
//                    alpha, A, lda,
//                           B, ldb,
//                    0);

// }
// void kblasZtrsm(char side, char uplo, char trans, char diag,
//                 int m, int n,
//                 hipDoubleComplex alpha, const hipDoubleComplex *A, int lda,
//                                              hipDoubleComplex *B, int ldb){

//   kblasZtrsm_async(side, uplo, trans, diag,
//                    m, n,
//                    alpha, A, lda,
//                           B, ldb,
//                    0);
// }

//==============================================================================================

hipblasStatus_t kblasStrsm(hipblasHandle_t handle,
                          hipblasSideMode_t side, hipblasFillMode_t uplo,
                          hipblasOperation_t trans, hipblasDiagType_t diag,
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
hipblasStatus_t kblasDtrsm(hipblasHandle_t handle,
                          hipblasSideMode_t side, hipblasFillMode_t uplo,
                          hipblasOperation_t trans, hipblasDiagType_t diag,
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
// hipblasStatus_t kblasCtrsm(hipblasHandle_t handle,
//                           hipblasSideMode_t side, hipblasFillMode_t uplo,
//                           hipblasOperation_t trans, hipblasDiagType_t diag,
//                           int m, int n,
//                           const hipComplex *alpha,
//                           const hipComplex *A, int lda,
//                                 hipComplex *B, int ldb){
//   return kblasXtrsm(handle,
//                     side, uplo, trans, diag,
//                     m, n,
//                     alpha, A, lda,
//                            B, ldb);
// }
// hipblasStatus_t kblasZtrsm(hipblasHandle_t handle,
//                           hipblasSideMode_t side, hipblasFillMode_t uplo,
//                           hipblasOperation_t trans, hipblasDiagType_t diag,
//                           int m, int n,
//                           const hipDoubleComplex *alpha,
//                           const hipDoubleComplex *A, int lda,
//                                 hipDoubleComplex *B, int ldb){
//   return kblasXtrsm(handle,
//                     side, uplo, trans, diag,
//                     m, n,
//                     alpha, A, lda,
//                            B, ldb);
// }
//==============================================================================================

hipblasStatus_t kblas_strsm(hipblasHandle_t handle,
                           hipblasSideMode_t side, hipblasFillMode_t uplo,
                           hipblasOperation_t trans, hipblasDiagType_t diag,
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
hipblasStatus_t kblas_dtrsm(hipblasHandle_t handle,
                           hipblasSideMode_t side, hipblasFillMode_t uplo,
                           hipblasOperation_t trans, hipblasDiagType_t diag,
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
// hipblasStatus_t kblas_ctrsm(hipblasHandle_t handle,
//                            hipblasSideMode_t side, hipblasFillMode_t uplo,
//                            hipblasOperation_t trans, hipblasDiagType_t diag,
//                            int m, int n,
//                            const hipComplex *alpha,
//                            const hipComplex *A, int lda,
//                                  hipComplex *B, int ldb){
//   return kblasXtrsm_cpu(handle,
//                         side, uplo, trans, diag,
//                         m, n,
//                         alpha, A, lda,
//                               B, ldb);
// }
// hipblasStatus_t kblas_ztrsm(hipblasHandle_t handle,
//                            hipblasSideMode_t side, hipblasFillMode_t uplo,
//                            hipblasOperation_t trans, hipblasDiagType_t diag,
//                            int m, int n,
//                            const hipDoubleComplex *alpha,
//                            const hipDoubleComplex *A, int lda,
//                                  hipDoubleComplex *B, int ldb){
//   return kblasXtrsm_cpu(handle,
//                         side, uplo, trans, diag,
//                         m, n,
//                         alpha, A, lda,
//                               B, ldb);
// }

//==============================================================================================
hipblasStatus_t kblas_strsm_mgpu(hipblasHandle_t handle,
                                hipblasSideMode_t side, hipblasFillMode_t uplo,
                                hipblasOperation_t trans, hipblasDiagType_t diag,
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
hipblasStatus_t kblas_dtrsm_mgpu(hipblasHandle_t handle,
                                hipblasSideMode_t side, hipblasFillMode_t uplo,
                                hipblasOperation_t trans, hipblasDiagType_t diag,
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

// hipblasStatus_t kblas_ctrsm_mgpu(hipblasHandle_t handle,
//                                 hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                 hipblasOperation_t trans, hipblasDiagType_t diag,
//                                 int m, int n,
//                                 const hipComplex *alpha,
//                                 const hipComplex *A, int lda,
//                                       hipComplex *B, int ldb,
//                                 int ngpu){
//   return kblasXtrsm_cpu_m(handle,
//                         side, uplo, trans, diag,
//                         m, n,
//                         alpha, A, lda,
//                               B, ldb,
//                         ngpu);
// }
// hipblasStatus_t kblas_ztrsm_mgpu(hipblasHandle_t handle,
//                                 hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                 hipblasOperation_t trans, hipblasDiagType_t diag,
//                                 int m, int n,
//                                 const hipDoubleComplex *alpha,
//                                 const hipDoubleComplex *A, int lda,
//                                       hipDoubleComplex *B, int ldb,
//                                 int ngpu){
//   return kblasXtrsm_cpu_m(handle,
//                         side, uplo, trans, diag,
//                         m, n,
//                         alpha, A, lda,
//                               B, ldb,
//                         ngpu);
// }



