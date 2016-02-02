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
#include "Xtr_common.ch"
#include "operators.h"

//==============================================================================================

cublasStatus_t cublasXtrsm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const float *alpha,
                           const float *A, int lda,
                                 float *B, int ldb){
  return cublasStrsm(handle,
                     side, uplo, trans, diag,
                     m, n,
                     alpha, A, lda,
                            B, ldb );
}
cublasStatus_t cublasXtrsm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t      diag,
                           int m, int n,
                           const double *alpha,
                           const double *A, int lda,
                                 double *B, int ldb){
  return cublasDtrsm(handle,
                     side, uplo, trans, diag,
                     m, n,
                     alpha, A, lda,
                            B, ldb );
}
cublasStatus_t cublasXtrsm (cublasHandle_t handle,
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
                            B, ldb );
}
cublasStatus_t cublasXtrsm (cublasHandle_t handle,
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
                            B, ldb );
}

//==============================================================================================
int kblas_trsm_ib_cublas = 128;
bool kblas_trsm_use_custom = 0;
#define SIMPLE_SIZE(n) ( ((n) < WARP) || ( ((n) % WARP == 0) && ( (n) <= kblas_trsm_ib_cublas ) ) )
//==============================================================================================
template<class T>
cublasStatus_t Xtrsm(cublasHandle_t handle,
                     cublasSideMode_t side, cublasFillMode_t uplo,
                     cublasOperation_t trans, cublasDiagType_t diag,
                     int m, int n,
                     const T *alpha,
                     const T *A, int incA,
                           T *B, int incB){
  /*
  //handle odd cases with cublas
  if(  (*alpha == make_zero<T>())
    || (!kblas_trsm_use_custom)
    || (side == CUBLAS_SIDE_LEFT && m < WARP)
    || (side == CUBLAS_SIDE_RIGHT && n < WARP))*/
  {
    return cublasXtrsm(handle,
                       side, uplo, trans, diag,
                       m, n,
                       alpha, A, incA,
                              B, incB );
  }

  /*typedef void (*trmm_kernels_type)(int M, int N, T alpha, const T* A, int incA, T* B, int incB, int mb);

  #define WARPS_PER_BLOCK 8
  #define B_COLS_PER_WARP 1

  trmm_kernels_type trmm_kernels[16] = {// T, WARPS_PER_BLOCK, B_COLS_PER_WARP, LEFT, LOWER, TRANS, CONJG
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

  if((status = cublasGetStream( handle, &curStream )) != CUBLAS_STATUS_SUCCESS ) return status;

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
    trmm_kernels[func_idx]<<< gridDim, blockDim, 0, curStream>>> (m, n, *alpha, A, incA, B, incB, mb);
    if(!_kblas_error( (cudaGetLastError()), __func__, __FILE__, __LINE__ ))
      return CUBLAS_STATUS_EXECUTION_FAILED;
  }else{
    //error: we should not reach this case
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }*/
  return CUBLAS_STATUS_SUCCESS;
}
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
  T mInvAlpha = mone / alpha;
  cublasStatus_t status;

  if(*alpha == make_zero<T>()){//TODO
    return Xtrsm(handle,
                 side, uplo, trans, diag,
                 m, n,
                 alpha, A, incA,
                        B, incB );
  }
  cublasOperation_t noTrans = CUBLAS_OP_N;//Trans = CUBLAS_OP_T,
  
  if(side == CUBLAS_SIDE_LEFT){

    if(SIMPLE_SIZE(m)){
      return Xtrsm(handle,
                   side, uplo, trans, diag,
                   m, n,
                   alpha, A, incA,
                          B, incB );
    }

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
                                 trans, noTrans,
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
                                 trans, noTrans,
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
      if(trans == KBLAS_NoTrans){
        if((status = kblasXtrsm(handle,
                                side, uplo, trans, diag,
                                m1, n,
                                alpha, A, incA,
                                       B, incB
                                )) != CUBLAS_STATUS_SUCCESS) return status;
        
        if((status = cublasXgemm(handle,
                                 trans, noTrans,
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
          trans, KBLAS_NoTrans,
                    m1, n, m2,
                    mone,  A+m1, incA,
                           B+m1, incB,
                    alpha, B, incB);
        
        if((status = kblasXtrsm(handle,
          side, uplo, trans, diag,
                             m1, n,
                             one, A, incA, B, incB
        )) != CUBLAS_STATUS_SUCCESS) return status;
      }//transa == KBLAS_Trans
      
    }
    
  }
  else{//side == KBLAS_Right
    if(n <= kblas_trsm_ib_cublas){
      cublasXtrsm(side, uplo, trans, diag,
                  m, n,
                  alpha, A, incA,
                         B, incB );
      return 1;
    }
    int n1, n2;
    
    if(REG_SIZE(n))
      n1 = n2 = n/2;
    else{
      n1 = CLOSEST_REG_SIZE(n);
      n2 = n-n1;
    }
    
    if(uplo == KBLAS_Upper){
      //Right / Upper / NoTrans
      if(trans == KBLAS_NoTrans){
        if((status = kblasXtrsm(handle,
          side, uplo, trans, diag,
                       m, n1,
                       alpha, A, incA,
                       B, incB
        )) != CUBLAS_STATUS_SUCCESS) return status;
        
        if((status = cublasXgemm(handle,
          KBLAS_NoTrans, trans,
                    m, n2, n1,
                    mone,  B, incB,
                           A+n1*incA, incA,
                    alpha, B+n1*incB, incB);
        
        if((status = kblasXtrsm(handle,
          side, uplo, trans, diag,
                       m, n2,
                       one, A+n1+n1*incA, incA,
                       B+n1*incB, incB
        )) != CUBLAS_STATUS_SUCCESS) return status;
      }
      //Right / Upper / [Conj]Trans
      else{
        if((status = kblasXtrsm(handle,
          side, uplo, trans, diag,
                             m, n2,
                             alpha, A+n1+n1*incA, incA, B+n1*incB, incB
        )) != CUBLAS_STATUS_SUCCESS) return status;
        
        if((status = cublasXgemm(handle,
          KBLAS_NoTrans, trans,
                    m, n1, n2,
                    minvalpha, B+n1*incB, incB,
                               A+n1*incA, incA,
                    one,       B, incB);
        
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
      if(trans == KBLAS_NoTrans){
        if((status = kblasXtrsm(handle,
          side, uplo, trans, diag,
                       m, n2,
                       alpha, A+n1+n1*incA, incA,
                       B+n1*incB, incB
        )) != CUBLAS_STATUS_SUCCESS) return status;
        
        if((status = cublasXgemm(handle,
          KBLAS_NoTrans, trans,
                    m, n1, n2,
                    mone,  B+n1*incB, incB,
                           A+n1, incA,
                    alpha, B, incB);
        
        if((status = kblasXtrsm(handle,
          side, uplo, trans, diag,
                       m, n1,
                       one, A, incA,
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
          KBLAS_NoTrans, trans,
                    m, n2, n1,
                    minvalpha, B, incB,
                               A+n1, incA,
                    one,       B+n1*incB, incB);
        
        if((status = kblasXtrsm(handle,
          side, uplo, trans, diag,
                       m, n2,
                       alpha, A+n1+n1*incA, incA,
                       B+n1*incB, incB
        )) != CUBLAS_STATUS_SUCCESS) return status;
      }
    }
    
  }//side == KBLAS_Right
  
  return 1;
}

//==============================================================================================

extern "C" {
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
  
}




