/**
  --* (C) Copyright 2013 King Abdullah University of Science and Technology
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
#ifndef __XTRSM_BATCH_DRIVERS_H__
#define __XTRSM_BATCH_DRIVERS_H__


#include "Xgemm_batch_core.cuh"
#include "Xtrsm_batch_kernels.cuh"

//==============================================================================================
template<class T, bool STRIDED>
void Xtrsm_batch_wsquery_core(int batchCount,
                              char side, int m, int n,
                              kblasWorkspaceState_t ws)
{
  if( ( (side == KBLAS_Right) && (n > 16) ) ||
      ( (side == KBLAS_Left ) && (m > 16) ) ){
    if(STRIDED){
      Xgemm_batch_strided_wsquery_core<T>(batchCount, ws);
    }else{
      Xgemm_batch_wsquery_core<T>(batchCount,
                              1, 1, 1, 1, 1, 1,
                              ws);
    }
  }else{
    ws->reset();
  }
}

//TODO tuning variable
#define BY 64

#define TRSM_R_NUM_VARIANTS 8
#define TRSM_kernel_variant_R(__T, __T_PTR, __STRIDED, __TX, __BY)                            \
      kernel_trsm_U_RLXN_registers_fixN_mulM<__T, __T_PTR, __STRIDED,  true,  __TX, __BY>,    \
      kernel_trsm_U_RLXN_registers_fixN_mulM<__T, __T_PTR, __STRIDED, false,  __TX, __BY>,    \
      kernel_trsm_U_RLXN_registers_fixN_varM<__T, __T_PTR, __STRIDED,  true,  __TX, __BY>,    \
      kernel_trsm_U_RLXN_registers_fixN_varM<__T, __T_PTR, __STRIDED, false,  __TX, __BY>,    \
      kernel_trsm_U_RLXN_registers_varN_varM<__T, __T_PTR, __STRIDED,  true,  __TX, __BY>,    \
      kernel_trsm_U_RLXN_registers_varN_varM<__T, __T_PTR, __STRIDED, false,  __TX, __BY>,    \
      kernel_trsm_U_RLXN_registers_varN_varM<__T, __T_PTR, __STRIDED,  true,  __TX, __BY>,    \
      kernel_trsm_U_RLXN_registers_varN_varM<__T, __T_PTR, __STRIDED, false,  __TX, __BY>

#define TRSM_L_NUM_VARIANTS 4
#define TRSM_kernel_variant_L(__T, __T_PTR, __STRIDED, __TX, __BY)                            \
      kernel_trsm_U_LLXN_registers_Mfix_Nvar<__T, __T_PTR, __STRIDED,  true,  __TX, __BY>,    \
      kernel_trsm_U_LLXN_registers_Mfix_Nvar<__T, __T_PTR, __STRIDED, false,  __TX, __BY>,    \
      kernel_trsm_U_LLXN_registers_MNvar    <__T, __T_PTR, __STRIDED,  true,  __TX, __BY>,    \
      kernel_trsm_U_LLXN_registers_MNvar    <__T, __T_PTR, __STRIDED, false,  __TX, __BY>

#define offA(i,j) A + A_row_off + (i) + (A_col_off + (j)) * lda
#define offB(i,j) B + B_row_off + (i) + (B_col_off + (j)) * ldb
#define Aoff(_i,_j) A, A_row_off + (_i), A_col_off + (_j)
#define Boff(_i,_j) B, B_row_off + (_i), B_col_off + (_j)

template<class T, class T_PTR, bool STRIDED>
int Xtrsm_batch_core( kblasHandle_t handle,
                      char side, char uplo, char trans, char diag,
                      const int m, const int n,
                      const T alpha,
                      T_PTR A, int A_row_off, int A_col_off, int lda, long strideA,
                      T_PTR B, int B_row_off, int B_col_off, int ldb, long strideB,
                      int batchCount)
{

  if( uplo == KBLAS_Upper || diag == KBLAS_Unit ){
    printf("(Upper | Unit) TRSM_BATCH is not implemented yet\n");
    return KBLAS_NotImplemented;
  }

  if( ((side == KBLAS_Right) && (0 < n && n <= 16)) ||
      ((side == KBLAS_Left ) && (0 < m && m <= 16)) )
  {
    int2 dims[] = {
      { 8, 8},//2 warps
      { 8,12},//3 warps
      {16, 2},//1 warps
      {16, 4},//2 warps
      { 8, 4},//1 warps
      {32, 1},//4 warps
      { 8,16},//4 warps
      {16, 6},//3 warps
      {16, 8},//4 warps
      {16,10} //5 warps
    };
    int func_idx = 0;
    int dim_idx = 0;

    typedef void (*trsm_kernels_type)( const int m, const int n, int batchCount,
                                       const T alpha, T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                                                   T_PTR B_array, int B_row_off, int B_col_off, int ldb, long strideB);

    trsm_kernels_type trsm_kernels[] = {
      TRSM_kernel_variant_R(T, T_PTR, STRIDED,  8, BY),
      TRSM_kernel_variant_R(T, T_PTR, STRIDED, 16, BY),
      TRSM_kernel_variant_L(T, T_PTR, STRIDED,  8, BY),
      TRSM_kernel_variant_L(T, T_PTR, STRIDED, 16, BY)
    };
    int mmul = ((n == 16) && (m % 16 == 0)) || ((n == 8) && (m % 8 == 0)), nvar = (n != 8) && (n != 16);
    int mvar = (m != 8) && (m != 16);
    func_idx = (side == KBLAS_Right) * ((trans == KBLAS_NoTrans) + 2 * (mmul == 0) + 8 * (n > 8) + 4 * (nvar == 1))
               +
               (side == KBLAS_Left) * (2*TRSM_R_NUM_VARIANTS + (trans == KBLAS_NoTrans) + 4 * (m > 8) + 2 * (mvar == 1));
    // printf("func_idx(%d) ",func_idx); fflush( stdout );
    int IS_SINGLE = (typeid(T) == typeid(float));
    dim_idx = (side == KBLAS_Right) * (n > 8) * 2 + (side == KBLAS_Left) * (m > 8) * 2 + (IS_SINGLE == 1);
    //dim_idx = (side == KBLAS_Right) * (n > 8) + (side == KBLAS_Left) * (m > 8);
    // if(handle->back_door[0] >= 0){
    //   dim_idx = handle->back_door[0];
    //   //printf("kblas_back_door %d\n",kblas_back_door); fflush( stdout );
    // }
    dim3 blockDim( dims[dim_idx].x, dims[dim_idx].y );
    dim3 gridDim( batchCount / blockDim.y + (batchCount % blockDim.y != 0),
                  (side == KBLAS_Right) * ( m/BY + ((m % BY) != 0) ) +
                  (side == KBLAS_Left ) * ( n/BY + ((n % BY) != 0) )
                  );//TODO
    // printf("blockDim(%d,%d), gridDim(%d,%d) ", blockDim.x, blockDim.y, gridDim.x, gridDim.y);fflush( stdout );
    long sh_mem = blockDim.x*(blockDim.x+1)*blockDim.y*sizeof(T);
    long trsm_kernels_sharedMem[] = {
      0,
      sh_mem
    };
    // printf("%s(%d): STRIDED(%d), A_row_off(%d), A_col_off(%d), B_row_off(%d), B_col_off(%d), m(%d), n(%d)\n",
    //         __FILE__, __LINE__, STRIDED, A_row_off, A_col_off, B_row_off, B_col_off, m, n);

    //cudaStream_t curStream;
    //check_error( cublasGetStream( handle, &curStream ), KBLAS_cuBLAS_Error);

    trsm_kernels[func_idx]<<< gridDim, blockDim, trsm_kernels_sharedMem[(side == KBLAS_Left)], handle->stream>>>
                              ( m, n, batchCount,
                                alpha, A, A_row_off, A_col_off, lda, strideA,
                                       B, B_row_off, B_col_off, ldb, strideB);
    check_error_ret( cudaGetLastError(), KBLAS_UnknownError);
  }
  else
  //try recurssion
  if( ( (side == KBLAS_Right) && (n > 16) ) ||
      ( (side == KBLAS_Left ) && (m > 16) ) )
  {

    int status;
    T one = make_one<T>(), mone = -one;

    if(side == KBLAS_Right){

      int n1, n2;

      if(REG_SIZE(n))
        n1 = n2 = n/2;
      else{
        n1 = CLOSEST_REG_SIZE(n);
        n2 = n-n1;
      }

      //Right / Lower / [Conj]Trans
      if(trans == KBLAS_Trans){
        //TRSM_BATCH
        check_error_ret((status = Xtrsm_batch_core<T, T_PTR, STRIDED>(
                                                  handle,
                                                  side, uplo, trans, diag,
                                                  m, n1,
                                                  alpha, Aoff(0, 0), lda, strideA,
                                                         Boff(0, 0), ldb, strideB,
                                                  batchCount)), status);

        T mInvAlpha = mone / alpha;

        //GEMM_BATCH
        if(STRIDED){
          check_error_ret (status = Xgemm_batch_strided(handle,
                                                        KBLAS_NoTrans, trans,
                                                        m, n2, n1,
                                                        mInvAlpha, (const T*)offB( 0,  0), ldb, strideB,
                                                                   (const T*)offA(n1,  0), lda, strideA,
                                                        one,             (T*)offB( 0, n1), ldb, strideB,
                                                        batchCount), status);
        }else{
          check_error_ret (status = kblas_gemm_batch( handle,
                                                      KBLAS_NoTrans, trans,
                                                      m, n2, n1,
                                                      mInvAlpha, (const T**)Boff( 0,  0), ldb,
                                                                 (const T**)Aoff(n1,  0), lda,
                                                      one,             (T**)Boff( 0, n1), ldb,
                                                      batchCount), status);
        }

        //TRSM_BATCH
        check_error_ret ((status = Xtrsm_batch_core<T, T_PTR, STRIDED>(
                                                  handle,
                                                  side, uplo, trans, diag,
                                                  m, n2,
                                                  alpha, Aoff(n1, n1), lda, strideA,
                                                         Boff( 0, n1), ldb, strideB,
                                                  batchCount)), status);
      }
      //Right / Lower / NoTrans
      else{
        //TRSM_BATCH
        check_error_ret((status = Xtrsm_batch_core<T, T_PTR, STRIDED>(
                                                  handle,
                                                  side, uplo, trans, diag,
                                                  m, n2,
                                                  // alpha, A, A_row_off + n1, A_col_off + n1, lda, strideA,
                                                  //        B, B_row_off +  0, B_col_off + n1, ldb, strideB,
                                                  alpha, Aoff(n1, n1), lda, strideA,
                                                         Boff( 0, n1), ldb, strideB,
                                                  batchCount)), status);

        //GEMM_BATCH
        if(STRIDED){
          check_error_ret((status = Xgemm_batch_strided(handle,
                                                        KBLAS_NoTrans, trans,
                                                        m, n1, n2,
                                                        mone, (const T*)offB( 0, n1), ldb, strideB,
                                                              (const T*)offA(n1,  0), lda, strideA,
                                                        alpha,      (T*)offB( 0,  0), ldb, strideB,
                                                        batchCount)), status);
        }else{
          // printf("non-strided %d\n", __LINE__);
          check_error_ret((status = kblas_gemm_batch( handle,
                                                      KBLAS_NoTrans, trans,
                                                      m, n1, n2,
                                                      mone, (const T**)Boff( 0, n1), ldb,
                                                            (const T**)Aoff(n1,  0), lda,
                                                      alpha,      (T**)Boff( 0,  0), ldb,
                                                      batchCount)), status);
        }

        //TRSM_BATCH
        check_error_ret((status = Xtrsm_batch_core<T, T_PTR, STRIDED>(
                                                  handle,
                                                  side, uplo, trans, diag,
                                                  m, n1,
                                                  one, Aoff(0, 0), lda, strideA,
                                                       Boff(0, 0), ldb, strideB,
                                                  batchCount)), status);
      }
    }else{

      int m1, m2;
      if(REG_SIZE(m))
        m1 = m2 = m/2;
      else{
        m1 = CLOSEST_REG_SIZE(m);
        m2 = m-m1;
      }

      //Left / Lower / [Conj]Trans
      if(trans == KBLAS_Trans){
        //TRSM_BATCH
        check_error_ret((status = Xtrsm_batch_core<T, T_PTR, STRIDED>(
                                                  handle,
                                                  side, uplo, trans, diag,
                                                  m2, n,
                                                  alpha, Aoff(m1, m1), lda, strideA,
                                                         Boff(m1,  0), ldb, strideB,
                                                  batchCount)), status);

        //GEMM_BATCH
        if(STRIDED){
          check_error_ret (status = Xgemm_batch_strided(handle,
                                                        trans, KBLAS_NoTrans,
                                                        m1, n, m2,
                                                        mone, (const T*)offA(m1, 0), lda, strideA,
                                                              (const T*)offB(m1, 0), ldb, strideB,
                                                        alpha,      (T*)offB( 0, 0), ldb, strideB,
                                                        batchCount), status);
        }else{
          check_error_ret (status = kblas_gemm_batch( handle,
                                                      trans, KBLAS_NoTrans,
                                                      m1, n, m2,
                                                      mone, (const T**)Aoff(m1, 0), lda,
                                                            (const T**)Boff(m1, 0), ldb,
                                                      alpha,      (T**)Boff( 0, 0), ldb,
                                                      batchCount), status);
        }

        //TRSM_BATCH
        check_error_ret((status = Xtrsm_batch_core<T, T_PTR, STRIDED>(
                                                  handle,
                                                  side, uplo, trans, diag,
                                                  m1, n,
                                                  one, Aoff(0, 0), lda, strideA,
                                                       Boff(0, 0), ldb, strideB,
                                                  batchCount)), status);
      }
      //Left / Lower / NoTrans
      else{
        //TRSM_BATCH
        check_error_ret((status = Xtrsm_batch_core<T, T_PTR, STRIDED>(
                                                  handle,
                                                  side, uplo, trans, diag,
                                                  m1, n,
                                                  alpha, Aoff(0, 0), lda, strideA,
                                                         Boff(0, 0), ldb, strideB,
                                                  batchCount)), status);

        //GEMM_BATCH
        if(STRIDED){
          check_error_ret((status = Xgemm_batch_strided(handle,
                                                        trans, KBLAS_NoTrans,
                                                        m2, n, m1,
                                                        mone, (const T*)offA(m1, 0), lda, strideA,
                                                              (const T*)offB( 0, 0), ldb, strideB,
                                                        alpha,      (T*)offB(m1, 0), ldb, strideB,
                                                        batchCount)), status);
        }else{
          check_error_ret((status = kblas_gemm_batch( handle,
                                                      trans, KBLAS_NoTrans,
                                                      m2, n, m1,
                                                      mone, (const T**)Aoff(m1, 0), lda,
                                                            (const T**)Boff( 0, 0), ldb,
                                                      alpha,      (T**)Boff(m1, 0), ldb,
                                                      batchCount)), status);
        }

        //TRSM_BATCH
        check_error_ret((status = Xtrsm_batch_core<T, T_PTR, STRIDED>(
                                                  handle,
                                                  side, uplo, trans, diag,
                                                  m2, n,
                                                  one, Aoff(m1, m1), lda, strideA,
                                                       Boff(m1,  0), ldb, strideB,
                                                  batchCount)), status);
      }
    }
  }else{
    //should not reach this
    return KBLAS_NotImplemented;
  }
  return KBLAS_Success;
}

#endif //__XTRSM_BATCH_DRIVERS_H__