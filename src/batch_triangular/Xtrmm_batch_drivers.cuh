/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xtrmm_batch_drivers.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ali Charara
 * @date 2020-12-10
 **/

#ifndef __XTRMM_BATCH_DRIVERS_H__
#define __XTRMM_BATCH_DRIVERS_H__


#include "Xtrmm_batch_kernels.cuh"

//==============================================================================================

//TODO tuning variable
#define BY 64

#define TRMM_R_NUM_VARIANTS 8
#define TRMM_kernel_variant_R(__T, __T_PTR, __TX, __BY)                            \
      NULL, /*kernel_trsm_U_RLXN_registers_fixN_mulM<__T, __T_PTR, __STRIDED,  true,  __TX, __BY>,*/    \
      NULL, /*kernel_trsm_U_RLXN_registers_fixN_mulM<__T, __T_PTR, __STRIDED, false,  __TX, __BY>,*/    \
      NULL, /*kernel_trsm_U_RLXN_registers_fixN_varM<__T, __T_PTR, __STRIDED,  true,  __TX, __BY>,*/    \
      NULL, /*kernel_trsm_U_RLXN_registers_fixN_varM<__T, __T_PTR, __STRIDED, false,  __TX, __BY>,*/    \
      NULL, /*kernel_trsm_U_RLXN_registers_varN_varM<__T, __T_PTR, __STRIDED,  true,  __TX, __BY>,*/    \
      NULL, /*kernel_trsm_U_RLXN_registers_varN_varM<__T, __T_PTR, __STRIDED, false,  __TX, __BY>,*/    \
      NULL, /*kernel_trsm_U_RLXN_registers_varN_varM<__T, __T_PTR, __STRIDED,  true,  __TX, __BY>,*/    \
      NULL  /*kernel_trsm_U_RLXN_registers_varN_varM<__T, __T_PTR, __STRIDED, false,  __TX, __BY>*/

#define TRMM_L_NUM_VARIANTS 4
#define TRMM_kernel_variant_L(__T, __T_PTR, __TX, __BY)                            \
      kernel_trmm_U_LLXN_reg_shared_Mfix_Nvar<__T, __T_PTR,  true,  __TX, __BY>,   \
      kernel_trmm_U_LLXN_reg_shared_Mfix_Nvar<__T, __T_PTR, false,  __TX, __BY>,   \
      kernel_trmm_U_LLXN_reg_shared_MNvar    <__T, __T_PTR,  true,  __TX, __BY>,   \
      kernel_trmm_U_LLXN_reg_shared_MNvar    <__T, __T_PTR, false,  __TX, __BY>

#define offA(i,j) A + A_row_off + (i) + (A_col_off + (j)) * lda
#define offB(i,j) B + B_row_off + (i) + (B_col_off + (j)) * ldb
#define Aoff(_i,_j) A, A_row_off + (_i), A_col_off + (_j)
#define Boff(_i,_j) B, B_row_off + (_i), B_col_off + (_j)

template<class T, class T_PTR>
int Xtrmm_batch_core( kblasHandle_t handle,
                      char side, char uplo, char trans, char diag,
                      const int m, const int n,
                      const T alpha,
                      T_PTR A, int A_row_off, int A_col_off, int lda, long strideA,
                      T_PTR B, int B_row_off, int B_col_off, int ldb, long strideB,
                      int batchCount)
{

  if( side == KBLAS_Right || uplo == KBLAS_Upper || diag == KBLAS_Unit ){
    printf("(Right | Upper | Unit) TRMM_BATCH is not implemented yet\n");
    return KBLAS_NotImplemented;
  }

  if( ((side == KBLAS_Right) && (0 < n && n <= 16)) ||
      ((side == KBLAS_Left ) && (0 < m && m <= 16)) )
  {
    int2 dims[] = {
      { 8, 4},//1 warps
      {16, 2},//1 warps
      { 8, 8},//2 warps
      {32, 1},//4 warps
      { 8,12},//3 warps
      { 8,16},//4 warps
      {16, 4},//2 warps
      {16, 8} //4 warps
    };
    int func_idx = 0;
    int dim_idx = 0;

    typedef void (*trmm_kernels_type)( const int m, const int n, int batchCount,
                                       const T alpha, T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                                                   T_PTR B_array, int B_row_off, int B_col_off, int ldb, long strideB);

    trmm_kernels_type trmm_kernels[] = {
      TRMM_kernel_variant_R(T, T_PTR,  8, BY),
      TRMM_kernel_variant_R(T, T_PTR, 16, BY),
      TRMM_kernel_variant_L(T, T_PTR,  8, BY),
      TRMM_kernel_variant_L(T, T_PTR, 16, BY)
    };
    int mmul = ((n == 16) && (m % 16 == 0)) || ((n == 8) && (m % 8 == 0)), nvar = (n != 8) && (n != 16);
    int mvar = (m != 8) && (m != 16);
    func_idx = (side == KBLAS_Right) * ((trans == KBLAS_NoTrans) + 2 * (mmul == 0) + 8 * (n > 8) + 4 * (nvar == 1))
               +
               (side == KBLAS_Left) * (2*TRMM_R_NUM_VARIANTS + (trans == KBLAS_NoTrans) + 4 * (m > 8) + 2 * (mvar == 1));
    // printf("func_idx(%d) ",func_idx); fflush( stdout );
    int IS_SINGLE = (typeid(T) == typeid(float));
    dim_idx = (side == KBLAS_Right) * (n > 8) + (side == KBLAS_Left) * (m > 8);
    // dim_idx = (side == KBLAS_Right) * (n > 8) * 2 + (side == KBLAS_Left) * (m > 8) * 2 + (IS_SINGLE == 1);
    //dim_idx = (side == KBLAS_Right) * (n > 8) + (side == KBLAS_Left) * (m > 8);
    // if(handle->back_door[0] >= 0){
    //   dim_idx = handle->back_door[0];
    //   //printf("kblas_back_door %d\n",kblas_back_door); fflush( stdout );
    // }
    dim3 blockDim( dims[dim_idx].x, dims[dim_idx].y );
    dim3 gridDim( batchCount / blockDim.y + (batchCount % blockDim.y != 0),
                  (side == KBLAS_Right) * ( m/BY + ((m % BY) != 0) ) +
                  (side == KBLAS_Left ) * ( n/BY + ((n % BY) != 0) )
                  );

    trmm_kernels[func_idx]<<< gridDim, blockDim, blockDim.x*(blockDim.x+2)*blockDim.y*sizeof(T), handle->stream>>>
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

    // int status;
    T one = make_one<T>();

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
        //TRMM_BATCH
        check_ret_error(( Xtrmm_batch_core<T, T_PTR>(
                                          handle,
                                          side, uplo, trans, diag,
                                          m, n2,
                                          alpha, Aoff(n1, n1), lda, strideA,
                                                 Boff( 0, n1), ldb, strideB,
                                          batchCount)) );

        //GEMM_BATCH
        check_ret_error( Xgemm_batch( handle,
                                      KBLAS_NoTrans, trans,
                                      m, n2, n1,
                                      alpha, (T_PTR)Boff( 0,  0), ldb, strideB,
                                             (T_PTR)Aoff(n1,  0), lda, strideA,
                                      one,   (T_PTR)Boff( 0, n1), ldb, strideB,
                                      batchCount) );

        //TRMM_BATCH
        check_ret_error(( Xtrmm_batch_core<T, T_PTR>(
                                          handle,
                                          side, uplo, trans, diag,
                                          m, n1,
                                          alpha, Aoff(0, 0), lda, strideA,
                                                 Boff(0, 0), ldb, strideB,
                                          batchCount)) );
      }
      //Right / Lower / NoTrans
      else{
        //TRMM_BATCH
        check_ret_error(( Xtrmm_batch_core<T, T_PTR>(
                                          handle,
                                          side, uplo, trans, diag,
                                          m, n1,
                                          alpha, Aoff(0, 0), lda, strideA,
                                                 Boff(0, 0), ldb, strideB,
                                          batchCount)) );

        //GEMM_BATCH
        check_ret_error( Xgemm_batch( handle,
                                      KBLAS_NoTrans, trans,
                                      m, n1, n2,
                                      alpha, (T_PTR)Boff( 0, n1), ldb, strideB,
                                             (T_PTR)Aoff(n1,  0), lda, strideA,
                                      one,   (T_PTR)Boff( 0,  0), ldb, strideB,
                                      batchCount) );

        //TRMM_BATCH
        check_ret_error(( Xtrmm_batch_core<T, T_PTR>(
                                          handle,
                                          side, uplo, trans, diag,
                                          m, n2,
                                          alpha, Aoff(n1, n1), lda, strideA,
                                                 Boff( 0, n1), ldb, strideB,
                                          batchCount)) );
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
        //TRMM_BATCH
        check_ret_error(( Xtrmm_batch_core<T, T_PTR>(
                                          handle,
                                          side, uplo, trans, diag,
                                          m1, n,
                                          alpha, Aoff(0, 0), lda, strideA,
                                                 Boff(0, 0), ldb, strideB,
                                          batchCount)) );

        //GEMM_BATCH
        check_ret_error( Xgemm_batch( handle,
                                      trans, KBLAS_NoTrans,
                                      m1, n, m2,
                                      alpha, (T_PTR)Aoff(m1, 0), lda, strideA,
                                             (T_PTR)Boff(m1, 0), ldb, strideB,
                                      one,   (T_PTR)Boff( 0, 0), ldb, strideB,
                                      batchCount) );

        //TRMM_BATCH
        check_ret_error(( Xtrmm_batch_core<T, T_PTR>(
                                          handle,
                                          side, uplo, trans, diag,
                                          m2, n,
                                          alpha, Aoff(m1, m1), lda, strideA,
                                                 Boff(m1,  0), ldb, strideB,
                                          batchCount)) );
      }
      //Left / Lower / NoTrans
      else{
        //TRMM_BATCH
        check_ret_error(( Xtrmm_batch_core<T, T_PTR>(
                                          handle,
                                          side, uplo, trans, diag,
                                          m2, n,
                                          alpha, Aoff(m1, m1), lda, strideA,
                                                 Boff(m1,  0), ldb, strideB,
                                          batchCount)) );

        //GEMM_BATCH
        check_ret_error( Xgemm_batch( handle,
                                      trans, KBLAS_NoTrans,
                                      m2, n, m1,
                                      alpha, (T_PTR)Aoff(m1, 0), lda, strideA,
                                             (T_PTR)Boff( 0, 0), ldb, strideB,
                                      one,   (T_PTR)Boff(m1, 0), ldb, strideB,
                                      batchCount) );

        //TRMM_BATCH
        check_ret_error(( Xtrmm_batch_core<T, T_PTR>(
                                          handle,
                                          side, uplo, trans, diag,
                                          m1, n,
                                          alpha, Aoff(0, 0), lda, strideA,
                                                 Boff(0, 0), ldb, strideB,
                                          batchCount)) );
      }
    }
  }else{
    //should not reach this
    return KBLAS_NotImplemented;
  }
  return KBLAS_Success;
}

#endif //__XTRMM_BATCH_DRIVERS_H__
