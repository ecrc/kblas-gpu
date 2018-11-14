/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xtrsm_batch_drivers.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef __XTRSM_BATCH_DRIVERS_H__
#define __XTRSM_BATCH_DRIVERS_H__


// #include "Xgemm_batch_core.cuh"
#include "Xtrsm_batch_kernels.cuh"

//==============================================================================================

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

    int IS_SINGLE = (typeid(T) == typeid(float));
    dim_idx = (side == KBLAS_Right) * (n > 8) * 2 + (side == KBLAS_Left) * (m > 8) * 2 + (IS_SINGLE == 1);
    //dim_idx = (side == KBLAS_Right) * (n > 8) + (side == KBLAS_Left) * (m > 8);

    dim3 blockDim( dims[dim_idx].x, dims[dim_idx].y );
    dim3 gridDim( batchCount / blockDim.y + (batchCount % blockDim.y != 0),
                  (side == KBLAS_Right) * ( m/BY + ((m % BY) != 0) ) +
                  (side == KBLAS_Left ) * ( n/BY + ((n % BY) != 0) )
                  );//TODO

    long sh_mem = blockDim.x*(blockDim.x+1)*blockDim.y*sizeof(T);
    long trsm_kernels_sharedMem[] = {
      0,
      sh_mem
    };

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
        check_ret_error( (Xtrsm_batch_core<T, T_PTR, STRIDED>(
                                          handle,
                                          side, uplo, trans, diag,
                                          m, n1,
                                          alpha, Aoff(0, 0), lda, strideA,
                                                 Boff(0, 0), ldb, strideB,
                                          batchCount)) );
        T mInvAlpha = mone / alpha;

        //GEMM_BATCH
        check_ret_error( Xgemm_batch( handle,
                                      KBLAS_NoTrans, trans,
                                      m, n2, n1,
                                      mInvAlpha, (T_PTR)Boff( 0,  0), ldb, strideB,
                                                 (T_PTR)Aoff(n1,  0), lda, strideA,
                                      one,       (T_PTR)Boff( 0, n1), ldb, strideB,
                                      batchCount) );
        //TRSM_BATCH
        check_ret_error( (Xtrsm_batch_core<T, T_PTR, STRIDED>(
                                          handle,
                                          side, uplo, trans, diag,
                                          m, n2,
                                          alpha, Aoff(n1, n1), lda, strideA,
                                                 Boff( 0, n1), ldb, strideB,
                                          batchCount)) );
      }
      //Right / Lower / NoTrans
      else{
        //TRSM_BATCH
        check_ret_error( (Xtrsm_batch_core<T, T_PTR, STRIDED>(
                                          handle,
                                          side, uplo, trans, diag,
                                          m, n2,
                                          // alpha, A, A_row_off + n1, A_col_off + n1, lda, strideA,
                                          //        B, B_row_off +  0, B_col_off + n1, ldb, strideB,
                                          alpha, Aoff(n1, n1), lda, strideA,
                                                 Boff( 0, n1), ldb, strideB,
                                          batchCount)) );
        //GEMM_BATCH
        check_ret_error( Xgemm_batch( handle,
                                      KBLAS_NoTrans, trans,
                                      m, n1, n2,
                                      mone,  (T_PTR)Boff( 0, n1), ldb, strideB,
                                             (T_PTR)Aoff(n1,  0), lda, strideA,
                                      alpha, (T_PTR)Boff( 0,  0), ldb, strideB,
                                      batchCount) );
        //TRSM_BATCH
        check_ret_error( (Xtrsm_batch_core<T, T_PTR, STRIDED>(
                                          handle,
                                          side, uplo, trans, diag,
                                          m, n1,
                                          one, Aoff(0, 0), lda, strideA,
                                               Boff(0, 0), ldb, strideB,
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
        //TRSM_BATCH
        check_ret_error( (Xtrsm_batch_core<T, T_PTR, STRIDED>(
                                          handle,
                                          side, uplo, trans, diag,
                                          m2, n,
                                          alpha, Aoff(m1, m1), lda, strideA,
                                                 Boff(m1,  0), ldb, strideB,
                                          batchCount)) );
        //GEMM_BATCH
        check_ret_error( Xgemm_batch( handle,
                                      trans, KBLAS_NoTrans,
                                      m1, n, m2,
                                      mone,  (T_PTR)Aoff(m1, 0), lda, strideA,
                                             (T_PTR)Boff(m1, 0), ldb, strideB,
                                      alpha, (T_PTR)Boff( 0, 0), ldb, strideB,
                                      batchCount) );
        //TRSM_BATCH
        check_ret_error( (Xtrsm_batch_core<T, T_PTR, STRIDED>(
                                          handle,
                                          side, uplo, trans, diag,
                                          m1, n,
                                          one, Aoff(0, 0), lda, strideA,
                                               Boff(0, 0), ldb, strideB,
                                          batchCount)) );
      }
      //Left / Lower / NoTrans
      else{
        //TRSM_BATCH
        check_ret_error( (Xtrsm_batch_core<T, T_PTR, STRIDED>(
                                          handle,
                                          side, uplo, trans, diag,
                                          m1, n,
                                          alpha, Aoff(0, 0), lda, strideA,
                                                 Boff(0, 0), ldb, strideB,
                                          batchCount)) );
        //GEMM_BATCH
        check_ret_error( Xgemm_batch( handle,
                                      trans, KBLAS_NoTrans,
                                      m2, n, m1,
                                      mone,  (T_PTR)Aoff(m1, 0), lda, strideA,
                                             (T_PTR)Boff( 0, 0), ldb, strideB,
                                      alpha, (T_PTR)Boff(m1, 0), ldb, strideB,
                                      batchCount) );
        //TRSM_BATCH
        check_ret_error( (Xtrsm_batch_core<T, T_PTR, STRIDED>(
                                          handle,
                                          side, uplo, trans, diag,
                                          m2, n,
                                          one, Aoff(m1, m1), lda, strideA,
                                               Boff(m1,  0), ldb, strideB,
                                          batchCount)) );
      }
    }
  }else{
    //should not reach this
    return KBLAS_NotImplemented;
  }
  return KBLAS_Success;
}



//==============================================================================================
template<class T>
int Xtrsm_batch_nonuniform_core(kblasHandle_t handle,
                                char side, char uplo, char trans, char diag,
                                int *m, int *n,
                                T alpha,
                                T** A, int *lda,
                                T** B, int *ldb,
                                int max_m, int max_n,
                                int batchCount)
{
  if(handle->use_magma){
  #ifdef USE_MAGMA

    //TODO: it might be better to look up the maximum per 65k chunck, except that synchromizations will be forced
    // if(batchCount > 65535) return KBLAS_Error_WrongInput;
    KBlasWorkspaceState ws_needed;
    trsm_batch_nonuniform_wsquery_core((kblasWorkspaceState_t)&ws_needed);

    if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) ){
      return KBLAS_InsufficientWorkspace;
    }

    int h_max_mn[2];
    kblasWorkspace_t ws_current = &(handle->work_space);
    int* d_max_mn = (int*)(ws_current->d_data);

    //take care of batch size limitation with magma
    int batch_increment = 65535;
    int batch_start = 0;
    if(max_m > 0 || max_n > 0){
      h_max_mn[0] = max_m;
      h_max_mn[1] = max_n;
    }

    while(batch_start != batchCount)
    {
      int batch_size = kmin(batch_increment, batchCount - batch_start);

      if((batchCount > batch_increment) || (max_m <= 0 && max_n <= 0)){
        // compute the max. dimensions
        kblas_imax_size_2(handle, m, n, *d_max_mn, *(d_max_mn+1), batch_size);
        check_error_ret( cublasGetVectorAsync( 2, sizeof(int), d_max_mn, 1, h_max_mn, 1, handle->stream ), KBLAS_cuBLAS_Error);
        check_error_ret( cudaStreamSynchronize(handle->stream), KBLAS_CUDA_Error );
      }
      magmablas_Xtrsm_vbatched_max_nocheck(
                  (magma_side_t)(side == KBLAS_Left ? MagmaLeft : MagmaRight),
                  (magma_uplo_t)(uplo == KBLAS_Lower ? MagmaLower : MagmaUpper),
                  (magma_trans_t)(trans == KBLAS_Trans ? MagmaTrans : MagmaNoTrans),
                  (magma_diag_t)(diag == KBLAS_NonUnit? MagmaNonUnit : MagmaUnit),
                  m, n, alpha,
                  A, lda,
                  B, ldb,
                  batch_size,
                  h_max_mn[0], h_max_mn[1], handle->magma_queue);

      A += batch_size;
      B += batch_size;
      m += batch_size;
      n += batch_size;
      lda += batch_size;
      ldb += batch_size;

      batch_start += batch_size;
      check_error_ret( cudaGetLastError(), KBLAS_MAGMA_Error);
    }
  #else
    printf("Configuration error at %s in file %s at line %d, MAGMA required but KBLAS not compiled with it!\n", __func__, __FILE__, __LINE__ );
    return KBLAS_WrongConfig;
  #endif
  }
  else
  if(!handle->use_magma){
    printf("Configuration error at %s in file %s at line %d, MAGMA required but not enabled!\n", __func__, __FILE__, __LINE__ );
    return KBLAS_WrongConfig;
  }

  return KBLAS_Success;
}


#endif //__XTRSM_BATCH_DRIVERS_H__
