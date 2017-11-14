/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xpotrs_batch_drivers.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 2.0.0
 * @author Ali Charara
 * @date 2017-11-13
 **/

#ifndef __XPOTRS_BATCH_DRIVERS_H__
#define __XPOTRS_BATCH_DRIVERS_H__


#include "Xpotrs_batch_kernels.cuh"

//==============================================================================================

#define offA(i,j) A + A_row_off + (i) + (A_col_off + (j)) * lda
#define offB(i,j) B + B_row_off + (i) + (B_col_off + (j)) * ldb
#define Aoff(_i,_j) A, A_row_off + (_i), A_col_off + (_j)
#define Boff(_i,_j) B, B_row_off + (_i), B_col_off + (_j)

template<class T, class T_PTR, bool STRIDED>
int Xpotrs_batch_core(kblasHandle_t handle,
                      char side, char uplo,
                      const int m, const int n,
                      T_PTR A, int A_row_off, int A_col_off, int lda, long strideA,
                      T_PTR B, int B_row_off, int B_col_off, int ldb, long strideB,
                      int batchCount)
{
  if( side == KBLAS_Left || uplo == KBLAS_Upper){
    printf("(Left | Upper) POTRS_BATCH is not implemented yet\n");
    return KBLAS_NotImplemented;
  }
  #if 0
  //TODO fix this
  if( n <= 16 ){
    int2 dims[5] = {
      { 8, 8},//2 warps
      {16, 2},//1 warps
      { 8, 4},//1 warps
      {16, 4},//2 warps
      { 8,16} //4 warps
    };

    typedef void (*potrs_kernels_type)(const int m, const int n, int batchCount,
                                       T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                                    T_PTR B_array, int B_row_off, int B_col_off, int ldb, long strideB);

    potrs_kernels_type potrs_kernels[] = {
      kernel_potrs_U_RL_registers_Nfix_Mvar<T, T_PTR, STRIDED,  8>,
      kernel_potrs_U_RL_registers_NMvar    <T, T_PTR, STRIDED,  8>,
      kernel_potrs_U_RL_registers_Nfix_Mvar<T, T_PTR, STRIDED, 16>,
      kernel_potrs_U_RL_registers_NMvar    <T, T_PTR, STRIDED, 16>
    };
    int nvar = (n != 8) && (n != 16);
    int func_idx = 2 * (n > 8) + nvar;
    int dim_idx = (n > 8);
    dim3 blockDim( dims[dim_idx].x, dims[dim_idx].y );
    dim3 gridDim( batchCount / blockDim.y + (batchCount % blockDim.y != 0), 1);//TODO try 2D grid

    potrs_kernels[func_idx]<<< gridDim, blockDim, 0, handle->stream>>>
                              ( m, n, batchCount,
                                A, A_row_off, A_col_off, lda, strideA,
                                B, B_row_off, B_col_off, ldb, strideB);
    check_error_ret( cudaGetLastError(), KBLAS_UnknownError);
  }
  else
  #endif
  //invoke recurssion
  {

    int status;
    T one = make_one<T>(), mone = -one;

    int n1, n2;

    if(REG_SIZE(n))
      n1 = n2 = n/2;
    else{
      n1 = CLOSEST_REG_SIZE(n);
      n2 = n-n1;
    }

    //TRSM_BATCH
    if(STRIDED){
      check_error_ret (status = Xtrsm_batch_offset( handle,
                                                    side, uplo, KBLAS_Trans, KBLAS_NonUnit,
                                                    m, n1,
                                                    one, (const T*)Aoff(0, 0), lda, strideA,
                                                               (T*)Boff(0, 0), ldb, strideB,
                                                    batchCount), status);
    }else{
      check_error_ret (status = Xtrsm_batch_offset( handle,
                                                    side, uplo, KBLAS_Trans, KBLAS_NonUnit,
                                                    m, n1,
                                                    one, (const T**)Aoff(0, 0), lda,
                                                               (T**)Boff(0, 0), ldb,
                                                    batchCount), status);
    }

    //GEMM_BATCH
    if(STRIDED){
      check_error_ret (status = kblas_gemm_batch( handle,
                                                  KBLAS_NoTrans, KBLAS_Trans,
                                                  m, n2, n1,
                                                  mone, (const T*)offB( 0,  0), ldb, strideB,
                                                        (const T*)offA(n1,  0), lda, strideA,
                                                  one,        (T*)offB( 0, n1), ldb, strideB,
                                                  batchCount), status);
    }else{
      check_error_ret (status = kblas_gemm_batch( handle,
                                                  KBLAS_NoTrans, KBLAS_Trans,
                                                  m, n2, n1,
                                                  mone, (const T**)Boff( 0,  0), ldb,
                                                        (const T**)Aoff(n1,  0), lda,
                                                  one,        (T**)Boff( 0, n1), ldb,
                                                  batchCount), status);
    }
    #if 1
    //TODO replace by one POTRS call
    //TRSM_BATCH
    if(STRIDED){
      check_error_ret (status = Xtrsm_batch_offset( handle,
                                                    side, uplo, KBLAS_Trans, KBLAS_NonUnit,
                                                    m, n2,
                                                    one, (const T*)Aoff(n1, n1), lda, strideA,
                                                               (T*)Boff( 0, n1), ldb, strideB,
                                                    batchCount), status);
    }else{
      check_error_ret (status = Xtrsm_batch_offset( handle,
                                                    side, uplo, KBLAS_Trans, KBLAS_NonUnit,
                                                    m, n2,
                                                    one, (const T**)Aoff(n1, n1), lda,
                                                               (T**)Boff( 0, n1), ldb,
                                                    batchCount), status);
    }
    //TRSM_BATCH
    if(STRIDED){
      check_error_ret (status = Xtrsm_batch_offset( handle,
                                                    side, uplo, KBLAS_NoTrans, KBLAS_NonUnit,
                                                    m, n2,
                                                    one, (const T*)Aoff(n1, n1), lda, strideA,
                                                               (T*)Boff( 0, n1), ldb, strideB,
                                                    batchCount), status);
    }else{
      check_error_ret (status = Xtrsm_batch_offset( handle,
                                                    side, uplo, KBLAS_NoTrans, KBLAS_NonUnit,
                                                    m, n2,
                                                    one, (const T**)Aoff(n1, n1), lda,
                                                               (T**)Boff( 0, n1), ldb,
                                                    batchCount), status);
    }
    #else
    //POTRS_BATCH
    check_error_ret((status = Xpotrs_batch_core<T, T_PTR, STRIDED>(
                                                handle,
                                                side, uplo,
                                                m, n2,
                                                Aoff(n1, n1), lda, strideA,
                                                Boff( 0, n1), ldb, strideB,
                                                batchCount)), status);
    #endif

    //GEMM_BATCH
    if(STRIDED){
      check_error_ret (status = kblas_gemm_batch( handle,
                                                  KBLAS_NoTrans, KBLAS_NoTrans,
                                                  m, n1, n2,
                                                  mone, (const T*)offB( 0, n1), ldb, strideB,
                                                        (const T*)offA(n1,  0), lda, strideA,
                                                  one,        (T*)offB( 0,  0), ldb, strideB,
                                                  batchCount), status);
    }else{
      check_error_ret (status = kblas_gemm_batch( handle,
                                                  KBLAS_NoTrans, KBLAS_NoTrans,
                                                  m, n1, n2,
                                                  mone, (const T**)Boff( 0, n1), ldb,
                                                        (const T**)Aoff(n1,  0), lda,
                                                  one,        (T**)Boff( 0,  0), ldb,
                                                  batchCount), status);
    }

    //TRSM_BATCH
    if(STRIDED){
      check_error_ret (status = Xtrsm_batch_offset( handle,
                                                    side, uplo, KBLAS_NoTrans, KBLAS_NonUnit,
                                                    m, n1,
                                                    one, (const T*)Aoff(0, 0), lda, strideA,
                                                               (T*)Boff(0, 0), ldb, strideB,
                                                    batchCount), status);
    }else{
      check_error_ret (status = Xtrsm_batch_offset( handle,
                                                    side, uplo, KBLAS_NoTrans, KBLAS_NonUnit,
                                                    m, n1,
                                                    one, (const T**)Aoff(0, 0), lda,
                                                               (T**)Boff(0, 0), ldb,
                                                    batchCount), status);
    }
  }
  return KBLAS_Success;
}

#endif //__XPOTRS_BATCH_DRIVERS_H__
