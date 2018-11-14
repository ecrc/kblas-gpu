/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xposv_batch_drivers.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef __XPOSV_BATCH_DRIVERS_H__
#define __XPOSV_BATCH_DRIVERS_H__


#include "Xposv_batch_kernels.cuh"

//==============================================================================================

#define offA(i,j) A + A_row_off + (i) + (A_col_off + (j)) * lda
#define offB(i,j) B + B_row_off + (i) + (B_col_off + (j)) * ldb
#define Aoff(_i,_j) A, A_row_off + (_i), A_col_off + (_j)
#define Boff(_i,_j) B, B_row_off + (_i), B_col_off + (_j)

template<class T, class T_PTR, bool STRIDED>
int Xposv_batch_core( kblasHandle_t handle,
                      char side, char uplo,
                      const int m, const int n,
                      T_PTR A, int A_row_off, int A_col_off, int lda, long strideA,
                      T_PTR B, int B_row_off, int B_col_off, int ldb, long strideB,
                      int batchCount,
                      int *info_array)
{
  if( side == KBLAS_Left || uplo == KBLAS_Upper){
    printf("(Left | Upper) POSV_BATCH is not implemented yet\n");
    return KBLAS_NotImplemented;
  }
#if 0
  //TODO fix this
  if( n <= 8 ){
    int2 dims[6] = {
      { 8, 8},//2 warps
      {16, 2},//1 warps
      { 8, 4},//1 warps
      { 8,16} //4 warps
    };

    typedef void (*posv_kernels_type)(const int m, const int n, int batchCount,
                                      T_PTR A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                      T_PTR B_array, int B_row_off, int B_col_off, int ldb, long strideB);

    posv_kernels_type posv_kernels[] = {
      kernel_posv_U_RL_registers_Nfix_Mvar<T, T_PTR, STRIDED,  8>,
      kernel_posv_U_RL_registers_NMvar    <T, T_PTR, STRIDED,  8>,
      kernel_posv_U_RL_registers_Nfix_Mvar<T, T_PTR, STRIDED, 16>,
      kernel_posv_U_RL_registers_NMvar    <T, T_PTR, STRIDED, 16>
    };
    bool nvar = (n != 8) && (n != 16);
    int func_idx = 2 * (n > 8) + nvar;
    int dim_idx = (n > 8);

    dim3 blockDim( dims[dim_idx].x, dims[dim_idx].y );
    dim3 gridDim( batchCount / blockDim.y + (batchCount % blockDim.y != 0), 1);//TODO try 2D grid

    posv_kernels[func_idx]<<< gridDim, blockDim, 0, handle->stream>>>
                              ( m, n, batchCount,
                                A, A_row_off, A_col_off, lda, strideA,
                                B, B_row_off, B_col_off, ldb, strideB);
    check_error_ret( cudaGetLastError(), KBLAS_UnknownError);
  }
  else
  #endif
  {
    int status;

    //POTRF_BATCH
    if(STRIDED){
      check_error_ret (status = Xpotrf_batch_offset(handle,
                                                    uplo,
                                                    n,
                                                    (T*)Aoff( 0,  0), lda, strideA,
                                                    batchCount,
                                                    info_array), status);
    }else{
      check_error_ret (status = Xpotrf_batch_offset(handle,
                                                    uplo,
                                                    n,
                                                    (T**)Aoff( 0,  0), lda,
                                                    batchCount,
                                                    info_array), status);
    }

    if(STRIDED){
      check_error_ret (status = Xpotrs_batch_offset(handle,
                                                    side, uplo,
                                                    m, n,
                                                    (const T*)Aoff(0, 0), lda, strideA,
                                                          (T*)Boff(0, 0), ldb, strideB,
                                                    batchCount), status);
    }else{
      check_error_ret (status = Xpotrs_batch_offset(handle,
                                                    side, uplo,
                                                    m, n,
                                                    (const T**)Aoff(0, 0), lda,
                                                          (T**)Boff(0, 0), ldb,
                                                    batchCount), status);
    }
  }
  return KBLAS_Success;
}

#endif //__XPOSV_BATCH_DRIVERS_H__
