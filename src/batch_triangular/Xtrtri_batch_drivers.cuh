/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xtrtri_batch_drivers.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef __XTRTRI_BATCH_DRIVERS_H__
#define __XTRTRI_BATCH_DRIVERS_H__


#include "Xtrtri_batch_kernels.cuh"

//==============================================================================================

#define offA(i,j) A + A_row_off + (i) + (A_col_off + (j)) * lda
#define Aoff(_i,_j) A, A_row_off + (_i), A_col_off + (_j)

//==============================================================================================
template<class T, class T_PTR>
int Xtrtri_trsm_rec(kblasHandle_t handle,
                    char uplo, char diag,
                    const int m,
                    T_PTR A, int A_row_off, int A_col_off, int lda, long strideA,
                    int batchCount)
{
  //TODO these trsm calls can run in parallel per level, through streams or merged batch call
  if(m <= 16)
    return KBLAS_Success;

  T one = make_one<T>(), mone = -one;
  int m1, m2;

  if(REG_SIZE(m))
    m1 = m2 = m/2;
  else{
    m1 = CLOSEST_REG_SIZE(m);
    m2 = m-m1;
  }

  if(uplo == KBLAS_Lower)
  {
    //TRSM_BATCH
    check_ret_error( Xtrsm_batch( handle,
                                  KBLAS_Right, uplo, KBLAS_NoTrans, diag,
                                  m2, m1,
                                  mone, (T_PTR)Aoff( 0,  0), lda, strideA,
                                        (T_PTR)Aoff(m1,  0), lda, strideA,
                                  batchCount) );

    //TRSM_BATCH
    check_ret_error( Xtrsm_batch( handle,
                                  KBLAS_Left, uplo, KBLAS_NoTrans, diag,
                                  m2, m1,
                                  one, (T_PTR)Aoff(m1, m1), lda, strideA,
                                       (T_PTR)Aoff(m1,  0), lda, strideA,
                                  batchCount) );

    check_ret_error( (Xtrtri_trsm_rec<T, T_PTR>(
                                      handle,
                                      uplo, diag,
                                      m1,
                                      Aoff(0, 0), lda, strideA,
                                      batchCount)) );

    check_ret_error( (Xtrtri_trsm_rec<T, T_PTR>(
                                      handle,
                                      uplo, diag,
                                      m2,
                                      Aoff(m1, m1), lda, strideA,
                                      batchCount)) );
  }else{
    return KBLAS_NotImplemented;
  }
  return KBLAS_Success;
}

template<class T, class T_PTR, bool STRIDED>
int Xtrtri_batch_core(kblasHandle_t handle,
                      char uplo, char diag,
                      const int n,
                      T_PTR A, int A_row_off, int A_col_off, int lda, long strideA,
                      int batchCount,
                      int *info_array)
{
  if( uplo == KBLAS_Upper || diag == KBLAS_Unit ){
    printf("(Upper | DIAG) TRTRI_BATCH is not implemented yet\n");
    return KBLAS_NotImplemented;
  }

  if(n > 16){
    check_ret_error( (Xtrtri_trsm_rec<T, T_PTR>(
                                      handle,
                                      uplo, diag,
                                      n,
                                      (T_PTR)A, A_row_off, A_col_off, lda, strideA,
                                      batchCount)) );
  }

  if(1)
  {
    int2 dims[] = {
      { 8,16},//4 warps
      {16, 8},//4 warps
      {16, 2},//1 warps
      { 8, 4},//1 warps
      { 8, 8},//2 warps
      { 8,12},//3 warps
      {16, 4}//2 warps
    };
    typedef void (*trtri_kernels_type)( const int n, int batchCount,
                                        T_PTR A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                        int* info);

    trtri_kernels_type trtri_kernels[] = {
      kernel_trtri_U_registers_Nfix<T, T_PTR, STRIDED, 8>,
      kernel_trtri_U_registers_Nvar<T, T_PTR, STRIDED, 8>,
      kernel_trtri_U_registers_Nfix<T, T_PTR, STRIDED, 16>,
      kernel_trtri_U_registers_Nvar<T, T_PTR, STRIDED, 16>
    };

    int nvar = (n != 8) && (n % 16 != 0);
    int func_idx = 2 * (n > 8) + (nvar == 1);
    int dim_idx = (n > 8);

    dim3 blockDim( dims[dim_idx].x, dims[dim_idx].y );
    dim3 gridDim( batchCount / blockDim.y + (batchCount % blockDim.y != 0), (n / 16) + (n % 16 != 0));

    trtri_kernels[func_idx]<<< gridDim, blockDim, 0, handle->stream >>>
                              ( n, batchCount,
                                A, A_row_off, A_col_off, lda, strideA,
                                info_array);
    check_error_ret( hipGetLastError(), KBLAS_UnknownError);
  }
  return KBLAS_Success;
}

#endif //__XTRTRI_BATCH_DRIVERS_H__
