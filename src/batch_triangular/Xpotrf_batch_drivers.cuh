/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xpotrf_batch_drivers.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 2.0.0
 * @author Ali Charara
 * @date 2017-11-13
 **/

#ifndef __XPOTRF_BATCH_DRIVERS_H__
#define __XPOTRF_BATCH_DRIVERS_H__


// #include "Xgemm_batch_core.cuh"
#include "Xpotrf_batch_kernels.cuh"

//==============================================================================================

#define offA(i,j) A + A_row_off + (i) + (A_col_off + (j)) * lda
#define Aoff(_i,_j) A, A_row_off + (_i), A_col_off + (_j)

template<class T, class T_PTR, bool STRIDED>
int Xpotrf_batch_core(kblasHandle_t handle,
                      char uplo,
                      const int n,
                      T_PTR A, int A_row_off, int A_col_off, int lda, long strideA,
                      int batchCount,
                      int *info_array)
{
  if( uplo == KBLAS_Upper ){
    printf("Upper POTRF_BATCH is not implemented yet\n");
    return KBLAS_NotImplemented;
  }

  if( n <= 16 )
  {
    int2 dims[] = {
      { 8,16},//4 warps
      { 8, 8},//2 warps
      { 8, 4},//1 warps
      { 8, 4},//1 warps
      {24, 4},
      {16, 2},//1 warps
      {32, 1},//1 warp
      {16, 4},//2 warps
      { 8, 4},//1 warp
      {16, 1},//1/2 warp
      {32, 2},//2 warps
      { 8,20},//5 warps
      { 8,80},//20 warps
      { 8,64},//16 warps
      { 8,12} //3 warps
    };

    typedef void (*potrf_kernels_type)( const int n, int batchCount,
                                        T_PTR A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                        int* info);


    potrf_kernels_type potrf_kernels[] = {
      kernel_potrf_U_registers_fixN<T, T_PTR, STRIDED, 8>,
      kernel_potrf_U_registers_varN<T, T_PTR, STRIDED, 8>,
      kernel_potrf_U_registers_fixN_blocked_2<T, T_PTR, STRIDED, 16, 8>,
      kernel_potrf_U_registers_varN_blocked_2<T, T_PTR, STRIDED, 16, 8>
    };

    int IS_SINGLE = (typeid(T) == typeid(float));

    int nvar = (n != 8) && (n != 16);
    int func_idx = 2 * (n > 8) + (nvar == 1);//
    int dim_idx = (n > 8) + 2 * (IS_SINGLE == 1);

    dim3 blockDim( dims[dim_idx].x, dims[dim_idx].y );
    dim3 gridDim( batchCount / blockDim.y + (batchCount % blockDim.y != 0), 1);

    potrf_kernels[func_idx]<<< gridDim, blockDim, 0, handle->stream>>>
                              ( n, batchCount,
                                A, A_row_off, A_col_off, lda, strideA,
                                info_array);
    check_error_ret( cudaGetLastError(), KBLAS_UnknownError);
  }
  else
  //try recurssion
  {

    int n1,n2;
    if(REG_SIZE(n)){
      n1 = n2 = n/2;
    }else{
      n1 = CLOSEST_REG_SIZE(n);
      //TODO if(n-n1 < 16) n1 = CLOSEST_REG_SIZE(n1-1);
      n2 = n-n1;
    }

    int status;
    T one = make_one<T>();
    T mone = make_zero<T>() - one;

    //POTRF_BATCH
    check_error_ret((status = Xpotrf_batch_core<T, T_PTR, STRIDED>(
                                                handle,
                                                uplo, n1,
                                                Aoff(0, 0), lda, strideA,
                                                batchCount,
                                                info_array)), status);

    //TRSM_BATCH
    if(STRIDED){
      check_error_ret (status = Xtrsm_batch_offset( handle,
                                                    KBLAS_Right, uplo, KBLAS_Trans, KBLAS_NonUnit,
                                                    n2, n1,
                                                    one, (const T*)Aoff( 0,  0), lda, strideA,
                                                               (T*)Aoff(n1,  0), lda, strideA,
                                                    batchCount), status);
    }else{
      check_error_ret (status = Xtrsm_batch_offset( handle,
                                                    KBLAS_Right, uplo, KBLAS_Trans, KBLAS_NonUnit,
                                                    n2, n1,
                                                    one, (const T**)Aoff( 0,  0), lda,
                                                               (T**)Aoff(n1,  0), lda,
                                                    batchCount), status);
    }

    //SYRK_BATCH
    if(STRIDED){
      check_error_ret (status = kblas_syrk_batch( handle,
                                                  uplo, KBLAS_NoTrans,
                                                  n2, n1,
                                                  mone, (const T*)offA(n1,  0), lda, strideA,
                                                  one,        (T*)offA(n1, n1), lda, strideA,
                                                  batchCount), status);
    }else{
      check_error_ret (status = Xsyrk_batch_offset( handle,
                                                    uplo, KBLAS_NoTrans,
                                                    n2, n1,
                                                    mone, (const T**)Aoff(n1,  0), lda,
                                                    one,        (T**)Aoff(n1, n1), lda,
                                                    batchCount), status);
    }

    //POTRF_BATCH
    check_error_ret((status = Xpotrf_batch_core<T, T_PTR, STRIDED>(
                                                handle,
                                                uplo, n2,
                                                Aoff(n1, n1), lda, strideA,
                                                batchCount,
                                                info_array)), status);

  }
  return KBLAS_Success;
}

#endif //__XPOTRF_BATCH_DRIVERS_H__
