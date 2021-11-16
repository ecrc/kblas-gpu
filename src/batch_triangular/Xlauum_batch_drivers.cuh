/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xlauum_batch_drivers.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef __XLAUUM_BATCH_DRIVERS_H__
#define __XLAUUM_BATCH_DRIVERS_H__


#include "Xlauum_batch_kernels.cuh"

//==============================================================================================

#define offA(i,j) A + A_row_off + (i) + (A_col_off + (j)) * lda
#define Aoff(_i,_j) A, A_row_off + (_i), A_col_off + (_j)

template<class T, class T_PTR, bool STRIDED>
int Xlauum_batch_core(kblasHandle_t handle,
                      char uplo,
                      const int n,
                      T_PTR A, int A_row_off, int A_col_off, int lda, long strideA,
                      int batchCount,
                      int *info_array)
{
  if( uplo == KBLAS_Upper ){
    printf("Upper LAUUM_BATCH is not implemented yet\n");
    return KBLAS_NotImplemented;
  }

  if( n <= 16 )
  {
    int2 dims[] = {
      { 8, 4},//1 warp
      {16, 4},//2 warps
      { 8,16},//4 warps
      { 8, 8},//2 warps
      {24, 4},
      {16, 2},//1 warps
      {32, 1},//1 warp
      {16, 1},//1/2 warp
      {32, 2} //2 warps
    };

    typedef void (*lauum_kernels_type)( const int n, int batchCount,
                                        T_PTR A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                        int* info);


    lauum_kernels_type lauum_kernels[] = {
      kernel_lauum_U_reg_shared_Nfix<T, T_PTR, STRIDED, 8>,
      kernel_lauum_U_registers_Nvar<T, T_PTR, STRIDED, 8>,
      kernel_lauum_U_registers_Nfix<T, T_PTR, STRIDED, 16>,
      kernel_lauum_U_registers_Nvar<T, T_PTR, STRIDED, 16>
    };

    int nvar = (n != 8) && (n != 16);
    int func_idx = 2 * (n > 8) + (nvar == 1);
    int dim_idx = (n > 8);


    dim3 blockDim( dims[dim_idx].x, dims[dim_idx].y );
    dim3 gridDim( batchCount / blockDim.y + (batchCount % blockDim.y != 0), 1);

    long lauum_kernels_sharedMem[2] = {
      0,
      blockDim.x*(blockDim.x+2)*blockDim.y*sizeof(T)
    };

    lauum_kernels[func_idx]<<< gridDim, blockDim, lauum_kernels_sharedMem[(func_idx==0)], handle->stream>>>
                              ( n, batchCount,
                                A, A_row_off, A_col_off, lda, strideA,
                                info_array);
    check_error_ret( hipGetLastError(), KBLAS_UnknownError);
  }
  else
  //invoke recursion
  {

    int n1,n2;
    if(REG_SIZE(n)){
      n1 = n2 = n/2;
    }else{
      n1 = CLOSEST_REG_SIZE(n);
      //TODO if(n-n1 < 16) n1 = CLOSEST_REG_SIZE(n1-1);
      n2 = n-n1;
    }

    // int status;
    T one = make_one<T>();
    T mone = make_zero<T>() - one;

    //LAUUM_BATCH
    check_ret_error(( Xlauum_batch_core<T, T_PTR, STRIDED>(
                                        handle,
                                        uplo, n1,
                                        Aoff(0, 0), lda, strideA,
                                        batchCount,
                                        info_array)) );
    //SYRK_BATCH
    check_ret_error( Xsyrk_batch( handle,
                                  uplo, KBLAS_Trans,
                                  n1, n2,
                                  one, (T_PTR)Aoff(n1, 0), lda, strideA,
                                  one, (T_PTR)Aoff( 0, 0), lda, strideA,
                                  batchCount) );
    //TRMM_BATCH
    check_ret_error( Xtrmm_batch( handle,
                                  KBLAS_Left, uplo, KBLAS_Trans, KBLAS_NonUnit,
                                  n2, n1,
                                  one, (T_PTR)Aoff(n1, n1), lda, strideA,
                                       (T_PTR)Aoff(n1,  0), lda, strideA,
                                  batchCount) );
    // //TRMM_BATCH
    // if(STRIDED){
    //   check_error_ret (status = Xtrmm_batch_offset( handle,
    //                                                 KBLAS_Left, uplo, KBLAS_Trans, KBLAS_NonUnit,
    //                                                 n2, n1,
    //                                                 one, (const T*)Aoff(n1, n1), lda, strideA,
    //                                                            (T*)Aoff(n1,  0), lda, strideA,
    //                                                 batchCount), status);
    // }else{
    //   check_error_ret (status = Xtrmm_batch_offset( handle,
    //                                                 KBLAS_Left, uplo, KBLAS_Trans, KBLAS_NonUnit,
    //                                                 n2, n1,
    //                                                 one, (const T**)Aoff(n1, n1), lda,
    //                                                            (T**)Aoff(n1,  0), lda,
    //                                                 batchCount), status);
    // }

    //LAUUM_BATCH
    check_ret_error(( Xlauum_batch_core<T, T_PTR, STRIDED>(
                                        handle,
                                        uplo, n2,
                                        Aoff(n1, n1), lda, strideA,
                                        batchCount,
                                        info_array)) );

  }
  return KBLAS_Success;
}

#endif //__XLAUUM_BATCH_DRIVERS_H__
