/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xpotri_batch_drivers.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef __XPOTI_BATCH_DRIVERS_H__
#define __XPOTI_BATCH_DRIVERS_H__


#include "Xpoti_batch_kernels.cuh"

//==============================================================================================

#define offA(i,j) A + A_row_off + (i) + (A_col_off + (j)) * lda
#define Aoff(_i,_j) A, A_row_off + (_i), A_col_off + (_j)

template<class T, class T_PTR, bool STRIDED>
int Xpoti_batch_core(kblasHandle_t handle,
                      char uplo,
                      const int n,
                      T_PTR A, int A_row_off, int A_col_off, int lda, long strideA,
                      int batchCount,
                      int *info_array)
{
  if( uplo == KBLAS_Upper ){
    printf("Upper POTI_BATCH is not implemented yet\n");
    return KBLAS_NotImplemented;
  }

  if( n <= 8 )
  {
    int2 dims[6] = {
      { 8, 8},//2 warps
      {16, 2},//1 warps
      { 8, 4},//1 warps
      { 8,16} //4 warps
    };

    typedef void (*poti_kernels_type)( const int n, int batchCount,
                                        T_PTR A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                        int* info);

    poti_kernels_type poti_kernels[] = {
      kernel_poti_U_L_reg_shared_Nfix<T, T_PTR, STRIDED,  8>,
      kernel_poti_U_L_reg_shared_Nvar<T, T_PTR, STRIDED,  8>
      // kernel_poti_U_L_reg_shared_Nfix<T, T_PTR, STRIDED, 16>,
      // kernel_poti_U_L_reg_shared_Nvar<T, T_PTR, STRIDED, 16>
    };

    bool nvar = (n != 8) && (n != 16);
    int func_idx = 2 * (n > 8) + nvar;
    int dim_idx = (n > 8);

    dim3 blockDim( dims[dim_idx].x, dims[dim_idx].y );
    dim3 gridDim( batchCount / blockDim.y + (batchCount % blockDim.y != 0), 1);

    poti_kernels[func_idx]<<< gridDim, blockDim, blockDim.x*(blockDim.x+1)*blockDim.y*sizeof(T), handle->stream>>>
                              ( n, batchCount,
                                A, A_row_off, A_col_off, lda, strideA,
                                info_array);
    check_error_ret( cudaGetLastError(), KBLAS_UnknownError);
  }
  else
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
      check_error_ret (status = Xpotri_batch_offset(handle,
                                                    uplo,
                                                    n,
                                                    (T*)Aoff(0, 0), lda, strideA,
                                                    batchCount,
                                                    info_array), status);
    }else{
      check_error_ret (status = Xpotri_batch_offset(handle,
                                                    uplo,
                                                    n,
                                                    (T**)Aoff(0, 0), lda,
                                                    batchCount,
                                                    info_array), status);
    }
  }
  return KBLAS_Success;
}

#endif //__XPOTI_BATCH_DRIVERS_H__
