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
    check_error_ret( cudaGetLastError(), KBLAS_UnknownError);
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

    int status;
    T one = make_one<T>();
    T mone = make_zero<T>() - one;

    //LAUUM_BATCH
    check_error_ret((status = Xlauum_batch_core<T, T_PTR, STRIDED>(
                                                handle,
                                                uplo, n1,
                                                Aoff(0, 0), lda, strideA,
                                                batchCount,
                                                info_array)), status);

    //SYRK_BATCH
    if(STRIDED){
      check_error_ret (status = kblas_syrk_batch( handle,
                                                  uplo, KBLAS_Trans,
                                                  n1, n2,
                                                  one, (const T*)offA(n1, 0), lda, strideA,
                                                  one,       (T*)offA( 0, 0), lda, strideA,
                                                  batchCount), status);
    }else{
      check_error_ret (status = Xsyrk_batch_offset( handle,
                                                    uplo, KBLAS_Trans,
                                                    n1, n2,
                                                    one, (const T**)Aoff(n1, 0), lda,
                                                    one,       (T**)Aoff( 0, 0), lda,
                                                    batchCount), status);
    }

    //TRMM_BATCH
    if(STRIDED){
      check_error_ret (status = Xtrmm_batch_offset( handle,
                                                    KBLAS_Left, uplo, KBLAS_Trans, KBLAS_NonUnit,
                                                    n2, n1,
                                                    one, (const T*)Aoff(n1, n1), lda, strideA,
                                                               (T*)Aoff(n1,  0), lda, strideA,
                                                    batchCount), status);
    }else{
      check_error_ret (status = Xtrmm_batch_offset( handle,
                                                    KBLAS_Left, uplo, KBLAS_Trans, KBLAS_NonUnit,
                                                    n2, n1,
                                                    one, (const T**)Aoff(n1, n1), lda,
                                                               (T**)Aoff(n1,  0), lda,
                                                    batchCount), status);
    }

    //LAUUM_BATCH
    check_error_ret((status = Xlauum_batch_core<T, T_PTR, STRIDED>(
                                                handle,
                                                uplo, n2,
                                                Aoff(n1, n1), lda, strideA,
                                                batchCount,
                                                info_array)), status);

  }
  return KBLAS_Success;
}

#endif //__XLAUUM_BATCH_DRIVERS_H__