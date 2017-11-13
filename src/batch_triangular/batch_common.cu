/**
  - -* (C) Copyright 2013 King Abdullah University of Science and Technology
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
#include <stdlib.h>
#include <stdio.h>

#include "defs.h"
#include "kblas_struct.h"

#include "batch_common.ch"

//==============================================================================================
template<typename T>
__global__ void kernel_set_value_diff_1(T* output_array, const T* input1, const T* input2, long count){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < count)
    output_array[idx] = input1[idx] - input2[idx];
}
int iset_value_diff_1(int* output_array, const int* input_array1, const int* input_array2,
                      long batchCount,  cudaStream_t cuda_stream){
  dim3 block(128,1);
  dim3 grid(batchCount / block.x + ((batchCount % block.x) > 0),1);
  kernel_set_value_diff_1<int><<< grid, block, 0, cuda_stream>>>(
    output_array, input_array1, input_array2, batchCount);
  check_error_ret( cudaGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

//==============================================================================================
void gemm_batch_offset_wsquery_core(int batchCount,
                                    int A_row_off, int A_col_off,
                                    int B_row_off, int B_col_off,
                                    int C_row_off, int C_col_off,
                                    kblasWorkspaceState_t ws)
{
  if ( (A_row_off > 0) || (A_col_off > 0)
    || (B_row_off > 0) || (B_col_off > 0)
    || (C_row_off > 0) || (C_col_off > 0) )
  {
    ws->d_ptrs_bytes = kmax((batchCount > 1) * batchCount * 3 * sizeof(void*), ws->d_ptrs_bytes);
  }
}

void kblas_gemm_batch_offset_wsquery(kblasHandle_t handle,
                              int batchCount,
                              int A_row_off, int A_col_off,
                              int B_row_off, int B_col_off,
                              int C_row_off, int C_col_off){
  gemm_batch_offset_wsquery_core( batchCount,
                                  A_row_off, A_col_off,
                                  B_row_off, B_col_off,
                                  C_row_off, C_col_off,
                                  &(handle->work_space.requested_ws_state));
}

void gemm_batch_strided_wsquery_core(int batchCount, kblasWorkspaceState_t ws)
{
#if ( __CUDACC_VER_MAJOR__ < 8 )
  ws->d_ptrs_bytes = kmax((batchCount > 1) * batchCount * 3 * sizeof(void*), ws->d_ptrs_bytes);
#endif
}

void kblas_gemm_batch_strided_wsquery(kblasHandle_t handle, int batchCount)
{
  gemm_batch_strided_wsquery_core(batchCount, &(handle->work_space.requested_ws_state));
}

//==============================================================================================
void syrk_batch_wsquery_core(const int m, int batchCount, kblasWorkspaceState_t ws)
{
  if(m > 16)
  {
    int depth = 0, s = 16;
    while(s < m){
      s = s << 1;
      depth++;
    }
    ws->d_ptrs_bytes = kmax((1 << (depth-1) ) * batchCount * 3 * sizeof(void*), ws->d_ptrs_bytes);
  }
}

void kblas_syrk_batch_wsquery(kblasHandle_t handle, const int m, int batchCount){
  syrk_batch_wsquery_core(m, batchCount, &(handle->work_space.requested_ws_state));
}
//==============================================================================================
void kblas_trsm_batch_wsquery(kblasHandle_t handle, int batchCount, char side, int m, int n){
  trsm_batch_wsquery_core<false>( batchCount,
                                  side, m, n,
                                  &(handle->work_space.requested_ws_state));
}
void kblas_trsm_batch_strided_wsquery(kblasHandle_t handle, int batchCount, char side, int m, int n){
  trsm_batch_wsquery_core<true>(batchCount,
                                side, m, n,
                                &(handle->work_space.requested_ws_state));
}

//==============================================================================================
void kblas_trmm_batch_wsquery(kblasHandle_t handle, int batchCount, char side, int m, int n){
  trmm_batch_wsquery_core<false>( batchCount,
                                  side, m, n,
                                  &(handle->work_space.requested_ws_state));
}
void kblas_trmm_batch_strided_wsquery(kblasHandle_t handle, int batchCount, char side, int m, int n){
  trmm_batch_wsquery_core<true>(batchCount,
                                side, m, n,
                                &(handle->work_space.requested_ws_state));
}

//==============================================================================================
void kblas_potrf_batch_wsquery(kblasHandle_t handle, const int n, int batchCount){
  potrf_batch_wsquery_core<false>(n, batchCount, &(handle->work_space.requested_ws_state));
}

void kblas_potrf_batch_strided_wsquery(kblasHandle_t handle, const int n, int batchCount){
  potrf_batch_wsquery_core<true>(n, batchCount, &(handle->work_space.requested_ws_state));
}

//==============================================================================================
void kblas_lauum_batch_wsquery(kblasHandle_t handle, const int n, int batchCount){
  lauum_batch_wsquery_core<false>(n, batchCount, &(handle->work_space.requested_ws_state));
}

void kblas_lauum_batch_strided_wsquery(kblasHandle_t handle, const int n, int batchCount){
  lauum_batch_wsquery_core<true>(n, batchCount, &(handle->work_space.requested_ws_state));
}

