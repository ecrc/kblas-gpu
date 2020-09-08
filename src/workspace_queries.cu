/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/workspace_queries.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

/*
    -- MAGMA (version 2.3.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date 2018-11-14

       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah

*/
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>

#include "kblas_defs.h"
#include "kblas.h"
#include "kblas_struct.h"

#include "workspace_queries.ch"

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
/*#############################################################################*/
// adopted from MAGMA with minor modification
#define AUX_MAX_SEGMENT    (256)    // must be even
#define AUX_MAX_TX         (AUX_MAX_SEGMENT)
/*****************************************************************************
    Does max reduction of n-element array x, leaving total in x[0].
    Contents of x are destroyed in the process.
    With k threads, can reduce array up to 2*k in size.
    Assumes number of threads <= 1024 (which is max number of threads up to CUDA capability 3.0)
    Having n as template parameter allows compiler to evaluate some conditions at compile time.
    Calls __syncthreads before & after reduction.
    @ingroup magma_kernel
*******************************************************************************/
template< int n, typename T >
__device__ void
magma_max_reduce_copy( /*int n,*/ int i, T* x )
{
  __syncthreads();
  if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] = kmax( x[i], x[i+1024] ); }  __syncthreads(); }
  if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] = kmax( x[i], x[i+ 512] ); }  __syncthreads(); }
  if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] = kmax( x[i], x[i+ 256] ); }  __syncthreads(); }
  if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] = kmax( x[i], x[i+ 128] ); }  __syncthreads(); }
  if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] = kmax( x[i], x[i+  64] ); }  __syncthreads(); }
  if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] = kmax( x[i], x[i+  32] ); }  __syncthreads(); }
  // probably don't need __syncthreads for < 16 threads
  // because of implicit warp level synchronization.
  if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] = kmax( x[i], x[i+  16] ); }  __syncthreads(); }
  if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] = kmax( x[i], x[i+   8] ); }  __syncthreads(); }
  if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] = kmax( x[i], x[i+   4] ); }  __syncthreads(); }
  if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] = kmax( x[i], x[i+   2] ); }  __syncthreads(); }
  if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] = kmax( x[i], x[i+   1] ); }  __syncthreads(); }
}
// end max_reduce

/******************************************************************************/
__global__ void
magma_imax_size_kernel_2_copy(int *m, int *n,
                              int &max_m, int &max_n, int l)
{
    int *vec;
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    int i, value, lmax = 0;
    const int L = (l/AUX_MAX_SEGMENT) * AUX_MAX_SEGMENT;

    __shared__ int swork[AUX_MAX_SEGMENT];

    if     (bx == 0) vec = m;
    else if(bx == 1) vec = n;

    for(i = 0; i < L; i+= AUX_MAX_SEGMENT){
        value = (int)vec[i + tx];
        lmax = ( value > lmax ) ? value : lmax;
    }

    // last incomplete segment
    if(tx < l - L){
        value = (int)vec[L + tx];
        lmax = ( value > lmax ) ? value : lmax;
    }

    swork[tx] = lmax;
    __syncthreads();
    magma_max_reduce_copy<AUX_MAX_SEGMENT, int>(tx, swork);
    // no need to sync
    if(tx == 0){
      if     (bx == 0) max_m = (int)(swork[0]);
      else if(bx == 1) max_n = (int)(swork[0]);
    }
}

__global__ void
magma_imax_size_kernel_3_copy(int *m, int *n, int *k,
                              int &max_m, int &max_n, int &max_k, int l)
{
  int *vec;
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  int i, value, lmax = 0;
  const int L = (l/AUX_MAX_SEGMENT) * AUX_MAX_SEGMENT;

  __shared__ int swork[AUX_MAX_SEGMENT];

  if     (bx == 0) vec = m;
  else if(bx == 1) vec = n;
  else if(bx == 2) vec = k;

  for(i = 0; i < L; i+= AUX_MAX_SEGMENT){
      value = (int)vec[i + tx];
      lmax = ( value > lmax ) ? value : lmax;
  }

  // last incomplete segment
  if(tx < l - L){
      value = (int)vec[L + tx];
      lmax = ( value > lmax ) ? value : lmax;
  }

  swork[tx] = lmax;
  __syncthreads();
  magma_max_reduce_copy<AUX_MAX_SEGMENT, int>(tx, swork);
  // no need to sync
  if(tx == 0){
    if     (bx == 0) max_m = (int)(swork[0]);
    else if(bx == 1) max_n = (int)(swork[0]);
    else if(bx == 2) max_k = (int)(swork[0]);
  }
}

void kblas_imax_size_2( kblasHandle_t handle,
                        int *m, int *n,
                        int &max_m, int &max_n, int l)
{
  dim3 grid(2, 1, 1);
  dim3 threads(AUX_MAX_TX, 1, 1);
  magma_imax_size_kernel_2_copy<<< grid, threads, 0, handle->stream >>>(m, n, max_m, max_n, l);
}

void kblas_imax_size_3( kblasHandle_t handle,
                        int *m, int *n, int *k,
                        int &max_m, int &max_n, int &max_k, int l)
{
  dim3 grid(3, 1, 1);
  dim3 threads(AUX_MAX_TX, 1, 1);
  magma_imax_size_kernel_3_copy<<< grid, threads, 0, handle->stream >>>(m, n, k, max_m, max_n, max_k, l);
}
/*#############################################################################*/

//==============================================================================================
void gemm_batch_offset_wsquery_core(int batchCount, bool offseted, kblasWorkspaceState_t ws)
{
  if ( offseted )
  {
    ws->d_ptrs_bytes = kmax((batchCount > 1) * size_t(batchCount) * 3 * sizeof(void*), ws->d_ptrs_bytes);
  }
}

void kblas_gemm_batch_offset_wsquery( kblasHandle_t handle,
                                      int batchCount, bool offseted)
{
  gemm_batch_offset_wsquery_core( batchCount, offseted,
                                  &(handle->work_space.requested_ws_state));
}

void gemm_batch_strided_wsquery_core(int batchCount, kblasWorkspaceState_t ws)
{
#if ( __CUDACC_VER_MAJOR__ < 8 ) || (defined USE_MAGMA)
  ws->d_ptrs_bytes = kmax((batchCount > 1) * size_t(batchCount) * 3 * sizeof(void*), ws->d_ptrs_bytes);
#endif
}

void gemm_batch_nonuniform_wsquery_core(kblasWorkspaceState_t ws)
{
#if defined USE_MAGMA
  ws->d_data_bytes = kmax( 3 * sizeof(int), ws->d_data_bytes);
#endif
}
void kblas_gemm_batch_strided_wsquery(kblasHandle_t handle, int batchCount)
{
  gemm_batch_strided_wsquery_core(batchCount, &(handle->work_space.requested_ws_state));
}

void kblas_gemm_batch_nonuniform_wsquery(kblasHandle_t handle)
{
  gemm_batch_nonuniform_wsquery_core(&(handle->work_space.requested_ws_state));
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
    ws->d_ptrs_bytes = kmax(size_t(1 << (depth-1) ) * batchCount * 3 * sizeof(void*), ws->d_ptrs_bytes);
  }
}

void kblas_syrk_batch_wsquery(kblasHandle_t handle, const int m, int batchCount){
  syrk_batch_wsquery_core(m, batchCount, &(handle->work_space.requested_ws_state));
}

void syrk_batch_nonuniform_wsquery_core(kblasWorkspaceState_t ws)
{
#if defined USE_MAGMA
  ws->d_data_bytes = kmax( 2 * sizeof(int), ws->d_data_bytes);
#endif
}

void kblas_syrk_batch_nonuniform_wsquery(kblasHandle_t handle)
{
  syrk_batch_nonuniform_wsquery_core(&(handle->work_space.requested_ws_state));
}

//==============================================================================================
void kblas_trsm_batch_wsquery(kblasHandle_t handle, char side, int m, int n, int batchCount){
  trsm_batch_wsquery_core<false>( batchCount,
                                  side, m, n,
                                  &(handle->work_space.requested_ws_state));
}
void kblas_trsm_batch_strided_wsquery(kblasHandle_t handle, char side, int m, int n, int batchCount){
  trsm_batch_wsquery_core<true>(batchCount,
                                side, m, n,
                                &(handle->work_space.requested_ws_state));
}

void trsm_batch_nonuniform_wsquery_core(kblasWorkspaceState_t ws)
{
#if defined USE_MAGMA
  ws->d_data_bytes = kmax( 2 * sizeof(int), ws->d_data_bytes);
#endif
}

void kblas_trsm_batch_nonuniform_wsquery(kblasHandle_t handle)
{
  trsm_batch_nonuniform_wsquery_core(&(handle->work_space.requested_ws_state));
}

//==============================================================================================
void kblas_trmm_batch_wsquery(kblasHandle_t handle, char side, int m, int n, int batchCount){
  trmm_batch_wsquery_core<false>( batchCount,
                                  side, m, n,
                                  &(handle->work_space.requested_ws_state));
}
void kblas_trmm_batch_strided_wsquery(kblasHandle_t handle, char side, int m, int n, int batchCount){
  trmm_batch_wsquery_core<true>(batchCount,
                                side, m, n,
                                &(handle->work_space.requested_ws_state));
}

//==============================================================================================
void kblas_symm_batch_wsquery(kblasHandle_t handle, int batchCount){
  symm_batch_wsquery_core<false>( batchCount, &(handle->work_space.requested_ws_state));
}
void kblas_symm_batch_strided_wsquery(kblasHandle_t handle, int batchCount){
  symm_batch_wsquery_core<true>( batchCount, &(handle->work_space.requested_ws_state));
}

void symm_batch_nonuniform_wsquery_core(kblasWorkspaceState_t ws)
{
#if defined USE_MAGMA
  ws->d_data_bytes = kmax( 2 * sizeof(int), ws->d_data_bytes);
#endif
}

void kblas_symm_batch_nonuniform_wsquery(kblasHandle_t handle)
{
  symm_batch_nonuniform_wsquery_core(&(handle->work_space.requested_ws_state));
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

//==============================================================================================
void kblas_trtri_batch_wsquery(kblasHandle_t handle, const int n, int batchCount){
  trtri_batch_wsquery_core<false>(n, batchCount, &(handle->work_space.requested_ws_state));
}

void kblas_trtri_batch_strided_wsquery(kblasHandle_t handle, const int n, int batchCount){
  trtri_batch_wsquery_core<true>(n, batchCount, &(handle->work_space.requested_ws_state));
}

//==============================================================================================
void kblas_potrs_batch_wsquery(kblasHandle_t handle, const int m, const int n, int batchCount){
  potrs_batch_wsquery_core<false>(m, n, batchCount, &(handle->work_space.requested_ws_state));
}

void kblas_potrs_batch_strided_wsquery(kblasHandle_t handle, const int m, const int n, int batchCount){
  potrs_batch_wsquery_core<true>(m, n, batchCount, &(handle->work_space.requested_ws_state));
}

//==============================================================================================
void kblas_potri_batch_wsquery(kblasHandle_t handle, const int n, int batchCount){
  potri_batch_wsquery_core<false>(n, batchCount, &(handle->work_space.requested_ws_state));
}

void kblas_potri_batch_strided_wsquery(kblasHandle_t handle, const int n, int batchCount){
  potri_batch_wsquery_core<true>(n, batchCount, &(handle->work_space.requested_ws_state));
}

//==============================================================================================
void kblas_poti_batch_wsquery(kblasHandle_t handle, const int n, int batchCount){
  poti_batch_wsquery_core<false>(n, batchCount, &(handle->work_space.requested_ws_state));
}

void kblas_poti_batch_strided_wsquery(kblasHandle_t handle, const int n, int batchCount){
  poti_batch_wsquery_core<true>(n, batchCount, &(handle->work_space.requested_ws_state));
}

//==============================================================================================
void kblas_posv_batch_wsquery(kblasHandle_t handle, char side, const int m, const int n, int batchCount){
  posv_batch_wsquery_core<false>(m, n, side, batchCount, &(handle->work_space.requested_ws_state));
}

void kblas_posv_batch_strided_wsquery(kblasHandle_t handle, char side, const int m, const int n, int batchCount){
  posv_batch_wsquery_core<true>(m, n, side, batchCount, &(handle->work_space.requested_ws_state));
}
