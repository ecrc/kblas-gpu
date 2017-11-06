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

#ifndef __XGEMM_BATCH_CORE__
#define __XGEMM_BATCH_CORE__

//==============================================================================================

//shuffle intrinsic is not supported before KEPLER
#if (SM >= 30)

//==============================================================================================
#define WARP 32
#define WARP1 33
#define tx threadIdx.x
#define ty threadIdx.y


//==============================================================================================;

//TODO IMPORTANT: stride should be long long int since it is a memory address measure
//==============================================================================================;
template<typename T, bool TRANSB, int TX, int AY, int BY>
__global__ void  //__launch_bounds__(256)
kernel_gemm_NX_registers_MNmulTX_2d(const int m, const int n, const int k,
                                    const T alpha,
                                    const T* __restrict__ A_array, int lda, long strideA,
                                    const T* __restrict__ B_array, int ldb, long strideB,
                                    const T beta, T*      C_array, int ldc, long strideC)
{
  if( (m % TX) || (n % TX) || (k % TX) || (k % AY) ) return;
  int nn = n / TX;
  const T *A = A_array + ((blockIdx.x * blockDim.y + ty) / nn) * strideA + blockIdx.y * TX;
  const T *B = B_array + ((blockIdx.x * blockDim.y + ty) / nn) * strideB + TX * (ty % nn) * (TRANSB? 1: ldb);
        T *C = C_array + ((blockIdx.x * blockDim.y + ty) / nn) * strideC + TX * (ty % nn) * ldc + blockIdx.y * TX;
  T rA0[AY], rB0[BY], rC0[TX], s, r;
  int blockCount = k / AY, ind;

  if(beta == make_zero<T>()){
    #pragma unroll
    for(int i = 0; i < TX; i++){
      rC0[ i ] = beta;
    }
  }else{
    #pragma unroll
    for(int i = 0; i < TX; i++){
      rC0[ i ] = __ldg(&(C[ tx + i * ldc ])) * beta;
    }
  }

  #pragma unroll
  for(int b = 0; b < blockCount; b++){

    ind = tx + AY * b * lda;
    #pragma unroll
    for(int i = 0; i < AY; i++){
      rA0[ i ] = __ldg(&(A[ ind + i * lda ]));
    }
    ind = tx + BY * b * (TRANSB? ldb : 1);
    #pragma unroll
    for(int i = 0; i < BY; i++){
      rB0[ i ] = __ldg(&(B[ ind + i * ldb ]));
    }

    //1. gemm
    {
      #pragma unroll
      for(int j = 0; j < TX; j++){
        s = make_zero<T>();
        #pragma unroll
        for(int i = 0; i < AY; i++){
          if(TRANSB)
            r = shfl(rB0[i], j, TX);
          else
            r = shfl(rB0[j], i, TX);
          s = FMA(rA0[i], r, s);
        }
        rC0[j] = FMA( alpha, s, rC0[j] );
      }
    }
  }
  //copy B[0] data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    C[ tx + i * ldc ] = rC0[ i ];
  }
}

//==============================================================================================
template<typename T, int TX>
int Xgemm_NX_batch( kblasHandle_t handle,
                    char transA, char transB,
                    const int m, const int n, const int k,
                    const T alpha,
                    const T* A, int lda, long strideA,
                    const T* B, int ldb, long strideB,
                    const T beta,
                          T* C, int ldc, long strideC,
                    int batchCount)
{
  cudaStream_t curStream = handle->stream;
  int2 dims[] = {
    {TX, 2},//1 warps
    {TX, 4},//2 warps
    {TX, 8} //4 warps
  };

  int dim_idx = 1, nn = n / TX;
  #ifdef KBLAS_ENABLE_BACKDOORS
  if(handle->back_door[0] >= 0){
    dim_idx = handle->back_door[0];
  }
  #endif
  dim3 blockDim( dims[dim_idx].x, dims[dim_idx].y );
  dim3 gridDim( batchCount * nn / blockDim.y, m / TX);

  if(transB == KBLAS_Trans)
    kernel_gemm_NX_registers_MNmulTX_2d<T, true, TX, 8, 8><<< gridDim, blockDim, 0, curStream>>> (
                            m, n, k,
                            alpha, A, lda, strideA,
                                   B, ldb, strideB,
                            beta,  C, ldc, strideC);
  else
    kernel_gemm_NX_registers_MNmulTX_2d<T, false, TX, TX, TX><<< gridDim, blockDim, 0, curStream>>> (
                            m, n, k,
                            alpha, A, lda, strideA,
                                   B, ldb, strideB,
                            beta,  C, ldc, strideC);
  check_error_ret( cudaGetLastError(), KBLAS_UnknownError)
  return KBLAS_Success;
}

#else
#endif

//==============================================================================================
//extern "C" {

// extern bool use_magma_gemm;
// extern bool use_cublas_gemm;

//==============================================================================================
//non-strided version
//workspace needed: none
// A_array, B_array, C_array: device pointers to device buffers
template<class T>
int Xgemm_batch_core( kblasHandle_t handle,
                      char transA,char transB,
                      const int m, const int n, const int k,
                      const T alpha,
                      const T** A_array, int lda,
                      const T** B_array, int ldb,
                      const T beta,
                            T** C_array, int ldc,
                      int batchCount)
{
  if(!handle->use_magma){
    check_error_ret( cublasXgemmBatched( handle->cublas_handle,
                          (transA == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N),
                          (transB == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N),
                          m, n, k,
                          &alpha, (const T**)A_array, lda,
                                  (const T**)B_array, ldb,
                          &beta,             C_array, ldc,
                          batchCount), KBLAS_cuBLAS_Error);
  }
  #ifdef USE_MAGMA
  else
  if(handle->use_magma){

    //take care of batch size limitation with magma
    int batch_increment = 65535;
    int batch_start = 0;

    while(batch_start != batchCount)
    {
      int batch_size = std::min(batch_increment, batchCount - batch_start);

      magmablas_Xgemm_batched((magma_trans_t)(MagmaNoTrans + (transA == KBLAS_Trans)),
                              (magma_trans_t)(MagmaNoTrans + (transB == KBLAS_Trans)),
                              m, n, k,
                              alpha, A_array, lda,
                                     B_array, ldb,
                              beta,  C_array, ldc,
                              batchCount, handle->magma_queue);

      A_array += batch_size;
      B_array += batch_size;
      C_array += batch_size;

      batch_start += batch_size;
      check_error_ret( cudaGetLastError(), KBLAS_MAGMA_Error);
    }
  }
  #endif
  #if 0
  else{
    if( transA == KBLAS_Trans )
      return KBLAS_NotSupported;

    if(typeid(T) == typeid(float)){
      return Xgemm_NX_batch<T, 16>(
                  transA, transB,
                  m, n, k,
                  alpha, A, lda, strideA,
                          B, ldb, strideB,
                  beta,  C, ldc, strideC,
                  batchCount, handle->cuda_stream);
    }else
    if(typeid(T) == typeid(double)){
      if(m < 16 || n < 16 || k < 16)
        return Xgemm_NX_batch<T, 8>(
                      transA, transB,
                      m, n, k,
                      alpha, A, lda, strideA,
                             B, ldb, strideB,
                      beta,  C, ldc, strideC,
                      batchCount, handle->cuda_stream);
      else
        return Xgemm_NX_batch<T, 16>(
                      transA, transB,
                      m, n, k,
                      alpha, A, lda, strideA,
                             B, ldb, strideB,
                      beta,  C, ldc, strideC,
                      batchCount, handle->cuda_stream);
    }else{
      return KBLAS_NotImplemented;
    }
  }
  #else
  else
  {
    printf("Configuration error at %s in file %s at line %d\n", __func__, __FILE__, __LINE__ );
    return KBLAS_WrongConfig;
  }
  #endif

  return KBLAS_Success;
}

//###############################################################################################
#if ( __CUDACC_VER_MAJOR__ < 8 )

template<class T>
void Xgemm_batch_strided_wsquery_core(int batchCount, kblasWorkspace_t ws)
{
  ws->d_ptrs_bytes_req = (batchCount > 1) * batchCount * 3 * sizeof(T*);
}

//workspace needed: device pointers
// A, B, C: host pointers to device buffers
template<class T>
int Xgemm_batch_strided_core( kblasHandle_t handle,
                              char transA,char transB,
                              const int m, const int n, const int k,
                              const T alpha,
                              const T* A, int lda, long strideA,
                              const T* B, int ldb, long strideB,
                              const T beta,
                                    T* C, int ldc, long strideC,
                              int batchCount)
{
  if(batchCount < 1)//TODO should not accept batch of size one
    return KBLAS_Error_WrongInput;

  KBlasWorkspace ws_needed;
  Xgemm_batch_strided_wsquery_core<T>(batchCount, (kblasWorkspace_t)&ws_needed);

  // int work_ptrs_bytes = (batchCount > 1) * batchCount * 3 * sizeof(T*);
  bool suffWorkspace = (ws_needed.d_ptrs_bytes_req <= handle->work_space.d_ptrs_bytes);

  if(!suffWorkspace){
    return KBLAS_InsufficientWorkspace;
  }
  kblasWorkspace_t ws_current = &(handle->work_space);

  if(batchCount == 1){
    check_error_ret( cublasXgemm( handle->cublas_handle,
                              (transA == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N),
                              (transB == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N),
                              m, n, k,
                              &alpha, (const T*) A, lda,
                                      (const T*) B, ldb,
                              &beta,             C, ldc), KBLAS_cuBLAS_Error);
    return KBLAS_Success;
  }


  T **A_array, **B_array, **C_array;
  A_array = (T**)(ws_current->d_ptrs);
  B_array = A_array + batchCount;
  C_array = B_array + batchCount;

  //if(use_cublas_gemm || use_magma_gemm)
  {
    check_error_ret( Xset_pointer_3(A_array, A, lda, strideA,
                                    B_array, B, ldb, strideB,
                                    C_array, C, ldc, strideC,
                                    batchCount, handle->stream), KBLAS_UnknownError);
  }
 //  int use_cublas = (m <= 64) || (n <= 64) || (k < 64);
  if(!handle->use_magma){
    check_error_ret( cublasXgemmBatched(handle->cublas_handle,
                                        (transA == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N),
                                        (transB == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N),
                                        m, n, k,
                                        &alpha, (const T**)A_array, lda,
                                                (const T**)B_array, ldb,
                                        &beta,             C_array, ldc,
                                        batchCount), KBLAS_cuBLAS_Error);
  }
  #ifdef USE_MAGMA
  else
  if(handle->use_magma){
    check_error_ret(handle->magma_queue != NULL, KBLAS_Error_NotInitialized);

    //take care of batch size limitation with magma
    int batch_increment = 65535;
    int batch_start = 0;

    while(batch_start != batchCount)
    {
      int batch_size = std::min(batch_increment, batchCount - batch_start);

      magmablas_Xgemm_batched((magma_trans_t)(MagmaNoTrans + (transA == KBLAS_Trans)),
                              (magma_trans_t)(MagmaNoTrans + (transB == KBLAS_Trans)),
                              m, n, k,
                              alpha, A_array, lda,
                                     B_array, ldb,
                              beta,  C_array, ldc,
                              batchCount, handle->magma_queue);

      A_array += batch_size;
      B_array += batch_size;
      C_array += batch_size;

      batch_start += batch_size;
      check_error_ret( cudaGetLastError(), KBLAS_MAGMA_Error);
    }
  }
  #endif
  #if 1
  else{
    //TODO unreachable code in some cases
    if( transA == KBLAS_Trans )
      return KBLAS_NotSupported;

    if(typeid(T) == typeid(float)){
      return Xgemm_NX_batch<T, 16>(
                  handle,
                  transA, transB,
                  m, n, k,
                  alpha, A, lda, strideA,
                          B, ldb, strideB,
                  beta,  C, ldc, strideC,
                  batchCount);
    }else
    if(typeid(T) == typeid(double)){
      if(m < 16 || n < 16 || k < 16)
        return Xgemm_NX_batch<T, 8>(
                      handle,
                      transA, transB,
                      m, n, k,
                      alpha, A, lda, strideA,
                             B, ldb, strideB,
                      beta,  C, ldc, strideC,
                      batchCount);
      else
        return Xgemm_NX_batch<T, 16>(
                      handle,
                      transA, transB,
                      m, n, k,
                      alpha, A, lda, strideA,
                             B, ldb, strideB,
                      beta,  C, ldc, strideC,
                      batchCount);
    }else{
      return KBLAS_NotImplemented;
    }
  }
  #else
  {
    printf("Configuration error at %s in file %s at line %d\n", __func__, __FILE__, __LINE__ );
    return KBLAS_WrongConfig;
  }
  #endif

  return KBLAS_Success;
}

//###############################################################################################
#else//__CUDACC_VER_MAJOR__ < 8

template<class T>
void Xgemm_batch_strided_wsquery_core(int batchCount, kblasWorkspace_t ws)
{
  ws->d_ptrs_bytes_req = 0;
}
//workspace needed: none
// A, B, C: host pointers to device buffers
template<class T>
int Xgemm_batch_strided_core( kblasHandle_t handle,
                              char transA,char transB,
                              const int m, const int n, const int k,
                              const T alpha,
                              const T* A, int lda, long strideA,
                              const T* B, int ldb, long strideB,
                              const T beta,
                                    T* C, int ldc, long strideC,
                              int batchCount)
{
  KBlasWorkspace ws_needed;
  Xgemm_batch_strided_wsquery_core<T>(batchCount, (kblasWorkspace_t)&ws_needed);

  // int work_ptrs_bytes = (batchCount > 1) * batchCount * 3 * sizeof(T*);
  bool suffWorkspace = (ws_needed.d_ptrs_bytes_req <= handle->work_space.d_ptrs_bytes);

  if(!suffWorkspace){
    return KBLAS_InsufficientWorkspace;
  }

  if(!handle->use_magma){
    check_error_ret( cublasXgemmStridedBatched( handle->cublas_handle,
                                                (transA == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N),
                                                (transB == KBLAS_Trans ? CUBLAS_OP_T : CUBLAS_OP_N),
                                                m, n, k,
                                                &alpha, (const T*)A, lda, strideA,
                                                        (const T*)B, ldb, strideB,
                                                &beta,            C, ldc, strideC,
                                                batchCount), KBLAS_cuBLAS_Error);
  }
  #if 1
  else{
    if( transA == KBLAS_Trans )
      return KBLAS_NotSupported;

    if(typeid(T) == typeid(float)){
      return Xgemm_NX_batch<T, 16>(
                  handle,
                  transA, transB,
                  m, n, k,
                  alpha, A, lda, strideA,
                          B, ldb, strideB,
                  beta,  C, ldc, strideC,
                  batchCount);
    }else
    if(typeid(T) == typeid(double)){
      if(m < 16 || n < 16 || k < 16)
        return Xgemm_NX_batch<T, 8>(
                      handle,
                      transA, transB,
                      m, n, k,
                      alpha, A, lda, strideA,
                             B, ldb, strideB,
                      beta,  C, ldc, strideC,
                      batchCount);
      else
        return Xgemm_NX_batch<T, 16>(
                      handle,
                      transA, transB,
                      m, n, k,
                      alpha, A, lda, strideA,
                             B, ldb, strideB,
                      beta,  C, ldc, strideC,
                      batchCount);
    }else{
      return KBLAS_NotImplemented;
    }
  }
  #else
  else
  {
    printf("Configuration error at %s in file %s at line %d\n", __func__, __FILE__, __LINE__ );
    return KBLAS_WrongConfig;
  }
  #endif

  return KBLAS_Success;
}
#endif//__CUDACC_VER_MAJOR__ < 8

#endif//__XGEMM_BATCH__