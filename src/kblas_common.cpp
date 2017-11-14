/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/kblas_common.cpp

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 2.0.0
 * @author Ali Charara
 * @author Wajih Halim Boukaram
 * @date 2017-11-13
 **/

#include <stdlib.h>
#include <stdio.h>
#include <set>
#include <cublas_v2.h>
#include "kblas.h"
#include "operators.h"
#include "kblas_common.h"
#include "kblas_struct.h"

//==============================================================================================
int kblasCreate(kblasHandle_t *handle){

  int dev_id;
  check_error(cudaGetDevice(&dev_id));
  *handle = new KBlasHandle(0, 0, dev_id);


  #ifdef KBLAS_ENABLE_BACKDOORS
    (*handle)->back_door = kblas_back_door;
  #endif

  return KBLAS_Success;
}

int kblasDestroy(kblasHandle_t *handle){

  free((*handle));

  return KBLAS_Success;
}

int kblasAllocateWorkspace(kblasHandle_t handle) {
	return handle->work_space.allocate();
}

int kblasFreeeWorkspace(kblasHandle_t handle) {
	return handle->work_space.deallocate();
}

void kblasTimerTic(kblasHandle_t handle){
	handle->tic();
}

void kblasTimerRecordEnd(kblasHandle_t handle){
	handle->recordEnd();
}

double kblasTimerToc(kblasHandle_t handle) {
	return handle->toc();
}

cudaStream_t kblasGetStream(kblasHandle_t handle) {
	return handle->stream;
}

void kblasSetStream(kblasHandle_t handle, cudaStream_t stream) {
	handle->stream = stream;
  if(handle->cublas_handle)
    check_error( cublasSetStream(handle->cublas_handle, stream) );
  // TODO need to set magma_queue stream also
}

cublasHandle_t kblasGetCublasHandle(kblasHandle_t handle) {
	return handle->cublas_handle;
}
//==============================================================================================
extern "C"{
const char* cublasGetErrorString( cublasStatus_t error )
{
  switch( error ) {
    case CUBLAS_STATUS_SUCCESS:
      return "success";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "not initialized";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "out of memory";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "invalid value";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "architecture mismatch";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "memory mapping error";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "execution failed";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "internal error";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "Not Supported";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "License Error";

    default:
      return "unknown CUBLAS error code";
  }
}
}//extern C
const char* kblasGetErrorString( int error )
{
  switch( error ) {
    case KBLAS_UnknownError:
      return "KBLAS: unknown error";
    case KBLAS_NotSupported:
      return "Operation not supported";
    case KBLAS_NotImplemented:
      return "Operation not implemented yet";
    case KBLAS_cuBLAS_Error:
      return "cuBLAS error";
    case KBLAS_WrongConfig:
      return "Wrong compilation flags configuration";
    case KBLAS_CUDA_Error:
      return "CUDA error";
    case KBLAS_InsufficientWorkspace:
      return "Insufficient workspace supplied to function";
    case KBLAS_Error_Allocation:
      return "Error allocating memory";
    case KBLAS_Error_Deallocation:
      return "Error de-allocating memory";
    case KBLAS_Error_NotInitialized:
      return "KBLAS handle not initialized";
    case KBLAS_Error_WrongInput:
      return "One of input parameter's value is wrong";
    default:
      return "unknown KBLAS error code";
  }
}

// ----------------------------------------
// C++ function is overloaded for different error types,
// which depends on error types being enums to be differentiable.
//inline
int _kblas_error( cudaError_t err, const char* func, const char* file, int line )
{
  if ( err != cudaSuccess ) {
    fprintf( stderr, "CUDA runtime error: %s (%d) in %s at %s:%d\n",
             cudaGetErrorString( err ), err, func, file, line );
    return 0;
  }
  return 1;
}

// --------------------
int _kblas_error( cublasStatus_t err, const char* func, const char* file, int line )
{
  if ( err != CUBLAS_STATUS_SUCCESS ) {
    fprintf( stderr, "CUBLAS error: %s (%d) in %s at %s:%d\n",
             cublasGetErrorString( err ), err, func, file, line );
    return 0;
  }
  return 1;
}

// --------------------
int _kblas_error( int err, const char* func, const char* file, int line )
{
  if ( err != KBLAS_Success ) {
    fprintf( stderr, "KBLAS error: %s (%d) in %s at %s:%d\n",
             kblasGetErrorString( err ), err, func, file, line );
    return 0;
  }
  return 1;
}

//==============================================================================================

//TODO remove these flags and use instead the back_door flags
//TODO have a smarter way to choose, auto tuning for example
// bool use_magma_gemm = 0;
// bool use_cublas_gemm = 1;

//==============================================================================================
bool REG_SIZE(int n){
  return ((n > 0) && !(n & (n - 1)));
}
int CLOSEST_REG_SIZE(int n){
  //TODO validate input
  if(n > 0){
    int res = 1;
    while (res < n){
      res = res << 1;
    }
    return res >> 1;
  }else{
    return 0;
  }
}

//==============================================================================================
#ifdef KBLAS_ENABLE_BACKDOORS
int kblas_back_door[KBLAS_BACKDOORS] = {-1};
#endif
//==============================================================================================
#if 1
cublasStatus_t cublasXgemm( cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const float *alpha, const float *A, int lda,
                                                const float *B, int ldb,
                            const float *beta,        float *C, int ldc){
  cublasStatus_t status;
  check_error_ret( status = cublasSgemm(handle,
                                    transa, transb,
                                    m, n, k,
                                    alpha, A, lda,
                                           B, ldb,
                                    beta,  C, ldc), status);
  check_error_ret( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasXgemm( cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const double *alpha, const double *A, int lda,
                                                 const double *B, int ldb,
                            const double *beta,        double *C, int ldc){
  cublasStatus_t status;
  check_error_ret( status = cublasDgemm(handle,
                                    transa, transb,
                                    m, n, k,
                                    alpha, A, lda,
                                           B, ldb,
                                    beta,  C, ldc), status);
  check_error_ret( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
  return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasXgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const cuComplex *alpha, const cuComplex *A, int lda,
                                                   const cuComplex *B, int ldb,
                           const cuComplex *beta,        cuComplex *C, int ldc){
  cublasStatus_t status;
  check_error_ret( status = cublasCgemm(handle,
                                    transa, transb,
                                    m, n, k,
                                    alpha, A, lda,
                                           B, ldb,
                                    beta,  C, ldc), status);
  check_error_ret( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
  return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasXgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                                                         const cuDoubleComplex *B, int ldb,
                           const cuDoubleComplex *beta,        cuDoubleComplex *C, int ldc){
  cublasStatus_t status;
  check_error_ret( status = cublasZgemm(handle,
                                    transa, transb,
                                    m, n, k,
                                    alpha, A, lda,
                                           B, ldb,
                                    beta,  C, ldc), status);
  check_error_ret( cudaGetLastError(), CUBLAS_STATUS_EXECUTION_FAILED );
  return CUBLAS_STATUS_SUCCESS;
}
#endif


