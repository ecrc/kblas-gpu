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
 * @version 3.0.0
 * @author Ali Charara
 * @author Wajih Halim Boukaram
 * @date 2018-11-14
 **/

#include <stdlib.h>
#include <stdio.h>
#include <set>
#include "hipblas.h"
#include "kblas.h"
#include "kblas_operators.h"
#include "kblas_common.h"
#include "kblas_struct.h"

#include "batch_triangular/Xhelper_funcs.cuh"

//==============================================================================================
/**
 * Creates a KBLAS handle and initialize internal structures including: cuda stream, cublas handle, magma handle, device id, workspace buffers.
 */
int kblasCreate(kblasHandle_t *handle){

  int dev_id;
  check_error(hipGetDevice(&dev_id));
  *handle = new KBlasHandle(0, 0, dev_id);


  #ifdef KBLAS_ENABLE_BACKDOORS
    (*handle)->back_door = kblas_back_door;
  #endif

  return KBLAS_Success;
}

#ifdef USE_MAGMA
/**
 * Enable MAGMA support and create magma queue within KBLAS handle.
 */
int kblasEnableMagma(kblasHandle_t handle){

  #ifdef USE_MAGMA
    handle->EnableMagma();
    return KBLAS_Success;
  #else
    printf("ERROR: KBLAS is compiled without magma!\n");
    return KBLAS_Error_NotInitialized;
  #endif
}


/**
 * Set MAGMA queue in KBLAS handle.
 */
int kblasSetMagma(kblasHandle_t handle, magma_queue_t queue){
    handle->SetMagma(queue);
    return KBLAS_Success;
}
#endif

/**
 * Destroys a KBLAS handle and frees internal structures including: cuda stream, cublas handle, magma handle, workspace buffers.
 */
int kblasDestroy(kblasHandle_t *handle){
  delete *handle;
  *handle = NULL;
  
  return KBLAS_Success;
}

/**
 * Allocates workspace buffer for various data types (host pointer array, host data, device pointer array, device data) to be used by KBLAS routines that need it.
 * To determine which workspace data type is needed and how much, use the corresponding *_wsquery() routine. You may call several *ws_query() routines to allocate the maximum buffers needed by the various KBLAS routines, then call this function to allocate the workspace buffers.
 */
int kblasAllocateWorkspace(kblasHandle_t handle) {
	return handle->work_space.allocate();
}

/**
 * Free workspace buffers hosted in the handle structure, and reset internal workspace sizes.
 */
int kblasFreeWorkspace(kblasHandle_t handle) {
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

int kblasCreateStreams(kblasHandle_t handle, int nStreams) {
  return handle->CreateStreams(nStreams);
}
hipStream_t kblasGetStream(kblasHandle_t handle) {
	return handle->stream;
}

void kblasSetStream(kblasHandle_t handle, hipStream_t stream) {
	handle->stream = stream;
  if(handle->cublas_handle)
    check_error( hipblasSetStream(handle->cublas_handle, stream) );
  #ifdef USE_MAGMA
  // TODO need to set magma_queue stream also, is that possible?
  #endif
}

hipblasHandle_t kblasGetCublasHandle(kblasHandle_t handle) {
	return handle->cublas_handle;
}
//==============================================================================================
extern "C"{
const char* cublasGetErrorString( hipblasStatus_t error )
{
  switch( error ) {
    case HIPBLAS_STATUS_SUCCESS:
      return "success";

    case HIPBLAS_STATUS_NOT_INITIALIZED:
      return "not initialized";

    case HIPBLAS_STATUS_ALLOC_FAILED:
      return "out of memory";

    case HIPBLAS_STATUS_INVALID_VALUE:
      return "invalid value";

    case HIPBLAS_STATUS_ARCH_MISMATCH:
      return "architecture mismatch";

    case HIPBLAS_STATUS_MAPPING_ERROR:
      return "memory mapping error";

    case HIPBLAS_STATUS_EXECUTION_FAILED:
      return "execution failed";

    case HIPBLAS_STATUS_INTERNAL_ERROR:
      return "internal error";

    case HIPBLAS_STATUS_NOT_SUPPORTED:
      return "Not Supported";
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
    case KBLAS_MAGMA_Error:
      return "MAGMA error";
    case KBLAS_SVD_NoConvergence:
      return "SVD-gram operation did not converge.";
    default:
      return "unknown KBLAS error code";
  }
}

// ----------------------------------------
// C++ function is overloaded for different error types,
// which depends on error types being enums to be differentiable.
//inline
int _kblas_error( hipError_t err, const char* func, const char* file, int line )
{
  if ( err != hipSuccess ) {
    fprintf( stderr, "CUDA runtime error: %s (%d) in %s at %s:%d\n",
             hipGetErrorString( err ), err, func, file, line );
    return 0;
  }
  return 1;
}

// --------------------
int _kblas_error( hipblasStatus_t err, const char* func, const char* file, int line )
{
  if ( err != HIPBLAS_STATUS_SUCCESS ) {
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

extern "C"
int kblas_roundup(int x, int y){
  return int( (x + y-1) / y ) * y;
}

long kblas_roundup_l(long x, long y){
  return long( (x + y-1) / y ) * y;
}

size_t kblas_roundup_s(size_t x, size_t y){
  return size_t( (x + y-1) / y ) * y;
}
//==============================================================================================
#ifdef KBLAS_ENABLE_BACKDOORS
int kblas_back_door[KBLAS_BACKDOORS] = {-1};
#endif
//==============================================================================================
#if 1
hipblasStatus_t cublasXgemm( hipblasHandle_t handle,
                            hipblasOperation_t transa, hipblasOperation_t transb,
                            int m, int n, int k,
                            const float *alpha, const float *A, int lda,
                                                const float *B, int ldb,
                            const float *beta,        float *C, int ldc){
  hipblasStatus_t status;
  check_error_ret( status = hipblasSgemm(handle,
                                    transa, transb,
                                    m, n, k,
                                    alpha, A, lda,
                                           B, ldb,
                                    beta,  C, ldc), status);
  check_error_ret( hipGetLastError(), HIPBLAS_STATUS_EXECUTION_FAILED );
  return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t cublasXgemm( hipblasHandle_t handle,
                            hipblasOperation_t transa, hipblasOperation_t transb,
                            int m, int n, int k,
                            const double *alpha, const double *A, int lda,
                                                 const double *B, int ldb,
                            const double *beta,        double *C, int ldc){
  hipblasStatus_t status;
  check_error_ret( status = hipblasDgemm(handle,
                                    transa, transb,
                                    m, n, k,
                                    alpha, A, lda,
                                           B, ldb,
                                    beta,  C, ldc), status);
  check_error_ret( hipGetLastError(), HIPBLAS_STATUS_EXECUTION_FAILED );
  return HIPBLAS_STATUS_SUCCESS;
}
// hipblasStatus_t cublasXgemm(hipblasHandle_t handle,
//                            hipblasOperation_t transa, hipblasOperation_t transb,
//                            int m, int n, int k,
//                            const hipComplex *alpha, const hipComplex *A, int lda,
//                                                    const hipComplex *B, int ldb,
//                            const hipComplex *beta,        hipComplex *C, int ldc){
//   hipblasStatus_t status;
//   check_error_ret( status = hipblasCgemm(handle,
//                                     transa, transb,
//                                     m, n, k,
//                                     alpha, A, lda,
//                                            B, ldb,
//                                     beta,  C, ldc), status);
//   check_error_ret( hipGetLastError(), HIPBLAS_STATUS_EXECUTION_FAILED );
//   return HIPBLAS_STATUS_SUCCESS;
// }
// hipblasStatus_t cublasXgemm(hipblasHandle_t handle,
//                            hipblasOperation_t transa, hipblasOperation_t transb,
//                            int m, int n, int k,
//                            const hipDoubleComplex *alpha, const hipDoubleComplex *A, int lda,
//                                                          const hipDoubleComplex *B, int ldb,
//                            const hipDoubleComplex *beta,        hipDoubleComplex *C, int ldc){
//   hipblasStatus_t status;
//   check_error_ret( status = hipblasZgemm(handle,
//                                     transa, transb,
//                                     m, n, k,
//                                     alpha, A, lda,
//                                            B, ldb,
//                                     beta,  C, ldc), status);
//   check_error_ret( hipGetLastError(), HIPBLAS_STATUS_EXECUTION_FAILED );
//   return HIPBLAS_STATUS_SUCCESS;
// }
#endif


//==============================================================================================
int iset_value_1( int *output_array, int input,
                  long batchCount, hipStream_t hip_stream)
{
  return Xset_value_1_core<int>(output_array, input, batchCount, hip_stream);
}

//==============================================================================================
int iset_value_2( int *output_array1, int input1,
                  int *output_array2, int input2,
                  long batchCount, hipStream_t hip_stream)
{
  return Xset_value_2_core<int>(output_array1, input1,
                                output_array2, input2,
                                batchCount, hip_stream);
}
//==============================================================================================
int iset_value_4( int *output_array1, int input1,
                  int *output_array2, int input2,
                  int *output_array3, int input3,
                  int *output_array4, int input4,
                  long batchCount, hipStream_t hip_stream)
{
  return Xset_value_4_core<int>(output_array1, input1,
                                output_array2, input2,
                                output_array3, input3,
                                output_array4, input4,
                                batchCount, hip_stream);
}
//==============================================================================================
int iset_value_5( int *output_array1, int input1,
                  int *output_array2, int input2,
                  int *output_array3, int input3,
                  int *output_array4, int input4,
                  int *output_array5, int input5,
                  long batchCount, hipStream_t hip_stream)
{
  return Xset_value_5_core<int>(output_array1, input1,
                                output_array2, input2,
                                output_array3, input3,
                                output_array4, input4,
                                output_array5, input5,
                                batchCount, hip_stream);
}
