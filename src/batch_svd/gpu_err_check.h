#ifndef __GPU_ERR_CHECK_H__
#define __GPU_ERR_CHECK_H__

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <assert.h>
#include <cusparse.h>
#include <stdio.h>

#ifndef __NO_NCCL__
#include <nccl.h>

#define ncclErrchk(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("NCCL failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
__device__ __host__
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) 
        printf("GPUassert: %s(%d) %s %d\n", cudaGetErrorString(code), (int)code, file, line);
}

__device__ __host__
inline const char* cublasGetErrorString(cublasStatus_t error)
{
    switch(error) 
    {
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
        default:
            return "unknown error code";
    }
}

__device__ __host__
inline const char* cusparseGetErrorString(cusparseStatus_t error)
{
    switch(error) 
    {
        case CUSPARSE_STATUS_SUCCESS:
            return "success";
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "not initialized";
        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "out of memory";
        case CUSPARSE_STATUS_INVALID_VALUE:
            return "invalid value";
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "architecture mismatch";
        case CUSPARSE_STATUS_MAPPING_ERROR:
            return "memory mapping error";
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "execution failed";
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "internal error";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "matrix type not supported";
        default:
            return "unknown error code";
    }
}

#define gpuCublasErrchk(ans) { gpuCublasAssert((ans), __FILE__, __LINE__); }
__device__ __host__
inline void gpuCublasAssert(cublasStatus_t code, const char *file, int line)
{
   if (code != CUBLAS_STATUS_SUCCESS) 
        printf("GPUassert: %s %s %d\n", cublasGetErrorString(code), file, line);
}

#define gpuCusparseErrchk(ans) { gpuCusparseAssert((ans), __FILE__, __LINE__); }
__device__ __host__
inline void gpuCusparseAssert(cusparseStatus_t code, const char *file, int line)
{
   if (code != CUSPARSE_STATUS_SUCCESS) 
        printf("GPUassert: %s %s %d\n", cusparseGetErrorString(code), file, line);
}

#endif
