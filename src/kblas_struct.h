 /**
  - -* (C) Copyright 2013 King Abdullah University of Science and Technology
  Authors:
  Ali Charara (ali.charara@kaust.edu.sa)
  Wajih Halim Boukaram (wajih.boukaram@kaust.edu.sa)
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
#ifndef __KBLAS_STRUCT__
#define __KBLAS_STRUCT__

#include <cuda_runtime.h>
#include <cublas_v2.h>
#ifdef USE_MAGMA
  #include <cusparse.h>
#endif

#include "kblas_error.h"

#ifdef KBLAS_PROFILING_ENABLED
#include "kblas_gpu_timer.h"
#endif

#ifdef KBLAS_ENABLE_BACKDOORS
#define KBLAS_BACKDOORS       16
extern int kblas_back_door[KBLAS_BACKDOORS];
#endif


struct KBlasWorkspace
{
  void* h_data;//host workspace
  long h_data_bytes;//current host workspace in bytes
  long h_data_bytes_req;//requested host workspace in bytes
  void** h_ptrs;//host pointer workspace
  long h_ptrs_bytes;//current host pointer workspace in bytes
  long h_ptrs_bytes_req;//requested host pointer workspace in bytes
  void* d_data;//device workspace
  long d_data_bytes;//current device workspace in bytes
  long d_data_bytes_req;//requested device workspace in bytes
  void** d_ptrs;//device pointer workspace
  long d_ptrs_bytes;//current device pointer workspace in bytes
  long d_ptrs_bytes_req;//requested device pointer workspace in bytes
  bool allocated;

  KBlasWorkspace()
  {
    reset();
  }

  void reset()
  {
    allocated = 0;
    h_data_bytes = 0;
    h_data_bytes_req = 0;
    h_data = NULL;
    h_ptrs_bytes = 0;
    h_ptrs_bytes_req = 0;
    h_ptrs = NULL;
    d_data_bytes = 0;
    d_data_bytes_req = 0;
    d_data = NULL;
    d_ptrs_bytes = 0;
    d_ptrs_bytes_req = 0;
    d_ptrs = NULL;
  }

  int allocate()
  {

    #ifdef DEBUG_ON
    printf("Workspace allocated:");
    #endif

    if(h_data_bytes_req > 0 && h_data_bytes < h_data_bytes_req ){
      h_data_bytes = h_data_bytes_req;
      if(h_data != NULL)
        check_error( cudaFreeHost(h_data) );
      check_error_ret( cudaHostAlloc ( (void**)&h_data, h_data_bytes, cudaHostAllocPortable  ), KBLAS_Error_Allocation);
      h_data_bytes_req = 0;
      #ifdef DEBUG_ON
      printf(" host bytes %d,", h_data_bytes);
      #endif
    }

    if(h_ptrs_bytes_req > 0 && h_ptrs_bytes < h_ptrs_bytes_req ){
      h_ptrs_bytes = h_ptrs_bytes_req;
      if(h_ptrs != NULL)
        check_error( cudaFreeHost(h_ptrs) );
      check_error_ret( cudaHostAlloc ( (void**)&h_ptrs, h_ptrs_bytes, cudaHostAllocPortable  ), KBLAS_Error_Allocation);
      h_ptrs_bytes_req = 0;
      #ifdef DEBUG_ON
      printf(" host pointer bytes %d,", h_ptrs_bytes);
      #endif
    }


    if(d_data_bytes_req > 0 && d_data_bytes < d_data_bytes_req ){
      d_data_bytes = d_data_bytes_req;
      // check_error( cudaSetDevice(device_id) );
      if(d_data != NULL){
        check_error_ret( cudaFree(d_data), KBLAS_Error_Deallocation);
      }
      #ifdef DEBUG_ON
      printf(" device bytes %d,", d_ptrs_bytes);
      #endif
      check_error_ret( cudaMalloc( (void**)&d_data, d_data_bytes ), KBLAS_Error_Allocation);
      d_data_bytes_req = 0;
    }

    if(d_ptrs_bytes_req > 0 && d_ptrs_bytes < d_ptrs_bytes_req ){
      d_ptrs_bytes = d_ptrs_bytes_req;
      // check_error( cudaSetDevice(device_id) );
      if(d_ptrs != NULL){
        check_error_ret( cudaFree(d_ptrs), KBLAS_Error_Deallocation);
      }
      check_error_ret( cudaMalloc( (void**)&d_ptrs, d_ptrs_bytes ), KBLAS_Error_Allocation);
      d_ptrs_bytes_req = 0;
      #ifdef DEBUG_ON
      printf(" device pointer bytes %d,", d_ptrs_bytes);
      #endif
    }
    #ifdef DEBUG_ON
    printf("\n");
    #endif
    allocated = 1;
    return KBLAS_Success;
  }

  void deallocate()
  {
    if(h_data)
      free(h_data);
    if(h_ptrs)
      free(h_ptrs);
    if(d_data)
      check_error( cudaFree(d_data) );
    if(d_ptrs)
      check_error( cudaFree(d_ptrs) );
  }
  ~KBlasWorkspace()
  {
    if(allocated)
      deallocate();
  }
};

struct KBlasHandle
{
  typedef unsigned char WS_Byte;

  cublasHandle_t cublas_handle;
  cudaStream_t stream;
  #ifdef USE_MAGMA
    cusparseHandle_t cusparse_handle;
    magma_queue_t  magma_queue;
  #endif
  int use_magma, device_id, create_cublas;

  #ifdef KBLAS_PROFILING_ENABLED
  kblas_gpu_timer timer;
  #endif
  // Workspace in bytes
  void* workspace;
  unsigned int workspace_bytes;

  KBlasWorkspace work_space;

  #ifdef KBLAS_ENABLE_BACKDOORS
  int *back_door;
  #endif

  //-----------------------------------------------------------
  KBlasHandle(int use_magma, cudaStream_t stream = 0, int device_id = 0)
  {
    #ifdef USE_MAGMA
    this->use_magma = use_magma;
    #else
    this->use_magma = 0;
    #endif
    create_cublas = 1;
    check_error( cublasCreate(&cublas_handle) );
    check_error( cublasSetStream(cublas_handle, stream) );

    #ifdef USE_MAGMA
    if(use_magma){
      check_error( cusparseCreate(&cusparse_handle) );
      magma_queue_create_from_cuda(
        device_id, stream, cublas_handle,
        cusparse_handle, &magma_queue
      );
    }
    else
      cusparse_handle = NULL;
    #endif

    #ifdef KBLAS_PROFILING_ENABLED
    timer.init();
    #endif

    this->device_id = device_id;
    this->stream = stream;
    workspace_bytes = 0;
    workspace = NULL;
  }
  //-----------------------------------------------------------
  #ifdef KBLAS_PROFILING_ENABLED
  void tic()   { timer.start(stream); }
  double toc() { return timer.stop(stream);  }
  #endif

  //-----------------------------------------------------------
  KBlasHandle(cublasHandle_t& cublas_handle)
  {
    this->use_magma = 0;
    this->create_cublas = 0;
    #ifdef USE_MAGMA
    cusparse_handle = NULL;
    #endif
    this->cublas_handle = cublas_handle;
    check_error( cublasGetStream(cublas_handle, &stream) );

    check_error( cudaGetDevice(&device_id) );
    workspace_bytes = 0;
    workspace = NULL;
  }

  ~KBlasHandle()
  {
    if(cublas_handle != NULL && create_cublas)
      check_error( cublasDestroy(cublas_handle) );
    #ifdef USE_MAGMA
    if(cusparse_handle != NULL)
      check_error( cusparseDestroy(cusparse_handle) );
    #endif
    if(stream && create_cublas)
      check_error( cudaStreamDestroy(stream) );

    #ifdef USE_MAGMA
    if(use_magma)
      magma_queue_destroy(magma_queue);
    #endif
    #ifdef KBLAS_PROFILING_ENABLED
    timer.destroy();
    #endif

    if(workspace)
      check_error( cudaFree(workspace) );
  }

  void setWorkspace(unsigned int bytes)
  {
    if(workspace)
      check_error( cudaFree(workspace) );
    workspace_bytes = bytes;
    check_error( cudaMalloc(&workspace, bytes) );
  }
};

typedef struct KBlasWorkspace *kblasWorkspace_t;
typedef struct KBlasHandle *kblasHandle_t;
#define GPUBlasHandle KBlasHandle

int kblasCreate(kblasHandle_t *handle);
int kblasDestroy(kblasHandle_t *handle);

#endif //__KBLAS_STRUCT__