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
  #include "magma.h"
  #include <cusparse.h>
#endif

#include <assert.h>

#include "kblas_error.h"
#include "kblas_gpu_timer.h"

#ifdef KBLAS_ENABLE_BACKDOORS
#define KBLAS_BACKDOORS       16
extern int kblas_back_door[KBLAS_BACKDOORS];
#endif

// Structure defining the a state of the workspace, whether its allocated,
// requested by a query routine, or consumed by a routine holding the handle
struct KBlasWorkspaceState
{
	long h_data_bytes, h_ptrs_bytes;		// host data and pointer allocations
	long d_data_bytes, d_ptrs_bytes;		// device data and pointer allocations

	KBlasWorkspaceState()
	{
		reset();
	}

	KBlasWorkspaceState(long h_data_bytes, long h_ptrs_bytes, long d_data_bytes, long d_ptrs_bytes)
	{
		this->h_data_bytes = h_data_bytes;
		this->h_ptrs_bytes = h_ptrs_bytes;

		this->d_data_bytes = d_data_bytes;
		this->d_ptrs_bytes = d_ptrs_bytes;
	}

	void reset()
	{
		h_data_bytes = h_ptrs_bytes = 0;
		d_data_bytes = d_ptrs_bytes = 0;
	}
};

struct KBlasWorkspace
{
  typedef unsigned char WS_Byte;

  void* h_data;//host workspace
  void** h_ptrs;//host pointer workspace

  void* d_data;//device workspace
  void** d_ptrs;//device pointer workspace

  KBlasWorkspaceState allocated_ws_state;  	// current allocated workspace state
  KBlasWorkspaceState requested_ws_state;	// requested workspace state set by any workspace query routine
  KBlasWorkspaceState consumed_ws_state;	// workspace currently being used by a routine holding the handle

  bool allocated;

  KBlasWorkspace()
  {
    reset();
  }

  void reset()
  {
    allocated = 0;
    h_data = NULL;
    h_ptrs = NULL;
    d_data = NULL;
    d_ptrs = NULL;
  	allocated_ws_state.reset();
  	requested_ws_state.reset();
  	consumed_ws_state.reset();
  }

  KBlasWorkspaceState getAvailable()
  {
  	KBlasWorkspaceState available;

  	available.h_data_bytes = allocated_ws_state.h_data_bytes - consumed_ws_state.h_data_bytes;
  	available.h_ptrs_bytes = allocated_ws_state.h_ptrs_bytes - consumed_ws_state.h_ptrs_bytes;

  	available.d_data_bytes = allocated_ws_state.d_data_bytes - consumed_ws_state.d_data_bytes;
  	available.d_ptrs_bytes = allocated_ws_state.d_ptrs_bytes - consumed_ws_state.d_ptrs_bytes;

  	return available;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Workspace management - handle with care!
  // Make sure to pop after using the workspace and in a FILO way
  /////////////////////////////////////////////////////////////////////////////
  // Device workspace
  void* push_d_data(long bytes)
  {
    assert(bytes + consumed_ws_state.d_data_bytes <= allocated_ws_state.d_data_bytes);

    void* ret_ptr = (WS_Byte*)d_data + consumed_ws_state.d_data_bytes;
  	consumed_ws_state.d_data_bytes += bytes;
  	return ret_ptr;
  }

  void pop_d_data(long bytes)
  {
  	assert(consumed_ws_state.d_data_bytes >= bytes);
  	consumed_ws_state.d_data_bytes -= bytes;
  }

  void* push_d_ptrs(long bytes)
  {
    assert(bytes + consumed_ws_state.d_ptrs_bytes <= allocated_ws_state.d_ptrs_bytes);

    void* ret_ptr = (WS_Byte*)d_ptrs + consumed_ws_state.d_ptrs_bytes;
  	consumed_ws_state.d_ptrs_bytes += bytes;
  	return ret_ptr;
  }

  void pop_d_ptrs(long bytes)
  {
  	assert(consumed_ws_state.d_ptrs_bytes >= bytes);

  	consumed_ws_state.d_ptrs_bytes -= bytes;
  }
  // Host workspace
  void* push_h_data(long bytes)
  {
    assert(bytes + consumed_ws_state.h_data_bytes <= allocated_ws_state.h_data_bytes);

    void* ret_ptr = (WS_Byte*)h_data + consumed_ws_state.h_data_bytes;
  	consumed_ws_state.h_data_bytes += bytes;
  	return ret_ptr;
  }

  void pop_h_data(long bytes)
  {
  	assert(consumed_ws_state.h_data_bytes >= bytes);
  	consumed_ws_state.h_data_bytes -= bytes;
  }

  void* push_h_ptrs(long bytes)
  {
    assert(bytes + consumed_ws_state.h_ptrs_bytes <= allocated_ws_state.h_ptrs_bytes);

    void* ret_ptr = (WS_Byte*)h_ptrs + consumed_ws_state.h_ptrs_bytes;
  	consumed_ws_state.h_ptrs_bytes += bytes;
  	return ret_ptr;
  }

  void pop_h_ptrs(long bytes)
  {
  	assert(consumed_ws_state.h_ptrs_bytes >= bytes);
  	consumed_ws_state.h_ptrs_bytes -= bytes;
  }

  int allocate()
  {

    #ifdef DEBUG_ON
    printf("Workspace allocated:");
    #endif

    if(requested_ws_state.h_data_bytes > 0 && allocated_ws_state.h_data_bytes < requested_ws_state.h_data_bytes ){
      allocated_ws_state.h_data_bytes = requested_ws_state.h_data_bytes;
      if(h_data != NULL)
        check_error( cudaFreeHost(h_data) );
      check_error_ret( cudaHostAlloc ( (void**)&h_data, allocated_ws_state.h_data_bytes, cudaHostAllocPortable  ), KBLAS_Error_Allocation);
      #ifdef DEBUG_ON
      printf(" host bytes %d,", allocated_ws_state.h_data_bytes);
      #endif
      requested_ws_state.h_data_bytes = 0;
    }

    if(requested_ws_state.h_ptrs_bytes > 0 && allocated_ws_state.h_ptrs_bytes < requested_ws_state.h_ptrs_bytes ){
      allocated_ws_state.h_ptrs_bytes = requested_ws_state.h_ptrs_bytes;
      if(h_ptrs != NULL)
        check_error( cudaFreeHost(h_ptrs) );
      check_error_ret( cudaHostAlloc ( (void**)&h_ptrs, allocated_ws_state.h_ptrs_bytes, cudaHostAllocPortable  ), KBLAS_Error_Allocation);
      #ifdef DEBUG_ON
      printf(" host pointer bytes %d,", allocated_ws_state.h_ptrs_bytes);
      #endif
      requested_ws_state.h_ptrs_bytes = 0;
    }


    if(requested_ws_state.d_data_bytes > 0 && allocated_ws_state.d_data_bytes < requested_ws_state.d_data_bytes ){
      allocated_ws_state.d_data_bytes = requested_ws_state.d_data_bytes;
      // check_error( cudaSetDevice(device_id) );
      if(d_data != NULL){
        check_error_ret( cudaFree(d_data), KBLAS_Error_Deallocation);
      }
      #ifdef DEBUG_ON
      printf(" device bytes %d,", allocated_ws_state.d_data_bytes);
      #endif
      check_error_ret( cudaMalloc( (void**)&d_data, allocated_ws_state.d_data_bytes ), KBLAS_Error_Allocation);
  	  requested_ws_state.d_data_bytes = 0;
    }

    if(requested_ws_state.d_ptrs_bytes > 0 && allocated_ws_state.d_ptrs_bytes < requested_ws_state.d_ptrs_bytes ){
      allocated_ws_state.d_ptrs_bytes = requested_ws_state.d_ptrs_bytes;
      // check_error( cudaSetDevice(device_id) );
      if(d_ptrs != NULL){
        check_error_ret( cudaFree(d_ptrs), KBLAS_Error_Deallocation);
      }
      check_error_ret( cudaMalloc( (void**)&d_ptrs, allocated_ws_state.d_ptrs_bytes ), KBLAS_Error_Allocation);
      #ifdef DEBUG_ON
      printf(" device pointer bytes %d,", allocated_ws_state.d_ptrs_bytes);
      #endif
      requested_ws_state.d_ptrs_bytes = 0;
    }
    #ifdef DEBUG_ON
    printf("\n");
    #endif

    allocated = 1;
    return KBLAS_Success;
  }

  int deallocate()
  {
    if(h_data)
      cudaFreeHost(h_data);
    if(h_ptrs)
      cudaFreeHost(h_ptrs);
    if(d_data)
      check_error_ret( cudaFree(d_data), KBLAS_Error_Deallocation );
    if(d_ptrs)
      check_error_ret( cudaFree(d_ptrs), KBLAS_Error_Deallocation );

    return KBLAS_Success;
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

  kblas_gpu_timer timer;
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

    timer.init();

    this->device_id = device_id;
    this->stream = stream;
  }
  //-----------------------------------------------------------
  void tic()   { timer.start(stream); }
  double toc() { return timer.stop(stream);  }

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

    timer.init();

    check_error( cudaGetDevice(&device_id) );
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
    timer.destroy();
  }
};

typedef struct KBlasWorkspaceState *kblasWorkspaceState_t;
typedef struct KBlasWorkspace *kblasWorkspace_t;
typedef struct KBlasHandle *kblasHandle_t;
#define GPUBlasHandle KBlasHandle

int kblasCreate(kblasHandle_t *handle);
int kblasDestroy(kblasHandle_t *handle);

#endif //__KBLAS_STRUCT__
