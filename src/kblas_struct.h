/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/kblas_struct.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @author Wajih Halim Boukaram
 * @date 2018-11-14
 **/

#ifndef __KBLAS_STRUCT__
#define __KBLAS_STRUCT__
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "hipblas.h"

#ifdef USE_MAGMA
  // #include "magma.h"
  #include "magma_v2.h"
  // #include <hipsparse.h>
#endif

#include <assert.h>

#include "kblas_error.h"
#include "kblas_gpu_timer.h"
#include "kblas_common.h"

#ifdef KBLAS_ENABLE_BACKDOORS
#define KBLAS_BACKDOORS       16
extern int kblas_back_door[KBLAS_BACKDOORS];
#endif

// Structure defining the a state of the workspace, whether its allocated,
// requested by a query routine, or consumed by a routine holding the handle
struct KBlasWorkspaceState
{
  size_t h_data_bytes, h_ptrs_bytes;		// host data and pointer allocations
  size_t d_data_bytes, d_ptrs_bytes;		// device data and pointer allocations

  KBlasWorkspaceState()
  {
    reset();
  }

  KBlasWorkspaceState(size_t h_data_bytes, size_t h_ptrs_bytes, size_t d_data_bytes, size_t d_ptrs_bytes)
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

  void pad(KBlasWorkspaceState* wss)
  {
    h_data_bytes = kmax(h_data_bytes, wss->h_data_bytes);
    h_ptrs_bytes = kmax(h_ptrs_bytes, wss->h_ptrs_bytes);
    d_data_bytes = kmax(d_data_bytes, wss->d_data_bytes);
    d_ptrs_bytes = kmax(d_ptrs_bytes, wss->d_ptrs_bytes);
  }

  void set(KBlasWorkspaceState* wss)
  {
    h_data_bytes = wss->h_data_bytes;
    h_ptrs_bytes = wss->h_ptrs_bytes;
    d_data_bytes = wss->d_data_bytes;
    d_ptrs_bytes = wss->d_ptrs_bytes;
  }

  bool isSufficient(KBlasWorkspaceState* wss)
  {
    return (h_data_bytes <= wss->h_data_bytes)
        && (h_ptrs_bytes <= wss->h_ptrs_bytes)
        && (d_data_bytes <= wss->d_data_bytes)
        && (d_ptrs_bytes <= wss->d_ptrs_bytes);
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
    // TODO need to deallocate first
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
  void* push_d_data(size_t bytes)
  {
    assert(bytes + consumed_ws_state.d_data_bytes <= allocated_ws_state.d_data_bytes);

    void* ret_ptr = (WS_Byte*)d_data + consumed_ws_state.d_data_bytes;
    consumed_ws_state.d_data_bytes += bytes;
    return ret_ptr;
  }

  void pop_d_data(size_t bytes)
  {
    assert(consumed_ws_state.d_data_bytes >= bytes);
    consumed_ws_state.d_data_bytes -= bytes;
  }

  void* push_d_ptrs(size_t bytes)
  {
    assert(bytes + consumed_ws_state.d_ptrs_bytes <= allocated_ws_state.d_ptrs_bytes);

    void* ret_ptr = (WS_Byte*)d_ptrs + consumed_ws_state.d_ptrs_bytes;
    consumed_ws_state.d_ptrs_bytes += bytes;
    return ret_ptr;
  }

  void pop_d_ptrs(size_t bytes)
  {
    assert(consumed_ws_state.d_ptrs_bytes >= bytes);

    consumed_ws_state.d_ptrs_bytes -= bytes;
  }
  // Host workspace
  void* push_h_data(size_t bytes)
  {
    assert(bytes + consumed_ws_state.h_data_bytes <= allocated_ws_state.h_data_bytes);

    void* ret_ptr = (WS_Byte*)h_data + consumed_ws_state.h_data_bytes;
    consumed_ws_state.h_data_bytes += bytes;
    return ret_ptr;
  }

  void pop_h_data(size_t bytes)
  {
    assert(consumed_ws_state.h_data_bytes >= bytes);
    consumed_ws_state.h_data_bytes -= bytes;
  }

  void* push_h_ptrs(size_t bytes)
  {
    assert(bytes + consumed_ws_state.h_ptrs_bytes <= allocated_ws_state.h_ptrs_bytes);

    void* ret_ptr = (WS_Byte*)h_ptrs + consumed_ws_state.h_ptrs_bytes;
    consumed_ws_state.h_ptrs_bytes += bytes;
    return ret_ptr;
  }

  void pop_h_ptrs(size_t bytes)
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
        check_error( hipHostFree(h_data) );
      check_error_ret( hipHostMalloc ( (void**)&h_data, allocated_ws_state.h_data_bytes, hipHostMallocPortable  ), KBLAS_Error_Allocation);
      #ifdef DEBUG_ON
      printf(" host bytes %d,", allocated_ws_state.h_data_bytes);
      #endif
      requested_ws_state.h_data_bytes = 0;
    }

    if(requested_ws_state.h_ptrs_bytes > 0 && allocated_ws_state.h_ptrs_bytes < requested_ws_state.h_ptrs_bytes ){
      allocated_ws_state.h_ptrs_bytes = requested_ws_state.h_ptrs_bytes;
      if(h_ptrs != NULL)
        check_error( hipHostFree(h_ptrs) );
      check_error_ret( hipHostMalloc ( (void**)&h_ptrs, allocated_ws_state.h_ptrs_bytes, hipHostMallocPortable  ), KBLAS_Error_Allocation);
      #ifdef DEBUG_ON
      printf(" host pointer bytes %d,", allocated_ws_state.h_ptrs_bytes);
      #endif
      requested_ws_state.h_ptrs_bytes = 0;
    }


    if(requested_ws_state.d_data_bytes > 0 && allocated_ws_state.d_data_bytes < requested_ws_state.d_data_bytes ){
      allocated_ws_state.d_data_bytes = requested_ws_state.d_data_bytes;
      // check_error( hipSetDevice(device_id) );
      if(d_data != NULL){
        check_error_ret( hipFree(d_data), KBLAS_Error_Deallocation);
      }
      #ifdef DEBUG_ON
      printf(" device bytes %d,", allocated_ws_state.d_data_bytes);
      #endif
      check_error_ret( hipMalloc( (void**)&d_data, allocated_ws_state.d_data_bytes ), KBLAS_Error_Allocation);
  	  requested_ws_state.d_data_bytes = 0;
    }

    if(requested_ws_state.d_ptrs_bytes > 0 && allocated_ws_state.d_ptrs_bytes < requested_ws_state.d_ptrs_bytes ){
      allocated_ws_state.d_ptrs_bytes = requested_ws_state.d_ptrs_bytes;
      // check_error( hipSetDevice(device_id) );
      if(d_ptrs != NULL){
        check_error_ret( hipFree(d_ptrs), KBLAS_Error_Deallocation);
      }
      check_error_ret( hipMalloc( (void**)&d_ptrs, allocated_ws_state.d_ptrs_bytes ), KBLAS_Error_Allocation);
      #ifdef DEBUG_ON
      printf(" device pointer bytes %d,", allocated_ws_state.d_ptrs_bytes);
      #endif
      requested_ws_state.d_ptrs_bytes = 0;
    }

    requested_ws_state.reset();
    #ifdef DEBUG_ON
    printf("\n");
    #endif

    allocated = 1;
    return KBLAS_Success;
  }

  int deallocate()
  {
    if(h_data)
      hipHostFree(h_data);
    if(h_ptrs)
      hipHostFree(h_ptrs);
    if(d_data)
      check_error_ret( hipFree(d_data), KBLAS_Error_Deallocation );
    if(d_ptrs)
      check_error_ret( hipFree(d_ptrs), KBLAS_Error_Deallocation );
    reset();

    return KBLAS_Success;
  }
  ~KBlasWorkspace()
  {
    if(allocated)
      deallocate();
  }
};

struct KBlasWorkspaceGuard
{
	KBlasWorkspaceState pushed_ws;
	KBlasWorkspace* ws_ptr;

	KBlasWorkspaceGuard(KBlasWorkspaceState& pushed_ws, KBlasWorkspace& ws)
	{
		this->pushed_ws = pushed_ws;
		ws_ptr = &ws;
	}
	~KBlasWorkspaceGuard()
	{
		ws_ptr->pop_d_data(pushed_ws.d_data_bytes);
		ws_ptr->pop_d_ptrs(pushed_ws.d_ptrs_bytes);
		ws_ptr->pop_h_data(pushed_ws.h_data_bytes);
		ws_ptr->pop_h_ptrs(pushed_ws.h_ptrs_bytes);
	}
};

struct KBlasHandle
{
  typedef unsigned char WS_Byte;

  hipblasHandle_t cublas_handle;
  hipStream_t stream;
  hipStream_t streams[KBLAS_NSTREAMS];
  int nStreams;
  #ifdef USE_MAGMA
    magma_queue_t  magma_queue;
  #endif
  int use_magma, device_id, create_cublas;

  kblas_gpu_timer timer;
  KBlasWorkspace work_space;

  #ifdef KBLAS_ENABLE_BACKDOORS
  int *back_door;
  #endif

  //-----------------------------------------------------------
  KBlasHandle(int use_magma, hipStream_t stream = 0, int device_id = 0)
  {
    #ifdef USE_MAGMA
    this->use_magma = use_magma;
    #else
    this->use_magma = 0;
    #endif
    create_cublas = 1;
    check_error( hipblasCreate(&cublas_handle) );
    check_error( hipblasSetStream(cublas_handle, stream) );

    #ifdef USE_MAGMA
    if(use_magma){
      magma_queue_create_from_hip(
        device_id, stream, cublas_handle,
        NULL, &magma_queue
      );
    }
    else
    #endif

    timer.init();

    work_space.reset();

    this->device_id = device_id;
    this->stream = stream;
    this->nStreams = 0;
  }
  //-----------------------------------------------------------
  void tic()   		{ timer.start(stream); 		 }
  void recordEnd()	{ timer.recordEnd(stream);   }
  double toc() 		{ return timer.stop(stream); }

  //-----------------------------------------------------------
  KBlasHandle(hipblasHandle_t& cublas_handle)
  {
    this->use_magma = 0;
    this->create_cublas = 0;
    #ifdef USE_MAGMA
    #endif
    this->cublas_handle = cublas_handle;
    check_error( hipblasGetStream(cublas_handle, &stream) );

    timer.init();

    work_space.reset();

    check_error( hipGetDevice(&device_id) );
  }

  //-----------------------------------------------------------
  #ifdef USE_MAGMA
  void EnableMagma()
  {
    #ifdef USE_MAGMA
    if(use_magma == 0){
      use_magma = 1;
      magma_queue_create_from_hip(
        device_id, stream, cublas_handle,
        NULL, &magma_queue
      );
    }
    #else
      printf("ERROR: KBLAS is compiled without magma!\n");
    #endif
  }

  //-----------------------------------------------------------
  void SetMagma(magma_queue_t queue)
  {
    #ifdef USE_MAGMA
      magma_queue = queue;
      if (queue != NULL)
        use_magma = 1;
      else
        use_magma = 0;
    #else
      printf("ERROR: KBLAS is compiled without magma!\n");
    #endif
  }
  #endif

  //-----------------------------------------------------------
  int SetStream(hipStream_t stream)
  {
    this->stream = stream;
    //if(cublas_handle)
      check_error( hipblasSetStream(cublas_handle, stream) );
    #ifdef USE_MAGMA
    // TODO need to set magma_queue stream also, is that possible?
    #endif
    return KBLAS_Success;
  }

  //-----------------------------------------------------------
  int CreateStreams(int nStreams)
  {
    if(nStreams > KBLAS_NSTREAMS) return KBLAS_WrongConfig;
    this->nStreams = nStreams;
    for (int i = 0; i < nStreams; ++i)
    {
      check_error( hipStreamCreateWithFlags( &(this->streams[i]), hipStreamNonBlocking) );
    }
    return KBLAS_Success;
  }

  //-----------------------------------------------------------
  ~KBlasHandle()
  {
    if(cublas_handle != NULL && create_cublas)
      check_error( hipblasDestroy(cublas_handle) );
    
    if(nStreams > 0){
      for (int i = 0; i < nStreams; ++i){
        check_error( hipStreamDestroy(streams[i]) );
      }
    }
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
