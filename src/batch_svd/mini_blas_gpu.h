#ifndef __MINI_BLAS_GPU_H__
#define __MINI_BLAS_GPU_H__

#include <algorithm>
#include <gpu_err_check.h>
#ifdef USE_MAGMA
#include <magma.h>
#endif
#include <curand.h>

#ifdef HLIB_PROFILING_ENABLED
// #include <timer_gpu.h>
#include <perf_counter.h>
#endif

#include "kblas_struct.h"

#if 0
struct GPUBlasHandle
{
	typedef unsigned char WS_Byte;

	cublasHandle_t cublas_handle;
	cusparseHandle_t cusparse_handle;
	#ifdef USE_MAGMA
	magma_queue_t  magma_queue;
	#endif
	cudaStream_t stream;
	int use_magma, device_id, create_cublas;

	#ifdef HLIB_PROFILING_ENABLED
	Timer_GPU timer;
	#endif

	// Workspace in bytes
	void* workspace;
	unsigned int workspace_bytes;

	GPUBlasHandle(int use_magma, cudaStream_t stream = 0, int device_id = 0)
	{
		#ifdef USE_MAGMA
		this->use_magma = use_magma;
		#else
		this->use_magma = 0;
		#endif
		create_cublas = 1;
		gpuCublasErrchk( cublasCreate(&cublas_handle) );
		gpuCublasErrchk( cublasSetStream(cublas_handle, stream) );

		#ifdef USE_MAGMA
		if(use_magma){
			gpuCusparseErrchk( cusparseCreate(&cusparse_handle) );
			magma_queue_create_from_cuda(
				device_id, stream, cublas_handle,
				cusparse_handle, &magma_queue
			);
		}else
		#endif
			cusparse_handle = NULL;

		#ifdef HLIB_PROFILING_ENABLED
		timer.init();
		#endif

		this->device_id = device_id;
		this->stream = stream;
		workspace_bytes = 0;
		workspace = NULL;
	}

	#ifdef HLIB_PROFILING_ENABLED
	void tic()   { timer.start(stream); }
	double toc() { return timer.stop(stream);  }
	#endif

	GPUBlasHandle(cublasHandle_t& cublas_handle)
 	{
 		this->use_magma = 0;
 		this->create_cublas = 0;
 		// gpuCublasErrchk( cublasCreate(&cublas_handle) );
 		// gpuCublasErrchk( cublasSetStream(cublas_handle, stream) );
 		cusparse_handle = NULL;
 		this->cublas_handle = cublas_handle;
 		cublasGetStream(cublas_handle, &stream);

 		this->device_id = 0;
 		workspace_bytes = 0;
 		workspace = NULL;
 	}

	~GPUBlasHandle()
	{
		if(cublas_handle != NULL && create_cublas)
			gpuCublasErrchk( cublasDestroy(cublas_handle) );
		if(cusparse_handle != NULL)
			gpuCusparseErrchk( cusparseDestroy(cusparse_handle) );
		if(stream && create_cublas)
			gpuErrchk( cudaStreamDestroy(stream) );

		#ifdef USE_MAGMA
		if(use_magma)
			magma_queue_destroy(magma_queue);
		#endif
		#ifdef HLIB_PROFILING_ENABLED
		timer.destroy();
		#endif

		if(workspace)
			gpuErrchk( cudaFree(workspace) );
	}

	void setWorkspace(unsigned int bytes)
	{
		if(workspace)
			gpuErrchk( cudaFree(workspace) );
		workspace_bytes = bytes;
		gpuErrchk( cudaMalloc(&workspace, bytes) );
	}
};
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////
// Utility function to properly align workspace
// ws_bytes is the number of bytes needed per op for each allocation type
// and is overwritten by the aligned bytes for num_ops operations
// num_ops is the number of operations we'd like to align, but the alignment may cause
// the needed memory to exceed our total workspace, so we have to reduce the number of ops
//////////////////////////////////////////////////////////////////////////////////////////////////
inline void alignWorkspace(unsigned int* ws_bytes, int N, unsigned int avail_ws, int& num_ops, int bytes)
{
	unsigned int sum_ws;
	do
	{
		sum_ws = 0;
		for(int i = 0; i < N; i++)
		{
			unsigned int op_bytes = ws_bytes[i] * num_ops;
			op_bytes += op_bytes % bytes;
			sum_ws += op_bytes;
		}
		if(sum_ws > avail_ws)
			num_ops--;
	} while(sum_ws > avail_ws && num_ops > 0);

	for(int i = 0; i < N; i++)
	{
		ws_bytes[i] = ws_bytes[i] * num_ops;
		ws_bytes[i] += ws_bytes[i] % bytes;
		if(i > 0) ws_bytes[i] += ws_bytes[i - 1];
	}
}

inline void alignWorkspace(unsigned int* ws_bytes, int N, int bytes)
{
	for(int i = 0; i < N; i++)
	{
		ws_bytes[i] += ws_bytes[i] % bytes;
		if(i > 0) ws_bytes[i] += ws_bytes[i - 1];
	}
}

inline void push_workspace(GPUBlasHandle& handle, unsigned int bytes)
{
	assert(bytes <= handle.workspace_bytes);
	handle.workspace = (void*)((GPUBlasHandle::WS_Byte*)handle.workspace + bytes);
	handle.workspace_bytes -= bytes;
	//printf("Push %d left\n", handle.workspace_bytes);
}

inline void pop_workspace(GPUBlasHandle& handle, unsigned int bytes)
{
	handle.workspace = (void*)((GPUBlasHandle::WS_Byte*)handle.workspace - bytes);
	handle.workspace_bytes += bytes;
	//printf("Pop %d left\n", handle.workspace_bytes);
}

template<class T>
struct TStridedMatrixArray {
	T* base_ptr;
	int num_matrices;
	int rows, cols, ld, stride;

	void init(T* ptr, int num_matrices, int rows, int cols, int ld, int stride) {
		this->base_ptr = ptr;
		this->num_matrices = num_matrices;
		this->rows = rows;
		this->cols = cols;
		this->ld = ld;
		this->stride = stride;
	}

	TStridedMatrixArray() {}

	TStridedMatrixArray(T* ptr, int num_matrices, int rows, int cols, int ld, int stride) {
		init(ptr, num_matrices, rows, cols, ld, stride);
	}

	TStridedMatrixArray(T* ptr, int num_matrices, int rows, int cols, int ld) {
		init(ptr, num_matrices, rows, cols, ld, ld * cols);
	}

	TStridedMatrixArray(T* ptr, int num_matrices, int rows, int cols) {
		init(ptr, num_matrices, rows, cols, rows, rows * cols);
	}

	__forceinline__ __host__ __device__
	T* getMatrix(int matrix_index)
	{
		return base_ptr + stride * matrix_index;
	}
};

typedef TStridedMatrixArray<float> StridedMatrixArrayf;
typedef TStridedMatrixArray<double> StridedMatrixArrayd;

template<class T>
inline void generateRandomMatrices(T* d_m, int rows, int cols, unsigned int seed, int num_ops, cudaStream_t stream = 0);

template<>
inline void generateRandomMatrices(float* d_m, int rows, int cols, unsigned int seed, int num_ops, cudaStream_t stream)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
	curandSetStream(gen, stream);

    curandGenerateNormal(gen, d_m, num_ops * rows * cols, 0, 1);

    curandDestroyGenerator(gen);
}

template<>
inline void generateRandomMatrices(double* d_m, int rows, int cols, unsigned int seed, int num_ops, cudaStream_t stream)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
	curandSetStream(gen, stream);

    curandGenerateNormalDouble(gen, d_m, num_ops * rows * cols, 0, 1);

    curandDestroyGenerator(gen);
}

template<class T>
inline __host__ __device__ T** advanceOperationPtr(T** array, int op_id, int stride) {return array + op_id;}
template<class T>
inline __host__ __device__ T* advanceOperationPtr(T* array, int op_id, int stride) {return array + op_id * stride;}

template<class T>
inline __host__ __device__ T* getOperationPtr(T* array, int op_id, int stride) { return array + op_id * stride; }
template<class T>
inline __host__ __device__ T* getOperationPtr(T** array, int op_id, int stride) { return array[op_id]; }

template<class T, class T_ptr>
inline __host__ __device__ T_ptr selectPointerData(T* strided_data, T** array_data);

template<>
inline __host__ __device__ float* selectPointerData<float, float*>(float* strided_data, float** array_data) { return strided_data; }
template<>
inline __host__ __device__ float** selectPointerData<float, float**>(float* strided_data, float** array_data) { return array_data; }

template<>
inline __host__ __device__ double* selectPointerData<double, double*>(double* strided_data, double** array_data) { return strided_data; }
template<>
inline __host__ __device__ double** selectPointerData<double, double**>(double* strided_data, double** array_data) { return array_data; }

template<class T> struct KBlasEpsilon;
template<> struct KBlasEpsilon<float>  {static const float  eps = 1.1920928955078125e-07; };
template<> struct KBlasEpsilon<double> {static const double eps = 2.2204460492503131e-16; };

#endif
