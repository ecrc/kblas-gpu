/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/kblas_gpu_util.ch

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Wajih Halim Boukaram
 * @date 2020-12-10
 **/

#ifndef __KBLAS_GPU_UTIL_H__
#define __KBLAS_GPU_UTIL_H__

#include <cuda_runtime_api.h>

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION <= 9000)
#define kblas_sync_warp() do { } while(0)
#else
#define kblas_sync_warp() __syncwarp()
#endif

#define WARP_SIZE 32
#define MAX_GRID_DIM_Z 65535

template<int BS>
inline __device__ void kblas_sync_threads()
{
    if(BS <= WARP_SIZE) kblas_sync_warp();
    else                  __syncthreads();
}

template<class T>
inline __host__ __device__ T** advanceOperationPtr(T** array, int op_id, int stride) {return array + op_id;}
template<class T>
inline __host__ __device__ T* advanceOperationPtr(T* array, int op_id, int stride) {return array + op_id * stride;}

template<class T>
inline __host__ __device__ T* getOperationPtr(T* array, int op_id, int stride) { return array + op_id * stride; }
template<class T>
inline __host__ __device__ T* getOperationPtr(T** array, int op_id, int stride) { return array[op_id]; }

template<class T>
inline __host__ __device__ T getOperationVal(T val, int op_id) { return val; }
template<class T>
inline __host__ __device__ T getOperationVal(T* array, int op_id) { return array[op_id]; }

inline __host__ __device__ int getOperationDim(int* dim_array, int op_id) { return dim_array[op_id]; }
inline __host__ __device__ int getOperationDim(int dim, int op_id) { return dim; }
inline __host__ __device__ int* advanceOperationDim(int* dim_array, int op_id) { return dim_array + op_id; }
inline __host__ __device__ int advanceOperationDim(int dim, int op_id) { return dim; }

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
#if __cplusplus < 201103L
template<> struct KBlasEpsilon<float>  {static const float  eps = 1.1920928955078125e-07; };
template<> struct KBlasEpsilon<double> {static const double eps = 2.2204460492503131e-16; };
#else
template<> struct KBlasEpsilon<float>  {static constexpr float  eps = 1.1920928955078125e-07; };
template<> struct KBlasEpsilon<double> {static constexpr double eps = 2.2204460492503131e-16; };
#endif

__device__ __host__
inline int iDivUp( int a, int b ) { return (a % b != 0) ? (a / b + 1) : (a / b); }

template<class T, int BLOCK_SIZE>
__inline__ __device__
T blockAllReduceSum(T val, int warp_tid, int warp_id, volatile T* temp_storage)
{
    const int warps = BLOCK_SIZE / WARP_SIZE;

    // First do a reduction within each warp
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);

    if(warps > 1)
    {
        if(warp_tid == 0) temp_storage[warp_id] = val;
        kblas_sync_threads<BLOCK_SIZE>();

        T final_sum = 0;
        #pragma unroll
        for(int i = 0; i < warps; i++)
            final_sum += temp_storage[i];
        return final_sum;
    }
    else
        return val;
}

// Partial tree reduction used in the multireduction gemv 
template<class T, int BLOCK_SIZE, int PANEL_WIDTH>
__inline__ __device__
void partialTreeReduceSum(volatile T* smem_w, int threadId)
{
	if(BLOCK_SIZE >= 544                    ) { if(threadId < 512) smem_w[0] += smem_w[16 * (WARP_SIZE + 1)]; kblas_sync_threads<1024>(); }
	if(BLOCK_SIZE >= 288                    ) { if(threadId < 256) smem_w[0] += smem_w[ 8 * (WARP_SIZE + 1)]; kblas_sync_threads< 512>(); }
	if(BLOCK_SIZE >= 160                    ) { if(threadId < 128) smem_w[0] += smem_w[ 4 * (WARP_SIZE + 1)]; kblas_sync_threads< 256>(); }
    if(BLOCK_SIZE >= 96                     ) { if(threadId <  64) smem_w[0] += smem_w[ 2 * (WARP_SIZE + 1)]; kblas_sync_threads< 128>(); }
    if(BLOCK_SIZE >= 64                     ) { if(threadId <  32) smem_w[0] += smem_w[WARP_SIZE + 1];        kblas_sync_threads<  64>(); }
    if(BLOCK_SIZE >= 32 && PANEL_WIDTH <= 16) { if(threadId <  16) smem_w[0] += smem_w[16];                   kblas_sync_threads<  32>(); }
    if(BLOCK_SIZE >= 16 && PANEL_WIDTH <= 8 ) { if(threadId <   8) smem_w[0] += smem_w[8 ];                   kblas_sync_threads<  16>(); }
    if(BLOCK_SIZE >=  8 && PANEL_WIDTH <= 4 ) { if(threadId <   4) smem_w[0] += smem_w[4 ];                   kblas_sync_threads<   8>(); }
    if(BLOCK_SIZE >=  4 && PANEL_WIDTH <= 2 ) { if(threadId <   2) smem_w[0] += smem_w[2 ];                   kblas_sync_threads<   4>(); }
    if(BLOCK_SIZE >=  2 && PANEL_WIDTH <= 1 ) { if(threadId <   1) smem_w[0] += smem_w[1 ];                   kblas_sync_threads<   2>(); }
}

// Get the smallest index i of the largest value in x
template<class T>
__inline__ __device__ 
void arg_max(int tid, T* x, int* i, int n)
{
    if(n > 512) { if (tid < 512 && tid + 512 < n) { if(x[tid] < x[tid + 512] || (x[tid] == x[tid + 512] && i[tid + 512] < i[tid])) { i[tid] = i[tid + 512]; x[tid] = x[tid + 512]; } }  __syncthreads(); }
    if(n > 256) { if (tid < 256 && tid + 256 < n) { if(x[tid] < x[tid + 256] || (x[tid] == x[tid + 256] && i[tid + 256] < i[tid])) { i[tid] = i[tid + 256]; x[tid] = x[tid + 256]; } }  __syncthreads(); }
    if(n > 128) { if (tid < 128 && tid + 128 < n) { if(x[tid] < x[tid + 128] || (x[tid] == x[tid + 128] && i[tid + 128] < i[tid])) { i[tid] = i[tid + 128]; x[tid] = x[tid + 128]; } }  __syncthreads(); }
    if(n >  64) { if (tid <  64 && tid +  64 < n) { if(x[tid] < x[tid +  64] || (x[tid] == x[tid +  64] && i[tid +  64] < i[tid])) { i[tid] = i[tid +  64]; x[tid] = x[tid +  64]; } }  __syncthreads(); }
    if(n >  32) { if (tid <  32 && tid +  32 < n) { if(x[tid] < x[tid +  32] || (x[tid] == x[tid +  32] && i[tid +  32] < i[tid])) { i[tid] = i[tid +  32]; x[tid] = x[tid +  32]; } }  __syncthreads(); }
    if(n >  16) { if (tid <  16 && tid +  16 < n) { if(x[tid] < x[tid +  16] || (x[tid] == x[tid +  16] && i[tid +  16] < i[tid])) { i[tid] = i[tid +  16]; x[tid] = x[tid +  16]; } }  __syncthreads(); }
    if(n >   8) { if (tid <   8 && tid +   8 < n) { if(x[tid] < x[tid +   8] || (x[tid] == x[tid +   8] && i[tid +   8] < i[tid])) { i[tid] = i[tid +   8]; x[tid] = x[tid +   8]; } }  __syncthreads(); }
    if(n >   4) { if (tid <   4 && tid +   4 < n) { if(x[tid] < x[tid +   4] || (x[tid] == x[tid +   4] && i[tid +   4] < i[tid])) { i[tid] = i[tid +   4]; x[tid] = x[tid +   4]; } }  __syncthreads(); }
    if(n >   2) { if (tid <   2 && tid +   2 < n) { if(x[tid] < x[tid +   2] || (x[tid] == x[tid +   2] && i[tid +   2] < i[tid])) { i[tid] = i[tid +   2]; x[tid] = x[tid +   2]; } }  __syncthreads(); }
    if(n >   1) { if (tid <   1 && tid +   1 < n) { if(x[tid] < x[tid +   1] || (x[tid] == x[tid +   1] && i[tid +   1] < i[tid])) { i[tid] = i[tid +   1]; x[tid] = x[tid +   1]; } }  __syncthreads(); }
}

// Same as above, but for warps - can be used with multiple warps 
template<class T>
__inline__ __device__ 
void arg_max_warp(int tid, T* x, int* i, int n)
{
    if(n >  16) { if (tid <  16 && tid +  16 < n) { if(x[tid] < x[tid +  16] || (x[tid] == x[tid +  16] && i[tid +  16] < i[tid])) { i[tid] = i[tid +  16]; x[tid] = x[tid +  16]; } }  kblas_sync_warp(); }
    if(n >   8) { if (tid <   8 && tid +   8 < n) { if(x[tid] < x[tid +   8] || (x[tid] == x[tid +   8] && i[tid +   8] < i[tid])) { i[tid] = i[tid +   8]; x[tid] = x[tid +   8]; } }  kblas_sync_warp(); }
    if(n >   4) { if (tid <   4 && tid +   4 < n) { if(x[tid] < x[tid +   4] || (x[tid] == x[tid +   4] && i[tid +   4] < i[tid])) { i[tid] = i[tid +   4]; x[tid] = x[tid +   4]; } }  kblas_sync_warp(); }
    if(n >   2) { if (tid <   2 && tid +   2 < n) { if(x[tid] < x[tid +   2] || (x[tid] == x[tid +   2] && i[tid +   2] < i[tid])) { i[tid] = i[tid +   2]; x[tid] = x[tid +   2]; } }  kblas_sync_warp(); }
    if(n >   1) { if (tid <   1 && tid +   1 < n) { if(x[tid] < x[tid +   1] || (x[tid] == x[tid +   1] && i[tid +   1] < i[tid])) { i[tid] = i[tid +   1]; x[tid] = x[tid +   1]; } }  kblas_sync_warp(); }
}


template<class T, int columns, int i, int BLOCK_SIZE, int PANEL_WIDTH>
inline __device__ void multreduction_gemv(T matrix_row[columns], T v, int threadId, int warp_tid, int warp_id, volatile T* smem)
{
    int thread_smem_offset = threadId * PANEL_WIDTH + (threadId * PANEL_WIDTH) / WARP_SIZE;
    volatile T* smem_w = smem + threadId + warp_id;
    const int warps = BLOCK_SIZE / WARP_SIZE;

    // Fill in shared memory
    #pragma unroll
    for(int j = 0; j < PANEL_WIDTH; j++)
        smem[j + thread_smem_offset] = v * matrix_row[i + j];

    kblas_sync_threads<BLOCK_SIZE>();

    // Now do the partial reduction starting with the serial part
    #pragma unroll
    for(int j = 1; j < PANEL_WIDTH; j++)
        smem_w[0] += smem_w[warps * (WARP_SIZE + 1) * j];

	// Pad with zeros if the block size is not a power of two
	if(warps != 1 && warps != 2 && warps != 4 && warps != 8 && warps != 16 && warps != 32)
		smem_w[warps * (WARP_SIZE + 1)] = 0;

    kblas_sync_threads<BLOCK_SIZE>();

    // Now the tree partial reduction
    partialTreeReduceSum<T, BLOCK_SIZE, PANEL_WIDTH>(smem_w, threadId);
    
    kblas_sync_threads<BLOCK_SIZE>();
}

template<class T>
__inline__ __device__
T warpReduceSum(T val)
{
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

template<class T>
__inline__ __device__
T warpAllReduceSum(T val)
{
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

template<class T>
__inline__ __device__
T warpAllReduceMax(T val)
{
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
	{
		T t = __shfl_xor_sync(0xFFFFFFFF, val, mask);
        if(val < t) val = t;
	}
    return val;
}

#endif // __KBLAS_GPU_UTIL_H__
