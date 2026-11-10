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

#define WARP_SIZE 32

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

template<class T>
__inline__ __device__
T blockAllReduceSum(T val, int warp_id, int warp_tid, T* temp_storage, int blocksize)
{
    const int warps = blocksize / WARP_SIZE;

    // First do a reduction within each warp
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);

    if(warps > 1)
    {
        if(warp_tid == 0) temp_storage[warp_id] = val;
        __syncthreads();

        T final_sum = 0;
        #pragma unroll
        for(int i = 0; i < warps; i++)
            final_sum += temp_storage[i];

        // Is this sync necessary? I think so since if we call the routine again
        // while one warp is already on a second reduction, the temp values will
        // be overwritten before another warp has a chance to tally the results
        __syncthreads();
        return final_sum;
    }
    else
        return val;
}

template<class T, int BLOCKSIZE>
__inline__ __device__
T blockAllReduceSum(T val, int warp_tid, int warp_id, volatile T* temp_storage)
{
    const int warps = BLOCKSIZE / WARP_SIZE;

    // First do a reduction within each warp
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);

    if(warps > 1)
    {
        if(warp_tid == 0) temp_storage[warp_id] = val;
        __syncthreads();

        T final_sum = 0;
        #pragma unroll
        for(int i = 0; i < warps; i++)
            final_sum += temp_storage[i];

        // Is this sync necessary? I think so since if we call the routine again
        // while one warp is already on a second reduction, the temp values will
        // be overwritten before another warp has a chance to tally the results
        __syncthreads();
        return final_sum;
    }
    else
        return val;
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
