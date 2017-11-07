#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_scan.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <iostream>

#include "thrust_wrappers.h"
#include "kblas_gpu_util.ch"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Some array utility functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void exclusiveScan(int* array, int num_entries, int* result, int init, cudaStream_t stream)
{
    thrust::exclusive_scan(thrust::cuda::par.on(stream), array, array + num_entries, result, init);
}

void inclusiveScan(int* array, int num_entries, int* result, cudaStream_t stream)
{
    thrust::inclusive_scan(thrust::cuda::par.on(stream), array, array + num_entries, result);
}

template<class T>
T getMaxElementT(T* a, int elements, cudaStream_t stream)
{
    thrust::device_ptr<T> dev_a(a);
    return *(thrust::max_element(thrust::cuda::par.on(stream), dev_a, dev_a + elements));
}

int getMaxElement(int* a, int elements, cudaStream_t stream)
{
    return getMaxElementT<int>(a, elements, stream);
}

float getMaxElement(float* a, int elements, cudaStream_t stream)
{
    return getMaxElementT<float>(a, elements, stream);
}

double getMaxElement(double* a, int elements, cudaStream_t stream)
{
    return getMaxElementT<double>(a, elements, stream);
}

template<class T>
T reduceSumT(T* a, int elements, cudaStream_t stream)
{
    thrust::device_ptr<T> dev_a(a);
    return thrust::reduce(thrust::cuda::par.on(stream), dev_a, dev_a + elements);
}

double reduceSum(double* a, int elements, cudaStream_t stream)
{
	return reduceSumT<double>(a, elements, stream);
}

float reduceSum(float* a, int elements, cudaStream_t stream)
{
	return reduceSumT<float>(a, elements, stream);
}

template<class Real>
void fillArrayT(Real* array, int num_entries, Real val, cudaStream_t stream)
{
	thrust::device_ptr<Real> dev_start(array);
    thrust::device_ptr<Real> dev_end(array + num_entries);
	thrust::fill(thrust::cuda::par.on(stream), dev_start, dev_end, val);
}

void fillArray(float* array, int num_entries, float val, cudaStream_t stream)
{
	fillArrayT<float>(array, num_entries, val, stream);
}

void fillArray(double* array, int num_entries, double val, cudaStream_t stream)
{
	fillArrayT<double>(array, num_entries, val, stream);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generating array of pointers from either a strided array or another array of pointers
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T, class T_ptr>
struct UnaryAoAAssign : public thrust::unary_function<int, T*> 
{
    T_ptr original_array;
    int stride, offset;
    
    UnaryAoAAssign(T_ptr original_array, int stride, int offset) 
    {
        this->original_array = original_array;
        this->stride = stride;
        this->offset = offset;
    }
    
    __host__ __device__ 
    T* operator()(const unsigned int& thread_id) const
    {
        return getOperationPtr<T>(original_array, thread_id, stride) + offset;
    }
};

template<class T, class T_ptr>
void generateArrayOfPointersT(T_ptr original_array, T** array_of_arrays, int stride, int offset, int num_arrays, cudaStream_t stream)
{
    thrust::device_ptr<T*> dev_data(array_of_arrays);
    
    thrust::transform(  
		thrust::cuda::par.on(stream), 
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(num_arrays),
        dev_data,
        UnaryAoAAssign<T, T_ptr>(original_array, stride, offset)
    );
}

void generateArrayOfPointers(double* original_array, double** array_of_arrays, int stride, int offset, int num_arrays, cudaStream_t stream)
{ generateArrayOfPointersT<double, double*>(original_array, array_of_arrays, stride, offset, num_arrays, stream); }

void generateArrayOfPointers(double* original_array, double** array_of_arrays, int stride, int num_arrays, cudaStream_t stream)
{ generateArrayOfPointersT<double, double*>(original_array, array_of_arrays, stride, 0, num_arrays, stream); }

void generateArrayOfPointers(double** original_array, double** array_of_arrays, int stride, int offset, int num_arrays, cudaStream_t stream)
{ generateArrayOfPointersT<double, double**>(original_array, array_of_arrays, stride, offset, num_arrays, stream); }

void generateArrayOfPointers(double** original_array, double** array_of_arrays, int stride, int num_arrays, cudaStream_t stream)
{ generateArrayOfPointersT<double, double**>(original_array, array_of_arrays, stride, 0, num_arrays, stream); }

void generateArrayOfPointers(float* original_array, float** array_of_arrays, int stride, int offset, int num_arrays, cudaStream_t stream)
{ generateArrayOfPointersT<float, float*>(original_array, array_of_arrays, stride, offset, num_arrays, stream); }

void generateArrayOfPointers(float* original_array, float** array_of_arrays, int stride, int num_arrays, cudaStream_t stream)
{ generateArrayOfPointersT<float, float*>(original_array, array_of_arrays, stride, 0, num_arrays, stream); }

void generateArrayOfPointers(float** original_array, float** array_of_arrays, int stride, int offset, int num_arrays, cudaStream_t stream)
{ generateArrayOfPointersT<float, float**>(original_array, array_of_arrays, stride, offset, num_arrays, stream); }

void generateArrayOfPointers(float** original_array, float** array_of_arrays, int stride, int num_arrays, cudaStream_t stream)
{ generateArrayOfPointersT<float, float**>(original_array, array_of_arrays, stride, 0, num_arrays, stream); }

void generateArrayOfPointers(int* original_array, int** array_of_arrays, int stride, int offset, int num_arrays, cudaStream_t stream)
{ generateArrayOfPointersT<int, int*>(original_array, array_of_arrays, stride, offset, num_arrays, stream); }

void generateArrayOfPointers(int* original_array, int** array_of_arrays, int stride, int num_arrays, cudaStream_t stream)
{ generateArrayOfPointersT<int, int*>(original_array, array_of_arrays, stride, 0, num_arrays, stream); }

void generateArrayOfPointers(int** original_array, int** array_of_arrays, int stride, int offset, int num_arrays, cudaStream_t stream)
{ generateArrayOfPointersT<int, int**>(original_array, array_of_arrays, stride, offset, num_arrays, stream); }

void generateArrayOfPointers(int** original_array, int** array_of_arrays, int stride, int num_arrays, cudaStream_t stream)
{ generateArrayOfPointersT<int, int**>(original_array, array_of_arrays, stride, 0, num_arrays, stream); }
