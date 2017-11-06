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
#include "gpu_util.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Some hmatrix related stuff
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class Real>
struct UnaryProjectionAoAAssign : public thrust::unary_function<int, Real*> 
{
    Real* basis_tree_data;
    int increment, tree_node_offset, coupling_start, coupling_end;
    int *leaf_node_index, *basis_tree_nodes;
    
    UnaryProjectionAoAAssign(int* leaf_node_index, int* basis_tree_nodes, Real* basis_tree_data, int increment, int coupling_start, int tree_node_offset) 
    {
        this->basis_tree_data = basis_tree_data;
        this->increment = increment;
        this->leaf_node_index = leaf_node_index;
        this->basis_tree_nodes = basis_tree_nodes;
        this->tree_node_offset = tree_node_offset;
        this->coupling_start = coupling_start;
    }
    
    __host__ __device__ 
    Real* operator()(const unsigned int& thread_id) const
    {
        int coupling_leaf_index = coupling_start + thread_id;
        int node_index = basis_tree_nodes[leaf_node_index[coupling_leaf_index]] - tree_node_offset;
        return basis_tree_data + node_index * increment;
    }
};

template<class Real>
struct UnaryRelativeRankFinder : public thrust::unary_function<int, int> 
{
    Real* s, eps;
    int current_rank, stride_s;
    
    UnaryRelativeRankFinder(Real* s, int stride_s, int current_rank, Real eps) 
    {
        this->s = s;
        this->current_rank = current_rank;
		this->stride_s = stride_s;
        this->eps = eps;
    }
    
    __host__ __device__ 
    int operator()(const unsigned int& thread_id) const
    {
        Real* s_node = s + thread_id * stride_s;
        Real rel_eps = s_node[0] * eps;
        int new_rank = 0;
			
        while(new_rank < current_rank && s_node[new_rank] > rel_eps)
            new_rank++;
        return new_rank;
    }
};

template<class Real>
struct UnaryAbsoluteRankFinder : public thrust::unary_function<int, int> 
{
    Real* s, eps;
    int current_rank, stride_s;
    
    UnaryAbsoluteRankFinder(Real* s, int stride_s, int current_rank, Real eps) 
    {
        this->s = s;
        this->current_rank = current_rank;
		this->stride_s = stride_s;
        this->eps = eps;
    }
    
    __host__ __device__ 
    int operator()(const unsigned int& thread_id) const
    {
        Real* s_node = s + thread_id * stride_s;
        int new_rank = 0;
        while(new_rank < current_rank && s_node[new_rank] > eps)
            new_rank++;
        return new_rank;
    }
};

template<class Real>
void getNewRankFromErrorThresholdT(Real* s, int stride_s, int* new_rank, int current_rank, Real eps, int relative, int num_ops)
{
    thrust::device_ptr<int> dev_data(new_rank);
    
    if(relative == 0)
    {
        thrust::transform(  
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(num_ops),
            dev_data,
            UnaryAbsoluteRankFinder<Real>(s, stride_s, current_rank, eps)
        );
    }
    else
    {
        thrust::transform(  
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(num_ops),
            dev_data,
            UnaryRelativeRankFinder<Real>(s, stride_s, current_rank, eps)
        );
    }
}

void getNewRankFromErrorThreshold(float* s, int stride_s, int* new_rank, int current_rank, float eps, int relative, int num_ops)
{
	return getNewRankFromErrorThresholdT<float>(s, stride_s, new_rank, current_rank, eps, relative, num_ops);
}

void getNewRankFromErrorThreshold(double* s, int stride_s, int* new_rank, int current_rank, double eps, int relative, int num_ops)
{
	return getNewRankFromErrorThresholdT<double>(s, stride_s, new_rank, current_rank, eps, relative, num_ops);
}

template<class Real>
void generateLevelProjectionPointersT(
    int* leaf_node_index, int* basis_tree_nodes, Real* basis_tree_data, Real** array_of_tree_nodes, 
    int increment, int coupling_start, int tree_node_offset, int num_arrays
)
{
    thrust::device_ptr<Real*> dev_data(array_of_tree_nodes);
    
    thrust::transform(  
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(num_arrays),
        dev_data,
        UnaryProjectionAoAAssign<Real>(
            leaf_node_index, basis_tree_nodes, basis_tree_data, 
            increment, coupling_start, tree_node_offset
        )
    );
}

void generateLevelProjectionPointers(
    int* leaf_node_index, int* basis_tree_nodes, float* basis_tree_data, float** array_of_tree_nodes, 
    int increment, int coupling_start, int tree_node_offset, int num_arrays
)
{
	generateLevelProjectionPointersT<float>(
		leaf_node_index, basis_tree_nodes, basis_tree_data, array_of_tree_nodes, 
		increment, coupling_start, tree_node_offset, num_arrays
	);
}

void generateLevelProjectionPointers(
    int* leaf_node_index, int* basis_tree_nodes, double* basis_tree_data, double** array_of_tree_nodes, 
    int increment, int coupling_start, int tree_node_offset, int num_arrays
)
{
	generateLevelProjectionPointersT<double>(
		leaf_node_index, basis_tree_nodes, basis_tree_data, array_of_tree_nodes, 
		increment, coupling_start, tree_node_offset, num_arrays
	);
}

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
