#ifndef __KBLAS_VBATCH_THRUST_UTIL_H__
#define __KBLAS_VBATCH_THRUST_UTIL_H__
#include <thrust/tuple.h>

struct dim_product_functor
{
	__host__ __device__
	size_t operator()(const thrust::tuple<int, int>& t) const { 
		return (size_t)(thrust::get<0>(t) * thrust::get<1>(t));
	}
};

template<class T>
struct PointerScanAssign
{
	size_t *offsets;
	T **ptrs, *base_ptr;
	
	PointerScanAssign(size_t *offsets, T **ptrs, T *base_ptr)
	{
		this->offsets = offsets;
		this->ptrs = ptrs;
		this->base_ptr = base_ptr;
	}
	
	inline __host__ __device__ 
	void operator()(size_t i)
	{
		ptrs[i] = base_ptr + offsets[i];
	}
};

#endif 
