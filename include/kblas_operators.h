/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file include/kblas_operators.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ali Charara
 * @author Ahmad Abdelfattah
 * @date 2020-12-10
 **/

#ifndef _KBLAS_OPERATORS_
#define _KBLAS_OPERATORS_

/*************************************************************
/**
	zero and conjugate functions
**/
__host__ __device__ static __inline__ void make_zero_(float* x) {*x = 0.0;}
__host__ __device__ static __inline__ void make_zero_(double* x) {*x = 0.0;}
__host__ __device__ static __inline__ void make_zero_(cuFloatComplex* a) {(*a).x = 0.0; (*a).y = 0.0;}
__host__ __device__ static __inline__ void make_zero_(cuDoubleComplex* a) {(*a).x = 0.0; (*a).y = 0.0;}

__host__ __device__ static __inline__ void make_one_(float* x) {*x = 1.0;}
__host__ __device__ static __inline__ void make_one_(double* x) {*x = 1.0;}
__host__ __device__ static __inline__ void make_one_(cuFloatComplex* a) {(*a).x = 1.0; (*a).y = 0.0;}
__host__ __device__ static __inline__ void make_one_(cuDoubleComplex* a) {(*a).x = 1.0; (*a).y = 0.0;}

template<class T>
__host__ __device__ static __inline__ T make_zero()
{
  T a;
  make_zero_(&a);
  return a;
}

template<class T>
__host__ __device__ T make_one()
{
  T a;
  make_one_(&a);
  return a;
}

__device__ static __inline__ float FMA(float a, float b, float c){return fmaf(a,b,c);}
__device__ static __inline__ double FMA(double a, double b, double c){return fma(a,b,c);}
__device__ static __inline__ cuFloatComplex FMA(cuFloatComplex a, cuFloatComplex b, cuFloatComplex c){
  return make_cuFloatComplex( fmaf(a.x, b.x, fmaf(-a.y, b.y, c.x)), fmaf(a.y, b.x, fmaf(a.x, b.y, c.y)) );}
__device__ static __inline__ cuDoubleComplex FMA(cuDoubleComplex a, cuDoubleComplex b, cuDoubleComplex
c){  return make_cuDoubleComplex( fma(a.x, b.x, fma(-a.y, b.y, c.x)), fma(a.y, b.x, fma(a.x, b.y, c.y)) );}

__device__ static __inline__ float           conjugate(float x){return x;}
__device__ static __inline__ double          conjugate(double x){return x;}
__device__ static __inline__ cuFloatComplex  conjugate(cuFloatComplex x){return cuConjf(x);}
__device__ static __inline__ cuDoubleComplex conjugate(cuDoubleComplex x){return cuConj(x);}

__device__ static __inline__ float           conj_if(int _if_, float x){return x;}
__device__ static __inline__ double          conj_if(int _if_, double x){return x;}
__device__ static __inline__ cuFloatComplex  conj_if(int _if_, cuFloatComplex x){if(_if_==0)return x; else return cuConjf(x);}
__device__ static __inline__ cuDoubleComplex conj_if(int _if_, cuDoubleComplex x){if(_if_==0)return x; else return cuConj(x);}

__device__ static __inline__ float           make_real(float x){return x;}
__device__ static __inline__ double          make_real(double x){return x;}
__device__ static __inline__ cuFloatComplex  make_real(cuFloatComplex a){return make_cuFloatComplex(a.x, 0.0);}
__device__ static __inline__ cuDoubleComplex make_real(cuDoubleComplex a){return make_cuDoubleComplex(a.x, 0.0);}

__host__ __device__ static __inline__ float absolute(float x){return fabsf(x);}
__host__ __device__ static __inline__ double absolute(double x){return fabs(x);}
// TODO: absolute complex is not correct this way
// __host__ __device__ static __inline__ cuFloatComplex absolute(cuFloatComplex x){return cuCabsf(x);}
// __host__ __device__ static __inline__ cuDoubleComplex absolute(cuDoubleComplex x){return cuCabs(x);}

//==============================================================================================
#if defined(__CUDACC__)
#if (TARGET_SM >= 30)
__device__ __inline__ float shfl(float x, int lane, int ws = 32)
{
  return __shfl_sync(0xFFFFFFFF, x, lane, ws);
}
__device__ __inline__ double shfl(double x, int lane, int ws = 32)
{
  // Split the double number into 2 32b registers.
  int lo = __double2loint(x), hi = __double2hiint(x);
  // Shuffle the two 32b registers.
  lo = __shfl_sync(0xFFFFFFFF, lo, lane, ws);
  hi = __shfl_sync(0xFFFFFFFF, hi, lane, ws);
  // Recreate the 64b number.
  return __hiloint2double(hi,lo);
}
__device__ __inline__ cuComplex shfl(cuComplex x, int lane, int ws = 32)
{
  return make_cuFloatComplex( __shfl_sync(0xFFFFFFFF, x.x, lane, ws), __shfl_sync(0xFFFFFFFF, x.y, lane, ws) );
}
__device__ __inline__ cuDoubleComplex shfl(cuDoubleComplex x, int lane, int ws = 32)
{
  return make_cuDoubleComplex( shfl(x.x, lane, ws), shfl(x.y, lane, ws) );
}
#endif//__CUDA_ARCH__
/*************************************************************/
#if ( __CUDACC_VER_MAJOR__ >= 8 ) && ( !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600 )
#include "sm_60_atomic_functions.h"
#else
/**
Atomic add on double precision, as suggested by the CUDA programming Guide
**/
__device__ static __inline__ double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull =
  (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
          assumed = old;
          old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif//__CUDACC_VER_MAJOR__
/**
Atomic add for float complex ('C' precision)
**/
__device__ static __inline__ void atomicAdd(cuFloatComplex* address, cuFloatComplex val)
{
  atomicAdd( (float*) (&(*address).x) ,val.x);
  atomicAdd( (float*) (&(*address).y) ,val.y);
}

/**
Atomic add for double complex ('Z' precision)
**/
__device__ static __inline__ void atomicAdd(cuDoubleComplex* address, cuDoubleComplex val)
{
  atomicAdd( (double*) (&(*address).x) ,val.x);
  atomicAdd( (double*) (&(*address).y) ,val.y);
}
#endif//__CUDACC__

/*************************************************************
 *              cuDoubleComplex
 */

__host__ __device__ static __inline__ cuDoubleComplex
operator-(const cuDoubleComplex &a)
{
    return make_cuDoubleComplex(-a.x, -a.y);
}

__host__ __device__ static __inline__ cuDoubleComplex
operator+(const cuDoubleComplex a, const cuDoubleComplex b)
{
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

__host__ __device__ static __inline__ cuDoubleComplex
operator+(const cuDoubleComplex a, int b)
{
    return make_cuDoubleComplex(a.x + b, a.y);
}

__host__ __device__ static __inline__ void
operator+=(cuDoubleComplex &a, const cuDoubleComplex b)
{
    a.x += b.x; a.y += b.y;
}

__host__ __device__ static __inline__ cuDoubleComplex
operator-(const cuDoubleComplex a, const cuDoubleComplex b)
{
    return make_cuDoubleComplex(a.x - b.x, a.y - b.y);
}

__host__ __device__ static __inline__ void
operator-=(cuDoubleComplex &a, const cuDoubleComplex b)
{
    a.x -= b.x; a.y -= b.y;
}

__host__ __device__ static __inline__ cuDoubleComplex
operator*(const cuDoubleComplex a, const cuDoubleComplex b)
{
    return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}

__host__ __device__ static __inline__ cuDoubleComplex
operator*(const cuDoubleComplex a, const double s)
{
    return make_cuDoubleComplex(a.x * s, a.y * s);
}

__host__ __device__ static __inline__ cuDoubleComplex
operator*(const double s, const cuDoubleComplex a)
{
    return make_cuDoubleComplex(a.x * s, a.y * s);
}

__host__ __device__ static __inline__ void
operator*=(cuDoubleComplex &a, const cuDoubleComplex b)
{
  double tmp = a.y * b.x + a.x * b.y;
  a.x = a.x * b.x - a.y * b.y;
  a.y = tmp;
}

__host__ __device__ static __inline__ void
operator*=(cuDoubleComplex &a, const double s)
{
    a.x *= s; a.y *= s;
}

__host__ __device__ static __inline__ bool
operator==(const cuDoubleComplex a, const cuDoubleComplex b)
{
  return ((a.x == b.x) && (a.y == b.y));
}

__host__ __device__ static __inline__ bool
operator!=(const cuDoubleComplex a, const cuDoubleComplex b)
{
  return ((a.x != b.x) || (a.y != b.y));
}

__host__ __device__ static __inline__ cuDoubleComplex
operator/(const cuDoubleComplex a, const float b)
{
  //return (a * conjugate(b)) / (b * conjugate(b));
  return make_cuDoubleComplex(a.x / b, a.y / b);
}

__host__ __device__ static __inline__ cuDoubleComplex
operator/(const cuDoubleComplex a, const cuDoubleComplex b)
{
  //return (a * conjugate(b)) / (b * conjugate(b));
  return make_cuDoubleComplex(a.x*b.x + a.y*b.y, a.y*b.x - a.x*b.y) / (b.x*b.x + b.y*b.y);
}

__host__ __device__ static __inline__ cuDoubleComplex sqrt(cuDoubleComplex x)
{
  double radius = cuCabs(x);
  double cosA = x.x / radius;
  cuDoubleComplex out;
  out.x = sqrt(radius * (cosA + 1.0) / 2.0);
  out.y = sqrt(radius * (1.0 - cosA) / 2.0);
  // signbit should be false if x.y is negative
  if (signbit(x.y))
    out.y *= -1.0;

  return out;
}

__host__ __device__ static __inline__ void
operator/=(cuDoubleComplex &a, const float s)
{
    a.x /= s; a.y /= s;
}

__host__ __device__ static __inline__ void
operator/=(cuDoubleComplex a, const cuDoubleComplex b)
{
  double d = (b.x*b.x + b.y*b.y);
  a.x = (a.x*b.x + a.y*b.y) / d;
  a.y = (a.y*b.x - a.x*b.y) / d;
}

/*************************************************************
 *              cuFloatComplex
 */

__host__ __device__ static __inline__ cuFloatComplex
operator-(const cuFloatComplex &a)
{
    return make_cuFloatComplex(-a.x, -a.y);
}

__host__ __device__ static __inline__ cuFloatComplex
operator+(const cuFloatComplex a, const cuFloatComplex b)
{
    return make_cuFloatComplex(a.x + b.x, a.y + b.y);
}

__host__ __device__ static __inline__ cuFloatComplex
operator+(const cuFloatComplex a, int b)
{
    return make_cuFloatComplex(a.x + b, a.y);
}

__host__ __device__ static __inline__ void
operator+=(cuFloatComplex &a, const cuFloatComplex b)
{
    a.x += b.x; a.y += b.y;
}

__host__ __device__ static __inline__ cuFloatComplex
operator-(const cuFloatComplex a, const cuFloatComplex b)
{
    return make_cuFloatComplex(a.x - b.x, a.y - b.y);
}

__host__ __device__ static __inline__ void
operator-=(cuFloatComplex &a, const cuFloatComplex b)
{
    a.x -= b.x; a.y -= b.y;
}

__host__ __device__ static __inline__ cuFloatComplex
operator*(const cuFloatComplex a, const cuFloatComplex b)
{
    return make_cuFloatComplex(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}

__host__ __device__ static __inline__ cuFloatComplex
operator*(const cuFloatComplex a, const float s)
{
    return make_cuFloatComplex(a.x * s, a.y * s);
}

__host__ __device__ static __inline__ cuFloatComplex
operator*(const float s, const cuFloatComplex a)
{
    return make_cuFloatComplex(a.x * s, a.y * s);
}

__host__ __device__ static __inline__ void
operator*=(cuFloatComplex &a, const cuFloatComplex b)
{
  float tmp = a.y * b.x + a.x * b.y;
  a.x = a.x * b.x - a.y * b.y;
  a.y = tmp;
}

__host__ __device__ static __inline__ void
operator*=(cuFloatComplex &a, const float s)
{
    a.x *= s; a.y *= s;
}

__host__ __device__ static __inline__ bool
operator==(const cuFloatComplex a, const cuFloatComplex b)
{
  return ((a.x == b.x) && (a.y == b.y));
}

__host__ __device__ static __inline__ bool
operator!=(const cuFloatComplex a, const cuFloatComplex b)
{
  return ((a.x != b.x) || (a.y != b.y));
}

__host__ __device__ static __inline__ cuFloatComplex
operator/(const cuFloatComplex a, const float b)
{
    //return (a * conjugate(b)) / (b * conjugate(b));
    return make_cuFloatComplex(a.x / b, a.y / b);
}

__host__ __device__ static __inline__ cuFloatComplex
operator/(const cuFloatComplex a, const cuFloatComplex b)
{
    //return (a * conjugate(b)) / (b * conjugate(b));
    return make_cuFloatComplex(a.x*b.x + a.y*b.y, a.y*b.x - a.x*b.y) / (b.x*b.x + b.y*b.y);
}

__host__ __device__ static __inline__ cuFloatComplex sqrt(cuFloatComplex x)
{
  float radius = cuCabsf(x);
  float cosA = x.x / radius;
  cuFloatComplex out;
  out.x = sqrt(radius * (cosA + 1.0) / 2.0);
  out.y = sqrt(radius * (1.0 - cosA) / 2.0);
  // signbit should be false if x.y is negative
  if (signbit(x.y))
    out.y *= -1.0;

  return out;
}
__host__ __device__ static __inline__ void
operator/=(cuFloatComplex &a, const float s)
{
    a.x /= s; a.y /= s;
}

__host__ __device__ static __inline__ void
operator/=(cuFloatComplex a, const cuFloatComplex b)
{
  float d = (b.x*b.x + b.y*b.y);
  a.x = (a.x*b.x + a.y*b.y) / d;
  a.y = (a.y*b.x - a.x*b.y) / d;
}

#endif  // _KBLAS_OPERATORS_
