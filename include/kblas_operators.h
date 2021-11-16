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
 * @version 3.0.0
 * @author Ali Charara
 * @author Ahmad Abdelfattah
 * @date 2018-11-14
 **/

#ifndef _KBLAS_OPERATORS_
#define _KBLAS_OPERATORS_


#ifdef __HIP_PLATFORM_HCC__
#define SHFL(val, offset) __shfl(val, offset)
#else
#define SHFL(val, offset) __shfl_sync(0xffffffff, val, offset)
#endif

__host__ __device__ static __inline__ void make_zero_(float* x) {*x = 0.0;}
__host__ __device__ static __inline__ void make_zero_(double* x) {*x = 0.0;}
// __host__ __device__ static __inline__ void make_zero_(hipFloatComplex* a) {(*a).x = 0.0; (*a).y = 0.0;}
// __host__ __device__ static __inline__ void make_zero_(hipDoubleComplex* a) {(*a).x = 0.0; (*a).y = 0.0;}

__host__ __device__ static __inline__ void make_one_(float* x) {*x = 1.0;}
__host__ __device__ static __inline__ void make_one_(double* x) {*x = 1.0;}
// __host__ __device__ static __inline__ void make_one_(hipFloatComplex* a) {(*a).x = 1.0; (*a).y = 0.0;}
// __host__ __device__ static __inline__ void make_one_(hipDoubleComplex* a) {(*a).x = 1.0; (*a).y = 0.0;}

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
// __device__ static __inline__ hipFloatComplex FMA(hipFloatComplex a, hipFloatComplex b, hipFloatComplex c){
//   return make_hipFloatComplex( fmaf(a.x, b.x, fmaf(-a.y, b.y, c.x)), fmaf(a.y, b.x, fmaf(a.x, b.y, c.y)) );}
// __device__ static __inline__ hipDoubleComplex FMA(hipDoubleComplex a, hipDoubleComplex b, hipDoubleComplex
// c){  return make_hipDoubleComplex( fma(a.x, b.x, fma(-a.y, b.y, c.x)), fma(a.y, b.x, fma(a.x, b.y, c.y)) );}

__device__ static __inline__ float           conjugate(float x){return x;}
__device__ static __inline__ double          conjugate(double x){return x;}
// __device__ static __inline__ hipFloatComplex  conjugate(hipFloatComplex x){return hipConjf(x);}
// __device__ static __inline__ hipDoubleComplex conjugate(hipDoubleComplex x){return hipConj(x);}

__device__ static __inline__ float           conj_if(int _if_, float x){return x;}
__device__ static __inline__ double          conj_if(int _if_, double x){return x;}
// __device__ static __inline__ hipFloatComplex  conj_if(int _if_, hipFloatComplex x){if(_if_==0)return x; else return hipConjf(x);}
// __device__ static __inline__ hipDoubleComplex conj_if(int _if_, hipDoubleComplex x){if(_if_==0)return x; else return hipConj(x);}

__device__ static __inline__ float           make_real(float x){return x;}
__device__ static __inline__ double          make_real(double x){return x;}
// __device__ static __inline__ hipFloatComplex  make_real(hipFloatComplex a){return make_hipFloatComplex(a.x, 0.0);}
// __device__ static __inline__ hipDoubleComplex make_real(hipDoubleComplex a){return make_hipDoubleComplex(a.x, 0.0);}

__host__ __device__ static __inline__ float absolute(float x){return fabsf(x);}
__host__ __device__ static __inline__ double absolute(double x){return fabs(x);}
// TODO: absolute complex is not correct this way
// __host__ __device__ static __inline__ hipFloatComplex absolute(hipFloatComplex x){return hipCabsf(x);}
// __host__ __device__ static __inline__ hipDoubleComplex absolute(hipDoubleComplex x){return hipCabs(x);}

//==============================================================================================
#if defined(__HIPCC__)
#if (TARGET_SM >= 30)
__device__ __inline__ float shfl(float x, int lane, int ws = 32)
{
  return SHFL(x, lane);
}
__device__ __inline__ double shfl(double x, int lane, int ws = 32)
{
  // Split the double number into 2 32b registers.
  int lo = __double2loint(x), hi = __double2hiint(x);
  // Shuffle the two 32b registers.
  lo = SHFL(lo, lane);
  hi = SHFL(hi, lane);
  // Recreate the 64b number.
  return __hiloint2double(hi,lo);
}
// __device__ __inline__ hipComplex shfl(hipComplex x, int lane, int ws = 32)
// {
//   return make_hipFloatComplex( __shfl_sync(0xFFFFFFFF, x.x, lane, ws), __shfl_sync(0xFFFFFFFF, x.y, lane, ws) );
// }
// __device__ __inline__ hipDoubleComplex shfl(hipDoubleComplex x, int lane, int ws = 32)
// {
//   return make_hipDoubleComplex( shfl(x.x, lane, ws), shfl(x.y, lane, ws) );
// }
#endif//__CUDA_ARCH__
/*************************************************************/
// #if ( __CUDACC_VER_MAJOR__ >= 8 ) && ( !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600 )
// #include "sm_60_atomic_functions.h"
// #else
// /**
// Atomic add on double precision, as suggested by the CUDA programming Guide
// **/
// __device__ static __inline__ double atomicAdd(double* address, double val)
// {
//   unsigned long long int* address_as_ull =
//   (unsigned long long int*)address;
//   unsigned long long int old = *address_as_ull, assumed;
//   do {
//           assumed = old;
//           old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
//   } while (assumed != old);
//   return __longlong_as_double(old);
// }
// #endif//__CUDACC_VER_MAJOR__
// /**
// Atomic add for float complex ('C' precision)
// **/
// __device__ static __inline__ void atomicAdd(hipFloatComplex* address, hipFloatComplex val)
// {
//   atomicAdd( (float*) (&(*address).x) ,val.x);
//   atomicAdd( (float*) (&(*address).y) ,val.y);
// }

// /**
// Atomic add for double complex ('Z' precision)
// **/
// __device__ static __inline__ void atomicAdd(hipDoubleComplex* address, hipDoubleComplex val)
// {
//   atomicAdd( (double*) (&(*address).x) ,val.x);
//   atomicAdd( (double*) (&(*address).y) ,val.y);
// }
#endif//__HIPCC__

// /*************************************************************
//  *              hipDoubleComplex
//  */

// __host__ __device__ static __inline__ hipDoubleComplex
// operator-(const hipDoubleComplex &a)
// {
//     return make_hipDoubleComplex(-a.x, -a.y);
// }

// __host__ __device__ static __inline__ hipDoubleComplex
// operator+(const hipDoubleComplex a, const hipDoubleComplex b)
// {
//     return make_hipDoubleComplex(a.x + b.x, a.y + b.y);
// }

// __host__ __device__ static __inline__ hipDoubleComplex
// operator+(const hipDoubleComplex a, int b)
// {
//     return make_hipDoubleComplex(a.x + b, a.y);
// }

// __host__ __device__ static __inline__ void
// operator+=(hipDoubleComplex &a, const hipDoubleComplex b)
// {
//     a.x += b.x; a.y += b.y;
// }

// __host__ __device__ static __inline__ hipDoubleComplex
// operator-(const hipDoubleComplex a, const hipDoubleComplex b)
// {
//     return make_hipDoubleComplex(a.x - b.x, a.y - b.y);
// }

// __host__ __device__ static __inline__ void
// operator-=(hipDoubleComplex &a, const hipDoubleComplex b)
// {
//     a.x -= b.x; a.y -= b.y;
// }

// __host__ __device__ static __inline__ hipDoubleComplex
// operator*(const hipDoubleComplex a, const hipDoubleComplex b)
// {
//     return make_hipDoubleComplex(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
// }

// __host__ __device__ static __inline__ hipDoubleComplex
// operator*(const hipDoubleComplex a, const double s)
// {
//     return make_hipDoubleComplex(a.x * s, a.y * s);
// }

// __host__ __device__ static __inline__ hipDoubleComplex
// operator*(const double s, const hipDoubleComplex a)
// {
//     return make_hipDoubleComplex(a.x * s, a.y * s);
// }

// __host__ __device__ static __inline__ void
// operator*=(hipDoubleComplex &a, const hipDoubleComplex b)
// {
//   double tmp = a.y * b.x + a.x * b.y;
//   a.x = a.x * b.x - a.y * b.y;
//   a.y = tmp;
// }

// __host__ __device__ static __inline__ void
// operator*=(hipDoubleComplex &a, const double s)
// {
//     a.x *= s; a.y *= s;
// }

// __host__ __device__ static __inline__ bool
// operator==(const hipDoubleComplex a, const hipDoubleComplex b)
// {
//   return ((a.x == b.x) && (a.y == b.y));
// }

// __host__ __device__ static __inline__ bool
// operator!=(const hipDoubleComplex a, const hipDoubleComplex b)
// {
//   return ((a.x != b.x) || (a.y != b.y));
// }

// __host__ __device__ static __inline__ hipDoubleComplex
// operator/(const hipDoubleComplex a, const float b)
// {
//   //return (a * conjugate(b)) / (b * conjugate(b));
//   return make_hipDoubleComplex(a.x / b, a.y / b);
// }

// __host__ __device__ static __inline__ hipDoubleComplex
// operator/(const hipDoubleComplex a, const hipDoubleComplex b)
// {
//   //return (a * conjugate(b)) / (b * conjugate(b));
//   return make_hipDoubleComplex(a.x*b.x + a.y*b.y, a.y*b.x - a.x*b.y) / (b.x*b.x + b.y*b.y);
// }

// __host__ __device__ static __inline__ hipDoubleComplex sqrt(hipDoubleComplex x)
// {
//   double radius = hipCabs(x);
//   double cosA = x.x / radius;
//   hipDoubleComplex out;
//   out.x = sqrt(radius * (cosA + 1.0) / 2.0);
//   out.y = sqrt(radius * (1.0 - cosA) / 2.0);
//   // signbit should be false if x.y is negative
//   if (signbit(x.y))
//     out.y *= -1.0;

//   return out;
// }

// __host__ __device__ static __inline__ void
// operator/=(hipDoubleComplex &a, const float s)
// {
//     a.x /= s; a.y /= s;
// }

// __host__ __device__ static __inline__ void
// operator/=(hipDoubleComplex a, const hipDoubleComplex b)
// {
//   double d = (b.x*b.x + b.y*b.y);
//   a.x = (a.x*b.x + a.y*b.y) / d;
//   a.y = (a.y*b.x - a.x*b.y) / d;
// }

// /*************************************************************
//  *              hipFloatComplex
//  */

// __host__ __device__ static __inline__ hipFloatComplex
// operator-(const hipFloatComplex &a)
// {
//     return make_hipFloatComplex(-a.x, -a.y);
// }

// __host__ __device__ static __inline__ hipFloatComplex
// operator+(const hipFloatComplex a, const hipFloatComplex b)
// {
//     return make_hipFloatComplex(a.x + b.x, a.y + b.y);
// }

// __host__ __device__ static __inline__ hipFloatComplex
// operator+(const hipFloatComplex a, int b)
// {
//     return make_hipFloatComplex(a.x + b, a.y);
// }

// __host__ __device__ static __inline__ void
// operator+=(hipFloatComplex &a, const hipFloatComplex b)
// {
//     a.x += b.x; a.y += b.y;
// }

// __host__ __device__ static __inline__ hipFloatComplex
// operator-(const hipFloatComplex a, const hipFloatComplex b)
// {
//     return make_hipFloatComplex(a.x - b.x, a.y - b.y);
// }

// __host__ __device__ static __inline__ void
// operator-=(hipFloatComplex &a, const hipFloatComplex b)
// {
//     a.x -= b.x; a.y -= b.y;
// }

// __host__ __device__ static __inline__ hipFloatComplex
// operator*(const hipFloatComplex a, const hipFloatComplex b)
// {
//     return make_hipFloatComplex(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
// }

// __host__ __device__ static __inline__ hipFloatComplex
// operator*(const hipFloatComplex a, const float s)
// {
//     return make_hipFloatComplex(a.x * s, a.y * s);
// }

// __host__ __device__ static __inline__ hipFloatComplex
// operator*(const float s, const hipFloatComplex a)
// {
//     return make_hipFloatComplex(a.x * s, a.y * s);
// }

// __host__ __device__ static __inline__ void
// operator*=(hipFloatComplex &a, const hipFloatComplex b)
// {
//   float tmp = a.y * b.x + a.x * b.y;
//   a.x = a.x * b.x - a.y * b.y;
//   a.y = tmp;
// }

// __host__ __device__ static __inline__ void
// operator*=(hipFloatComplex &a, const float s)
// {
//     a.x *= s; a.y *= s;
// }

// __host__ __device__ static __inline__ bool
// operator==(const hipFloatComplex a, const hipFloatComplex b)
// {
//   return ((a.x == b.x) && (a.y == b.y));
// }

// __host__ __device__ static __inline__ bool
// operator!=(const hipFloatComplex a, const hipFloatComplex b)
// {
//   return ((a.x != b.x) || (a.y != b.y));
// }

// __host__ __device__ static __inline__ hipFloatComplex
// operator/(const hipFloatComplex a, const float b)
// {
//     //return (a * conjugate(b)) / (b * conjugate(b));
//     return make_hipFloatComplex(a.x / b, a.y / b);
// }

// __host__ __device__ static __inline__ hipFloatComplex
// operator/(const hipFloatComplex a, const hipFloatComplex b)
// {
//     //return (a * conjugate(b)) / (b * conjugate(b));
//     return make_hipFloatComplex(a.x*b.x + a.y*b.y, a.y*b.x - a.x*b.y) / (b.x*b.x + b.y*b.y);
// }

// __host__ __device__ static __inline__ hipFloatComplex sqrt(hipFloatComplex x)
// {
//   float radius = hipCabsf(x);
//   float cosA = x.x / radius;
//   hipFloatComplex out;
//   out.x = sqrt(radius * (cosA + 1.0) / 2.0);
//   out.y = sqrt(radius * (1.0 - cosA) / 2.0);
//   // signbit should be false if x.y is negative
//   if (signbit(x.y))
//     out.y *= -1.0;

//   return out;
// }
// __host__ __device__ static __inline__ void
// operator/=(hipFloatComplex &a, const float s)
// {
//     a.x /= s; a.y /= s;
// }

// __host__ __device__ static __inline__ void
// operator/=(hipFloatComplex a, const hipFloatComplex b)
// {
//   float d = (b.x*b.x + b.y*b.y);
//   a.x = (a.x*b.x + a.y*b.y) / d;
//   a.y = (a.y*b.x - a.x*b.y) / d;
// }

#endif  // _KBLAS_OPERATORS_
