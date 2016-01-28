 /**
 -- (C) Copyright 2013 King Abdullah University of Science and Technology
  Authors:
  Ahmad Abdelfattah (ahmad.ahmad@kaust.edu.sa)
  David Keyes (david.keyes@kaust.edu.sa)
  Hatem Ltaief (hatem.ltaief@kaust.edu.sa)

  Redistribution  and  use  in  source and binary forms, with or without
  modification,  are  permitted  provided  that the following conditions
  are met:

  * Redistributions  of  source  code  must  retain  the above copyright
    notice,  this  list  of  conditions  and  the  following  disclaimer.
  * Redistributions  in  binary  form must reproduce the above copyright
    notice,  this list of conditions and the following disclaimer in the
    documentation  and/or other materials provided with the distribution.
  * Neither  the  name of the King Abdullah University of Science and
    Technology nor the names of its contributors may be used to endorse 
    or promote products derived from this software without specific prior 
    written permission.

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

#ifndef _OPERATORS_
#define _OPERATORS_

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

__device__ static __inline__ float      conjugate(float x){return x;}
__device__ static __inline__ double conjugate(double x){return x;}
__device__ static __inline__ cuFloatComplex 	conjugate(cuFloatComplex x){return cuConjf(x);}
__device__ static __inline__ cuDoubleComplex 	conjugate(cuDoubleComplex x){return cuConj(x);}

__device__ static __inline__ float 				conj_if(int _if_, float x){return x;}
__device__ static __inline__ double 			conj_if(int _if_, double x){return x;}
__device__ static __inline__ cuFloatComplex 	conj_if(int _if_, cuFloatComplex x){if(_if_==0)return x; else return cuConjf(x);}
__device__ static __inline__ cuDoubleComplex 	conj_if(int _if_, cuDoubleComplex x){if(_if_==0)return x; else return cuConj(x);}

__device__ static __inline__ float 	make_real(float x){return x;}
__device__ static __inline__ double make_real(double x){return x;}
__device__ static __inline__ cuFloatComplex 	make_real(cuFloatComplex a){return make_cuFloatComplex(a.x, 0.0);}
__device__ static __inline__ cuDoubleComplex 	make_real(cuDoubleComplex a){return make_cuDoubleComplex(a.x, 0.0);}
/*************************************************************/
#if defined(__CUDACC__)
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
#endif

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

#endif  // _OPERATORS_


