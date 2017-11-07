#ifndef __UTIL_H__
#define __UTIL_H__

#include <cuda.h>
#include <cusolverDn.h>

int iDivUp(int, int);
void cusolveSafeCall(cusolverStatus_t);
double orthogonality(double* matrix, int m, int n);
float orthogonality(float* matrix, int m, int n);

#endif