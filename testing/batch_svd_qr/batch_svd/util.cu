#include <stdio.h>
#include <assert.h>
#include "util.h"
#include <vector>

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/********************/
/* CUDA ERROR CHECK */
/********************/
// --- Credit to http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) { exit(code); }
   }
}

void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }

/**************************/
/* CUSOLVE ERROR CHECKING */
/**************************/
static const char *_cudaGetErrorEnum(cusolverStatus_t error)
{
    switch (error)
    {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_SUCCESS";

        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_NOT_INITIALIZED";

        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";

        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";

        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";

        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";

        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";

        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

    }

    return "<unknown>";
}

inline void __cusolveSafeCall(cusolverStatus_t err, const char *file, const int line)
{
    if(CUSOLVER_STATUS_SUCCESS != err) 
	{
        fprintf(stderr, "CUSOLVE error in file '%s', line %d\n error %d: %s\nterminating!\n",__FILE__, __LINE__, err, _cudaGetErrorEnum(err));
        cudaDeviceReset(); 
		assert(0);
    }
}

void cusolveSafeCall(cusolverStatus_t err) { __cusolveSafeCall(err, __FILE__, __LINE__); }

// C = alpha * C + beta * A * B
template <typename Real, int transa, int transb>
inline void gemm(Real* c, int ldc, Real* a, int lda, Real* b, int ldb, int m, int n, int p, Real alpha, Real beta)
{
    if(m == 0 || n == 0 || p == 0) return;
    if(alpha == 0 && beta == 0)    return;
    
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            Real c_ij = 0;
            for(int k = 0; k < p; k++)
            {
                int index_a = (transa == 0 ? i + k * lda : k + i * lda);
                int index_b = (transb == 0 ? k + j * ldb : j + k * ldb);
                c_ij += a[index_a] * b[index_b];
            }
            int index_c = i + j * ldc;
            c[index_c] = alpha * c[index_c] + beta * c_ij;
        }
    }
}

void eye(std::vector<double>& identity, int m)
{
    identity.resize(m * m);
    std::fill(identity.begin(), identity.end(), 0);
    for(int i = 0; i < m; i++) 
        identity[i + i * m] = 1;
}

double frobeniusError(double* m1, double* m2, int m, int n)
{
    double err = 0;
    for(int i = 0; i < m * n; i++)
        err += (m1[i] - m2[i]) * (m1[i] - m2[i]);
    return sqrt(err);
}

double orthogonality(double* matrix, int m, int n)
{
    std::vector<double> prod(n * n, 0), identity;
    eye(identity, n);
    gemm<double, 1, 0>(&prod[0], n, matrix, m, matrix, m, n, n, m, 0, 1);
    return frobeniusError(&prod[0], &identity[0], n, n);
}

void eye(std::vector<float>& identity, int m)
{
    identity.resize(m * m);
    std::fill(identity.begin(), identity.end(), 0);
    for(int i = 0; i < m; i++) 
        identity[i + i * m] = 1;
}

float frobeniusError(float* m1, float* m2, int m, int n)
{
    float err = 0;
    for(int i = 0; i < m * n; i++)
        err += (m1[i] - m2[i]) * (m1[i] - m2[i]);
    return sqrt(err);
}

float orthogonality(float* matrix, int m, int n)
{
    std::vector<float> prod(n * n, 0), identity;
    eye(identity, n);
    gemm<float, 1, 0>(&prod[0], n, matrix, m, matrix, m, n, n, m, 0, 1);
    return frobeniusError(&prod[0], &identity[0], n, n);
}
