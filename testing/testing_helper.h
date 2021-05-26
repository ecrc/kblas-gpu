/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/testing_helper.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Wajih Halim Boukaram
 * @author Ali Charara
 * @date 2020-12-10
 **/

#ifndef __TESTING_HELPER_H__
#define __TESTING_HELPER_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <kblas.h>
#include <cusolverDn.h>

#ifdef USE_MAGMA
#include <magma_v2.h>
#endif
#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>		//TODO: if MKL not set we need to use other libs
#include <lapacke.h>
#endif
#ifdef USE_OPENMP
#include <omp.h>
#endif
#include <kblas.h>

#ifdef __cplusplus
extern "C" {
#endif

inline int iDivUp( int a, int b ) { return (a % b != 0) ? (a / b + 1) : (a / b); }

#define kmin(a,b) ((a)>(b)?(b):(a))
#define kmax(a,b) ((a)<(b)?(b):(a))
int kblas_roundup(int x, int y);

////////////////////////////////////////////////////////////
// Error checking
////////////////////////////////////////////////////////////
#define check_error(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line);

#define check_cublas_error(ans) { gpuCublasAssert((ans), __FILE__, __LINE__); }
void gpuCublasAssert(cublasStatus_t code, const char *file, int line);

#define check_kblas_error(ans) { gpuKblasAssert((ans), __FILE__, __LINE__); }
void gpuKblasAssert(int code, const char *file, int line);

#define check_cusolver_error(ans) { gpuCusolverAssert((ans), __FILE__, __LINE__); }
void gpuCusolverAssert(cusolverStatus_t code, const char *file, int line);

////////////////////////////////////////////////////////////
// Timers and related stuff
////////////////////////////////////////////////////////////
double gettime(void);

struct GPU_Timer;
typedef GPU_Timer* GPU_Timer_t;

GPU_Timer_t newGPU_Timer(cudaStream_t stream);
void deleteGPU_Timer(GPU_Timer_t timer);
void gpuTimerTic(GPU_Timer_t timer);
void gpuTimerRecordEnd(GPU_Timer_t timer);
double gpuTimerToc(GPU_Timer_t timer);

void avg_and_stdev(double* values, int num_vals, double* avg, double* std_dev, int warmup);

////////////////////////////////////////////////////////////
// Generate array of pointers from a strided array
////////////////////////////////////////////////////////////
void generateDArrayOfPointers(double* original_array, double** array_of_arrays, int stride, int num_arrays, cudaStream_t stream);
void generateSArrayOfPointers(float* original_array, float** array_of_arrays, int stride, int num_arrays, cudaStream_t stream);
void generateDArrayOfPointersHost(double* original_array, double** array_of_arrays, int stride, int num_arrays);
void generateSArrayOfPointersHost(float* original_array, float** array_of_arrays, int stride, int num_arrays);

// Reductions 
int getMaxElement(int* a, size_t elements, cudaStream_t stream);
int getMinElement(int* a, size_t elements, cudaStream_t stream);

////////////////////////////////////////////////////////////
// Allocations
////////////////////////////////////////////////////////////
#define TESTING_MALLOC_CPU( ptr, T, size)                       \
{			\
  if ( (ptr = (T*) malloc( (size)*sizeof( T ) ) ) == NULL) {    \
    fprintf( stderr, "!!!! malloc_cpu failed for: %s\n", #ptr ); \
    exit(-1);                                                   \
  } \
}
#define TESTING_MALLOC_DEV( ptr, T, size) check_error( cudaMalloc( (void**)&ptr, (size)*sizeof(T) ) )
#define TESTING_MALLOC_PIN( ptr, T, size) check_error( cudaHostAlloc ( (void**)&ptr, (size)*sizeof( T ), cudaHostAllocPortable  ))

#define TESTING_FREE_CPU(ptr)	{ if( (ptr) ) free( (ptr) ); }
#define TESTING_FREE_DEV(ptr)	check_error( cudaFree( (ptr) ) );

////////////////////////////////////////////////////////////
// Data generation
////////////////////////////////////////////////////////////
void generateDrandom(double* random_data, long num_elements, int num_ops);
void generateSrandom(float* random_data, long num_elements, int num_ops);
void srand_matrix(long rows, long cols, float* A, long LDA);
void drand_matrix(long rows, long cols, double* A, long LDA);
void crand_matrix(long rows, long cols, cuFloatComplex* A, long LDA);
void zrand_matrix(long rows, long cols, cuDoubleComplex* A, long LDA);

void smatrix_make_hpd(int N, float* A, int lda);
void dmatrix_make_hpd(int N, double* A, int lda);
void cmatrix_make_hpd(int N, cuFloatComplex* A, int lda);
void zmatrix_make_hpd(int N, cuDoubleComplex* A, int lda);

void generateRandDimensions(int minDim, int maxDim, int* randDims, int seed, int num_ops);
void fillIntArray(int* array_vals, int value, int num_elements);

void fillGPUIntArray(int* array_vals, int value, int num_elements, cudaStream_t stream);
void copyGPUPointerArray(void** originalPtrs, void** copyPtrs, int num_ptrs, cudaStream_t stream);

// set cond = 0 to use exp decay
void generateDrandomMatrices(
	double* M_strided, int stride_M, double* svals_strided, int stride_S, int rows, int cols,
	double cond, double exp_decay, int seed, int num_ops, int threads
);
void generateSrandomMatrices(
	float* M_strided, int stride_M, float* svals_strided, int stride_S, int rows, int cols,
	float cond, float exp_decay, int seed, int num_ops, int threads
);

// Mode = 0: use provided singualr values
// Mode = 1: sets singular values to random numbers in the range (1/cond, 1) 
// Mode = 2: generate singular values such that S(i) = exp(-exp_decay * i); 
//	     	 where exp_decay is a random value between exp_decay_min and exp_decay_max
void generateDrandomMatricesArray(
	double** M_ptrs, int* ldm_array, double** svals_ptrs, int *rows_array, int *cols_array,
	int mode, double cond, double exp_decay_min, double exp_decay_max, int seed, int num_ops, int threads
);
void generateSrandomMatricesArray(
	float** M_ptrs, int* ldm_array, float** svals_ptrs, int *rows_array, int *cols_array,
	int mode, float cond, float exp_decay_min, float exp_decay_max, int seed, int num_ops, int threads
);

void generateSsingular_values(float** svals_ptrs, int rank, float min_sval, float max_sval, int seed, int batchCount);
void generateDsingular_values(double** svals_ptrs, int rank, double min_sval, double max_sval, int seed, int batchCount);

////////////////////////////////////////////////////////////
// Result checking
////////////////////////////////////////////////////////////
// Vectors
float sget_max_error(float* ref, float *res, int n, int inc);
double dget_max_error(double* ref, double *res, int n, int inc);
float cget_max_error(cuFloatComplex* ref, cuFloatComplex *res, int n, int inc);
double zget_max_error(cuDoubleComplex* ref, cuDoubleComplex *res, int n, int inc);
// Matrices
float sget_max_error_matrix(float* ref, float *res, long m, long n, long lda);
double dget_max_error_matrix(double* ref, double *res, long m, long n, long lda);
float cget_max_error_matrix(cuFloatComplex* ref, cuFloatComplex *res, long m, long n, long lda);
double zget_max_error_matrix(cuDoubleComplex* ref, cuDoubleComplex *res, long m, long n, long lda);

double dget_max_error_matrix_uplo(double* ref, double *res, long m, long n, long lda, char uplo);
// float sget_max_error_matrix_uplo(float* ref, float *res, long m, long n, long lda, char uplo);
// double dget_max_error_matrix_symm(double* ref, double *res, long m, long n, long lda, char uplo);

#define printMatrix(m, n, A, lda, out) { \
  for(int r = 0; r < (m); r++){ \
    for(int c = 0; c < (n); c++){ \
      fprintf((out), "%.7e  ", (A)[r + c * (lda)]); \
    } \
    fprintf((out), "\n"); \
  } \
  fprintf((out), "\n"); \
}

////////////////////////////////////////////////////////////
// Command line parser
////////////////////////////////////////////////////////////
#define MAX_NTEST 1000

typedef struct kblas_opts
{
	// matrix size
	int ntest;
	int msize[ MAX_NTEST ];
	int nsize[ MAX_NTEST ];
	int ksize[ MAX_NTEST ];
	int rsize[ MAX_NTEST ];

	// scalars
	int devices[MAX_NGPUS];
	int nstream;
	int ngpu;
	int niter;
	int nruns;
	double      tolerance;
	int check;
	int verbose;
	int nb;
	int db;
	int custom;
	int warmup;
	int time;
	int lapack;
	int magma;
	int svd;
	int cuda;
	int nonUniform;
	//int bd[KBLAS_BACKDOORS];
	int batchCount;
	int strided;
	int btest, batch[MAX_NTEST];
	int rtest, rank[MAX_NTEST];
	int omp_numthreads;
  char LR;//Low rank format
  int version;

	// lapack flags
	char uplo;
	char transA;
	char transB;
	char side;
	char diag;
} kblas_opts;

int parse_opts(int argc, char** argv, kblas_opts *opts);

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus

#define kmin(a,b) ((a)>(b)?(b):(a))
#define kmax(a,b) ((a)<(b)?(b):(a))

#include "kblas_operators.h"
template<class T, class T_complex>
T get_magnitude(T_complex a){ return sqrt(a.x * a.x + a.y * a.y); }
template<class T>
T get_magnitude(T ax, T ay){ return sqrt(ax * ax + ay * ay); }

template<typename T>
bool kblas_laisnan(T val1, T val2){
  return val1 != val2;
}

template<typename T>
bool kblas_isnan(T val){
  return kblas_laisnan(val,val);
}

inline float Xabs(float a){return fabs(a);}
inline double Xabs(double a){return fabs(a);}
inline float Xabs(cuFloatComplex a){return get_magnitude<float,cuFloatComplex>(a);}
inline double Xabs(cuDoubleComplex a){return get_magnitude<double,cuDoubleComplex>(a);}

template<class T>
void matrix_make_hpd(int N, T* A, int lda, T diag)
{
  ssize_t ldas = ssize_t(lda);
  for(ssize_t i = 0; i < ssize_t(N); i++)
  {
    A[i + i * ldas] = A[i + i * ldas] + diag;
    for(ssize_t j = 0; j < i; j++){
      A[j + i*ldas] = A[i + j*ldas];
    }
  }
}

template<typename T, typename R>
R kblas_lange(char type, int M, int N, T* arr, int lda){
  R value = make_zero<R>();
  R temp;
  for(int j = 0; j < N; j++){
    for(int i = 0; i < M; i++){
      temp = Xabs(arr[i + j * lda]);
      if( kblas_isnan(temp) ){
        value = temp;
        printf("NAN encountered (%d,%d)\n", i, j);
        return value;
      }
      if( value < temp)
        value = temp;
    }
  }
  return value;
}

template<typename T>
void kblasXaxpy (int n, T alpha, const T *x, int incx, T *y, int incy){
  int ix = 0, iy = 0;
  if(incx < 0) ix = 1 - n * incx;
  if(incy < 0) iy = 1 - n * incy;
  for(int i = 0; i < n; i++, ix+=incx, iy+=incy){
    y[iy] += alpha * x[ix];
  }
}

template<class T>
void hilbertMatrix(int m, int n, T* A, int lda, T scal){
  for(int r = 0; r < m; r++){
    for(int c = 0; c < n; c++){
      A[r + c*lda] = T(scal)/T(r+c+1);
    }
  }
}

#endif //__cplusplus

#ifdef DBG_MSG
#define ECHO_I(_val) printf("%s(%d) ", #_val, (_val));fflush( stdout )
#define ECHO_f(_val) printf("%s(%e) ", #_val, (_val));fflush( stdout )
#define ECHO_p(_val) printf("%s(%p) ", #_val, (_val));fflush( stdout )
#define ECHO_LN printf("\n");fflush( stdout )
#else
#define ECHO_I(_val)
#define ECHO_f(_val)
#define ECHO_p(_val)
#define ECHO_LN
#endif

#endif // __TESTING_HELPER_H__
