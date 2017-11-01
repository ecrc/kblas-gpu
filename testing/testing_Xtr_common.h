#ifndef _TESTING_TR_COMMON_
#define _TESTING_TR_COMMON_


#include "testing_utils.h"
#include "operators.h"

//==============================================================================================
#define FMULS_GEMM(m_, n_, k_) ((m_) * (n_) * (k_))
#define FADDS_GEMM(m_, n_, k_) ((m_) * (n_) * (k_))

double FLOPS_GEMM(float p, char side, int m, int n, int k){
  return FMULS_GEMM((double)(m), (double)(n), (double)(k)) + FADDS_GEMM((double)(m), (double)(n), (double)(k));
}
double FLOPS_GEMM(double p, char side, int m, int n, int k){
  return FMULS_GEMM((double)(m), (double)(n), (double)(k)) + FADDS_GEMM((double)(m), (double)(n), (double)(k));
}
double FLOPS_GEMM(cuFloatComplex p, char side, int m, int n, int k){
  return 6. * FMULS_GEMM((double)(m), (double)(n), (double)(k)) + 2.0 * FADDS_GEMM((double)(m), (double)(n), (double)(k));
}
double FLOPS_GEMM(cuDoubleComplex p, char side, int m, int n, int k){
  return 6. * FMULS_GEMM((double)(m), (double)(n), (double)(k)) + 2.0 * FADDS_GEMM((double)(m), (double)(n), (double)(k));
}
//==============================================================================================
#define FMULS_TRMM_2(m_, n_) (0.5 * (n_) * (m_) * ((m_)+1))
#define FADDS_TRMM_2(m_, n_) (0.5 * (n_) * (m_) * ((m_)-1))
#define FMULS_TRMM(side_, m_, n_) ( ( (side_) == KBLAS_Left ) ? FMULS_TRMM_2((m_), (n_)) : FMULS_TRMM_2((n_), (m_)) )
#define FADDS_TRMM(side_, m_, n_) ( ( (side_) == KBLAS_Left ) ? FADDS_TRMM_2((m_), (n_)) : FADDS_TRMM_2((n_), (m_)) )


double FLOPS_TRMM(float p, char side, int m, int n){
  return FMULS_TRMM(side, (double)(m), (double)(n)) + FADDS_TRMM(side, (double)(m), (double)(n));
}
double FLOPS_TRMM(double p, char side, int m, int n){
  return FMULS_TRMM(side, (double)(m), (double)(n)) + FADDS_TRMM(side, (double)(m), (double)(n));
}
double FLOPS_TRMM(cuFloatComplex p, char side, int m, int n){
  return 6. * FMULS_TRMM(side, (double)(m), (double)(n)) + 2. * FADDS_TRMM(side, (double)(m), (double)(n));
}
double FLOPS_TRMM(cuDoubleComplex p, char side, int m, int n){
  return 6. * FMULS_TRMM(side, (double)(m), (double)(n)) + 2. * FADDS_TRMM(side, (double)(m), (double)(n));
}

//==============================================================================================
#define FMULS_TRSM_2(m_, n_) (0.5 * (n_) * (m_) * ((m_)+1))
#define FADDS_TRSM_2(m_, n_) (0.5 * (n_) * (m_) * ((m_)-1))
#define FMULS_TRSM(side_, m_, n_) ( ( (side_) == KBLAS_Left ) ? FMULS_TRSM_2((m_), (n_)) : FMULS_TRSM_2((n_), (m_)) )
#define FADDS_TRSM(side_, m_, n_) ( ( (side_) == KBLAS_Left ) ? FADDS_TRSM_2((m_), (n_)) : FADDS_TRSM_2((n_), (m_)) )


double FLOPS_TRSM(float p, char side, int m, int n){
  return FMULS_TRSM(side, (double)(m), (double)(n)) + FADDS_TRSM(side, (double)(m), (double)(n));
}
double FLOPS_TRSM(double p, char side, int m, int n){
  return FMULS_TRSM(side, (double)(m), (double)(n)) + FADDS_TRSM(side, (double)(m), (double)(n));
}
double FLOPS_TRSM(cuFloatComplex p, char side, int m, int n){
  return 6. * FMULS_TRSM(side, (double)(m), (double)(n)) + 2. * FADDS_TRSM(side, (double)(m), (double)(n));
}
double FLOPS_TRSM(cuDoubleComplex p, char side, int m, int n){
  return 6. * FMULS_TRSM(side, (double)(m), (double)(n)) + 2. * FADDS_TRSM(side, (double)(m), (double)(n));
}

//==============================================================================================
cublasStatus_t kblasXtrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const float *alpha,
                          const float *A, int lda,
                                float *B, int ldb){
  return kblasStrmm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
cublasStatus_t kblasXtrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const double *alpha,
                          const double *A, int lda,
                                double *B, int ldb){
  return kblasDtrmm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
cublasStatus_t kblasXtrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const cuComplex *alpha,
                          const cuComplex *A, int lda,
                                cuComplex *B, int ldb){
  return kblasCtrmm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
cublasStatus_t kblasXtrmm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const cuDoubleComplex *alpha,
                          const cuDoubleComplex *A, int lda,
                                cuDoubleComplex *B, int ldb){
  return kblasZtrmm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}

cublasStatus_t cublasXtrmm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const float *alpha,
                           const float *A, int lda,
                                 float *B, int ldb);
cublasStatus_t cublasXtrmm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t      diag,
                           int m, int n,
                           const double *alpha,
                           const double *A, int lda,
                                 double *B, int ldb);
cublasStatus_t cublasXtrmm (cublasHandle_t handle,
                            cublasSideMode_t side, cublasFillMode_t uplo,
                            cublasOperation_t trans, cublasDiagType_t diag,
                            int m, int n,
                            const cuComplex *alpha,
                            const cuComplex *A, int lda,
                                  cuComplex *B, int ldb);
cublasStatus_t cublasXtrmm (cublasHandle_t handle,
                            cublasSideMode_t side, cublasFillMode_t uplo,
                            cublasOperation_t trans, cublasDiagType_t diag,
                            int m, int n,
                            const cuDoubleComplex *alpha,
                            const cuDoubleComplex *A, int lda,
                                  cuDoubleComplex *B, int ldb);




//==============================================================================================

cublasStatus_t kblasXtrsm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const float *alpha,
                          const float *A, int lda,
                                float *B, int ldb){
  return kblasStrsm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
cublasStatus_t kblasXtrsm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const double *alpha,
                          const double *A, int lda,
                                double *B, int ldb){
  return kblasDtrsm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
cublasStatus_t kblasXtrsm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const cuComplex *alpha,
                          const cuComplex *A, int lda,
                                cuComplex *B, int ldb){
  return kblasCtrsm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}
cublasStatus_t kblasXtrsm(cublasHandle_t handle,
                          cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag,
                          int m, int n,
                          const cuDoubleComplex *alpha,
                          const cuDoubleComplex *A, int lda,
                                cuDoubleComplex *B, int ldb){
  return kblasZtrsm(handle,
                    side, uplo, trans, diag,
                    m, n,
                    alpha, A, lda,
                           B, ldb);
}

cublasStatus_t cublasXtrsm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const float *alpha,
                           const float *A, int lda,
                                 float *B, int ldb);
cublasStatus_t cublasXtrsm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t      diag,
                           int m, int n,
                           const double *alpha,
                           const double *A, int lda,
                                 double *B, int ldb);
cublasStatus_t cublasXtrsm (cublasHandle_t handle,
                            cublasSideMode_t side, cublasFillMode_t uplo,
                            cublasOperation_t trans, cublasDiagType_t diag,
                            int m, int n,
                            const cuComplex *alpha,
                            const cuComplex *A, int lda,
                                  cuComplex *B, int ldb);
cublasStatus_t cublasXtrsm (cublasHandle_t handle,
                            cublasSideMode_t side, cublasFillMode_t uplo,
                            cublasOperation_t trans, cublasDiagType_t diag,
                            int m, int n,
                            const cuDoubleComplex *alpha,
                            const cuDoubleComplex *A, int lda,
                                  cuDoubleComplex *B, int ldb);

cublasStatus_t cublasXgemm( cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const float *alpha, const float *A, int lda,
                            const float *B, int ldb,
                            const float *beta,        float *C, int ldc);
cublasStatus_t cublasXgemm( cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const double *alpha, const double *A, int lda,
                            const double *B, int ldb,
                            const double *beta,        double *C, int ldc);
cublasStatus_t cublasXgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const cuComplex *alpha, const cuComplex *A, int lda,
                           const cuComplex *B, int ldb,
                           const cuComplex *beta,        cuComplex *C, int ldc);
cublasStatus_t cublasXgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *B, int ldb,
                           const cuDoubleComplex *beta,        cuDoubleComplex *C, int ldc);


template<typename T>
void kblasXaxpy (int n, T alpha, const T *x, int incx, T *y, int incy){
  int ix = 0, iy = 0;
  if(incx < 0) ix = 1 - n * incx;
  if(incy < 0) iy = 1 - n * incy;
  for(int i = 0; i < n; i++, ix+=incx, iy+=incy){
    y[iy] += alpha * x[ix];
  }
}
/*void cublasXaxpy (int n, float alpha, const float *x, int incx, float *y, int incy){
  cublasSaxpy (n, alpha, x, incx, y, incy);
}
void cublasXaxpy (int n, double alpha, const double *x, int incx, double *y, int incy){
  cublasDaxpy (n, alpha, x, incx, y, incy);
}
void cublasXaxpy (int n, cuComplex alpha, const cuComplex *x, int incx, cuComplex *y, int incy){
  cublasCaxpy (n, alpha, x, incx, y, incy);
}
void cublasXaxpy (int n, cuDoubleComplex alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy){
  cublasZaxpy (n, alpha, x, incx, y, incy);
}*/

//==============================================================================================

template<typename T>
bool kblas_laisnan(T val1, T val2){
  return val1 != val2;
}

template<typename T>
bool kblas_isnan(T val){
  return kblas_laisnan(val,val);
}

float Xabs(float a){return fabs(a);}
double Xabs(double a){return fabs(a);}
float Xabs(cuFloatComplex a){return cget_magnitude(a);}
double Xabs(cuDoubleComplex a){return zget_magnitude(a);}

template<typename T, typename R>
R kblas_lange(char type, int M, int N, T* arr, int lda){
  R value = make_zero<R>();
  R temp;
  for(int j = 0; j < N; j++){
    for(int i = 0; i < N; i++){
      temp = Xabs(arr[i + j * lda]);
      if( kblas_isnan(temp) || value < temp)
        value = temp;
    }
  }
  return value;
}


//==============================================================================================
const char* cublasGetErrorString( cublasStatus_t error );
int _kblas_error( cudaError_t err, const char* func, const char* file, int line );
int _kblas_error( cublasStatus_t err, const char* func, const char* file, int line );

#define check_error( err ) \
{ \
  if(!_kblas_error( (err), __func__, __FILE__, __LINE__ )) \
    exit(1); \
}
 //   return 0;\

cudaEvent_t start, stop;
void start_timing(cudaStream_t curStream){
  check_error( cudaEventRecord(start, curStream) );
  check_error( cudaGetLastError() );
}
float get_elapsed_time(cudaStream_t curStream){
  check_error( cudaEventRecord(stop, curStream) );
  check_error( cudaGetLastError() );
  check_error( cudaEventSynchronize(stop) );
  check_error( cudaGetLastError() );
  float time = 0;
  check_error( cudaEventElapsedTime(&time, start, stop) );
  check_error( cudaGetLastError() );
  return time;
}

#define TESTING_MALLOC_CPU( ptr, T, size)                       \
  if ( (ptr = (T*) malloc( (size)*sizeof( T ) ) ) == NULL) {    \
    fprintf( stderr, "!!!! malloc_cpu failed for: %s\n", #ptr ); \
    exit(-1);                                                   \
  }

#define TESTING_MALLOC_DEV( ptr, T, size)                       \
{ \
  cudaError_t err = cudaMalloc( (void**)&ptr, (size)*sizeof(T) ); \
  if(!_kblas_error( (err), __func__, __FILE__, __LINE__ )) \
    return 0; \
}
//  check_error( cudaMalloc( (void**)&ptr, (size)*sizeof(T) ) )

#define TESTING_MALLOC_PIN( ptr, T, size)                       \
  check_error( cudaMallocHost((void**)&ptr, (size)*sizeof( T ) ))
  

#endif
