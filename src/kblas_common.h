/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/kblas_common.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 2.0.0
 * @author Ali Charara
 * @date 2017-11-13
 **/

#include "kblas_error.h"

//==============================================================================================
bool REG_SIZE(int n);
int CLOSEST_REG_SIZE(int n);

//==============================================================================================

#define kmin(a,b) ((a)>(b)?(b):(a))
#define kmax(a,b) ((a)<(b)?(b):(a))

//==============================================================================================
#if 1
/*void cublasXgemm(char transa, char transb, int m, int n, int k,
                 float alpha, const float *A, int lda,
                              const float *B, int ldb,
                 float beta,        float *C, int ldc );
void cublasXgemm(char transa, char transb, int m, int n, int k,
                 double alpha, const double *A, int lda,
                               const double *B, int ldb,
                 double beta,        double *C, int ldc);
void cublasXgemm(char transa, char transb, int m, int n, int k,
                 cuComplex alpha, const cuComplex *A, int lda,
                                  const cuComplex *B, int ldb,
                 cuComplex beta,        cuComplex *C, int ldc);
void cublasXgemm(char transa, char transb, int m, int n, int k,
                 cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                                        const cuDoubleComplex *B, int ldb,
                 cuDoubleComplex beta,        cuDoubleComplex *C, int ldc);*/

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
#endif
