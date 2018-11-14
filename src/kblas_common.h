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
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#include "kblas_error.h"

//==============================================================================================
bool REG_SIZE(int n);
int CLOSEST_REG_SIZE(int n);

#define kmin(a,b) ((a)>(b)?(b):(a))
#define kmax(a,b) ((a)<(b)?(b):(a))
extern "C"
int kblas_roundup(int x, int y);


long kblas_roundup_l(long x, long y);
size_t kblas_roundup_s(size_t x, size_t y);

//==============================================================================================
int iset_value_1( int *output_array, int input,
                  long batchCount, cudaStream_t cuda_stream);

int iset_value_2( int *output_array1, int input1,
                  int *output_array2, int input2,
                  long batchCount, cudaStream_t cuda_stream);
int iset_value_4( int *output_array1, int input1,
                  int *output_array2, int input2,
                  int *output_array3, int input3,
                  int *output_array4, int input4,
                  long batchCount, cudaStream_t cuda_stream);
int iset_value_5( int *output_array1, int input1,
                  int *output_array2, int input2,
                  int *output_array3, int input3,
                  int *output_array4, int input4,
                  int *output_array5, int input5,
                  long batchCount, cudaStream_t cuda_stream);

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
