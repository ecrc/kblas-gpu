/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_l3/l3_common.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef _TESTING_L3_COMMON_
#define _TESTING_L3_COMMON_



//==============================================================================================
hipblasStatus_t kblasXtrmm(hipblasHandle_t handle,
                          hipblasSideMode_t side, hipblasFillMode_t uplo,
                          hipblasOperation_t trans, hipblasDiagType_t diag,
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
hipblasStatus_t kblasXtrmm(hipblasHandle_t handle,
                          hipblasSideMode_t side, hipblasFillMode_t uplo,
                          hipblasOperation_t trans, hipblasDiagType_t diag,
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
// hipblasStatus_t kblasXtrmm(hipblasHandle_t handle,
//                           hipblasSideMode_t side, hipblasFillMode_t uplo,
//                           hipblasOperation_t trans, hipblasDiagType_t diag,
//                           int m, int n,
//                           const hipComplex *alpha,
//                           const hipComplex *A, int lda,
//                                 hipComplex *B, int ldb){
//   return kblasCtrmm(handle,
//                     side, uplo, trans, diag,
//                     m, n,
//                     alpha, A, lda,
//                            B, ldb);
// }
// hipblasStatus_t kblasXtrmm(hipblasHandle_t handle,
//                           hipblasSideMode_t side, hipblasFillMode_t uplo,
//                           hipblasOperation_t trans, hipblasDiagType_t diag,
//                           int m, int n,
//                           const hipDoubleComplex *alpha,
//                           const hipDoubleComplex *A, int lda,
//                                 hipDoubleComplex *B, int ldb){
//   return kblasZtrmm(handle,
//                     side, uplo, trans, diag,
//                     m, n,
//                     alpha, A, lda,
//                            B, ldb);
// }

hipblasStatus_t cublasXtrmm(hipblasHandle_t handle,
                           hipblasSideMode_t side, hipblasFillMode_t uplo,
                           hipblasOperation_t trans, hipblasDiagType_t diag,
                           int m, int n,
                           const float *alpha,
                           const float *A, int lda,
                                 float *B, int ldb);
hipblasStatus_t cublasXtrmm(hipblasHandle_t handle,
                           hipblasSideMode_t side, hipblasFillMode_t uplo,
                           hipblasOperation_t trans, hipblasDiagType_t      diag,
                           int m, int n,
                           const double *alpha,
                           const double *A, int lda,
                                 double *B, int ldb);
// hipblasStatus_t cublasXtrmm (hipblasHandle_t handle,
//                             hipblasSideMode_t side, hipblasFillMode_t uplo,
//                             hipblasOperation_t trans, hipblasDiagType_t diag,
//                             int m, int n,
//                             const hipComplex *alpha,
//                             const hipComplex *A, int lda,
//                                   hipComplex *B, int ldb);
// hipblasStatus_t cublasXtrmm (hipblasHandle_t handle,
//                             hipblasSideMode_t side, hipblasFillMode_t uplo,
//                             hipblasOperation_t trans, hipblasDiagType_t diag,
//                             int m, int n,
//                             const hipDoubleComplex *alpha,
//                             const hipDoubleComplex *A, int lda,
//                                   hipDoubleComplex *B, int ldb);



hipblasStatus_t cublasXtrsm(hipblasHandle_t handle,
                           hipblasSideMode_t side, hipblasFillMode_t uplo,
                           hipblasOperation_t trans, hipblasDiagType_t diag,
                           int m, int n,
                           const float *alpha,
                           const float *A, int lda,
                                 float *B, int ldb);
hipblasStatus_t cublasXtrsm(hipblasHandle_t handle,
                           hipblasSideMode_t side, hipblasFillMode_t uplo,
                           hipblasOperation_t trans, hipblasDiagType_t      diag,
                           int m, int n,
                           const double *alpha,
                           const double *A, int lda,
                                 double *B, int ldb);
// hipblasStatus_t cublasXtrsm (hipblasHandle_t handle,
//                             hipblasSideMode_t side, hipblasFillMode_t uplo,
//                             hipblasOperation_t trans, hipblasDiagType_t diag,
//                             int m, int n,
//                             const hipComplex *alpha,
//                             const hipComplex *A, int lda,
//                                   hipComplex *B, int ldb);
// hipblasStatus_t cublasXtrsm (hipblasHandle_t handle,
//                             hipblasSideMode_t side, hipblasFillMode_t uplo,
//                             hipblasOperation_t trans, hipblasDiagType_t diag,
//                             int m, int n,
//                             const hipDoubleComplex *alpha,
//                             const hipDoubleComplex *A, int lda,
//                                   hipDoubleComplex *B, int ldb);

hipblasStatus_t cublasXgemm( hipblasHandle_t handle,
                            hipblasOperation_t transa, hipblasOperation_t transb,
                            int m, int n, int k,
                            const float *alpha, const float *A, int lda,
                            const float *B, int ldb,
                            const float *beta,        float *C, int ldc);
hipblasStatus_t cublasXgemm( hipblasHandle_t handle,
                            hipblasOperation_t transa, hipblasOperation_t transb,
                            int m, int n, int k,
                            const double *alpha, const double *A, int lda,
                            const double *B, int ldb,
                            const double *beta,        double *C, int ldc);
// hipblasStatus_t cublasXgemm(hipblasHandle_t handle,
//                            hipblasOperation_t transa, hipblasOperation_t transb,
//                            int m, int n, int k,
//                            const hipComplex *alpha, const hipComplex *A, int lda,
//                            const hipComplex *B, int ldb,
//                            const hipComplex *beta,        hipComplex *C, int ldc);
// hipblasStatus_t cublasXgemm(hipblasHandle_t handle,
//                            hipblasOperation_t transa, hipblasOperation_t transb,
//                            int m, int n, int k,
//                            const hipDoubleComplex *alpha, const hipDoubleComplex *A, int lda,
//                            const hipDoubleComplex *B, int ldb,
//                            const hipDoubleComplex *beta,        hipDoubleComplex *C, int ldc);

#endif
