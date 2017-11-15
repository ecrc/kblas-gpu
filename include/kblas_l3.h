/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file include/kblas_l3.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 2.0.0
 * @author Ali Charara
 * @date 2017-11-13
 **/

#ifndef _KBLAS_L3_H_
#define _KBLAS_L3_H_

#ifdef __cplusplus
extern "C" {
#endif

//============================================================================
//BLAS3 routines
//============================================================================

//cuBLAS_v2 API
#if defined(CUBLAS_V2_H_)

/** @addtogroup C_API
*  @{
*/
  /**
   * @name KBLAS TRMM routines.
   *
   * @{
   */

      /**
       * @brief TRMM dense single precision.
       *
       * GPU API, single device, data resides on Device memory
       */
      cublasStatus_t kblasStrmm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const float *alpha,
                                const float *A, int lda,
                                      float *B, int ldb);
      /**
       * @brief TRMM dense double precision.
       *
       * GPU API, single device, data resides on Device memory
       */
      cublasStatus_t kblasDtrmm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const double *alpha,
                                const double *A, int lda,
                                      double *B, int ldb);
      /**
       * @brief TRMM dense single-complex precision.
       *
       * GPU API, single device, data resides on Device memory
       */
      cublasStatus_t kblasCtrmm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuComplex *alpha,
                                const cuComplex *A, int lda,
                                      cuComplex *B, int ldb);
      /**
       * @brief TRMM dense double-complex precision.
       *
       * GPU API, single device, data resides on Device memory
       */
      cublasStatus_t kblasZtrmm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, int lda,
                                      cuDoubleComplex *B, int ldb);

      /**
       * @brief TRMM dense single precision.
       *
       * CPU API, single device, data resides on Host memory
       */
      cublasStatus_t kblas_strmm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const float *alpha,
                                const float *A, int lda,
                                      float *B, int ldb);
      /**
       * @brief TRMM dense double precision.
       *
       * CPU API, single device, data resides on Host memory
       */
      cublasStatus_t kblas_dtrmm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const double *alpha,
                                const double *A, int lda,
                                      double *B, int ldb);
      /**
       * @brief TRMM dense single-complex precision.
       *
       * CPU API, single device, data resides on Host memory
       */
      cublasStatus_t kblas_ctrmm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuComplex *alpha,
                                const cuComplex *A, int lda,
                                      cuComplex *B, int ldb);
      /**
       * @brief TRMM dense double-complex precision.
       *
       * CPU API, single device, data resides on Host memory
       */
      cublasStatus_t kblas_ztrmm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, int lda,
                                      cuDoubleComplex *B, int ldb);
      /**
       * @brief TRMM dense single precision.
       *
       * CPU API, multiple devices, data resides on Host memory
       */
      cublasStatus_t kblas_strmm_mgpu(cublasHandle_t handle,
                                      cublasSideMode_t side, cublasFillMode_t uplo,
                                      cublasOperation_t trans, cublasDiagType_t diag,
                                      int m, int n,
                                      const float *alpha,
                                      const float *A, int lda,
                                            float *B, int ldb,
                                      int ngpu);
      /**
       * @brief TRMM dense double precision.
       *
       * CPU API, multiple devices, data resides on Host memory
       */
      cublasStatus_t kblas_dtrmm_mgpu(cublasHandle_t handle,
                                      cublasSideMode_t side, cublasFillMode_t uplo,
                                      cublasOperation_t trans, cublasDiagType_t diag,
                                      int m, int n,
                                      const double *alpha,
                                      const double *A, int lda,
                                            double *B, int ldb,
                                      int ngpu);
      /**
       * @brief TRMM dense single-complex precision.
       *
       * CPU API, multiple devices, data resides on Host memory
       */
      cublasStatus_t kblas_ctrmm_mgpu(cublasHandle_t handle,
                                      cublasSideMode_t side, cublasFillMode_t uplo,
                                      cublasOperation_t trans, cublasDiagType_t diag,
                                      int m, int n,
                                      const cuComplex *alpha,
                                      const cuComplex *A, int lda,
                                            cuComplex *B, int ldb,
                                      int ngpu);
      /**
       * @brief TRMM dense double-complex precision.
       *
       * CPU API, multiple devices, data resides on Host memory
       */
      cublasStatus_t kblas_ztrmm_mgpu(cublasHandle_t handle,
                                      cublasSideMode_t side, cublasFillMode_t uplo,
                                      cublasOperation_t trans, cublasDiagType_t diag,
                                      int m, int n,
                                      const cuDoubleComplex *alpha,
                                      const cuDoubleComplex *A, int lda,
                                            cuDoubleComplex *B, int ldb,
                                      int ngpu);
  /** @} */


  /**
   * @name KBLAS TRSM routines.
   *
   * @{
   */
      /**
       * @brief TRSM dense single precision.
       *
       * GPU API, single device, data resides on Device memory
       */
      cublasStatus_t kblasStrsm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const float *alpha,
                                const float *A, int lda,
                                      float *B, int ldb);
      /**
       * @brief TRSM dense double precision.
       *
       * GPU API, single device, data resides on Device memory
       */
      cublasStatus_t kblasDtrsm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const double *alpha,
                                const double *A, int lda,
                                      double *B, int ldb);
      /**
       * @brief TRSM dense single-complex precision.
       *
       * GPU API, single device, data resides on Device memory
       */
      cublasStatus_t kblasCtrsm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuComplex *alpha,
                                const cuComplex *A, int lda,
                                      cuComplex *B, int ldb);
      /**
       * @brief TRSM dense double-complex precision.
       *
       * GPU API, single device, data resides on Device memory
       */
      cublasStatus_t kblasZtrsm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, int lda,
                                      cuDoubleComplex *B, int ldb);

      /**
       * @brief TRSM dense single precision.
       *
       * CPU API, single device, data resides on Host memory
       */
      cublasStatus_t kblas_strsm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const float *alpha,
                                const float *A, int lda,
                                      float *B, int ldb);
      /**
       * @brief TRSM dense double precision.
       *
       * CPU API, single device, data resides on Host memory
       */
      cublasStatus_t kblas_dtrsm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const double *alpha,
                                const double *A, int lda,
                                      double *B, int ldb);
      /**
       * @brief TRSM dense single-complex precision.
       *
       * CPU API, single device, data resides on Host memory
       */
      cublasStatus_t kblas_ctrsm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuComplex *alpha,
                                const cuComplex *A, int lda,
                                      cuComplex *B, int ldb);
      /**
       * @brief TRSM dense double-complex precision.
       *
       * CPU API, single device, data resides on Host memory
       */
      cublasStatus_t kblas_ztrsm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, int lda,
                                      cuDoubleComplex *B, int ldb);

      /**
       * @brief TRSM dense single precision.
       *
       * CPU API, multiple devices, data resides on Host memory
       */
      cublasStatus_t kblas_strsm_mgpu(cublasHandle_t handle,
                                      cublasSideMode_t side, cublasFillMode_t uplo,
                                      cublasOperation_t trans, cublasDiagType_t diag,
                                      int m, int n,
                                      const float *alpha,
                                      const float *A, int lda,
                                            float *B, int ldb,
                                      int ngpu);
      /**
       * @brief TRSM dense double precision.
       *
       * CPU API, multiple devices, data resides on Host memory
       */
      cublasStatus_t kblas_dtrsm_mgpu(cublasHandle_t handle,
                                      cublasSideMode_t side, cublasFillMode_t uplo,
                                      cublasOperation_t trans, cublasDiagType_t diag,
                                      int m, int n,
                                      const double *alpha,
                                      const double *A, int lda,
                                            double *B, int ldb,
                                      int ngpu);
      /**
       * @brief TRSM dense single-complex precision.
       *
       * CPU API, multiple devices, data resides on Host memory
       */
      cublasStatus_t kblas_ctrsm_mgpu(cublasHandle_t handle,
                                      cublasSideMode_t side, cublasFillMode_t uplo,
                                      cublasOperation_t trans, cublasDiagType_t diag,
                                      int m, int n,
                                      const cuComplex *alpha,
                                      const cuComplex *A, int lda,
                                            cuComplex *B, int ldb,
                                      int ngpu);
      /**
       * @brief TRSM dense double-complex precision.
       *
       * CPU API, multiple devices, data resides on Host memory
       */
      cublasStatus_t kblas_ztrsm_mgpu(cublasHandle_t handle,
                                      cublasSideMode_t side, cublasFillMode_t uplo,
                                      cublasOperation_t trans, cublasDiagType_t diag,
                                      int m, int n,
                                      const cuDoubleComplex *alpha,
                                      const cuDoubleComplex *A, int lda,
                                            cuDoubleComplex *B, int ldb,
                                      int ngpu);
  /** @} */

/** @} */
#else//CUBLAS_V2_H_
  //cuBLAS Legacy API
  //assumes cuBLAS default stream (NULL)
  //GPU API, data resides on Device memory {
  void kblasStrmm(char side, char uplo, char trans, char diag,
                  int m, int n,
                  float alpha, const float *A, int lda,
                                    float *B, int ldb);
  void kblasDtrmm(char side, char uplo, char trans, char diag,
                  int m, int n,
                  double alpha, const double *A, int lda,
                                      double *B, int ldb);
  void kblasCtrmm(char side, char uplo, char trans, char diag,
                  int m, int n,
                  cuComplex alpha, const cuComplex *A, int lda,
                                        cuComplex *B, int ldb);
  void kblasZtrmm(char side, char uplo, char trans, char diag,
                  int m, int n,
                  cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                  cuDoubleComplex *B, int ldb);
  //}
  /*/CPU API, data resides on Host memory {
  void kblas_strmm(char side, char uplo, char trans, char diag,
                  int m, int n,
                  float alpha, const float *A, int lda,
                                    float *B, int ldb);
  void kblas_dtrmm(char side, char uplo, char trans, char diag,
                  int m, int n,
                  double alpha, const double *A, int lda,
                                      double *B, int ldb);
  void kblas_ctrmm(char side, char uplo, char trans, char diag,
                  int m, int n,
                  cuComplex alpha, const cuComplex *A, int lda,
                                        cuComplex *B, int ldb);
  void kblas_ztrmm(char side, char uplo, char trans, char diag,
                  int m, int n,
                  cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                                              cuDoubleComplex *B, int ldb);
  //}*/
  void kblasStrsm(char side, char uplo, char trans, char diag,
                  int m, int n,
                  float alpha, const float *A, int lda,
                                     float *B, int ldb);
  void kblasDtrsm(char side, char uplo, char trans, char diag,
                  int m, int n,
                  double alpha, const double *A, int lda,
                                      double *B, int ldb);
  void kblasCtrsm(char side, char uplo, char trans, char diag,
                  int m, int n,
                  cuComplex alpha, const cuComplex *A, int lda,
                                         cuComplex *B, int ldb);
  void kblasZtrsm(char side, char uplo, char trans, char diag,
                  int m, int n,
                  cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                                               cuDoubleComplex *B, int ldb);


  //Asynchronous version, takes streamID as parameter
  void kblasStrmm_async(char side, char uplo, char trans, char diag,
                        int m, int n,
                        float alpha, const float *A, int lda,
                                           float *B, int ldb,
                        cudaStream_t stream);
  void kblasDtrmm_async(char side, char uplo, char trans, char diag,
                        int m, int n,
                        double alpha, const double *A, int lda,
                                            double *B, int ldb,
                        cudaStream_t stream);
  void kblasCtrmm_async(char side, char uplo, char trans, char diag,
                        int m, int n,
                        cuComplex alpha, const cuComplex *A, int lda,
                                               cuComplex *B, int ldb,
                        cudaStream_t stream);
  void kblasZtrmm_async(char side, char uplo, char trans, char diag,
                        int m, int n,
                        cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                                                     cuDoubleComplex *B, int ldb,
                        cudaStream_t stream);

  void kblasStrsm_async(char side, char uplo, char trans, char diag,
                        int m, int n,
                        float alpha, const float *A, int lda,
                                           float *B, int ldb,
                        cudaStream_t stream);
  void kblasDtrsm_async(char side, char uplo, char trans, char diag,
                        int m, int n,
                        double alpha, const double *A, int lda,
                                            double *B, int ldb,
                        cudaStream_t stream);
  void kblasCtrsm_async(char side, char uplo, char trans, char diag,
                        int m, int n,
                        cuComplex alpha, const cuComplex *A, int lda,
                                               cuComplex *B, int ldb,
                        cudaStream_t stream);
  void kblasZtrsm_async(char side, char uplo, char trans, char diag,
                        int m, int n,
                        cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                                                     cuDoubleComplex *B, int ldb,
                        cudaStream_t stream);
#endif//CUBLAS_V2_H_


#ifdef __cplusplus
}
#endif

#endif // _KBLAS_L3_H_
