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
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef _KBLAS_L3_H_
#define _KBLAS_L3_H_

#ifdef __cplusplus
extern "C" {
#endif

//============================================================================
//BLAS3 routines
//============================================================================


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
      hipblasStatus_t kblasStrmm(hipblasHandle_t handle,
                                hipblasSideMode_t side, hipblasFillMode_t uplo,
                                hipblasOperation_t trans, hipblasDiagType_t diag,
                                int m, int n,
                                const float *alpha,
                                const float *A, int lda,
                                      float *B, int ldb);
      /**
       * @brief TRMM dense double precision.
       *
       * GPU API, single device, data resides on Device memory
       */
      hipblasStatus_t kblasDtrmm(hipblasHandle_t handle,
                                hipblasSideMode_t side, hipblasFillMode_t uplo,
                                hipblasOperation_t trans, hipblasDiagType_t diag,
                                int m, int n,
                                const double *alpha,
                                const double *A, int lda,
                                      double *B, int ldb);

//       /**
//        * @brief TRMM dense single precision.
//        *
//        * CPU API, single device, data resides on Host memory
//        */
//       hipblasStatus_t kblas_strmm(hipblasHandle_t handle,
//                                 hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                 hipblasOperation_t trans, hipblasDiagType_t diag,
//                                 int m, int n,
//                                 const float *alpha,
//                                 const float *A, int lda,
//                                       float *B, int ldb);
//       /**
//        * @brief TRMM dense double precision.
//        *
//        * CPU API, single device, data resides on Host memory
//        */
//       hipblasStatus_t kblas_dtrmm(hipblasHandle_t handle,
//                                 hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                 hipblasOperation_t trans, hipblasDiagType_t diag,
//                                 int m, int n,
//                                 const double *alpha,
//                                 const double *A, int lda,
//                                       double *B, int ldb);
//       /**
//        * @brief TRMM dense single-complex precision.
//        *
//        * CPU API, single device, data resides on Host memory
//        */
//       hipblasStatus_t kblas_ctrmm(hipblasHandle_t handle,
//                                 hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                 hipblasOperation_t trans, hipblasDiagType_t diag,
//                                 int m, int n,
//                                 const hipComplex *alpha,
//                                 const hipComplex *A, int lda,
//                                       hipComplex *B, int ldb);
//       /**
//        * @brief TRMM dense double-complex precision.
//        *
//        * CPU API, single device, data resides on Host memory
//        */
//       hipblasStatus_t kblas_ztrmm(hipblasHandle_t handle,
//                                 hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                 hipblasOperation_t trans, hipblasDiagType_t diag,
//                                 int m, int n,
//                                 const hipDoubleComplex *alpha,
//                                 const hipDoubleComplex *A, int lda,
//                                       hipDoubleComplex *B, int ldb);
//       /**
//        * @brief TRMM dense single precision.
//        *
//        * CPU API, multiple devices, data resides on Host memory
//        */
//       hipblasStatus_t kblas_strmm_mgpu(hipblasHandle_t handle,
//                                       hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                       hipblasOperation_t trans, hipblasDiagType_t diag,
//                                       int m, int n,
//                                       const float *alpha,
//                                       const float *A, int lda,
//                                             float *B, int ldb,
//                                       int ngpu);
//       /**
//        * @brief TRMM dense double precision.
//        *
//        * CPU API, multiple devices, data resides on Host memory
//        */
//       hipblasStatus_t kblas_dtrmm_mgpu(hipblasHandle_t handle,
//                                       hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                       hipblasOperation_t trans, hipblasDiagType_t diag,
//                                       int m, int n,
//                                       const double *alpha,
//                                       const double *A, int lda,
//                                             double *B, int ldb,
//                                       int ngpu);
//       /**
//        * @brief TRMM dense single-complex precision.
//        *
//        * CPU API, multiple devices, data resides on Host memory
//        */
//       hipblasStatus_t kblas_ctrmm_mgpu(hipblasHandle_t handle,
//                                       hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                       hipblasOperation_t trans, hipblasDiagType_t diag,
//                                       int m, int n,
//                                       const hipComplex *alpha,
//                                       const hipComplex *A, int lda,
//                                             hipComplex *B, int ldb,
//                                       int ngpu);
//       /**
//        * @brief TRMM dense double-complex precision.
//        *
//        * CPU API, multiple devices, data resides on Host memory
//        */
//       hipblasStatus_t kblas_ztrmm_mgpu(hipblasHandle_t handle,
//                                       hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                       hipblasOperation_t trans, hipblasDiagType_t diag,
//                                       int m, int n,
//                                       const hipDoubleComplex *alpha,
//                                       const hipDoubleComplex *A, int lda,
//                                             hipDoubleComplex *B, int ldb,
//                                       int ngpu);
//   /** @} */


//   /**
//    * @name KBLAS TRSM routines.
//    *
//    * @{
//    */
//       /**
//        * @brief TRSM dense single precision.
//        *
//        * GPU API, single device, data resides on Device memory
//        */
      hipblasStatus_t kblasStrsm(hipblasHandle_t handle,
                                hipblasSideMode_t side, hipblasFillMode_t uplo,
                                hipblasOperation_t trans, hipblasDiagType_t diag,
                                int m, int n,
                                const float *alpha,
                                const float *A, int lda,
                                      float *B, int ldb);
      /**
       * @brief TRSM dense double precision.
       *
       * GPU API, single device, data resides on Device memory
       */
      hipblasStatus_t kblasDtrsm(hipblasHandle_t handle,
                                hipblasSideMode_t side, hipblasFillMode_t uplo,
                                hipblasOperation_t trans, hipblasDiagType_t diag,
                                int m, int n,
                                const double *alpha,
                                const double *A, int lda,
                                      double *B, int ldb);
//       /**
//        * @brief TRSM dense single-complex precision.
//        *
//        * GPU API, single device, data resides on Device memory
//        */
//       hipblasStatus_t kblasCtrsm(hipblasHandle_t handle,
//                                 hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                 hipblasOperation_t trans, hipblasDiagType_t diag,
//                                 int m, int n,
//                                 const hipComplex *alpha,
//                                 const hipComplex *A, int lda,
//                                       hipComplex *B, int ldb);
//       /**
//        * @brief TRSM dense double-complex precision.
//        *
//        * GPU API, single device, data resides on Device memory
//        */
//       hipblasStatus_t kblasZtrsm(hipblasHandle_t handle,
//                                 hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                 hipblasOperation_t trans, hipblasDiagType_t diag,
//                                 int m, int n,
//                                 const hipDoubleComplex *alpha,
//                                 const hipDoubleComplex *A, int lda,
//                                       hipDoubleComplex *B, int ldb);

//       /**
//        * @brief TRSM dense single precision.
//        *
//        * CPU API, single device, data resides on Host memory
//        */
//       hipblasStatus_t kblas_strsm(hipblasHandle_t handle,
//                                 hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                 hipblasOperation_t trans, hipblasDiagType_t diag,
//                                 int m, int n,
//                                 const float *alpha,
//                                 const float *A, int lda,
//                                       float *B, int ldb);
//       /**
//        * @brief TRSM dense double precision.
//        *
//        * CPU API, single device, data resides on Host memory
//        */
//       hipblasStatus_t kblas_dtrsm(hipblasHandle_t handle,
//                                 hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                 hipblasOperation_t trans, hipblasDiagType_t diag,
//                                 int m, int n,
//                                 const double *alpha,
//                                 const double *A, int lda,
//                                       double *B, int ldb);
//       /**
//        * @brief TRSM dense single-complex precision.
//        *
//        * CPU API, single device, data resides on Host memory
//        */
//       hipblasStatus_t kblas_ctrsm(hipblasHandle_t handle,
//                                 hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                 hipblasOperation_t trans, hipblasDiagType_t diag,
//                                 int m, int n,
//                                 const hipComplex *alpha,
//                                 const hipComplex *A, int lda,
//                                       hipComplex *B, int ldb);
//       /**
//        * @brief TRSM dense double-complex precision.
//        *
//        * CPU API, single device, data resides on Host memory
//        */
//       hipblasStatus_t kblas_ztrsm(hipblasHandle_t handle,
//                                 hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                 hipblasOperation_t trans, hipblasDiagType_t diag,
//                                 int m, int n,
//                                 const hipDoubleComplex *alpha,
//                                 const hipDoubleComplex *A, int lda,
//                                       hipDoubleComplex *B, int ldb);

//       /**
//        * @brief TRSM dense single precision.
//        *
//        * CPU API, multiple devices, data resides on Host memory
//        */
//       hipblasStatus_t kblas_strsm_mgpu(hipblasHandle_t handle,
//                                       hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                       hipblasOperation_t trans, hipblasDiagType_t diag,
//                                       int m, int n,
//                                       const float *alpha,
//                                       const float *A, int lda,
//                                             float *B, int ldb,
//                                       int ngpu);
//       /**
//        * @brief TRSM dense double precision.
//        *
//        * CPU API, multiple devices, data resides on Host memory
//        */
//       hipblasStatus_t kblas_dtrsm_mgpu(hipblasHandle_t handle,
//                                       hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                       hipblasOperation_t trans, hipblasDiagType_t diag,
//                                       int m, int n,
//                                       const double *alpha,
//                                       const double *A, int lda,
//                                             double *B, int ldb,
//                                       int ngpu);
//       /**
//        * @brief TRSM dense single-complex precision.
//        *
//        * CPU API, multiple devices, data resides on Host memory
//        */
//       hipblasStatus_t kblas_ctrsm_mgpu(hipblasHandle_t handle,
//                                       hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                       hipblasOperation_t trans, hipblasDiagType_t diag,
//                                       int m, int n,
//                                       const hipComplex *alpha,
//                                       const hipComplex *A, int lda,
//                                             hipComplex *B, int ldb,
//                                       int ngpu);
//       /**
//        * @brief TRSM dense double-complex precision.
//        *
//        * CPU API, multiple devices, data resides on Host memory
//        */
//       hipblasStatus_t kblas_ztrsm_mgpu(hipblasHandle_t handle,
//                                       hipblasSideMode_t side, hipblasFillMode_t uplo,
//                                       hipblasOperation_t trans, hipblasDiagType_t diag,
//                                       int m, int n,
//                                       const hipDoubleComplex *alpha,
//                                       const hipDoubleComplex *A, int lda,
//                                             hipDoubleComplex *B, int ldb,
//                                       int ngpu);
//   /** @} */

// /** @} */
// #else//CUBLAS_V2_H_
//   //cuBLAS Legacy API
//   //assumes cuBLAS default stream (NULL)
//   //GPU API, data resides on Device memory {
//   void kblasStrmm(char side, char uplo, char trans, char diag,
//                   int m, int n,
//                   float alpha, const float *A, int lda,
//                                     float *B, int ldb);
//   void kblasDtrmm(char side, char uplo, char trans, char diag,
//                   int m, int n,
//                   double alpha, const double *A, int lda,
//                                       double *B, int ldb);
//   void kblasCtrmm(char side, char uplo, char trans, char diag,
//                   int m, int n,
//                   hipComplex alpha, const hipComplex *A, int lda,
//                                         hipComplex *B, int ldb);
//   void kblasZtrmm(char side, char uplo, char trans, char diag,
//                   int m, int n,
//                   hipDoubleComplex alpha, const hipDoubleComplex *A, int lda,
//                   hipDoubleComplex *B, int ldb);
//   //}
//   /*/CPU API, data resides on Host memory {
//   void kblas_strmm(char side, char uplo, char trans, char diag,
//                   int m, int n,
//                   float alpha, const float *A, int lda,
//                                     float *B, int ldb);
//   void kblas_dtrmm(char side, char uplo, char trans, char diag,
//                   int m, int n,
//                   double alpha, const double *A, int lda,
//                                       double *B, int ldb);
//   void kblas_ctrmm(char side, char uplo, char trans, char diag,
//                   int m, int n,
//                   hipComplex alpha, const hipComplex *A, int lda,
//                                         hipComplex *B, int ldb);
//   void kblas_ztrmm(char side, char uplo, char trans, char diag,
//                   int m, int n,
//                   hipDoubleComplex alpha, const hipDoubleComplex *A, int lda,
//                                               hipDoubleComplex *B, int ldb);
//   //}*/
//   void kblasStrsm(char side, char uplo, char trans, char diag,
//                   int m, int n,
//                   float alpha, const float *A, int lda,
//                                      float *B, int ldb);
//   void kblasDtrsm(char side, char uplo, char trans, char diag,
//                   int m, int n,
//                   double alpha, const double *A, int lda,
//                                       double *B, int ldb);
//   void kblasCtrsm(char side, char uplo, char trans, char diag,
//                   int m, int n,
//                   hipComplex alpha, const hipComplex *A, int lda,
//                                          hipComplex *B, int ldb);
//   void kblasZtrsm(char side, char uplo, char trans, char diag,
//                   int m, int n,
//                   hipDoubleComplex alpha, const hipDoubleComplex *A, int lda,
//                                                hipDoubleComplex *B, int ldb);


//   //Asynchronous version, takes streamID as parameter
//   void kblasStrmm_async(char side, char uplo, char trans, char diag,
//                         int m, int n,
//                         float alpha, const float *A, int lda,
//                                            float *B, int ldb,
//                         hipStream_t stream);
//   void kblasDtrmm_async(char side, char uplo, char trans, char diag,
//                         int m, int n,
//                         double alpha, const double *A, int lda,
//                                             double *B, int ldb,
//                         hipStream_t stream);
//   void kblasCtrmm_async(char side, char uplo, char trans, char diag,
//                         int m, int n,
//                         hipComplex alpha, const hipComplex *A, int lda,
//                                                hipComplex *B, int ldb,
//                         hipStream_t stream);
//   void kblasZtrmm_async(char side, char uplo, char trans, char diag,
//                         int m, int n,
//                         hipDoubleComplex alpha, const hipDoubleComplex *A, int lda,
//                                                      hipDoubleComplex *B, int ldb,
//                         hipStream_t stream);

//   void kblasStrsm_async(char side, char uplo, char trans, char diag,
//                         int m, int n,
//                         float alpha, const float *A, int lda,
//                                            float *B, int ldb,
//                         hipStream_t stream);
//   void kblasDtrsm_async(char side, char uplo, char trans, char diag,
//                         int m, int n,
//                         double alpha, const double *A, int lda,
//                                             double *B, int ldb,
//                         hipStream_t stream);
//   void kblasCtrsm_async(char side, char uplo, char trans, char diag,
//                         int m, int n,
//                         hipComplex alpha, const hipComplex *A, int lda,
//                                                hipComplex *B, int ldb,
//                         hipStream_t stream);
//   void kblasZtrsm_async(char side, char uplo, char trans, char diag,
//                         int m, int n,
//                         hipDoubleComplex alpha, const hipDoubleComplex *A, int lda,
//                                                      hipDoubleComplex *B, int ldb,
//                         hipStream_t stream);



#ifdef __cplusplus
}
#endif

#endif // _KBLAS_L3_H_
