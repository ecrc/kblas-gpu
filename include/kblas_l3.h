 /**
 -- (C) Copyright 2013 King Abdullah University of Science and Technology
  Authors:
  Ali Charara (ali.charara@kaust.edu.sa)
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
  //TRMM {
    //GPU API, data resides on Device memory {
      cublasStatus_t kblasStrmm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const float *alpha,
                                const float *A, int lda,
                                      float *B, int ldb);
      cublasStatus_t kblasDtrmm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const double *alpha,
                                const double *A, int lda,
                                      double *B, int ldb);
      cublasStatus_t kblasCtrmm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuComplex *alpha,
                                const cuComplex *A, int lda,
                                      cuComplex *B, int ldb);
      cublasStatus_t kblasZtrmm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, int lda,
                                      cuDoubleComplex *B, int ldb);
    //}
    //CPU API, data resides on Host memory {
      cublasStatus_t kblas_strmm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const float *alpha,
                                const float *A, int lda,
                                      float *B, int ldb);
      cublasStatus_t kblas_dtrmm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const double *alpha,
                                const double *A, int lda,
                                      double *B, int ldb);
      cublasStatus_t kblas_ctrmm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuComplex *alpha,
                                const cuComplex *A, int lda,
                                      cuComplex *B, int ldb);
      cublasStatus_t kblas_ztrmm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, int lda,
                                      cuDoubleComplex *B, int ldb);
      cublasStatus_t kblas_strmm_mgpu(cublasHandle_t handle,
                                      cublasSideMode_t side, cublasFillMode_t uplo,
                                      cublasOperation_t trans, cublasDiagType_t diag,
                                      int m, int n,
                                      const float *alpha,
                                      const float *A, int lda,
                                            float *B, int ldb,
                                      int ngpu);
      cublasStatus_t kblas_dtrmm_mgpu(cublasHandle_t handle,
                                      cublasSideMode_t side, cublasFillMode_t uplo,
                                      cublasOperation_t trans, cublasDiagType_t diag,
                                      int m, int n,
                                      const double *alpha,
                                      const double *A, int lda,
                                            double *B, int ldb,
                                      int ngpu);
      cublasStatus_t kblas_ctrmm_mgpu(cublasHandle_t handle,
                                      cublasSideMode_t side, cublasFillMode_t uplo,
                                      cublasOperation_t trans, cublasDiagType_t diag,
                                      int m, int n,
                                      const cuComplex *alpha,
                                      const cuComplex *A, int lda,
                                            cuComplex *B, int ldb,
                                      int ngpu);
      cublasStatus_t kblas_ztrmm_mgpu(cublasHandle_t handle,
                                      cublasSideMode_t side, cublasFillMode_t uplo,
                                      cublasOperation_t trans, cublasDiagType_t diag,
                                      int m, int n,
                                      const cuDoubleComplex *alpha,
                                      const cuDoubleComplex *A, int lda,
                                            cuDoubleComplex *B, int ldb,
                                      int ngpu);
    //}
  //}
  //TRSM {
    //GPU API, data resides on Device memory {
      cublasStatus_t kblasStrsm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const float *alpha,
                                const float *A, int lda,
                                      float *B, int ldb);
      cublasStatus_t kblasDtrsm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const double *alpha,
                                const double *A, int lda,
                                      double *B, int ldb);
      cublasStatus_t kblasCtrsm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuComplex *alpha,
                                const cuComplex *A, int lda,
                                      cuComplex *B, int ldb);
      cublasStatus_t kblasZtrsm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, int lda,
                                      cuDoubleComplex *B, int ldb);
    //}
    //CPU API, data resides on Host memory {
      cublasStatus_t kblas_strsm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const float *alpha,
                                const float *A, int lda,
                                      float *B, int ldb);
      cublasStatus_t kblas_dtrsm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const double *alpha,
                                const double *A, int lda,
                                      double *B, int ldb);
      cublasStatus_t kblas_ctrsm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuComplex *alpha,
                                const cuComplex *A, int lda,
                                      cuComplex *B, int ldb);
      cublasStatus_t kblas_ztrsm(cublasHandle_t handle,
                                cublasSideMode_t side, cublasFillMode_t uplo,
                                cublasOperation_t trans, cublasDiagType_t diag,
                                int m, int n,
                                const cuDoubleComplex *alpha,
                                const cuDoubleComplex *A, int lda,
                                      cuDoubleComplex *B, int ldb);
      cublasStatus_t kblas_strsm_mgpu(cublasHandle_t handle,
                                      cublasSideMode_t side, cublasFillMode_t uplo,
                                      cublasOperation_t trans, cublasDiagType_t diag,
                                      int m, int n,
                                      const float *alpha,
                                      const float *A, int lda,
                                            float *B, int ldb,
                                      int ngpu);
      cublasStatus_t kblas_dtrsm_mgpu(cublasHandle_t handle,
                                      cublasSideMode_t side, cublasFillMode_t uplo,
                                      cublasOperation_t trans, cublasDiagType_t diag,
                                      int m, int n,
                                      const double *alpha,
                                      const double *A, int lda,
                                            double *B, int ldb,
                                      int ngpu);
      cublasStatus_t kblas_ctrsm_mgpu(cublasHandle_t handle,
                                      cublasSideMode_t side, cublasFillMode_t uplo,
                                      cublasOperation_t trans, cublasDiagType_t diag,
                                      int m, int n,
                                      const cuComplex *alpha,
                                      const cuComplex *A, int lda,
                                            cuComplex *B, int ldb,
                                      int ngpu);
      cublasStatus_t kblas_ztrsm_mgpu(cublasHandle_t handle,
                                      cublasSideMode_t side, cublasFillMode_t uplo,
                                      cublasOperation_t trans, cublasDiagType_t diag,
                                      int m, int n,
                                      const cuDoubleComplex *alpha,
                                      const cuDoubleComplex *A, int lda,
                                            cuDoubleComplex *B, int ldb,
                                      int ngpu);
    //}
  //}
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