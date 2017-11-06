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
#ifndef _KBLAS_BATCH_H_
#define _KBLAS_BATCH_H_

//============================================================================
//BATCH GEMM routines
//wrappers around cuBLAS/MAGMA batch GEMM routines
//============================================================================

#ifdef __cplusplus
int kblas_gemm_batch( kblasHandle_t handle,
                      char transA, char transB,
                      const int m, const int n, const int k,
                      const float alpha,
                      const float** A, int lda,
                      const float** B, int ldb,
                      const float beta,
                            float** C, int ldc,
                      int batchCount);

int kblas_gemm_batch( kblasHandle_t handle,
                      char transA, char transB,
                      const int m, const int n, const int k,
                      const double alpha,
                      const double** A, int lda,
                      const double** B, int ldb,
                      const double beta,
                            double** C, int ldc,
                      int batchCount);

int kblas_gemm_batch( kblasHandle_t handle,
                      char transA, char transB,
                      const int m, const int n, const int k,
                      const cuFloatComplex alpha,
                      const cuFloatComplex** A, int lda,
                      const cuFloatComplex** B, int ldb,
                      const cuFloatComplex beta,
                            cuFloatComplex** C, int ldc,
                      int batchCount);

int kblas_gemm_batch( kblasHandle_t handle,
                      char transA, char transB,
                      const int m, const int n, const int k,
                      const cuDoubleComplex alpha,
                      const cuDoubleComplex** A, int lda,
                      const cuDoubleComplex** B, int ldb,
                      const cuDoubleComplex beta,
                            cuDoubleComplex** C, int ldc,
                      int batchCount);
#endif


#ifdef __cplusplus
extern "C" {
#endif
int kblasSgemm_batch( kblasHandle_t handle,
                      char transA, char transB,
                      const int m, const int n, const int k,
                      const float alpha,
                      const float** A, int lda,
                      const float** B, int ldb,
                      const float beta,
                            float** C, int ldc,
                      int batchCount);

int kblasDgemm_batch( kblasHandle_t handle,
                      char transA, char transB,
                      const int m, const int n, const int k,
                      const double alpha,
                      const double** A, int lda,
                      const double** B, int ldb,
                      const double beta,
                            double** C, int ldc,
                      int batchCount);

int kblasCgemm_batch( kblasHandle_t handle,
                      char transA, char transB,
                      const int m, const int n, const int k,
                      const cuFloatComplex alpha,
                      const cuFloatComplex** A, int lda,
                      const cuFloatComplex** B, int ldb,
                      const cuFloatComplex beta,
                            cuFloatComplex** C, int ldc,
                      int batchCount);

int kblasZgemm_batch( kblasHandle_t handle,
                      char transA, char transB,
                      const int m, const int n, const int k,
                      const cuDoubleComplex alpha,
                      const cuDoubleComplex** A, int lda,
                      const cuDoubleComplex** B, int ldb,
                      const cuDoubleComplex beta,
                            cuDoubleComplex** C, int ldc,
                      int batchCount);

//============================================================================
//KBLAS BATCH routines
//============================================================================

void kblasSsyrk_batch_wsquery(const int m, int batchCount, kblasWorkspace_t ws);
void kblasDsyrk_batch_wsquery(const int m, int batchCount, kblasWorkspace_t ws);
void kblasCsyrk_batch_wsquery(const int m, int batchCount, kblasWorkspace_t ws);
void kblasZsyrk_batch_wsquery(const int m, int batchCount, kblasWorkspace_t ws);

int kblasSsyrk_batch( kblasHandle_t handle,
                      char uplo, char trans,
                      const int m, const int n,
                      const float alpha, const float** A, int lda,
                      const float beta,        float** B, int ldb,
                      int batchCount);

int kblasDsyrk_batch( kblasHandle_t handle,
                      char uplo, char trans,
                      const int m, const int n,
                      const double alpha, const double** A, int lda,
                      const double beta,        double** B, int ldb,
                      int batchCount);

int kblasCsyrk_batch( kblasHandle_t handle,
                      char uplo, char trans,
                      const int m, const int n,
                      const cuFloatComplex alpha, const cuFloatComplex** A, int lda,
                      const cuFloatComplex beta,        cuFloatComplex** B, int ldb,
                      int batchCount);

int kblasZsyrk_batch( kblasHandle_t handle,
                      char uplo, char trans,
                      const int m, const int n,
                      const cuDoubleComplex alpha, const cuDoubleComplex** A, int lda,
                      const cuDoubleComplex beta,        cuDoubleComplex** B, int ldb,
                      int batchCount);

//------------------------------------------------------------------------------
void kblasSsyrk_batch_strided_wsquery(const int m, int batchCount, kblasWorkspace_t ws);
void kblasDsyrk_batch_strided_wsquery(const int m, int batchCount, kblasWorkspace_t ws);
void kblasCsyrk_batch_strided_wsquery(const int m, int batchCount, kblasWorkspace_t ws);
void kblasZsyrk_batch_strided_wsquery(const int m, int batchCount, kblasWorkspace_t ws);

int kblasSsyrk_batch_strided( kblasHandle_t handle,
                              char uplo, char trans,
                              const int m, const int n,
                              const float alpha, const float* A, int lda, long strideA,
                              const float beta,        float* B, int ldb, long strideB,
                              int batchCount);

int kblasDsyrk_batch_strided( kblasHandle_t handle,
                              char uplo, char trans,
                              const int m, const int n,
                              const double alpha, const double* A, int lda, long strideA,
                              const double beta,        double* B, int ldb, long strideB,
                              int batchCount);

int kblasCsyrk_batch_strided( kblasHandle_t handle,
                              char uplo, char trans,
                              const int m, const int n,
                              const cuFloatComplex alpha, const cuFloatComplex* A, int lda, long strideA,
                              const cuFloatComplex beta,        cuFloatComplex* B, int ldb, long strideB,
                              int batchCount);

int kblasZsyrk_batch_strided( kblasHandle_t handle,
                              char uplo, char trans,
                              const int m, const int n,
                              const cuDoubleComplex alpha, const cuDoubleComplex* A, int lda, long strideA,
                              const cuDoubleComplex beta,        cuDoubleComplex* B, int ldb, long strideB,
                              int batchCount);

//============================================================================

#ifdef __cplusplus
}
#endif

#endif // _KBLAS_BATCH_H_