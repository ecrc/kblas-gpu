 /**
  - -* (C) Copyright 2013 King Abdullah University of Science and Technology
  Authors:
  Ali Charara (ali.charara@kaust.edu.sa)
  David Keyes (david.keyes@kaust.edu.sa)
  Hatem Ltaief (hatem.ltaief@kaust.edu.sa)
  
  Redistribution  and  use  in  source and binary forms, with or without
  modification,  are  permitted  provided  that the following conditions
  are met:
  
  * Redistributions  of  source  code  must  retain  the above copyright
  * notice,  this  list  of  conditions  and  the  following  disclaimer.
  * Redistributions  in  binary  form must reproduce the above copyright
  * notice,  this list of conditions and the following disclaimer in the
  * documentation  and/or other materials provided with the distribution.
  * Neither  the  name of the King Abdullah University of Science and
  * Technology nor the names of its contributors may be used to endorse
  * or promote products derived from this software without specific prior
  * written permission.
  * 
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
#include <stdlib.h>
#include <stdio.h>
#include <set>
#include <cublas_v2.h>
#include "kblas.h"
#include "operators.h"

//==============================================================================================

const char* cublasGetErrorString( cublasStatus_t error )
{
  switch( error ) {
    case CUBLAS_STATUS_SUCCESS:
      return "success";
      
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "not initialized";
      
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "out of memory";
      
    case CUBLAS_STATUS_INVALID_VALUE:
      return "invalid value";
      
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "architecture mismatch";
      
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "memory mapping error";
      
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "execution failed";
      
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "internal error";
      
    default:
      return "unknown CUBLAS error code";
  }
}


// ----------------------------------------
// C++ function is overloaded for different error types,
// which depends on error types being enums to be differentiable.
//inline
int _kblas_error( cudaError_t err, const char* func, const char* file, int line )
{
  if ( err != cudaSuccess ) {
    fprintf( stderr, "CUDA runtime error: %s (%d) in %s at %s:%d\n",
             cudaGetErrorString( err ), err, func, file, line );
    return 0;
  }
  return 1;
}

// --------------------
//inline
int _kblas_error( cublasStatus_t err, const char* func, const char* file, int line )
{
  if ( err != CUBLAS_STATUS_SUCCESS ) {
    fprintf( stderr, "CUBLAS error: %s (%d) in %s at %s:%d\n",
             cublasGetErrorString( err ), err, func, file, line );
    return 0;
  }
  return 1;
}

#define check_error( err ) \
{if(!_kblas_error( (err), __func__, __FILE__, __LINE__ )) return 0;}

//==============================================================================================
bool REG_SIZE(int n){
  return ((n > 0) && !(n & (n - 1)));
}
int CLOSEST_REG_SIZE(int n){
  //TODO validate input
  if(n > 0){
    int res = 1;
    while (res < n){
      res = res << 1;
    }
    return res >> 1;
  }else{
    return 0;    
  }
}

//==============================================================================================

cublasStatus_t cublasXgemm( cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const float *alpha, const float *A, int lda,
                                                const float *B, int ldb,
                            const float *beta,        float *C, int ldc){
  return cublasSgemm(handle,
                     transa, transb,
                     m, n, k,
                     alpha, A, lda,
                            B, ldb,
                     beta,  C, ldc);
}

cublasStatus_t cublasXgemm( cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const double *alpha, const double *A, int lda,
                                                 const double *B, int ldb,
                            const double *beta,        double *C, int ldc){
  return cublasDgemm(handle,
                     transa, transb,
                     m, n, k,
                     alpha, A, lda,
                            B, ldb,
                     beta,  C, ldc);
}
cublasStatus_t cublasXgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const cuComplex *alpha, const cuComplex *A, int lda,
                                                   const cuComplex *B, int ldb,
                           const cuComplex *beta,        cuComplex *C, int ldc){
  return cublasCgemm(handle,
                     transa, transb,
                     m, n, k,
                     alpha, A, lda,
                            B, ldb,
                     beta,  C, ldc);
}
cublasStatus_t cublasXgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
                                                         const cuDoubleComplex *B, int ldb,
                           const cuDoubleComplex *beta,        cuDoubleComplex *C, int ldc){
  return cublasZgemm(handle,
                     transa, transb,
                     m, n, k,
                     alpha, A, lda,
                            B, ldb,
                     beta,  C, ldc);
}



