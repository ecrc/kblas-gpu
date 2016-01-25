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
#include <cublas.h>
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
int kblas_reg_sizes_ar[] = {32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536};//,96
std::set<int> kblas_reg_sizes(kblas_reg_sizes_ar, kblas_reg_sizes_ar + sizeof(kblas_reg_sizes_ar) / sizeof(int) );

bool REG_SIZE(int n){
  return (kblas_reg_sizes.find(n) != kblas_reg_sizes.end());
}
int CLOSEST_REG_SIZE(int n){
  //TODO validate input
  return *(--kblas_reg_sizes.lower_bound(n));
}

//==============================================================================================

void cublasXgemm( char transa, char transb, int m, int n, int k,
                 float alpha, const float *A, int lda,
                 const float *B, int ldb,
                 float beta, float *C, int ldc ){
  cublasSgemm(transa, transb, m, n, k,
              alpha, A, lda,
              B, ldb,
              beta, C, ldc);
}

void cublasXgemm( char transa, char transb, int m, int n, int k,
                 double alpha, const double *A, int lda,
                 const double *B, int ldb,
                 double beta, double *C, int ldc){
  cublasDgemm(transa, transb, m, n, k,
              alpha, A, lda,
              B, ldb,
              beta, C, ldc);
}
void cublasXgemm( char transa, char transb, int m, int n, int k,
                 cuComplex alpha, const cuComplex *A, int lda,
                 const cuComplex *B, int ldb,
                 cuComplex beta, cuComplex *C, int ldc){
  cublasCgemm(transa, transb, m, n, k,
              alpha, A, lda,
              B, ldb,
              beta, C, ldc);
}
void cublasXgemm( char transa, char transb, int m, int n, int k,
                 cuDoubleComplex alpha, const cuDoubleComplex *A, int lda,
                 const cuDoubleComplex *B, int ldb,
                 cuDoubleComplex beta, cuDoubleComplex *C, int ldc){
  cublasZgemm(transa, transb, m, n, k,
              alpha, A, lda,
              B, ldb,
              beta, C, ldc);
}



