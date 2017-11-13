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
#ifndef __KBLAS_ERR_CHECK_H__
#define __KBLAS_ERR_CHECK_H__

// ----------------------------------------
#define KBLAS_Success 1
#define KBLAS_UnknownError 0
#define KBLAS_NotSupported -1
#define KBLAS_NotImplemented -2
#define KBLAS_cuBLAS_Error -3
#define KBLAS_WrongConfig -4
#define KBLAS_CUDA_Error -5
#define KBLAS_InsufficientWorkspace -6
#define KBLAS_Error_Allocation -7
#define KBLAS_Error_Deallocation -8
#define KBLAS_Error_NotInitialized -9
#define KBLAS_Error_WrongInput -10
#define KBLAS_MAGMA_Error -11
#define KBLAS_SVD_NoConvergence -12
// ----------------------------------------
// C++ function is overloaded for different error types,
// which depends on error types being enums to be differentiable.
//inline
int _kblas_error( cudaError_t err, const char* func, const char* file, int line );
int _kblas_error( cublasStatus_t err, const char* func, const char* file, int line );
int _kblas_error( int err, const char* func, const char* file, int line );

#define check_error_ret( err_, ret_ ) \
{if(!_kblas_error( (err_), __func__, __FILE__, __LINE__ )) return ret_;}

#define check_error( err_ ) \
{if(!_kblas_error( (err_), __func__, __FILE__, __LINE__ )) ;}

#define check_error_forward( err_ ) \
{ int ret_val = (err_); if(!_kblas_error( ret_val, __func__, __FILE__, __LINE__ )) return ret_val;}

#endif //__KBLAS_ERR_CHECK_H__
