/**
  --* (C) Copyright 2013 King Abdullah University of Science and Technology
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
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include <typeinfo>

#include "kblas.h"
#include "kblas_struct.h"
#include "operators.h"
#include "defs.h"
#include "kblas_common.h"
#include "batch_common.ch"

//==============================================================================================
#include "Xblas_core.ch"
#include "Xhelper_funcs.ch"
#include "Xtrtri_batch_drivers.cuh"

//==============================================================================================
//Non-Strided form

// workspace needed: device pointers
// A: host pointer to device buffer
int Xtrtri_batch_offset(kblasHandle_t handle,
                        char uplo, char diag,
                        const int n,
                        TYPE** A, int A_row_off, int A_col_off, int lda,
                        int batchCount,
                        int *info_array)
{
  KBlasWorkspaceState ws_needed;
  trtri_batch_wsquery_core<false>( n, batchCount, (kblasWorkspaceState_t)&ws_needed);

  if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) ){
    return KBLAS_InsufficientWorkspace;
  }

  return Xtrtri_batch_core<TYPE, TYPE**, false>(
                          handle,
                          uplo, diag, n,
                          (TYPE**)A, A_row_off, A_col_off, lda, (long)0,
                          batchCount,
                          info_array);
}

// workspace needed: device pointers
// A: host pointer to device buffer
int kblas_trtri_batch(kblasHandle_t handle,
                      char uplo, char diag,
                      const int n,
                      TYPE** A, int lda,
                      int batchCount,
                      int *info_array)
{
  return Xtrtri_batch_offset( handle,
                              uplo, diag, n,
                              A, 0, 0, lda,
                              batchCount,
                              info_array);
}


// workspace needed: device pointers
// A: host pointer to device buffer
extern "C"
int kblasXtrtri_batch(kblasHandle_t handle,
                      char uplo, char diag,
                      const int n,
                      TYPE** A, int lda,
                      int batchCount,
                      int *info_array)
{
  return Xtrtri_batch_offset( handle,
                              uplo, diag, n,
                              A, 0, 0, lda,
                              batchCount,
                              info_array);
}


//==============================================================================================
//Strided form
// template<>

// workspace needed: device pointers
// A: host pointer to device buffer
int Xtrtri_batch_offset(kblasHandle_t handle,
                        char uplo, char diag,
                        const int n,
                        TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                        int batchCount,
                        int *info_array)
{
  KBlasWorkspaceState ws_needed;
  trtri_batch_wsquery_core<true>( batchCount, n, (kblasWorkspaceState_t)&ws_needed);

  if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) ){
    return KBLAS_InsufficientWorkspace;
  }

  return Xtrtri_batch_core<TYPE, TYPE*, true>(
                          handle,
                          uplo, diag, n,
                          (TYPE*)A, A_row_off, A_col_off, lda, strideA,
                          batchCount,
                          info_array);
}

// workspace needed: device pointers
// A: host pointer to device buffer
int kblas_trtri_batch(kblasHandle_t handle,
                      char uplo, char diag,
                      const int n,
                      TYPE* A, int lda, long strideA,
                      int batchCount,
                      int *info_array)
{
  return Xtrtri_batch_offset( handle,
                              uplo, diag, n,
                              A, 0, 0, lda, strideA,
                              batchCount,
                              info_array);
}

// workspace needed: device pointers
// A: host pointer to device buffer
extern "C"
int kblasXtrtri_batch_strided(kblasHandle_t handle,
                              char uplo, char diag,
                              const int n,
                              TYPE* A, int lda, long strideA,
                              int batchCount,
                              int *info_array)
{
  return Xtrtri_batch_offset( handle,
                              uplo, diag, n,
                              A, 0, 0, lda, strideA,
                              batchCount,
                              info_array);
}
