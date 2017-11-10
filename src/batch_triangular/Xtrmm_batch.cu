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
#include "Xtrmm_batch_drivers.cuh"

//==============================================================================================
//Non-Strided form

// workspace needed: device pointers
// A, B: host pointer to array of device pointers to device buffers
int kblas_trmm_batch(kblasHandle_t handle,
                     char side, char uplo, char trans, char diag,
                     const int m, const int n,
                     const TYPE alpha,
                     const TYPE** A, int lda,
                           TYPE** B, int ldb,
                    int batchCount){

  KBlasWorkspaceState ws_needed;
  trmm_batch_wsquery_core<false>( batchCount,
                                  side, m, n,
                                  (kblasWorkspaceState_t)&ws_needed);

  bool suffWorkspace = (ws_needed.d_ptrs_bytes <= handle->work_space.allocated_ws_state.d_ptrs_bytes);

  if(!suffWorkspace){
    return KBLAS_InsufficientWorkspace;
  }

  return Xtrmm_batch_core<TYPE, TYPE**, false>(
                          handle,
                          side, uplo, trans, diag,
                          m, n,
                          alpha,
                          (TYPE**)A, 0, 0, lda, (long)0,
                          (TYPE**)B, 0, 0, ldb, (long)0,
                          batchCount);
}

extern "C" {

// workspace needed: device pointers
// A, B: host pointer to array of device pointers to device buffers
int kblasXtrmm_batch(kblasHandle_t handle,
                     char side, char uplo, char trans, char diag,
                     const int m, const int n,
                     const TYPE alpha,
                     const TYPE** A, int lda,
                           TYPE** B, int ldb,
                    int batchCount){

  KBlasWorkspaceState ws_needed;
  trmm_batch_wsquery_core<false>( batchCount,
                                  side, m, n,
                                  (kblasWorkspaceState_t)&ws_needed);

  bool suffWorkspace = (ws_needed.d_ptrs_bytes <= handle->work_space.allocated_ws_state.d_ptrs_bytes);

  if(!suffWorkspace){
    return KBLAS_InsufficientWorkspace;
  }

  return Xtrmm_batch_core<TYPE, TYPE**, false> (
                          handle,
                          side, uplo, trans, diag,
                          m, n,
                          alpha,
                          (TYPE**)A, 0, 0, lda, (long)0,
                          (TYPE**)B, 0, 0, ldb, (long)0,
                          batchCount);
}

} //extern "C"

//==============================================================================================
//Strided form
// template<>

// workspace needed: device pointers
// A, B: host pointer to array of device pointers to device buffers
int kblas_trmm_batch(kblasHandle_t handle,
                     char side, char uplo, char trans, char diag,
                     const int m, const int n,
                     const TYPE alpha,
                     const TYPE* A, int lda, long strideA,
                           TYPE* B, int ldb, long strideB,
                    int batchCount){

  KBlasWorkspaceState ws_needed;
  trmm_batch_wsquery_core<true>(batchCount,
                                side, m, n,
                                (kblasWorkspaceState_t)&ws_needed);

  bool suffWorkspace = (ws_needed.d_ptrs_bytes <= handle->work_space.allocated_ws_state.d_ptrs_bytes);

  if(!suffWorkspace){
    return KBLAS_InsufficientWorkspace;
  }

  return Xtrmm_batch_core<TYPE, TYPE*, true>(
                          handle,
                          side, uplo, trans, diag,
                          m, n,
                          alpha,
                          (TYPE*)A, 0, 0, lda, strideA,
                          (TYPE*)B, 0, 0, ldb, strideB,
                          batchCount);
}

extern "C" {

// workspace needed: device pointers
// A, B: host pointer to device buffers
int kblasXtrmm_batch_strided(kblasHandle_t handle,
                             char side, char uplo, char trans, char diag,
                             const int m, const int n,
                             const TYPE alpha,
                             const TYPE* A, int lda, long strideA,
                                   TYPE* B, int ldb, long strideB,
                             int batchCount){

  KBlasWorkspaceState ws_needed;
  trmm_batch_wsquery_core<true>(batchCount,
                                side, m, n,
                                (kblasWorkspaceState_t)&ws_needed);

  bool suffWorkspace = (ws_needed.d_ptrs_bytes <= handle->work_space.allocated_ws_state.d_ptrs_bytes);

  if(!suffWorkspace){
    return KBLAS_InsufficientWorkspace;
  }

  return Xtrmm_batch_core<TYPE, TYPE*, true> (
                          handle,
                          side, uplo, trans, diag,
                          m, n,
                          alpha,
                          (TYPE*)A, 0, 0, lda, strideA,
                          (TYPE*)B, 0, 0, ldb, strideB,
                          batchCount);
}

}//extern C
