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

#include "kblas.h"
#include "kblas_struct.h"
#include "operators.h"
#include "defs.h"
#include "kblas_common.h"

//==============================================================================================
#include "Xblas_core.ch"
#include "Xhelper_funcs.ch"
#include "Xsyrk_batch_drivers.cuh"

extern "C" {
//==============================================================================================
//Non-Strided form
// template<>
void kblasXsyrk_batch_wsquery(const int m, int batchCount, kblasWorkspace_t ws){
  Xsyrk_batch_wsquery_core<TYPE>(m, batchCount, ws);
}

// workspace needed: device pointers
// A, B: host pointer to array of device pointers to device buffers
int kblasXsyrk_batch(kblasHandle_t handle,
                    char uplo, char trans,
                    const int m, const int n,
                    const TYPE alpha, const TYPE** A, int lda,
                    const TYPE beta,        TYPE** B, int ldb,
                    int batchCount){
  return Xsyrk_batch_core(handle,
                          uplo, trans,
                          m, n,
                          alpha, A, lda,
                          beta,  B, ldb,
                          batchCount);
}

//==============================================================================================
//Strided form
// template<>
void kblasXsyrk_batch_strided_wsquery(const int m, int batchCount, kblasWorkspace_t ws){
  Xsyrk_batch_strided_wsquery_core<TYPE>(m, batchCount, ws);
}

// workspace needed: device pointers
// A, B: host pointer to device buffers
int kblasXsyrk_batch_strided(kblasHandle_t handle,
                            char uplo, char trans,
                            const int m, const int n,
                            const TYPE alpha, const TYPE* A, int lda, long strideA,
                            const TYPE beta,        TYPE* B, int ldb, long strideB,
                            int batchCount){
  return Xsyrk_batch_strided_core(handle,
                                  uplo, trans,
                                  m, n,
                                  alpha, A, lda, strideA,
                                  beta,  B, ldb, strideB,
                                  batchCount);
}

}//extern C
