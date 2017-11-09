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
#ifndef __TESTING_PREC_DEF__
#define __TESTING_PREC_DEF__

#include "kblas_prec_def.h"

#if defined   PREC_s
  #define cublasXtrsm_batched cublasStrsmBatched
  #define LAPACK_GEMM sgemm
  #define LAPACK_GEMM_BATCH sgemm_batch
  #define LAPACK_SYRK ssyrk
  #define LAPACK_AXPY saxpy
  #define LAPACK_LANSY slansy
  #define LAPACK_TRSM strsm
  #define LAPACK_TRMM strmm

#elif defined PREC_d
  #define cublasXtrsm_batched cublasDtrsmBatched
  #define LAPACK_GEMM dgemm
  #define LAPACK_GEMM_BATCH dgemm_batch
  #define LAPACK_SYRK dsyrk
  #define LAPACK_AXPY daxpy
  #define LAPACK_LANSY dlansy
  #define LAPACK_TRSM dtrsm
  #define LAPACK_TRMM dtrmm

#elif defined PREC_c
  #define cublasXtrsm_batched cublasCtrsmBatched
  #define LAPACK_GEMM cgemm
  #define LAPACK_GEMM_BATCH cgemm_batch
  #define LAPACK_SYRK csyrk
  #define LAPACK_AXPY caxpy
  #define LAPACK_LANSY clansy
  #define LAPACK_TRSM ctrsm
  #define LAPACK_TRMM ctrmm

#elif defined PREC_z
  #define cublasXtrsm_batched cublasZtrsmBatched
  #define LAPACK_GEMM zgemm
  #define LAPACK_GEMM_BATCH zgemm_batch
  #define LAPACK_SYRK zsyrk
  #define LAPACK_AXPY zaxpy
  #define LAPACK_LANSY zlansy
  #define LAPACK_TRSM ztrsm
  #define LAPACK_TRMM ztrmm
#endif

#endif //__KBLAS_PREC_DEF__
