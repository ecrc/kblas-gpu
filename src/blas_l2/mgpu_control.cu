/**
 -- (C) Copyright 2013 King Abdullah University of Science and Technology
  Authors:
  Ahmad Abdelfattah (ahmad.ahmad@kaust.edu.sa)
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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <stdio.h>
#include "mgpu_control.h"

/*****************************************************************************************/
extern "C"
void kblas_smalloc_mgpu_1D(	int rows, int cols, float** dA, int ngpus, int ldb, int block_size)
{
    kblas_malloc_mgpu_1D<float>(rows, cols, dA, ngpus, ldb, block_size);
}

extern "C"
void kblas_dmalloc_mgpu_1D(	int rows, int cols, double** dA, int ngpus, int ldb, int block_size)
{
    kblas_malloc_mgpu_1D<double>(rows, cols, dA, ngpus, ldb, block_size);
}

extern "C"
void kblas_cmalloc_mgpu_1D(	int rows, int cols, cuFloatComplex** dA, int ngpus, int ldb, int block_size)
{
    kblas_malloc_mgpu_1D<cuFloatComplex>(rows, cols, dA, ngpus, ldb, block_size);
}

extern "C"
void kblas_zmalloc_mgpu_1D(	int rows, int cols, cuDoubleComplex** dA, int ngpus, int ldb, int block_size)
{
    kblas_malloc_mgpu_1D<cuDoubleComplex>(rows, cols, dA, ngpus, ldb, block_size);
}

/*****************************************************************************************/
extern "C"
void kblas_ssetmatrix_mgpu_1D(int rows, int cols, float* A, int LDA, float** dA, int LDB, int ngpus, int block_size)
{
    kblas_setmatrix_mgpu_1D<float>(rows, cols, A, LDA, dA, LDB, ngpus, block_size);
}

extern "C"
void kblas_dsetmatrix_mgpu_1D(int rows, int cols, double* A, int LDA, double** dA, int LDB, int ngpus, int block_size)
{
    kblas_setmatrix_mgpu_1D<double>(rows, cols, A, LDA, dA, LDB, ngpus, block_size);
}

extern "C"
void kblas_csetmatrix_mgpu_1D(int rows, int cols, cuFloatComplex* A, int LDA, cuFloatComplex** dA, int LDB, int ngpus, int block_size)
{
    kblas_setmatrix_mgpu_1D<cuFloatComplex>(rows, cols, A, LDA, dA, LDB, ngpus, block_size);
}

extern "C"
void kblas_zsetmatrix_mgpu_1D(int rows, int cols, cuDoubleComplex* A, int LDA, cuDoubleComplex** dA, int LDB, int ngpus, int block_size)
{
    kblas_setmatrix_mgpu_1D<cuDoubleComplex>(rows, cols, A, LDA, dA, LDB, ngpus, block_size);
}
/*****************************************************************************************/
extern "C"
void kblas_ssetvector_mgpu_1D(int n, float* Y, float** dY, int ngpus, int block_size)
{
    kblas_setvector_mgpu_1D<float>(n, Y, dY, ngpus, block_size);
}

extern "C"
void kblas_dsetvector_mgpu_1D(int n, double* Y, double** dY, int ngpus, int block_size)
{
    kblas_setvector_mgpu_1D<double>(n, Y, dY, ngpus, block_size);
}

extern "C"
void kblas_csetvector_mgpu_1D(int n, cuFloatComplex* Y, cuFloatComplex** dY, int ngpus, int block_size)
{
    kblas_setvector_mgpu_1D<cuFloatComplex>(n, Y, dY, ngpus, block_size);
}

extern "C"
void kblas_zsetvector_mgpu_1D(int n, cuDoubleComplex* Y, cuDoubleComplex** dY, int ngpus, int block_size)
{
    kblas_setvector_mgpu_1D<cuDoubleComplex>(n, Y, dY, ngpus, block_size);
}
