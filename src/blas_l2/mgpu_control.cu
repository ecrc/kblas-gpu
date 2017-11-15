/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/mgpu_control.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 2.0.0
 * @author Ahmad Abdelfattah
 * @date 2017-11-13
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
