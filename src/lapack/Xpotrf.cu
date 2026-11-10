/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/lapack/Xpotrf.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ali Charara
 * @author Wajih Halim Boukaram
 * @date 2020-12-10
 **/
 
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include "kblas.h"
#include "kblas_common.h"
#include "kblas_operators.h"

// #define DBG_MSG

#include "kblas_struct.h"
#include "kblas_prec_def.h"
#include "kblas_gpu_util.ch"

#include "Xblas_core.ch"
#include "Xhelper_funcs.ch"
#include "Xpotrf_core.cuh"

#include "kblas_potrf.h"

extern "C"
int kblasXpotrf(kblasHandle_t handle,
                char uplo, int n,
                TYPE *dA, int ldda,
                int* info)
{
  return Xpotrf(handle, uplo, n, dA, ldda, info);
}
