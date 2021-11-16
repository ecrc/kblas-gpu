/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_l3/test_ztrmm.c

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "hipblas.h"

#include "kblas.h"
#include "testing_utils.h"
#include "kblas_operators.h"
#include "test_trmm.ch"


extern int kblas_trmm_ib_custom;
extern int kblas_trmm_ib_cublas;
extern bool kblas_trmm_use_custom;

//==============================================================================================
int main(int argc, char** argv)
{
  kblas_opts opts;
  if(!parse_opts( argc, argv, &opts )){
    USAGE;
    return -1;
  }
  
  hipblasHandle_t cublas_handle;
  hipblasCreate(&cublas_handle);
  
  kblas_trmm_ib_custom = opts.nb;
  kblas_trmm_ib_cublas = opts.nb;
  kblas_trmm_use_custom = (bool)opts.custom;
  hipDoubleComplex alpha = make_hipDoubleComplex(1.2, -0.6);
  test_trmm<hipDoubleComplex>(opts, alpha, cublas_handle);
  
  hipblasDestroy(cublas_handle);
}


