/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_l3/test_dgemm.c

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
#include "test_gemm.ch"


//==============================================================================================
int main(int argc, char** argv)
{
  hipblasHandle_t cublas_handle;
  hipblasCreate(&cublas_handle);
  
  kblas_opts opts;
  if(!parse_opts( argc, argv, &opts )){
    USAGE;
    return -1;
  }
  
  double alpha = 1.2;
  double beta = 0.6;
  test_gemm<double>(opts, alpha, beta, cublas_handle);
  
  hipblasDestroy(cublas_handle);
}


