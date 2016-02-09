#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"

#include "kblas.h"
#include "testing_utils.h"
#include "operators.h"
#include "test_trmm_cpu.ch"


extern int kblas_trmm_ib_custom;
extern int kblas_trmm_ib_cublas;
extern bool kblas_trmm_use_custom;

//==============================================================================================
int main(int argc, char** argv)
{
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  kblas_opts opts;
  if(!parse_opts( argc, argv, &opts )){
    USAGE;
    return -1;
  }

  cuFloatComplex alpha = make_cuFloatComplex(1.2, -0.6);
  kblas_trmm_ib_custom = opts.nb;
  kblas_trmm_ib_cublas = opts.nb;
  kblas_trmm_use_custom = (bool)opts.custom;
  test_trmm<cuFloatComplex>(opts, alpha, cublas_handle);
  
  cublasDestroy(cublas_handle);
}


