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
#include "test_trsm.ch"


extern int kblas_trsm_ib_cublas;
extern bool kblas_trsm_use_custom;

//==============================================================================================
int main(int argc, char** argv)
{
  kblas_opts opts;
  if(!parse_opts( argc, argv, &opts )){
    USAGE;
    return -1;
  }

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  float alpha = 0.29;
  kblas_trsm_ib_cublas = opts.nb;
  kblas_trsm_use_custom = (bool)opts.custom;
  test_trsm<float>(opts, alpha, cublas_handle);
  
  cublasDestroy(cublas_handle);
}


