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
#include "test_gemm.ch"


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
  
  float alpha = 1.2;
  float beta = 0.6;
  test_gemm<float>(opts, alpha, beta, cublas_handle);
  
  cublasDestroy(cublas_handle);
}


