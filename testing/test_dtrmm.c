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
#include <cublas.h>

#include "kblas.h"
#include "testing_utils.h"
#include "test_trmm.ch"

extern int kblas_trmm_ib_custom;
extern int kblas_trmm_ib_cublas;
extern bool kblas_trmm_use_custom;

//==============================================================================================
int main(int argc, char** argv)
{
  if( CUBLAS_STATUS_SUCCESS != cublasInit() ) {
    fprintf(stderr, "ERROR: cublasInit failed\n");
    exit(-1);
  }
  
  kblas_opts opts;
  if(!parse_opts( argc, argv, &opts )){
    USAGE;
    return -1;
  }
  
  double alpha = 0.29;
  kblas_trmm_ib_custom = opts.nb;
  kblas_trmm_ib_cublas = opts.nb;
  kblas_trmm_use_custom = (bool)opts.custom;
  test_trmm<double>(opts, alpha);
  
  cublasShutdown();
}


