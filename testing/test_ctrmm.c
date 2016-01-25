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
  
  cuFloatComplex alpha = make_cuFloatComplex(1.2, -0.6);
  test_trmm<cuFloatComplex>(opts, alpha);
  
  cublasShutdown();
}


