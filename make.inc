
#_SUPPORT_BLAS2_=TRUE
_SUPPORT_BLAS3_=TRUE

#_SUPPORT_CUBLAS_=TRUE
_SUPPORT_MKL_=TRUE

ifndef _SUPPORT_BLAS2_
  ifndef _SUPPORT_BLAS3_
    $(error: Please enable BLAS2 and/or BLAS3 support. Please edit your make.inc file)
  endif
endif
ifdef _SUPPORT_BLAS3_
  ifndef _SUPPORT_CUBLAS_
    ifndef _SUPPORT_MKL_
      $(error: Please enable MKL and/or CUBLAS for BLAS3 support. Please edit your make.inc file)
    endif
  endif
endif

# specify compilers
ifdef _SUPPORT_BLAS3_
  ifdef _SUPPORT_MKL_
    CC=icpc
  else
    CC=g++
  endif
else
  CC=gcc
endif

#customize the name of the generated kblas library
LIB_KBLAS_NAME=libkblas.a

ifdef _SUPPORT_BLAS2_
  
  _USE_CUDA_=TRUE
  
  NVOPTS+=-DSUPPORT_BLAS2
  COPTS+=-DSUPPORT_BLAS2

  ifeq (${CC}, gcc)
    COPTS+=-std=c99
  endif

endif #_SUPPORT_BLAS2_

######################## TRMM & TRSM support options

ifdef _SUPPORT_BLAS3_

  NVOPTS+=-DSUPPORT_BLAS3
  COPTS+=-DSUPPORT_BLAS3

  ifdef _SUPPORT_CUBLAS_
    _USE_CUDA_=TRUE
    NVOPTS += -DSUPPORT_CUBLAS
    COPTS += -DSUPPORT_CUBLAS
  endif

  ifdef _SUPPORT_MKL_
    _USE_MKL_=TRUE
    COPTS += -DSUPPORT_MKL -DUSE_MKL
    _MKL_ROOT_=/opt/share/INTEL/v15.app/mkl
  endif

endif #_SUPPORT_BLAS3_


ifdef _USE_CUDA_

  NVCC=$(_CUDA_ROOT_)/bin/nvcc

  #specify cuda directory
  _CUDA_ROOT_=/usr/local/cuda

  # specify the gpu architecture (fermi, kepler)
  GPU_ARCH=kepler

  NVOPTS+=-O3 --compiler-options -fno-strict-aliasing -DUSE_CUDA
  COPTS += -DUSE_CUDA

  ifeq (${GPU_ARCH}, kepler)
    NVOPTS += -DSM=35 -arch sm_35 
  else ifeq (${GPU_ARCH}, fermi)
    NVOPTS += -DSM=20 -arch sm_20
  else
    $(error GPU_ARCH, currently ${GPU_TARGET}, must be either fermi or kepler. Please edit your make.inc file)
  endif

endif
