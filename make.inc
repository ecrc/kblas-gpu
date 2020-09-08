#
# @copyright (c) 2012- King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#


#
# @file make.inc
#
# KBLAS is a high performance CUDA library for subset of BLAS
#    and LAPACK routines optimized for NVIDIA GPUs.
# KBLAS is provided by KAUST.
#
# @version 3.0.0
# @author Ali Charara
# @author Ahmad Abdelfattah
# @date 2018-11-14
#

#-----------------------------------------
_SUPPORT_BLAS2_?=TRUE
_SUPPORT_BLAS3_?=TRUE
_SUPPORT_BATCH_TR_?=TRUE
_SUPPORT_TLR_?=TRUE
_SUPPORT_SVD_?=TRUE

#_DEBUG_?=TRUE

#-----------------------------------------
ifndef _USE_OPENBLAS_
_USE_MKL_?=TRUE
endif

#-----------------------------------------
_USE_MAGMA_?=TRUE

ifdef _SUPPORT_TLR_
ifndef _SUPPORT_SVD_
  $(error SVD support required for TLR support)
endif
ifndef _USE_MAGMA_
  $(error MAGMA required for TLR support)
endif
endif

#-----------------------------------------
#specify cuda directory
_CUDA_ROOT_=$(CUDA_HOME)
_CUDA_ARCH_ ?= 70

# specify compilers
CXX ?= g++
CC ?= gcc
NVCC=$(_CUDA_ROOT_)/bin/nvcc

#customize the name of the static kblas library
LIB_KBLAS_NAME=kblas-gpu

#-----------------------------------------
NVOPTS = -ccbin $(CXX) --compiler-options -fno-strict-aliasing
COPTS = -fopenmp

NVOPTS_3 = -DTARGET_SM=$(_CUDA_ARCH_) -arch sm_$(_CUDA_ARCH_) -Xcompiler -fopenmp

#-----------------------------------------
ifdef _DEBUG_
  COPTS += -g -Xcompiler -rdynamic
  NVOPTS += -G -g -lineinfo
else
  COPTS += -O3
  NVOPTS += -O3
endif

#-----------------------------------------
ifdef _SUPPORT_SVD_
  _CUB_DIR_?=$(HOME)/scratch/codes/cub-1.8.0
  #COPTS += -DSUPPORT_SVD -D__NO_NCCL__
  #NVOPTS += -DSUPPORT_SVD -D__NO_NCCL__
endif

#-----------------------------------------
ifdef _USE_MAGMA_
  COPTS += -DUSE_MAGMA
  _MAGMA_ROOT_?=$(HOME)/scratch/codes/magma-2.5.2
  NVOPTS += -DUSE_MAGMA
endif

#-----------------------------------------
ifdef _SUPPORT_BLAS2_
  # specify the gpu architecture for BLAS2 routines (fermi, kepler, volta)
  GPU_ARCH_2=volta

  ifeq (${GPU_ARCH_2}, volta)
    NVOPTS_2 = -DTARGET_SM=70 -arch sm_70
  else ifeq (${GPU_ARCH_2}, kepler)
	NVOPTS_2 = -DTARGET_SM=35 -arch sm_35
  else ifeq (${GPU_ARCH_2}, fermi)
    NVOPTS_2 = -DTARGET_SM=20 -arch sm_20
  else
    $(error GPU_ARCH_2, currently ${GPU_TARGET_2}, must be either volta, fermi or kepler for BLAS2 routines. Please edit your make.inc file)
  endif
endif #_SUPPORT_BLAS2_

#-----------------------------------------
ifdef _USE_MKL_
  COPTS += -DUSE_MKL
  NVOPTS += -DUSE_MKL
  _MKL_ROOT_?=${MKLROOT}
endif

#-----------------------------------------
ifdef _USE_OPENBLAS_
  COPTS += -DUSE_OPENBLAS
  NVOPTS += -DUSE_OPENBLAS

  OPENBLAS_INCLUDE?=${OPENBLAS_ROOT}/include
  OPENBLAS_LIB?=${OPENBLAS_ROOT}/lib

  _OPENBLAS_ROOT_?=${OPENBLAS_ROOT}
  _OPENBLAS_INCLUDE_?=${OPENBLAS_INCLUDE}
  _OPENBLAS_LIB_?=${OPENBLAS_LIB}
endif
