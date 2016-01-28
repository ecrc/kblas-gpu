
#specify cuda directory
_CUDA_ROOT_=/usr/local/cuda

# specify compilers
CXX=g++
CC=gcc
NVCC=$(_CUDA_ROOT_)/bin/nvcc

#customize the name of the generated kblas library
LIB_KBLAS_NAME=libkblas.a

NVOPTS =-O3 --compiler-options -fno-strict-aliasing 
ifeq (${CC}, gcc)
  COPTS+=-std=c99
endif

NVOPTS_3 = -DSM=35 -arch sm_35 -lineinfo -Xptxas -v 

# specify the gpu architecture for BLAS2 routines (fermi, kepler)
GPU_ARCH_2=kepler

ifeq (${GPU_ARCH_2}, kepler)
  NVOPTS_2 = -DSM=35 -arch sm_35 
else ifeq (${GPU_ARCH}, fermi)
  NVOPTS_2 = -DSM=20 -arch sm_20
else
  $(error GPU_ARCH_2, currently ${GPU_TARGET_2}, must be either fermi or kepler for BLAS2 routines. Please edit your make.inc file)
endif
