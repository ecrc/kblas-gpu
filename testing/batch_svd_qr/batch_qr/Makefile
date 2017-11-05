################################################################################
# Location of the CUDA Toolkit
CUDA_PATH=$(CUDA_ROOT)
MAGMA_DIR=/home/chararam/codes/magma-2.2.0
KBLAS_DIR=../../..
SRC_DIR=$(KBLAS_DIR)/src
NVCC=nvcc
CC=gcc
KBLAS_LIB=-lkblas-gpu
################################################################################

################################################################################
# internal flags
NVCCFLAGS   := -m64 -O3
NVCCLDFLAGS := -lcudart -lcublas -lmagma -lcusparse

CCFLAGS     := -fopenmp -m64 -DMKL_INT=int -D__NO_NCCL__ -DHLIB_PROFILING_ENABLED -DDOUBLE_PRECISION
LDFLAGS     := -ldl -lpthread -lm -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -liomp5 -lgomp

ALL_CCFLAGS := $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS := $(ALL_CCFLAGS)
ALL_LDFLAGS += $(KBLAS_LIB) $(NVCCLDFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(EXTRA_NVCCLDFLAGS)
################################################################################

################################################################################
# Common includes and paths for CUDA
INCLUDES  := -I$(SRC_DIR)/batch_svd -I$(CUDA_PATH)/include/ -I$(SRC_DIR)/ -I$(KBLAS_DIR)/include/
INCLUDES  += -I$(MAGMA_DIR)/include
INCLUDES  += -I$(MKLROOT)/include

LIBRARIES := -L$(MAGMA_DIR)/lib -L$(MKLROOT)/lib/intel64 -L$(KBLAS_DIR)/lib
################################################################################

################################################################################
# CUDA code generation flags
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_FLAGS   := $(GENCODE_SM35) $(GENCODE_SM60)
################################################################################

################################################################################
# Regular C files
OBJDIR := ../../obj
VPATH = $(SRC_DIR)/:$(SRC_DIR)/batch_svd

OBJ_LIST :=
OBJS := $(addprefix $(OBJDIR)/,$(OBJ_LIST))

CUDA_OBJ_LIST = test_batch_qr.o
# CUDA_OBJ_LIST = test_batch_qr.o batch_qr.o thrust_wrappers.o

CUDA_OBJS := $(addprefix $(OBJDIR)/,$(CUDA_OBJ_LIST))
################################################################################

################################################################################
# Target rules
all: build

build: test_batch_qr

$(OBJDIR)/%.o: %.cpp
	$(CC) $(INCLUDES) $(CCFLAGS) -o $@ -c $<

$(OBJDIR)/%.o: %.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

test_batch_qr: $(OBJS) $(CUDA_OBJS)
	$(NVCC) $(ALL_LDFLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f horthog $(OBJS) $(CUDA_OBJS)

clobber: clean
################################################################################