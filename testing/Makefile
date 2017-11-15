include ../make.inc

# print-%  : ; @echo $* = $($*)

# ROOTS
_KBLAS_ROOT_=..

# include and lib paths
INCLUDES=
INCLUDES+= -I.
INCLUDES+= -I${_CUDA_ROOT_}/include
INCLUDES+= -I${_KBLAS_ROOT_}/include -I${_KBLAS_ROOT_}/src
ifdef _USE_MAGMA_
INCLUDES+= -I$(_MAGMA_ROOT_)/include
endif
ifdef _USE_MKL_
INCLUDES+= -I${_MKL_ROOT_}/include
endif

LIB_PATH=
LIB_PATH+= -L${_CUDA_ROOT_}/lib64
LIB_PATH+= -L${_KBLAS_ROOT_}/lib
ifdef _USE_MAGMA_
LIB_PATH+= -L${_MAGMA_ROOT_}/lib
endif
ifdef _USE_MKL_
LIB_PATH+= -L${_MKL_ROOT_}/lib/intel64
endif

# libraries to link against
LIB= -lm -l${LIB_KBLAS_NAME}
ifdef _USE_MAGMA_
LIB+= -lmagma -lcusparse
endif
LIB+= -lcublas -lcudart
ifdef _USE_MKL_
LIB+= -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
# LIB+= -mkl=sequential
# LIB+= -mkl=parallel
endif
LIB+= -lgomp
LIB+= -lstdc++

KBLAS_LIB=../lib/lib${LIB_KBLAS_NAME}.a

C_SRC=
INC_SRC=testing_utils.h testing_Xtr_common.h

OBJ_DIR=./obj
# C_OBJ =

BIN_DIR=./bin

ifdef _SUPPORT_BLAS2_
KBLAS_TEST2_DIR=blas_l2
include $(KBLAS_TEST2_DIR)/Makefile
endif

ifdef _SUPPORT_BLAS3_
KBLAS_TEST3_DIR=blas_l3
include $(KBLAS_TEST3_DIR)/Makefile
endif

ifdef _SUPPORT_BATCH_TR_
KBLAS_BATCH_DIR=batch_triangular
include $(KBLAS_BATCH_DIR)/Makefile
endif

# C_OBJ= $(patsubst %.c,$(OBJ_DIR)/%.o,$(C_SRC))
# EXE= $(patsubst %.c,$(BIN_DIR)/%,$(C_SRC))

all: $(EXE_l2) $(EXE_l3) $(EXE_BATCH)

$(OBJ_DIR)/testing_helper.o: testing_helper.cu
	$(NVCC) $(NVOPTS) $(NVOPTS_3) $(INCLUDES) $(NVCCFLAGS) -c $< -o $@

$(EXE_BATCH): $(BIN_DIR)/%: $(OBJ_DIR)/%.o $(KBLAS_LIB) $(OBJ_DIR)/testing_helper.o
	$(CC) $(COPTS) $(OBJ_DIR)/testing_helper.o $< -o $@ $(LIB_PATH) $(LIB)

$(EXE_l2): $(BIN_DIR)/%: $(OBJ_DIR)/%.o $(KBLAS_LIB)
	$(CC) $(COPTS) $< -o $@ $(LIB_PATH) $(LIB)

$(EXE_l3): $(BIN_DIR)/%: $(OBJ_DIR)/%.o $(KBLAS_LIB)
	$(CC) $(COPTS) $< -o $@ $(LIB_PATH) $(LIB)

clean:
	rm -f $(OBJ_DIR)/*.o $(EXE_l2) $(EXE_l3) $(EXE_BATCH)
