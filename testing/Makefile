include ../make.inc

# print-%  : ; @echo $* = $($*)

# ROOTS
_KBLAS_ROOT_=..

# include and lib paths
INCLUDES=
INCLUDES+= -I.
INCLUDES+= -I${_CUDA_ROOT_}/include
INCLUDES+= -I${_KBLAS_ROOT_}/include -I${_KBLAS_ROOT_}/src
ifdef _USE_MKL_
INCLUDES+= -I${_MKL_ROOT_}/include
endif

LIB_PATH=
LIB_PATH+= -L${_CUDA_ROOT_}/lib64
LIB_PATH+= -L${_KBLAS_ROOT_}/lib
ifdef _USE_MKL_
LIB_PATH+= -L${_MKL_ROOT_}/lib/intel64
endif

# libraries to link against
LIB= -lm -l${LIB_KBLAS_NAME}
LIB+= -lcublas -lcudart
ifdef _USE_MKL_
# LIB+= -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
# LIB+= -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
# LIB+= -mkl=sequential
LIB+= -mkl=parallel
endif
# LIB+= -liomp5
LIB+= -lstdc++

KBLAS_LIB=../lib/lib${LIB_KBLAS_NAME}.a

C_SRC=
INC_SRC=testing_utils.h testing_Xtr_common.h

OBJ_DIR=./obj
C_OBJ =

BIN_DIR=./bin

KBLAS_TEST2_DIR=blas_l2
KBLAS_TEST3_DIR=blas_l3
KBLAS_BATCH_DIR=batch_triangular

include $(KBLAS_TEST2_DIR)/Makefile
include $(KBLAS_TEST3_DIR)/Makefile
include $(KBLAS_BATCH_DIR)/Makefile

C_OBJ= $(patsubst %.c,$(OBJ_DIR)/%.o,$(C_SRC))

$(BIN_DIR)/test_%: $(OBJ_DIR)/test_%.o
	$(CC) $(COPTS) $< -o $@ $(LIB_PATH) $(LIB)

EXE= $(patsubst %.c,$(BIN_DIR)/%,$(C_SRC))

all: $(EXE)

$(EXE): $(C_OBJ)

clean:
	rm -f $(OBJ_DIR)/*.o $(EXE)
