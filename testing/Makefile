include ../make.inc

# print-%  : ; @echo $* = $($*)

# ROOTS
_KBLAS_ROOT_=..

# include and lib paths
INCLUDES=
INCLUDES+= -I.
INCLUDES+= -I${_CUDA_ROOT_}/include
INCLUDES+= -I${_KBLAS_ROOT_}/include

LIB_PATH=
LIB_PATH+= -L${_CUDA_ROOT_}/lib64
LIB_PATH+= -L${_KBLAS_ROOT_}/lib

# libraries to link against
LIB= -lm -l${LIB_KBLAS_NAME}
LIB+= -lcublas -lcudart
# LIB+= -liomp5
LIB+= -lstdc++

KBLAS_LIB=../lib/lib${LIB_KBLAS_NAME}.a

C_SRC=
INC_SRC=testing_utils.h

OBJ_DIR=./obj
C_OBJ =

BIN_DIR=./bin

KBLAS_TEST2_DIR=blas_l2
KBLAS_TEST3_DIR=blas_l3

include $(KBLAS_TEST2_DIR)/Makefile
include $(KBLAS_TEST3_DIR)/Makefile

C_OBJ= $(patsubst %.c,$(OBJ_DIR)/%.o,$(C_SRC))
EXE= $(patsubst %.c,$(BIN_DIR)/%,$(C_SRC))

all: $(EXE)

$(EXE): $(BIN_DIR)/%: $(OBJ_DIR)/%.o
	$(CC) $(COPTS) $< -o $@ $(LIB_PATH) $(LIB)
clean:
	rm -f $(OBJ_DIR)/*.o $(EXE)
