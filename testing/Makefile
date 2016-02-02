include ../make.inc

# ROOTS
_KBLAS_ROOT_=..

# include and lib paths
INCLUDES=
INCLUDES+= -I${_CUDA_ROOT_}/include
INCLUDES+= -I${_KBLAS_ROOT_}/include

LIB_PATH=
LIB_PATH+= -L${_CUDA_ROOT_}/lib64
LIB_PATH+= -L${_KBLAS_ROOT_}/lib

# libraries to link against
LIB= -lm -lkblas
LIB+= -lcublas -lcudart

KBLAS_LIB=../lib/${LIB_KBLAS_NAME}

C_SRC=
C_SRC+=	test_sscal.c \
	test_dscal.c \
	test_cscal.c \
	test_zscal.c \
	test_ssymv.c \
	test_dsymv.c \
	test_chemv.c \
	test_zhemv.c \
	test_sgemv.c \
	test_dgemv.c \
	test_cgemv.c \
	test_zgemv.c \
	test_ssymv_mgpu.c \
	test_dsymv_mgpu.c \
	test_chemv_mgpu.c \
	test_zhemv_mgpu.c \
	test_sgemv_mgpu.c \
	test_dgemv_mgpu.c \
	test_cgemv_mgpu.c \
	test_zgemv_mgpu.c \
	test_sgemv_offset.c \
	test_dgemv_offset.c \
	test_cgemv_offset.c \
	test_zgemv_offset.c	\
	test_ssymv_offset.c \
	test_dsymv_offset.c \
	test_chemv_offset.c \
	test_zhemv_offset.c \
	test_dgemm_mgpu.c \
	test_sgemm_mgpu.c \
	test_cgemm_mgpu.c \
	test_zgemm_mgpu.c \
	test_dgemm_mgpu_dim.c \
	test_sgemv2.c \
	test_dgemv2.c \
	test_cgemv2.c \
	test_zgemv2.c \
	test_sgemv2_offset.c \
	test_dgemv2_offset.c \
	test_cgemv2_offset.c \
	test_zgemv2_offset.c \
	test_sgemv2_dim.c \
	test_dgemv2_dim.c

C_SRC+=	              \
	test_dtrmm.c \
	test_strmm.c \
	test_ctrmm.c \
	test_ztrmm.c


INC_SRC=testing_utils.h

OBJ_DIR=./obj
C_OBJ = $(patsubst %.c,$(OBJ_DIR)/%.o,$(C_SRC))

#C_OBJ=$(C_SRC:.c=.o) $(C_SRC:.cpp=.o)
EXE=$(C_SRC:.c=)

$(OBJ_DIR)/test_%trmm.o: test_%trmm.c test_trmm.ch $(INC_SRC) $(KBLAS_LIB)
	$(CXX) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/%.o: %.c $(INC_SRC) $(KBLAS_LIB)
	$(CC) $(COPTS) $(INCLUDES) -c $< -o $@

test_%trmm: $(OBJ_DIR)/test_%trmm.o
	$(CC) $< -o $@ $(LIB_PATH) $(LIB)

test_%: $(OBJ_DIR)/test_%.o 
	$(CC) $< -o $@ $(LIB_PATH) $(LIB)

all: $(EXE)

$(EXE): $(C_OBJ)

clean:
	rm -f $(OBJ_DIR)/*.o $(EXE)

