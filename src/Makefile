include ../make.inc

KBLAS_ROOT= ..

INC=-I${KBLAS_ROOT}/include
INC+= -I.
INC+= -I${_CUDA_ROOT_}/include

OBJ_DIR=./obj
ALL_OBJ=

DEPS_COM= ../make.inc kblas_common.h

KBLAS_SRC2_DIR=blas_l2
KBLAS_SRC3_DIR=blas_l3

include $(KBLAS_SRC2_DIR)/Makefile
include $(KBLAS_SRC3_DIR)/Makefile

KBLAS3_COM_SRC= kblas_common.cpp
ALL_OBJ+= $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(KBLAS3_COM_SRC))

$(OBJ_DIR)/%blas_common.o: %blas_common.cpp $(DEPS_COM)
	$(CXX) $(COPTS) $(INC) -c $< -o $@


KBLAS_DIR=$(KBLAS_ROOT)/lib
LIBKBLAS=$(KBLAS_DIR)/lib$(LIB_KBLAS_NAME).a

all: $(LIBKBLAS)

$(LIBKBLAS): $(ALL_OBJ)
	ar -cru $(LIBKBLAS) $(ALL_OBJ)
	ranlib $@

#cp $(LIBKBLAS) ${KBLAS_ROOT}/lib

clean:
	rm -f $(OBJ_DIR)/*.cu_o
