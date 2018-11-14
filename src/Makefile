include ../make.inc

KBLAS_ROOT= ..

INC=-I${KBLAS_ROOT}/include
INC+= -I.
INC+= -I${_CUDA_ROOT_}/include
ifdef _USE_MAGMA_
INC+= -I$(_MAGMA_ROOT_)/include
endif

OBJ_DIR=./obj
ALL_OBJ=

DEPS_COM= ../make.inc kblas_common.h

ifdef _SUPPORT_BLAS2_
KBLAS_SRC2_DIR=./blas_l2
include $(KBLAS_SRC2_DIR)/Makefile
endif

ifdef _SUPPORT_BLAS3_
KBLAS_SRC3_DIR=./blas_l3
include $(KBLAS_SRC3_DIR)/Makefile
endif

ifdef _SUPPORT_BATCH_TR_
KBLAS_BATCH_DIR=./batch_triangular
include $(KBLAS_BATCH_DIR)/Makefile
endif

ifdef _SUPPORT_TLR_
KBLAS_TLR_DIR=./blas_tlr
include $(KBLAS_TLR_DIR)/Makefile
endif

ifdef _SUPPORT_SVD_
KBLAS_BATCH_SVD_DIR=./batch_svd
include $(KBLAS_BATCH_SVD_DIR)/Makefile
endif

KBLAS_COM_CU = kblas_common.cu workspace_queries.cu
KBLAS_COM_CU += sblas_core.cu dblas_core.cu cblas_core.cu zblas_core.cu

ALL_OBJ+= $(patsubst %.cu,$(OBJ_DIR)/%.cu_o,$(KBLAS_COM_CU))

$(OBJ_DIR)/%blas_common.cu_o: %blas_common.cu $(DEPS_COM)
	$(NVCC) $(NVOPTS) $(NVOPTS_3) $(INC) -c $< -o $@
	# $(CXX) $(COPTS) $(INC) -c $< -o $@

$(OBJ_DIR)/%blas_core.cu_o: Xblas_core.cu Xblas_core.ch $(DEPS_COM)
	$(NVCC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) $(INC) -c $< -o $@

$(OBJ_DIR)/%_queries.cu_o: %_queries.cu workspace_queries.ch kblas_struct.h $(DEPS_COM)
	$(NVCC) $(NVOPTS) $(NVOPTS_3) $(INC) -c $< -o $@


KBLAS_DIR=$(KBLAS_ROOT)/lib
LIBKBLAS=$(KBLAS_DIR)/lib$(LIB_KBLAS_NAME).a

all: $(LIBKBLAS)

$(LIBKBLAS): $(ALL_OBJ)
	ar -cru $(LIBKBLAS) $(ALL_OBJ)
	ranlib $@

#cp $(LIBKBLAS) ${KBLAS_ROOT}/lib

clean:
	rm -f $(OBJ_DIR)/*.cu_o $(OBJ_DIR)/*.o
