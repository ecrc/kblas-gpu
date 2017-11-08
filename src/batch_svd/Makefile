#WARNING: Don't invoke this file independently, must be included into parent folder makefile

#TODO what is minimum cuda version needed

#TODO list any further dependencies needed
DEPS_BATCH=$(DEPS_COM) kblas_struct.h kblas_gpu_timer.h kblas_error.h

KBLAS_BATCH_SVD_CU_SRC=  \
			thrust_wrappers.cu \
			batch_qr.cu \
			batch_svd.cu \
			batch_transpose.cu \
			batch_block_copy.cu 

ALL_OBJ+= $(patsubst %.cu,$(OBJ_DIR)/%.cu_o,$(KBLAS_BATCH_SVD_CU_SRC))

SVD_INC  := $(INC) -I$(KBLAS_BATCH_SVD_DIR)
SVD_INC  += -I$(_CUB_DIR_)/

NVCCFLAGS   := -m64 --use_fast_math --maxrregcount=255

$(OBJ_DIR)/%.cu_o: $(KBLAS_BATCH_SVD_DIR)/%.cu
	$(NVCC) $(SVD_INC) $(NVOPTS) $(NVOPTS_3) $(NVCCFLAGS) -c $< -o $@
