#WARNING: Don't invoke this file independently, must be included into parent folder makefile

#TODO what is minimum cuda version needed

#TODO list any further dependencies needed
# DEPS_BATCH=$(DEPS_COM) kblas_struct.h kblas_gpu_timer.h kblas_error.h

KBLAS_BATCH_SVD_CU_SRC=  \
			thrust_wrappers.cu \
			batch_qr.cu \
			batch_transpose.cu \
			batch_block_copy.cu \
			batch_rand.cu

ifdef _SUPPORT_SVD_
KBLAS_BATCH_SVD_CU_SRC+=batch_svd.cu \
												ssvd_full_batch.cu  \
												dsvd_full_batch.cu  \
												saca_batch.cu 			\
												daca_batch.cu
endif

ALL_OBJ+= $(patsubst %.cu,$(OBJ_DIR)/%.cu_o,$(KBLAS_BATCH_SVD_CU_SRC))

SVD_INC  := $(INC) -I$(KBLAS_BATCH_SVD_DIR)
SVD_INC  += -I$(_CUB_DIR_)/

NVCCFLAGS   := -m64 --use_fast_math 
# --maxrregcount=255

$(OBJ_DIR)/%aca_batch.cu_o: $(KBLAS_BATCH_SVD_DIR)/Xaca_batch.cu $(KBLAS_BATCH_SVD_DIR)/Xaca_batch_core.cuh
	$(NVCC) $(SVD_INC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) -c $< -o $@

$(OBJ_DIR)/%svd_full_batch.cu_o: $(KBLAS_BATCH_SVD_DIR)/Xsvd_full_batch.cu $(KBLAS_BATCH_SVD_DIR)/Xsvd_full_batch_core.cuh
	$(NVCC) $(SVD_INC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) -c $< -o $@

$(OBJ_DIR)/%.cu_o: $(KBLAS_BATCH_SVD_DIR)/%.cu
	$(NVCC) $(SVD_INC) $(NVOPTS) $(NVOPTS_3) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%_qr.cu_o: $(KBLAS_BATCH_SVD_DIR)/%_qr.cu $(KBLAS_BATCH_SVD_DIR)/qr_kernels.cuh
	$(NVCC) $(SVD_INC) $(NVOPTS) $(NVOPTS_3) $(NVCCFLAGS) -c $< -o $@
