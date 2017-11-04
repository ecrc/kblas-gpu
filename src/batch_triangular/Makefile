#WARNING: Don't invoke this file independently, must be included into parent folder makefile

DEPS_BATCH=$(DEPS_COM) kblas_struct.h kblas_prec_def.h kblas_error.h

KBLAS_BATCH_SRC = \
			shelper_funcs.cu dhelper_funcs.cu chelper_funcs.cu zhelper_funcs.cu \
			sgemm_batch.cu dgemm_batch.cu cgemm_batch.cu zgemm_batch.cu \
			ssyrk_batch.cu dsyrk_batch.cu csyrk_batch.cu zsyrk_batch.cu

ALL_OBJ+= $(patsubst %.cu,$(OBJ_DIR)/%.cu_o,$(KBLAS_BATCH_SRC))


$(OBJ_DIR)/%helper_funcs.cu_o: $(KBLAS_BATCH_DIR)/Xhelper_funcs.cu $(DEPS_BATCH)
	$(NVCC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) $(INC) -c $< -o $@

$(OBJ_DIR)/%gemm_batch.cu_o: $(KBLAS_BATCH_DIR)/Xgemm_batch.cu $(KBLAS_BATCH_DIR)/Xgemm_batch_core.cuh $(DEPS_BATCH)
	$(NVCC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) $(INC) -c $< -o $@

$(OBJ_DIR)/%syrk_batch.cu_o: $(KBLAS_BATCH_DIR)/Xsyrk_batch.cu $(KBLAS_BATCH_DIR)/Xsyrk_batch_drivers.cuh $(KBLAS_BATCH_DIR)/Xsyrk_batch_kernels.cuh $(DEPS_BATCH)
	$(NVCC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) $(INC) -c $< -o $@

