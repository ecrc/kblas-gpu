#WARNING: Don't invoke this file independently, must be included into parent folder makefile

DEPS_BATCH=$(DEPS_COM) kblas_struct.h kblas_prec_def.h kblas_error.h

KBLAS_BATCH_SRC =
KBLAS_BATCH_SRC+= 		 \
			shelper_funcs.cu \
			sgemm_batch.cu   \
			ssyrk_batch.cu   \
			strsm_batch.cu
KBLAS_BATCH_SRC+= 		 \
			dhelper_funcs.cu \
			dgemm_batch.cu   \
			dsyrk_batch.cu   \
			dtrsm_batch.cu
KBLAS_BATCH_SRC+= 		 \
			chelper_funcs.cu \
			cgemm_batch.cu   \
			csyrk_batch.cu   \
			ctrsm_batch.cu
KBLAS_BATCH_SRC+=      \
			zhelper_funcs.cu \
			zgemm_batch.cu   \
			zsyrk_batch.cu   \
			ztrsm_batch.cu

ALL_OBJ+= $(patsubst %.cu,$(OBJ_DIR)/%.cu_o,$(KBLAS_BATCH_SRC))


$(OBJ_DIR)/%helper_funcs.cu_o: $(KBLAS_BATCH_DIR)/Xhelper_funcs.cu $(DEPS_BATCH)
	$(NVCC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) $(INC) -c $< -o $@

$(OBJ_DIR)/%gemm_batch.cu_o: $(KBLAS_BATCH_DIR)/Xgemm_batch.cu $(KBLAS_BATCH_DIR)/Xgemm_batch_core.cuh $(DEPS_BATCH)
	$(NVCC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) $(INC) -c $< -o $@

$(OBJ_DIR)/%syrk_batch.cu_o: $(KBLAS_BATCH_DIR)/Xsyrk_batch.cu $(KBLAS_BATCH_DIR)/Xsyrk_batch_drivers.cuh $(KBLAS_BATCH_DIR)/Xsyrk_batch_kernels.cuh $(DEPS_BATCH)
	$(NVCC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) $(INC) -c $< -o $@

$(OBJ_DIR)/%trsm_batch.cu_o: $(KBLAS_BATCH_DIR)/Xtrsm_batch.cu $(KBLAS_BATCH_DIR)/Xtrsm_batch_drivers.cuh $(KBLAS_BATCH_DIR)/Xtrsm_batch_kernels.cuh $(DEPS_BATCH)
	$(NVCC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) $(INC) -c $< -o $@

