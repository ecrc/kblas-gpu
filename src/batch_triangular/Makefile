#WARNING: Don't invoke this file independently, must be included into parent folder makefile

DEPS_BATCH=$(DEPS_COM) kblas_struct.h kblas_prec_def.h kblas_error.h $(KBLAS_BATCH_DIR)/batch_common.ch
#batch_common.ch
KBLAS_BATCH_SRC = batch_common.cu
KBLAS_BATCH_SRC+= 		 \
			shelper_funcs.cu \
			sgemm_batch.cu   \
			ssyrk_batch.cu   \
			strsm_batch.cu   \
			strmm_batch.cu   \
			spotrf_batch.cu  \
			strtri_batch.cu  \
			slauum_batch.cu	 \
			spotrs_batch.cu
KBLAS_BATCH_SRC+= 		 \
			dhelper_funcs.cu \
			dgemm_batch.cu   \
			dsyrk_batch.cu 	 \
			dtrsm_batch.cu   \
			dtrmm_batch.cu   \
			dpotrf_batch.cu  \
			dlauum_batch.cu  \
			dtrtri_batch.cu	 \
			dpotrs_batch.cu
KBLAS_BATCH_SRC+= 		 \
			chelper_funcs.cu \
			cgemm_batch.cu   \
			csyrk_batch.cu   \
			ctrsm_batch.cu   \
			ctrmm_batch.cu   \
			cpotrf_batch.cu  \
			ctrtri_batch.cu  \
			clauum_batch.cu	 \
			cpotrs_batch.cu
KBLAS_BATCH_SRC+=      \
			zhelper_funcs.cu \
			zgemm_batch.cu   \
			zsyrk_batch.cu   \
			ztrsm_batch.cu   \
			ztrmm_batch.cu   \
			zpotrf_batch.cu  \
			ztrtri_batch.cu  \
			zlauum_batch.cu	 \
			zpotrs_batch.cu

ALL_OBJ+= $(patsubst %.cu,$(OBJ_DIR)/%.cu_o,$(KBLAS_BATCH_SRC))


$(OBJ_DIR)/%_common.cu_o: $(KBLAS_BATCH_DIR)/%_common.cu $(DEPS_BATCH)
	$(NVCC) $(NVOPTS) $(NVOPTS_3) $(INC) -c $< -o $@

$(OBJ_DIR)/%helper_funcs.cu_o: $(KBLAS_BATCH_DIR)/Xhelper_funcs.cu $(DEPS_BATCH)
	$(NVCC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) $(INC) -c $< -o $@

$(OBJ_DIR)/%gemm_batch.cu_o: $(KBLAS_BATCH_DIR)/Xgemm_batch.cu $(KBLAS_BATCH_DIR)/Xgemm_batch_core.cuh $(DEPS_BATCH)
	$(NVCC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) $(INC) -c $< -o $@

$(OBJ_DIR)/%syrk_batch.cu_o: $(KBLAS_BATCH_DIR)/Xsyrk_batch.cu $(KBLAS_BATCH_DIR)/Xsyrk_batch_drivers.cuh $(KBLAS_BATCH_DIR)/Xsyrk_batch_kernels.cuh $(DEPS_BATCH)
	$(NVCC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) $(INC) -c $< -o $@

$(OBJ_DIR)/%trsm_batch.cu_o: $(KBLAS_BATCH_DIR)/Xtrsm_batch.cu $(KBLAS_BATCH_DIR)/Xtrsm_batch_drivers.cuh $(KBLAS_BATCH_DIR)/Xtrsm_batch_kernels.cuh $(DEPS_BATCH)
	$(NVCC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) $(INC) -c $< -o $@

$(OBJ_DIR)/%trmm_batch.cu_o: $(KBLAS_BATCH_DIR)/Xtrmm_batch.cu $(KBLAS_BATCH_DIR)/Xtrmm_batch_drivers.cuh $(KBLAS_BATCH_DIR)/Xtrmm_batch_kernels.cuh $(DEPS_BATCH)
	$(NVCC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) $(INC) -c $< -o $@

$(OBJ_DIR)/%potrf_batch.cu_o: $(KBLAS_BATCH_DIR)/Xpotrf_batch.cu $(KBLAS_BATCH_DIR)/Xpotrf_batch_drivers.cuh $(KBLAS_BATCH_DIR)/Xpotrf_batch_kernels.cuh $(DEPS_BATCH)
	$(NVCC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) $(INC) -c $< -o $@

$(OBJ_DIR)/%lauum_batch.cu_o: $(KBLAS_BATCH_DIR)/Xlauum_batch.cu $(KBLAS_BATCH_DIR)/Xlauum_batch_drivers.cuh $(KBLAS_BATCH_DIR)/Xlauum_batch_kernels.cuh $(DEPS_BATCH)
	$(NVCC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) $(INC) -c $< -o $@

$(OBJ_DIR)/%trtri_batch.cu_o: $(KBLAS_BATCH_DIR)/Xtrtri_batch.cu $(KBLAS_BATCH_DIR)/Xtrtri_batch_drivers.cuh $(KBLAS_BATCH_DIR)/Xtrtri_batch_kernels.cuh $(DEPS_BATCH)
	$(NVCC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) $(INC) -c $< -o $@

$(OBJ_DIR)/%potrs_batch.cu_o: $(KBLAS_BATCH_DIR)/Xpotrs_batch.cu $(KBLAS_BATCH_DIR)/Xpotrs_batch_drivers.cuh $(KBLAS_BATCH_DIR)/Xpotrs_batch_kernels.cuh $(DEPS_BATCH)
	$(NVCC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) $(INC) -c $< -o $@

