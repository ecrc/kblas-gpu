#WARNING: Don't invoke this file independently, must be included into parent folder makefile

DEPS_TLR=$(DEPS_COM) kblas_struct.h kblas_prec_def.h kblas_error.h Xblas_core.ch workspace_queries.ch

KBLAS_TLR_SRC =   \
		sgemm_tlr.cu  \
		dgemm_tlr.cu  \

ALL_OBJ+= $(patsubst %.cu,$(OBJ_DIR)/%.cu_o,$(KBLAS_TLR_SRC))


$(OBJ_DIR)/%gemm_tlr.cu_o: $(KBLAS_TLR_DIR)/Xgemm_tlr.cu $(KBLAS_TLR_DIR)/Xgemm_tlr_core.cuh $(DEPS_TLR)
	$(NVCC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) $(INC) -c $< -o $@

