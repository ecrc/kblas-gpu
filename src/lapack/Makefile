#WARNING: Don't invoke this file independently, must be included into parent folder makefile

DEPS_LAP=$(DEPS_COM) kblas_struct.h kblas_prec_def.h kblas_error.h Xblas_core.ch workspace_queries.ch

KBLAS_LAP_SRC =   \
		dpotrf.cu  \
        spotrf.cu  \

ALL_OBJ+= $(patsubst %.cu,$(OBJ_DIR)/%.cu_o,$(KBLAS_LAP_SRC))


$(OBJ_DIR)/%potrf.cu_o: $(KBLAS_LAP_DIR)/Xpotrf.cu $(KBLAS_LAP_DIR)/Xpotrf_core.cuh $(DEPS_LAP)
	$(NVCC) $(NVOPTS) -DPREC_$* $(NVOPTS_3) $(INC) -c $< -o $@
