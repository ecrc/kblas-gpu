#WARNING: Don't invoke this file independently, must be included into parent folder makefile

DEPS_SRC3=$(DEPS_COM)
KBLAS3_CU_SRC= Xtrmm.cu Xtrsm.cu

ALL_OBJ+= $(patsubst %.cu,$(OBJ_DIR)/%.cu_o,$(KBLAS3_CU_SRC))


$(OBJ_DIR)/%trmm.cu_o: $(KBLAS_SRC3_DIR)/%trmm.cu $(DEPS_SRC3)
	$(NVCC) $(NVOPTS) $(NVOPTS_3) $(INC) -c $< -o $@

$(OBJ_DIR)/%trsm.cu_o: $(KBLAS_SRC3_DIR)/%trsm.cu $(DEPS_SRC3)
	$(NVCC) $(NVOPTS) $(NVOPTS_3) $(INC) -c $< -o $@

