#WARNING: Don't invoke this file independently, must be included into parent folder makefile


KBLAS3_CU_SRC= Xtrmm.cu Xtrsm.cu

ALL_OBJ+= $(patsubst %.cu,$(OBJ_DIR)/%.cu_o,$(KBLAS3_CU_SRC))


$(OBJ_DIR)/%trmm.cu_o: $(KBLAS_SRC3_DIR)/%trmm.cu %tr_common.ch
	$(NVCC) $(NVOPTS) $(NVOPTS_3) $(INC) -c $< -o $@

$(OBJ_DIR)/%trsm.cu_o: $(KBLAS_SRC3_DIR)/%trsm.cu %tr_common.ch
	$(NVCC) $(NVOPTS) $(NVOPTS_3) $(INC) -c $< -o $@

