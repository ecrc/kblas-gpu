#WARNING: Don't invoke this file independently, must be included into parent folder makefile

# KBLAS_L3_CSRC=
C_SRC+=	               \
	test_sgemm_batch.c test_dgemm_batch.c test_cgemm_batch.c test_zgemm_batch.c \
	test_ssyrk_batch.c test_dsyrk_batch.c test_csyrk_batch.c test_zsyrk_batch.c


$(OBJ_DIR)/test_%gemm_batch.o: $(KBLAS_BATCH_DIR)/test_Xgemm_batch.cpp $(INC_COMMON) $(INC_SRC) $(KBLAS_LIB)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%syrk_batch.o: $(KBLAS_BATCH_DIR)/test_Xsyrk_batch.cpp $(INC_COMMON) $(INC_SRC) $(KBLAS_LIB)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

$(BIN_DIR)/test_%_batch: $(OBJ_DIR)/test_%_batch.o
	$(CXX) $(COPTS) $< -o $@ $(LIB_PATH) $(LIB)