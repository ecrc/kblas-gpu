#WARNING: Don't invoke this file independently, must be included into parent folder makefile

# KBLAS_L3_CSRC=
C_SRC+=	             \
	test_sgemm_batch.c \
	test_ssyrk_batch.c \
	test_strsm_batch.c \
	test_strmm_batch.c
C_SRC+=	             \
	test_dgemm_batch.c \
	test_dsyrk_batch.c \
	test_dtrsm_batch.c \
	test_dtrmm_batch.c
C_SRC+=	             \
	test_cgemm_batch.c \
	test_csyrk_batch.c \
	test_ctrsm_batch.c \
	test_ctrmm_batch.c
C_SRC+=	             \
	test_zgemm_batch.c \
	test_zsyrk_batch.c \
	test_ztrsm_batch.c \
	test_ztrmm_batch.c


$(OBJ_DIR)/test_%gemm_batch.o: $(KBLAS_BATCH_DIR)/test_Xgemm_batch.cpp $(INC_SRC)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%syrk_batch.o: $(KBLAS_BATCH_DIR)/test_Xsyrk_batch.cpp $(INC_SRC)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%trsm_batch.o: $(KBLAS_BATCH_DIR)/test_Xtrsm_batch.cpp $(INC_SRC)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%trmm_batch.o: $(KBLAS_BATCH_DIR)/test_Xtrmm_batch.cpp $(INC_SRC)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

$(BIN_DIR)/test_%_batch: $(OBJ_DIR)/test_%_batch.o
	$(CXX) $(COPTS) $< -o $@ $(LIB_PATH) $(LIB)