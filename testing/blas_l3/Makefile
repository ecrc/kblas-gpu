#WARNING: Don't invoke this file independently, must be included into parent folder makefile

# KBLAS_L3_CSRC=
C_SRC+=	              \
	test_strmm.c \
	test_strmm_cpu.c \
	test_strmm_mgpu.c \
	test_dtrmm.c \
	test_dtrmm_cpu.c \
	test_dtrmm_mgpu.c \
	test_ctrmm.c \
	test_ctrmm_cpu.c \
	test_ctrmm_mgpu.c \
	test_ztrmm.c \
	test_ztrmm_cpu.c \
	test_ztrmm_mgpu.c \
	test_strsm.c \
	test_strsm_cpu.c \
	test_strsm_mgpu.c \
	test_dtrsm.c \
	test_dtrsm_cpu.c \
	test_dtrsm_mgpu.c \
	test_ctrsm.c \
	test_ctrsm_cpu.c \
	test_ctrsm_mgpu.c \
	test_ztrsm.c \
	test_ztrsm_cpu.c \
	test_ztrsm_mgpu.c \
	test_sgemm.c \
	test_dgemm.c \
	test_dgemmXt.c


# C_OBJ+= $(patsubst %.c,$(OBJ_DIR)/%.o,$(KBLAS_L3_CSRC))
# EXE+= $(patsubst %.c,$(BIN_DIR)/%,$(KBLAS_L3_CSRC))


$(OBJ_DIR)/test_%trmm.o: $(KBLAS_TEST3_DIR)/test_%trmm.c $(KBLAS_TEST3_DIR)/test_trmm.ch testing_Xtr_common.h $(INC_SRC) $(KBLAS_LIB)
	$(CXX) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%trmm_cpu.o: $(KBLAS_TEST3_DIR)/test_%trmm_cpu.c $(KBLAS_TEST3_DIR)/test_trmm_cpu.ch testing_Xtr_common.h $(INC_SRC) $(KBLAS_LIB)
	$(CXX) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%trmm_mgpu.o: $(KBLAS_TEST3_DIR)/test_%trmm_mgpu.c $(KBLAS_TEST3_DIR)/test_trmm_mgpu.ch testing_Xtr_common.h $(INC_SRC) $(KBLAS_LIB)
	$(CXX) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%trsm.o: $(KBLAS_TEST3_DIR)/test_%trsm.c $(KBLAS_TEST3_DIR)/test_trsm.ch testing_Xtr_common.h $(INC_SRC) $(KBLAS_LIB)
	$(CXX) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%trsm_cpu.o: $(KBLAS_TEST3_DIR)/test_%trsm_cpu.c $(KBLAS_TEST3_DIR)/test_trsm_cpu.ch testing_Xtr_common.h $(INC_SRC) $(KBLAS_LIB)
	$(CXX) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%trsm_mgpu.o: $(KBLAS_TEST3_DIR)/test_%trsm_mgpu.c $(KBLAS_TEST3_DIR)/test_trsm_mgpu.ch testing_Xtr_common.h $(INC_SRC) $(KBLAS_LIB)
	$(CXX) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%gemm.o: $(KBLAS_TEST3_DIR)/test_%gemm.c $(KBLAS_TEST3_DIR)/test_gemm.ch testing_Xtr_common.h $(INC_SRC) $(KBLAS_LIB)
	$(CXX) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%gemmXt.o: $(KBLAS_TEST3_DIR)/test_%gemmXt.c $(INC_SRC) $(KBLAS_LIB)
	$(CXX) $(INCLUDES) -c $< -o $@

