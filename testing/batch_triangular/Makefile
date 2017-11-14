#WARNING: Don't invoke this file independently, must be included into parent folder makefile

BATCH_INC_DEP= $(INC_SRC) testing_prec_def.h
C_SRC_BATCH=
C_SRC_BATCH+=         \
	test_sgemm_batch.c  \
	test_ssyrk_batch.c  \
	test_strsm_batch.c  \
	test_strmm_batch.c  \
	test_spotrf_batch.c \
	test_strtri_batch.c \
	test_slauum_batch.c \
	test_spotrs_batch.c \
	test_spotri_batch.c
C_SRC_BATCH+=         \
	test_dgemm_batch.c  \
	test_dsyrk_batch.c  \
	test_dtrsm_batch.c  \
	test_dtrmm_batch.c  \
	test_dpotrf_batch.c \
	test_dlauum_batch.c \
	test_dtrtri_batch.c \
	test_dpotrs_batch.c \
	test_dpotri_batch.c
C_SRC_BATCH+=         \
	test_cgemm_batch.c  \
	test_csyrk_batch.c  \
	test_ctrsm_batch.c  \
	test_ctrmm_batch.c  \
	test_cpotrf_batch.c \
	test_ctrtri_batch.c \
	test_clauum_batch.c \
	test_cpotrs_batch.c \
	test_cpotri_batch.c
C_SRC_BATCH+=         \
	test_zgemm_batch.c  \
	test_zsyrk_batch.c  \
	test_ztrsm_batch.c  \
	test_ztrmm_batch.c  \
	test_zpotrf_batch.c \
	test_ztrtri_batch.c \
	test_zlauum_batch.c \
	test_zpotrs_batch.c \
	test_zpotri_batch.c

ifdef _SUPPORT_SVD_
C_SRC_BATCH+= test_dgesvj_batch.c test_sgesvj_batch.c

$(OBJ_DIR)/test_%gesvj_batch.o: $(KBLAS_BATCH_DIR)/test_Xgesvj_batch.cpp $(BATCH_INC_DEP)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

C_SRC_BATCH+= test_dgeqrf_batch.c test_sgeqrf_batch.c

$(OBJ_DIR)/test_%geqrf_batch.o: $(KBLAS_BATCH_DIR)/test_Xgeqrf_batch.cpp $(BATCH_INC_DEP)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

LIB+= -lcusolver -lcurand
endif

$(OBJ_DIR)/test_%gemm_batch.o: $(KBLAS_BATCH_DIR)/test_Xgemm_batch.cpp $(BATCH_INC_DEP)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%syrk_batch.o: $(KBLAS_BATCH_DIR)/test_Xsyrk_batch.cpp $(BATCH_INC_DEP)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%trsm_batch.o: $(KBLAS_BATCH_DIR)/test_Xtrsm_batch.cpp $(BATCH_INC_DEP)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%trmm_batch.o: $(KBLAS_BATCH_DIR)/test_Xtrmm_batch.cpp $(BATCH_INC_DEP)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%potrf_batch.o: $(KBLAS_BATCH_DIR)/test_Xpotrf_batch.cpp $(BATCH_INC_DEP)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%lauum_batch.o: $(KBLAS_BATCH_DIR)/test_Xlauum_batch.cpp $(BATCH_INC_DEP)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%trtri_batch.o: $(KBLAS_BATCH_DIR)/test_Xtrtri_batch.cpp $(BATCH_INC_DEP)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%potrs_batch.o: $(KBLAS_BATCH_DIR)/test_Xpotrs_batch.cpp $(BATCH_INC_DEP)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%potri_batch.o: $(KBLAS_BATCH_DIR)/test_Xpotri_batch.cpp $(BATCH_INC_DEP)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

$(BIN_DIR)/test_%_batch: $(OBJ_DIR)/test_%_batch.o
	$(CXX) $(COPTS) $< -o $@ $(LIB_PATH) $(LIB)

EXE_BATCH= $(patsubst %.c,$(BIN_DIR)/%,$(C_SRC_BATCH))
