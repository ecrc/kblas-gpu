#WARNING: Don't invoke this file independently, must be included into parent folder makefile

# C_SRC+=
C_SRC_TLR=	              \
	test_sgemm_lr.c 				\
	test_sgemm_tlr.c  \
	test_dgemm_lr.c 				\
	test_dgemm_tlr.c  \

$(OBJ_DIR)/test_%gemm_lr.o: $(KBLAS_TLR_DIR)/test_Xgemm_lr.cpp $(INC_SRC)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test_%gemm_tlr.o: $(KBLAS_TLR_DIR)/test_Xgemm_tlr.cpp $(INC_SRC)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@

EXE_TLR= $(patsubst %.c,$(BIN_DIR)/%,$(C_SRC_TLR))
