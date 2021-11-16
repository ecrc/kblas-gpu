#WARNING: Don't invoke this file independently, must be included into parent folder makefile
INCLUDES+= -I$(HIPBLASROOT)/include -D__HIP_PLATFORM_AMD__
CXX=/opt/rocm/bin/hipcc
# C_SRC+=
C_SRC_L3=	              \
	test_strmm.cc \
	test_dtrmm.cc \
	test_strsm.cc \
	test_dtrsm.cc
#	test_strmm_cpu.c 
	# #test_strmm_mgpu.c \
	# test_dtrmm.c \
	# test_dtrmm_cpu.c \
	# test_dtrmm_mgpu.c \
#	test_ctrmm.c \
#	test_ctrmm_cpu.c \
#	test_ctrmm_mgpu.c \
#	test_ztrmm.c \
#	test_ztrmm_cpu.c \
#	test_ztrmm_mgpu.c 
	# #test_strsm.c \
	# test_strsm_cpu.c \
	# test_strsm_mgpu.c \
	# test_dtrsm.c \
	# test_dtrsm_cpu.c \
	# test_dtrsm_mgpu.c 
#	test_ctrsm.c \
#	test_ctrsm_cpu.c \
#	test_ctrsm_mgpu.c \
#	test_ztrsm.c \
#	test_ztrsm_cpu.c \
#	test_ztrsm_mgpu.c \
#	test_sgemm.c \
#	test_dgemm.c \
#	test_dgemmXt.c



$(OBJ_DIR)/test_%trmm.o: $(KBLAS_TEST3_DIR)/test_%trmm.cc $(KBLAS_TEST3_DIR)/test_trmm.ch testing_Xtr_common.h $(INC_SRC)
	$(CXX) -fPIC $(INCLUDES) -c $< -o $@

#$(OBJ_DIR)/test_%trmm_cpu.o: $(KBLAS_TEST3_DIR)/test_%trmm_cpu.c $(KBLAS_TEST3_DIR)/test_trmm_cpu.ch testing_Xtr_common.h $(INC_SRC)
#	$(CXX) $(INCLUDES) -c $< -o $@
#
#$(OBJ_DIR)/test_%trmm_mgpu.o: $(KBLAS_TEST3_DIR)/test_%trmm_mgpu.c $(KBLAS_TEST3_DIR)/test_trmm_mgpu.ch testing_Xtr_common.h $(INC_SRC)
#	$(CXX) $(INCLUDES) -c $< -o $@
#
$(OBJ_DIR)/test_%trsm.o: $(KBLAS_TEST3_DIR)/test_%trsm.cc $(KBLAS_TEST3_DIR)/test_trsm.ch testing_Xtr_common.h $(INC_SRC)
	$(CXX) -fPIC $(INCLUDES) -c $< -o $@
#
#$(OBJ_DIR)/test_%trsm_cpu.o: $(KBLAS_TEST3_DIR)/test_%trsm_cpu.c $(KBLAS_TEST3_DIR)/test_trsm_cpu.ch testing_Xtr_common.h $(INC_SRC)
#	$(CXX) $(INCLUDES) -c $< -o $@
#
#$(OBJ_DIR)/test_%trsm_mgpu.o: $(KBLAS_TEST3_DIR)/test_%trsm_mgpu.c $(KBLAS_TEST3_DIR)/test_trsm_mgpu.ch testing_Xtr_common.h $(INC_SRC)
#	$(CXX) $(INCLUDES) -c $< -o $@
#
#$(OBJ_DIR)/test_%gemm.o: $(KBLAS_TEST3_DIR)/test_%gemm.c $(KBLAS_TEST3_DIR)/test_gemm.ch testing_Xtr_common.h $(INC_SRC)
#	$(CXX) $(INCLUDES) -c $< -o $@
#
#$(OBJ_DIR)/test_%gemmXt.o: $(KBLAS_TEST3_DIR)/test_%gemmXt.c $(INC_SRC)
#	$(CXX) $(INCLUDES) -c $< -o $@

EXE_L3= $(patsubst %.cc,$(BIN_DIR)/%,$(C_SRC_L3))
