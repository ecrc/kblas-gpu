#WARNING: Don't invoke this file independently, must be included into parent folder makefile

# C_SRC+=
C_SRC_l2=	\
	test_sscal.c \
	test_dscal.c \
	test_cscal.c \
	test_zscal.c \
	test_ssymv.c \
	test_dsymv.c \
	test_chemv.c \
	test_zhemv.c \
	test_sgemv.c \
	test_dgemv.c \
	test_cgemv.c \
	test_zgemv.c \
	test_ssymv_mgpu.c \
	test_dsymv_mgpu.c \
	test_chemv_mgpu.c \
	test_zhemv_mgpu.c \
	test_sgemv_mgpu.c \
	test_dgemv_mgpu.c \
	test_cgemv_mgpu.c \
	test_zgemv_mgpu.c \
	test_sgemv_offset.c \
	test_dgemv_offset.c \
	test_cgemv_offset.c \
	test_zgemv_offset.c	\
	test_ssymv_offset.c \
	test_dsymv_offset.c \
	test_chemv_offset.c \
	test_zhemv_offset.c \
	test_dgemm_mgpu.c \
	test_sgemm_mgpu.c \
	test_cgemm_mgpu.c \
	test_zgemm_mgpu.c \
	test_dgemm_mgpu_dim.c \
	test_sgemv2.c \
	test_dgemv2.c \
	test_cgemv2.c \
	test_zgemv2.c \
	test_sgemv2_offset.c \
	test_dgemv2_offset.c \
	test_cgemv2_offset.c \
	test_zgemv2_offset.c \
	test_sgemv2_dim.c \
	test_dgemv2_dim.c


$(OBJ_DIR)/test_%.o: $(KBLAS_TEST2_DIR)/test_%.c $(INC_SRC)
	$(CC) $(COPTS) $(INCLUDES) -c $< -o $@

EXE_L2= $(patsubst %.c,$(BIN_DIR)/%,$(C_SRC_l2))
