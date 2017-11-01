#WARNING: Don't invoke this file independently, must be included into parent folder makefile

KBLAS2_CU_SRC =

SCAL_SRC=sscal.cu dscal.cu cscal.cu zscal.cu
SYHEMV_SRC= ssymv.cu dsymv.cu chemv.cu zhemv.cu
SYHEMV_OFFSET_SRC=ssymv_offset.cu dsymv_offset.cu chemv_offset.cu zhemv_offset.cu
GEMV_SRC= sgemv.cu dgemv.cu cgemv.cu zgemv.cu
GEMV2_SRC= sgemv2.cu dgemv2.cu cgemv2.cu zgemv2.cu
GEMV2_OFFSET_SRC= sgemv2_offset.cu dgemv2_offset.cu cgemv2_offset.cu zgemv2_offset.cu
GEMV_OFFSET_SRC= sgemv_offset.cu dgemv_offset.cu cgemv_offset.cu zgemv_offset.cu
SYHEMV_MGPU_SRC= ssymv_mgpu.cu dsymv_mgpu.cu chemv_mgpu.cu zhemv_mgpu.cu
GEMV_MGPU_SRC= sgemv_mgpu.cu dgemv_mgpu.cu cgemv_mgpu.cu zgemv_mgpu.cu
GEMM_MGPU_SRC= sgemm_mgpu.cu dgemm_mgpu.cu cgemm_mgpu.cu zgemm_mgpu.cu
OTHER_SRC=mgpu_control.cu

KBLAS2_CU_SRC=	$(SCAL_SRC)          \
		$(SYHEMV_SRC) 	     \
		$(SYHEMV_OFFSET_SRC) \
		$(GEMV_SRC)   	     \
		$(GEMV2_SRC)   	     \
		$(GEMV_OFFSET_SRC)   \
		$(GEMV2_OFFSET_SRC)  \
		$(SYHEMV_MGPU_SRC)   \
		$(GEMV_MGPU_SRC)     \
		$(GEMM_MGPU_SRC)     \
		$(OTHER_SRC)

ALL_OBJ+= $(patsubst %.cu,$(OBJ_DIR)/%.cu_o,$(KBLAS2_CU_SRC))

$(OBJ_DIR)/%scal.cu_o: $(KBLAS_SRC2_DIR)/%scal.cu $(KBLAS_SRC2_DIR)/scal_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@

$(OBJ_DIR)/%symv.cu_o: $(KBLAS_SRC2_DIR)/%symv.cu $(KBLAS_SRC2_DIR)/syhemv_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@

$(OBJ_DIR)/%symv_offset.cu_o: $(KBLAS_SRC2_DIR)/%symv_offset.cu $(KBLAS_SRC2_DIR)/syhemv_offset_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@

$(OBJ_DIR)/%hemv.cu_o: $(KBLAS_SRC2_DIR)/%hemv.cu $(KBLAS_SRC2_DIR)/syhemv_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@

$(OBJ_DIR)/%hemv_offset.cu_o: $(KBLAS_SRC2_DIR)/%hemv_offset.cu $(KBLAS_SRC2_DIR)/syhemv_offset_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@

$(OBJ_DIR)/%gemv.cu_o: $(KBLAS_SRC2_DIR)/%gemv.cu $(KBLAS_SRC2_DIR)/gemv_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@

$(OBJ_DIR)/%gemv2.cu_o: $(KBLAS_SRC2_DIR)/%gemv2.cu $(KBLAS_SRC2_DIR)/gemv2_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@

$(OBJ_DIR)/%gemv_offset.cu_o: $(KBLAS_SRC2_DIR)/%gemv_offset.cu $(KBLAS_SRC2_DIR)/gemv_offset_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@

$(OBJ_DIR)/%gemv2_offset.cu_o: $(KBLAS_SRC2_DIR)/%gemv2_offset.cu $(KBLAS_SRC2_DIR)/gemv2_offset_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@

$(OBJ_DIR)/%symv_mgpu.cu_o: $(KBLAS_SRC2_DIR)/%symv_mgpu.cu $(KBLAS_SRC2_DIR)/syhemv_mgpu_core.cuh $(KBLAS_SRC2_DIR)/syhemv_mgpu_offset_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@

$(OBJ_DIR)/%hemv_mgpu.cu_o: $(KBLAS_SRC2_DIR)/%hemv_mgpu.cu $(KBLAS_SRC2_DIR)/syhemv_mgpu_core.cuh $(KBLAS_SRC2_DIR)/syhemv_mgpu_offset_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@

$(OBJ_DIR)/%gemv_mgpu.cu_o: $(KBLAS_SRC2_DIR)/%gemv_mgpu.cu $(KBLAS_SRC2_DIR)/gemv_mgpu_core.cuh $(KBLAS_SRC2_DIR)/gemv_mgpu_offset_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@

$(OBJ_DIR)/%gemm_mgpu.cu_o: $(KBLAS_SRC2_DIR)/%gemm_mgpu.cu $(KBLAS_SRC2_DIR)/gemm_aux.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@

$(OBJ_DIR)/%_control.cu_o: $(KBLAS_SRC2_DIR)/%_control.cu
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@

