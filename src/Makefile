include ../make.inc

KBLAS_ROOT= ..
KBLAS2_CU_SRC = 
KBLAS3_CU_SRC =

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

KBLAS3_CU_SRC= Xtrmm.cu Xtr_common.cu Xtrsm.cu

OBJ_DIR=./obj
#ALL_OBJ=$(KBLAS2_CU_SRC:.cu=.cu_o) $(KBLAS3_CU_SRC:.cu=.cu_o)
ALL_OBJ = $(patsubst %.cu,$(OBJ_DIR)/%.cu_o,$(KBLAS2_CU_SRC))
ALL_OBJ+= $(patsubst %.cu,$(OBJ_DIR)/%.cu_o,$(KBLAS3_CU_SRC))

INC=-I${KBLAS_ROOT}/include
INC+= -I${_CUDA_ROOT_}/include

$(OBJ_DIR)/%scal.cu_o: %scal.cu scal_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@ 

$(OBJ_DIR)/%symv.cu_o: %symv.cu syhemv_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@ 

$(OBJ_DIR)/%symv_offset.cu_o: %symv_offset.cu syhemv_offset_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@ 

$(OBJ_DIR)/%hemv.cu_o: %hemv.cu syhemv_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@ 

$(OBJ_DIR)/%hemv_offset.cu_o: %hemv_offset.cu syhemv_offset_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@ 

$(OBJ_DIR)/%gemv.cu_o: %gemv.cu gemv_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@ 

$(OBJ_DIR)/%gemv2.cu_o: %gemv2.cu gemv2_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@ 

$(OBJ_DIR)/%gemv_offset.cu_o: %gemv_offset.cu gemv_offset_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@ 

$(OBJ_DIR)/%gemv2_offset.cu_o: %gemv2_offset.cu gemv2_offset_core.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@ 

$(OBJ_DIR)/%symv_mgpu.cu_o: %symv_mgpu.cu syhemv_mgpu_core.cuh syhemv_mgpu_offset_core.cuh 
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@ 

$(OBJ_DIR)/%hemv_mgpu.cu_o: %hemv_mgpu.cu syhemv_mgpu_core.cuh syhemv_mgpu_offset_core.cuh 
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@ 

$(OBJ_DIR)/%gemv_mgpu.cu_o: %gemv_mgpu.cu gemv_mgpu_core.cuh gemv_mgpu_offset_core.cuh 
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@

$(OBJ_DIR)/%gemm_mgpu.cu_o: %gemm_mgpu.cu gemm_aux.cuh
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@

$(OBJ_DIR)/%_control.cu_o: %_control.cu
	$(NVCC) $(NVOPTS) $(NVOPTS_2) $(INC) -c $< -o $@


$(OBJ_DIR)/%tr_common.cu_o: %tr_common.cu %tr_common.ch
	$(NVCC) $(NVOPTS) $(NVOPTS_3) $(INC) -c $< -o $@

$(OBJ_DIR)/%trmm.cu_o: %trmm.cu %tr_common.ch
	$(NVCC) $(NVOPTS) $(NVOPTS_3) $(INC) -c $< -o $@

$(OBJ_DIR)/%trsm.cu_o: %trsm.cu %tr_common.ch
	$(NVCC) $(NVOPTS) $(NVOPTS_3) $(INC) -c $< -o $@

KBLAS_DIR=$(KBLAS_ROOT)/lib
LIBKBLAS=$(KBLAS_DIR)/lib$(LIB_KBLAS_NAME).a

all: $(LIBKBLAS)

$(LIBKBLAS): $(ALL_OBJ) 
	ar -cru $(LIBKBLAS) $(ALL_OBJ)
	ranlib $@
	
#cp $(LIBKBLAS) ${KBLAS_ROOT}/lib
#rm -f ${KBLAS_ROOT}/testing/*.o
	
clean:
	rm -f $(OBJ_DIR)/*.cu_o

