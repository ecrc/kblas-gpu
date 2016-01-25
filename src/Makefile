include ../make.inc

KBLAS_ROOT= ..
ALL_CU_SRC = 
ALL_CPP_SRC =

ifdef _SUPPORT_BLAS2_

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

ALL_CU_SRC= 	$(SCAL_SRC) 	   \
			$(SYHEMV_SRC) 			 \
			$(SYHEMV_OFFSET_SRC) \
			$(GEMV_SRC)   			 \
			$(GEMV2_SRC)   			 \
			$(GEMV_OFFSET_SRC)   \
			$(GEMV2_OFFSET_SRC)	 \
			$(SYHEMV_MGPU_SRC)   \
			$(GEMV_MGPU_SRC)     \
			$(GEMM_MGPU_SRC)		 \
			$(OTHER_SRC)

endif #_SUPPORT_BLAS2_

ifdef _SUPPORT_BLAS3_
  ifdef _SUPPORT_CUBLAS_
    TR_CU_SRC= Xtrmm.cu
#Xtrsm.cu
ALL_CU_SRC+=		            \
			$(TR_CU_SRC)
  endif

  TR_CPP_SRC= Xtr_common.cpp
  
  ifdef _SUPPORT_MKL_
    TR_CPP_SRC+= Xtrmm.cpp
  endif

ALL_CPP_SRC+=		            \
			$(TR_CPP_SRC)
endif

ALL_OBJ=
ifdef _USE_CUDA_
ALL_OBJ+=$(ALL_CU_SRC:.cu=.cu_o)
endif
#ifdef _USE_GCC_
ALL_OBJ+=$(ALL_CPP_SRC:.cpp=.c_o)
#endif


OBJ_DIR=./obj
INC=-I${KBLAS_ROOT}/include
ifdef _USE_CUDA_
INC+= -I${_CUDA_ROOT_}/include
endif
ifdef _USE_MKL_
INC+= -I${_MKL_ROOT_}/include
endif

ifdef _SUPPORT_BLAS2_

%scal.cu_o: %scal.cu scal_core.cuh
	$(NVCC) $(NVOPTS) $(INC) -c $< -o $@ 

%symv.cu_o: %symv.cu syhemv_core.cuh
	$(NVCC) $(NVOPTS) $(INC) -c $< -o $@ 

%symv_offset.cu_o: %symv_offset.cu syhemv_offset_core.cuh
	$(NVCC) $(NVOPTS) $(INC) -c $< -o $@ 

%hemv.cu_o: %hemv.cu syhemv_core.cuh
	$(NVCC) $(NVOPTS) $(INC) -c $< -o $@ 

%hemv_offset.cu_o: %hemv_offset.cu syhemv_offset_core.cuh
	$(NVCC) $(NVOPTS) $(INC) -c $< -o $@ 

%gemv.cu_o: %gemv.cu gemv_core.cuh
	$(NVCC) $(NVOPTS) $(INC) -c $< -o $@ 

%gemv2.cu_o: %gemv2.cu gemv2_core.cuh
	$(NVCC) $(NVOPTS) $(INC) -c $< -o $@ 

%gemv_offset.cu_o: %gemv_offset.cu gemv_offset_core.cuh
	$(NVCC) $(NVOPTS) $(INC) -c $< -o $@ 

%gemv2_offset.cu_o: %gemv2_offset.cu gemv2_offset_core.cuh
	$(NVCC) $(NVOPTS) $(INC) -c $< -o $@ 

%symv_mgpu.cu_o: %symv_mgpu.cu syhemv_mgpu_core.cuh syhemv_mgpu_offset_core.cuh 
	$(NVCC) $(NVOPTS) $(INC) -c $< -o $@ 

%hemv_mgpu.cu_o: %hemv_mgpu.cu syhemv_mgpu_core.cuh syhemv_mgpu_offset_core.cuh 
	$(NVCC) $(NVOPTS) $(INC) -c $< -o $@ 

%gemv_mgpu.cu_o: %gemv_mgpu.cu gemv_mgpu_core.cuh gemv_mgpu_offset_core.cuh 
	$(NVCC) $(NVOPTS) $(INC) -c $< -o $@

%gemm_mgpu.cu_o: %gemm_mgpu.cu gemm_aux.cuh
	$(NVCC) $(NVOPTS) $(INC) -c $< -o $@

%_control.cu_o: %_control.cu
	$(NVCC) $(NVOPTS) $(INC) -c $< -o $@
endif #_SUPPORT_BLAS2_


ifdef _SUPPORT_BLAS3_
%tr_common.c_o: %tr_common.cpp %tr_common.ch
	$(CC) $(COPTS) $(INC) -c $< -o $@

  ifdef _SUPPORT_CUBLAS_

%trmm.cu_o: %trmm.cu %trmm.hxx %tr_common.ch
	$(NVCC) $(NVOPTS) $(INC) -c $< -o $@

#%trsm.cu_o: %trsm.cu
#	$(NVCC) $(NVOPTS) $(INC) -c $< -o $@
  endif #_SUPPORT_CUBALS_

  ifdef _SUPPORT_MKL_
%trmm.c_o: %trmm.cpp %trmm.hxx %tr_common.ch
	$(CC) $(COPTS) $(INC) -c -o $@ $<
  endif #_SUPPORT_MKL_
endif #_SUPPORT_BLAS3_

KBLAS_DIR=$(KBLAS_ROOT)/lib
LIBKBLAS=$(KBLAS_DIR)/$(LIB_KBLAS_NAME) 

all: $(LIBKBLAS)

$(LIBKBLAS): $(ALL_OBJ) 
	ar -cru $(LIBKBLAS) $(ALL_OBJ)
	ranlib $@
	
#cp $(LIBKBLAS) ${KBLAS_ROOT}/lib
#rm -f ${KBLAS_ROOT}/testing/*.o
	
clean:
	rm -f *.cu_o

