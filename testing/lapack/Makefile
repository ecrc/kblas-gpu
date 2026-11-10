#WARNING: Don't invoke this file independently, must be included into parent folder makefile

# C_SRC+=
C_SRC_LAP=	      \
	test_dpotrf.c \
	test_spotrf.c \

$(OBJ_DIR)/test_%potrf.o: $(KBLAS_LAP_DIR)/test_Xpotrf.cpp $(INC_SRC)
	$(CXX) $(COPTS) -DPREC_$* $(INCLUDES) -c $< -o $@
    
EXE_LAP= $(patsubst %.c,$(BIN_DIR)/%,$(C_SRC_LAP))
