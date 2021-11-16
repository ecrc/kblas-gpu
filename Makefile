.PHONY: all testing lib clean

all: lib tests


tests:
	(export KBLAS_ROOT=$(pwd) && cd testing && make -j$(KBLAS_MAKE_NP))

lib:
	(export KBLAS_ROOT=$(pwd) && cd src && make -j$(KBLAS_MAKE_NP))

clean:
	rm -f -v ./lib/*.a ./lib/*.so
	(cd src && make clean)
	(cd testing && make clean)
