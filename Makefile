.PHONY: all testing lib clean

all: lib

tests:
	(cd testing && make -j$(KBLAS_MAKE_NP))

lib:
	(cd src && make -j$(KBLAS_MAKE_NP))

clean:
	rm -f -v ./lib/*.a
	(cd src && make clean)
	(cd testing && make clean)
