all:
	(cd src && make -j)
	(cd testing && make -j)
	
clean:
	rm -f -v ./lib/*.a
	(cd src && make clean)
	(cd testing && make clean)