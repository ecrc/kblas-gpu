all:
	(cd src && make -j)
	(cd testing && make -j)
	(cd testing/batch_svd_qr && make)

clean:
	rm -f -v ./lib/*.a
	(cd src && make clean)
	(cd testing && make clean)
	(cd testing/batch_svd_qr && make clean)