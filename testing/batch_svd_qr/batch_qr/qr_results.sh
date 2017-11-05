#!/bin/bash

start_n=32
end_n=512

printf "Size\t\tKBLAS_GFLOPS\tCUBLAS_GFLOPS\tMAGMA_GFLOPS\n"

for (( i=$start_n; i <= $end_n; i+=32)); do
    output=$(./test_batch_qr $i $i 1000)
    
    kblas_gflops=$(echo "$output" | grep "KBLAS:" | awk '{print $7}')
	cublas_gflops=$(echo "$output" | grep "CUBLAS:" | awk '{print $7}')
	magma_gflops=$(echo "$output" | grep "MAGMA:" | awk '{print $7}')

    printf "$i\t\t$kblas_gflops\t$cublas_gflops\t$magma_gflops\n"
done
