#!/bin/bash

start_n=96
end_n=512

printf "Size\t\tCuSolver_Time\tOSBJ_Time\tCuSolver_Dev\tOSBJ_Dev\n"

for (( i=$start_n; i <= $end_n; i+=32)); do
    output=$(./batch_svd $i $i 200)
    
    cusolver_time=$(echo "$output" | grep "Streamed SVD:" | awk '{print $3}')
	cusolver_dev=$(echo "$output" | grep "Streamed SVD:" | awk '{print $5}')
    osbj_time=$(echo "$output" | grep "Batch SVD total time:" | awk '{print $5}')
	osbj_dev=$(echo "$output" | grep "Batch SVD total time:" | awk '{print $10}')
    
    printf "$i\t\t$cusolver_time\t$osbj_time\t\t$cusolver_dev\t$osbj_dev\n"
done