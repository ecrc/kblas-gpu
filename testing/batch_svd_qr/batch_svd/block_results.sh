#!/bin/bash

start_n=96
end_n=512

printf "%15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s\n" "Size" "GEMM_Time" "GEMM_GFLOPs" "SVD_Time" "SVD_GFLOPs" "QR_Time" "QR_GFLOPs" "Time" "GFLOPS" "Time_Dev" "GFLOPs_Dev" "CPU_Time" "CPU_Time_Dev"

for (( i=$start_n; i <= $end_n; i+=32)); do
    output=$(./batch_svd $i $i 200 10)
    
    gemm_time=$(echo "$output" | grep "GEMM performance:" | awk '{print $3}')
    gemm_flops=$(echo "$output" | grep "GEMM performance:" | awk '{print $9}')
	svd_time=$(echo "$output" | grep "SVD performance:" | awk '{print $3}')
    svd_flops=$(echo "$output" | grep "SVD performance:" | awk '{print $9}')
    qr_time=$(echo "$output" | grep "QR performance:" | awk '{print $3}')
	qr_flops=$(echo "$output" | grep "QR performance:" | awk '{print $9}')
    
	overall_time=$(echo "$output" | grep "Batch SVD total time:" | awk '{print $5}')
    overall_flops=$(echo "$output" | grep "Batch SVD total time:" | awk '{print $7}')
	time_dev=$(echo "$output" | grep "Batch SVD total time:" | awk '{print $10}')
	gflops_dev=$(echo "$output" | grep "Batch SVD total time:" | awk '{print $11}')
	
	cpu_time=$(echo "$output" | grep "CPU Time:" | awk '{print $3}')
    cpu_time_dev=$(echo "$output" | grep "CPU Time:" | awk '{print $5}')
    
    
    printf "%15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s\n"  $i $gemm_time $gemm_flops $svd_time $svd_flops $qr_time $qr_flops $overall_time $overall_flops $time_dev $gflops_dev $cpu_time $cpu_time_dev
done