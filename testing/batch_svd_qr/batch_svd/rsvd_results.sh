#!/bin/bash

start_n=96
end_n=512

printf "Size\t\tGEMM_Time\tGEMM_GFLOPs\tSVD_Time\tSVD_GFLOPs\tQR_Time\t\tQR_GFLOPs\tTime\t\tGFLOPS\t\tTime_Dev\tGFLOPs_Dev\tCPU_Time\tCPU_Time_Dev\n"

for (( i=$start_n; i <= $end_n; i+=32)); do
    output=$(./batch_svd $i $i 200 1)
    
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
    
    printf "$i\t\t$gemm_time\t$gemm_flops\t$svd_time\t$svd_flops\t$qr_time\t$qr_flops\t$overall_time\t$overall_flops\t$time_dev\t$gflops_dev\t$cpu_time\t$cpu_time_dev\n"
done