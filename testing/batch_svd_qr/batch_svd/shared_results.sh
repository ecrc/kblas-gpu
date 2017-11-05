#!/bin/bash

start_n=2
end_n=64
streams=24
batch_size=1000

printf "Size\t\tTime\t\tGFLOPs\t\tTime_Dev\tGFLOPs_Dev\tCPU_Time\tCPU_Time_Dev\tOccupancy\n"

for (( i=$start_n; i <= $end_n; i+=2)); do
    output=$(./batch_svd $i $i $batch_size $streams)
    nvp_output=$(nvprof --kernels blockSVDKernel --metrics achieved_occupancy ./batch_svd $i $i $batch_size $streams 2>&1)
    
    time=$(echo "$output" | grep "Batch SVD total time:" | awk '{print $5}')
    flops=$(echo "$output" | grep "Batch SVD total time:" | awk '{print $7}')
	time_dev=$(echo "$output" | grep "Batch SVD total time:" | awk '{print $10}')
	gflops_dev=$(echo "$output" | grep "Batch SVD total time:" | awk '{print $11}')
    
    cpu_time=$(echo "$output" | grep "CPU Time:" | awk '{print $3}')
    cpu_time_dev=$(echo "$output" | grep "CPU Time:" | awk '{print $5}')
    
	occupancy=$(echo "$nvp_output" | grep "achieved_occupancy" | awk '{print $7}')
    
    printf "$i\t\t$time\t$flops\t$time_dev\t$gflops_dev\t$cpu_time\t$cpu_time_dev\t$occupancy\n"
done
