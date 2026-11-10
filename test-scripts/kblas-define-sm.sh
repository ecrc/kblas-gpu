#!/bin/bash 
#
# Prints the compute capability of the first CUDA device installed
# on the system, or alternatively the device whose index is the
# first command-line argument
# modified from https://gist.github.com/eyalroz/71ce52fa80acdd1c3b192e43a6c1d930

device_index=${1:-0}
timestamp=$(date +%s.%N)
generated_binary="/tmp/cuda-compute-version-helper-$$-$timestamp"
source_code="$(cat << EOF 
#include <stdio.h>
#include <cuda_runtime_api.h>

int main()
{
	cudaDeviceProp prop;
	cudaError_t status;
	int device_count;
	status = cudaGetDeviceCount(&device_count);
	if (status != cudaSuccess) { 
		fprintf(stderr,"cudaGetDeviceCount() failed: %s\n", cudaGetErrorString(status)); 
		return -1;
	}
	if (${device_index} >= device_count) {
		fprintf(stderr, "Specified device index %d exceeds the maximum (the device count on this system is %d)\n", ${device_index}, device_count);
		return -1;
	}
	status = cudaGetDeviceProperties(&prop, ${device_index});
	if (status != cudaSuccess) { 
		fprintf(stderr,"cudaGetDeviceProperties() for device ${device_index} failed: %s\n", cudaGetErrorString(status)); 
		return -1;
	}
	int v = prop.major * 10 + prop.minor;
	printf("%d\\n", v);
}
EOF
)"

echo "$source_code" | nvcc -x c++ -I"$CUDA_ROOT/include" -L"$CUDA_ROOT/lib64" -lcudart -o "$generated_binary" -

export _CUDA_ARCH_=`$generated_binary`
rm $generated_binary