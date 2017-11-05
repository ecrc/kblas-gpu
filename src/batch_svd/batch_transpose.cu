#include <algorithm>
#include <gpu_util.h>
#include <batch_transpose.h>

#define TRANSPOSE_TILE_DIM      32
#define TRANSPOSE_BLOCK_ROWS    8
#define TRANSPOSE_LOAD(m)       __ldg(&(m))

template<class T>
__global__ 
void transpose_kernel(int m, int n, T** matrix_data, int ldm, T** transpose_data, int ldt, int op_start, int ops)
{
    __shared__ T tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];
    
    int x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;
    int op_index = op_start + blockIdx.z;
    
    if(op_index >= ops) return;
    
    T* matrix = matrix_data[op_index];
    T* transpose = transpose_data[op_index];
    
    #pragma unroll
    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS)
        if(x < m && y + j < n)
            tile[threadIdx.y + j][threadIdx.x] = TRANSPOSE_LOAD(matrix[x + (y + j) * ldm]);

    __syncthreads();
    
    x = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x; 
    y = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;
    
    #pragma unroll
    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS)
        if(y + j < m && x < n)
            transpose[x + (y + j) * ldt] = tile[threadIdx.x][threadIdx.y + j];
}

template<class T>
void batch_transpose_template(int m, int n, T** matrix_data, int ldm, T** transpose_data, int ldt, int ops, GPUBlasHandle& handle)
{
	int ops_per_kernel = 32768;
		
    int block_rows = iDivUp(m, TRANSPOSE_TILE_DIM);
    int block_cols = iDivUp(n, TRANSPOSE_TILE_DIM);
    
	dim3 blockDim(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS, 1);
    dim3 gridDim(block_rows, block_cols, ops_per_kernel);
	
	int op_start = 0;

	while(op_start < ops)
    {
		gridDim.z = std::min(ops_per_kernel, ops - op_start);
		transpose_kernel<T><<< gridDim, blockDim, 0, handle.stream >>>(m, n, matrix_data, ldm, transpose_data, ldt, op_start, ops);
		op_start += ops_per_kernel;
	}	
	gpuErrchk( cudaGetLastError() );
}

void batch_transpose(int m, int n, double** matrix_data, int ldm, double** transpose_data, int ldt, int ops, GPUBlasHandle& handle)
{
	batch_transpose_template<double>(m, n, matrix_data, ldm, transpose_data, ldt, ops, handle);
}

void batch_transpose(int m, int n, float** matrix_data, int ldm, float** transpose_data, int ldt, int ops, GPUBlasHandle& handle)
{
	batch_transpose_template<float>(m, n, matrix_data, ldm, transpose_data, ldt, ops, handle);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////TEST
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    
    int test_m = 58, test_n = 93;
    int num_ops = 10000;
    thrust::host_vector<Real> matrix_data(test_m * test_n * num_ops), matrix_data_transpose(test_m * test_n * num_ops);
    randomData(vec_ptr(matrix_data), test_m * test_n * num_ops);
    
    for(int i = 0; i < num_ops; i++)
    {
        Real *matrix = vec_ptr(matrix_data) + i * test_m * test_n;
        Real *trans_matrix = vec_ptr(matrix_data_transpose) + i * test_m * test_n;
        transpose(trans_matrix, matrix, test_m, test_m, test_n);
    }
    
    thrust::device_vector<Real> dev_data = matrix_data, dev_trans_data(test_m * test_n * num_ops);
    thrust::device_vector<Real*> dev_ptrs(num_ops), dev_trans_ptrs(num_ops);
    generateArrayOfPointers(vec_ptr(dev_data), dev_ptrs, test_m * test_n, num_ops);
    generateArrayOfPointers(vec_ptr(dev_trans_data), dev_trans_ptrs, test_m * test_n, num_ops);
    
    batch_transpose(test_m, test_n, vec_ptr(dev_ptrs), test_m, vec_ptr(dev_trans_ptrs), test_n, num_ops);

    matrix_data = dev_trans_data;
    for(int i = 0; i < num_ops; i++)
    {
        Real *matrix = vec_ptr(matrix_data) + i * test_m * test_n;
        Real *trans_matrix = vec_ptr(matrix_data_transpose) + i * test_m * test_n;
        Real diff = 0; 
        for(int j = 0; j < test_m * test_n; j++)
            diff += abs(matrix[j] - trans_matrix[j]);
        if(diff > 1e-10) printf("Error %e in matrix %d\n", diff, i);
    }
*/
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
