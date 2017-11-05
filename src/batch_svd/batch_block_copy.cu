#include <mini_blas_gpu.h>
#include <batch_block_copy.h>
#include <gpu_util.h>

template<class T, class T_ptr>
__global__
void batchCopyMatrixBlockKernel(
	T_ptr orig_array, int row_offset_orig, int col_offset_orig, int ld_orig, int stride_orig,
	T_ptr copy_array, int row_offset_copy, int col_offset_copy, int ld_copy, int stride_copy,
	int rows, int cols, int ops, int rows_per_thread
)
{
    int op_id = blockIdx.x * blockDim.y + threadIdx.y;
    if(op_id >= ops) return;
	
    T* orig_matrix = getOperationPtr<T>(orig_array, op_id, stride_orig) + row_offset_orig + col_offset_orig * ld_orig;
    T* copy_matrix = getOperationPtr<T>(copy_array, op_id, stride_copy) + row_offset_copy + col_offset_copy * ld_copy;
    
    int tid = threadIdx.x;
    
    for(int j = 0; j < cols; j++)
    {
        for(int i = 0; i < rows_per_thread; i++)
        {
            int row_index = WARP_SIZE * i + tid;
            if(row_index < rows)
                copy_matrix[row_index + j * ld_copy] = orig_matrix[row_index + j * ld_orig];
        }
    }
}

template<class T, class T_ptr>
void batchCopyMatrixBlock(
	T_ptr orig_array, int row_offset_orig, int col_offset_orig, int ld_orig, int stride_orig,
	T_ptr copy_array, int row_offset_copy, int col_offset_copy, int ld_copy, int stride_copy,
	int rows, int cols, int ops, GPUBlasHandle& handle
)
{
	if(ops == 0 || rows == 0 || cols == 0) return;
	
    int ops_per_block = 8;
    int rows_per_thread = iDivUp(rows, WARP_SIZE);
    int blocks = iDivUp(ops, ops_per_block);
    
    dim3 dimBlock(WARP_SIZE, ops_per_block);
    dim3 dimGrid(blocks, 1);
    
    batchCopyMatrixBlockKernel<T, T_ptr><<< dimGrid, dimBlock, 0, handle.stream >>> (
		orig_array, row_offset_orig, col_offset_orig, ld_orig, stride_orig,
		copy_array, row_offset_copy, col_offset_copy, ld_copy, stride_copy,
		rows, cols, ops, rows_per_thread	
	);
    
    gpuErrchk( cudaGetLastError() );
}

// Array of pointers interface
void batchCopyMatrixBlock(
	double** orig_array, int row_offset_orig, int col_offset_orig, int ld_orig, 
	double** copy_array, int row_offset_copy, int col_offset_copy, int ld_copy,
	int rows, int cols, int ops, GPUBlasHandle& handle
)
{
	batchCopyMatrixBlock<double, double**>(
		orig_array, row_offset_orig, col_offset_orig, ld_orig, 0,
		copy_array, row_offset_copy, col_offset_copy, ld_copy, 0,
		rows, cols, ops, handle
	);
}

void batchCopyMatrixBlock(
	float** orig_array, int row_offset_orig, int col_offset_orig, int ld_orig, 
	float** copy_array, int row_offset_copy, int col_offset_copy, int ld_copy,
	int rows, int cols, int ops, GPUBlasHandle& handle
)
{
	batchCopyMatrixBlock<float, float**>(
		orig_array, row_offset_orig, col_offset_orig, ld_orig, 0,
		copy_array, row_offset_copy, col_offset_copy, ld_copy, 0,
		rows, cols, ops, handle
	);
}

// Strided interface
void batchCopyMatrixBlock(
	double* orig_array, int row_offset_orig, int col_offset_orig, int ld_orig, int stride_orig, 
	double* copy_array, int row_offset_copy, int col_offset_copy, int ld_copy, int stride_copy, 
	int rows, int cols, int ops, GPUBlasHandle& handle
)
{
	batchCopyMatrixBlock<double, double*>(
		orig_array, row_offset_orig, col_offset_orig, ld_orig, stride_orig,
		copy_array, row_offset_copy, col_offset_copy, ld_copy, stride_copy,
		rows, cols, ops, handle
	);
}

void batchCopyMatrixBlock(
	float* orig_array, int row_offset_orig, int col_offset_orig, int ld_orig, int stride_orig, 
	float* copy_array, int row_offset_copy, int col_offset_copy, int ld_copy, int stride_copy, 
	int rows, int cols, int ops, GPUBlasHandle& handle
)
{	
	batchCopyMatrixBlock<float, float*>(
		orig_array, row_offset_orig, col_offset_orig, ld_orig, stride_orig,
		copy_array, row_offset_copy, col_offset_copy, ld_copy, stride_copy,
		rows, cols, ops, handle
	);
}
