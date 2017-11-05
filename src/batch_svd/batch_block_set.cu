#include <batch_block_set.h>
#include <gpu_util.h>

template<class T>
__global__
void batchBlockSetIdentityKernel(
	T** block_ptrs, int row_offset, int col_offset, int ld, 
	int rows, int cols, int ops, int rows_per_thread
)
{
    int op_id = blockIdx.x * blockDim.y + threadIdx.y;
    if(op_id >= ops) return;
    
    T* block_ptr = block_ptrs[op_id] + row_offset + col_offset * ld;

    int tid = threadIdx.x;
    
    for(int j = 0; j < cols; j++)
    {
        for(int i = 0; i < rows_per_thread; i++)
        {
            int row_index = WARP_SIZE * i + tid;
			T val = (row_index == j ? 1 : 0);
            if(row_index < rows)
                block_ptr[row_index + j * ld] = val;
        }
    }
}

template<class T>
__global__
void batchBlockSetZeroKernel(
	T** block_ptrs, int row_offset, int col_offset, int ld, 
	int rows, int cols, int ops, int rows_per_thread
)
{
    int op_id = blockIdx.x * blockDim.y + threadIdx.y;
    if(op_id >= ops) return;
	
	// Grab the current operation pointer 
    T* block_ptr = block_ptrs[op_id] + row_offset + col_offset * ld;
	
	// Advance the row pointer according to the block index
	block_ptr += blockIdx.y * blockDim.x;

    int tid = threadIdx.x;
	
    for(int j = 0; j < cols; j++)
    {
        for(int i = 0; i < rows_per_thread; i++)
        {
            int row_index = blockDim.x * i + tid;
			if(row_index < rows)
				block_ptr[row_index + j * ld] = 0;
        }
    }
}

template<class T>
void batchBlockSetIdentityT(T** block_ptrs, int row_offset, int col_offset, int ld, int rows, int cols, int ops, GPUBlasHandle& handle)
{
	if(ops == 0 || rows == 0 || cols == 0) return;
	
    int ops_per_block = 8;
    int rows_per_thread = iDivUp(rows, WARP_SIZE);
    int blocks = iDivUp(ops, ops_per_block);

    dim3 dimBlock(WARP_SIZE, ops_per_block);
    dim3 dimGrid(blocks, 1);
    
    batchBlockSetIdentityKernel<T><<< dimGrid, dimBlock, 0, handle.stream >>> (
		block_ptrs, row_offset, col_offset, ld, 
		rows, cols, ops, rows_per_thread
	);
    
    gpuErrchk( cudaGetLastError() );
}

template<class T>
void batchBlockSetZeroT(T** block_ptrs, int row_offset, int col_offset, int ld, int rows, int cols, int ops, GPUBlasHandle& handle)
{
	if(ops == 0 || rows == 0 || cols == 0) return;
	
	const int MAX_TPB = 512;
	
	int ops_per_block, rows_per_thread, thread_x;
	int blocks_x, blocks_y;
	
	if(rows > MAX_TPB)
	{
		ops_per_block = 1;
		rows_per_thread = 1;
		thread_x = MAX_TPB;
		blocks_x = ops;
		blocks_y = iDivUp(rows, MAX_TPB);
	}
	else
	{
		ops_per_block = 8;
		rows_per_thread = iDivUp(rows, WARP_SIZE);
		thread_x = WARP_SIZE;
		blocks_x = iDivUp(ops, ops_per_block);
		blocks_y = 1;
	}
	
    dim3 dimBlock(thread_x, ops_per_block);
    dim3 dimGrid(blocks_x, blocks_y);
    
    batchBlockSetZeroKernel<T><<< dimGrid, dimBlock, 0, handle.stream >>> (
		block_ptrs, row_offset, col_offset, ld, 
		rows, cols, ops, rows_per_thread
	);
    
    gpuErrchk( cudaGetLastError() );
}

void batchBlockSetIdentity(double** block_ptrs, int row_offset, int col_offset, int ld, int rows, int cols, int ops, GPUBlasHandle& handle)
{batchBlockSetIdentityT<double>(block_ptrs, row_offset, col_offset, ld, rows, cols, ops, handle);}

void batchBlockSetZero(double** block_ptrs, int row_offset, int col_offset, int ld, int rows, int cols, int ops, GPUBlasHandle& handle)
{batchBlockSetZeroT<double>(block_ptrs, row_offset, col_offset, ld, rows, cols, ops, handle);}

void batchBlockSetIdentity(float** block_ptrs, int row_offset, int col_offset, int ld, int rows, int cols, int ops, GPUBlasHandle& handle)
{batchBlockSetIdentityT<float>(block_ptrs, row_offset, col_offset, ld, rows, cols, ops, handle);}

void batchBlockSetZero(float** block_ptrs, int row_offset, int col_offset, int ld, int rows, int cols, int ops, GPUBlasHandle& handle)
{batchBlockSetZeroT<float>(block_ptrs, row_offset, col_offset, ld, rows, cols, ops, handle);}
