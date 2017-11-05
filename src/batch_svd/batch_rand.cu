#include <gpu_util.h>
#include <batch_rand.h>
#include <curand.h>
#include <curand_kernel.h>

template<class T> __device__ __forceinline__ T tcurand_normal(curandState* state);
template<> __device__ __forceinline__  float tcurand_normal<float>(curandState* state) { return curand_normal(state); }
template<> __device__ __forceinline__  double tcurand_normal<double>(curandState* state) { return curand_normal_double(state); }

template<class T>
__global__
void batch_rand_blocks_kernel(T** M_ptrs, int ldm, int rows, int cols, int num_ops)
{
	int op_id = blockIdx.z;
	if(op_id >= num_ops) return;
	
	int thread_row = blockIdx.x * blockDim.x + threadIdx.x;
	int thread_col = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(thread_row >= rows || thread_col >= cols) return;
	
	int entry_index = thread_row + thread_col * rows;
	unsigned long long seed = op_id * rows * cols + entry_index;
	
	curandState state;
	curand_init(seed, 0, 0, &state);
	M_ptrs[op_id][thread_row + thread_col * ldm] = tcurand_normal<T>(&state);
}

template<class T>
void batch_rand_blocks_template(T** M_ptrs, int ldm, int rows, int cols, int num_ops, GPUBlasHandle& handle)
{
	int max_thread_x = 64;
	int max_thread_y = 8;
	
	int thread_x = (rows < max_thread_x ? rows : max_thread_x);
	int thread_y = (cols < max_thread_y ? cols : max_thread_y);
	
	int grid_x = iDivUp(rows, thread_x);
	int grid_y = iDivUp(cols, thread_y);
	
	dim3 dimBlock(thread_x, thread_y);
	dim3 dimGrid(grid_x, grid_y, num_ops);
	
	batch_rand_blocks_kernel<T><<< dimGrid, dimBlock, 0, handle.stream >>>(M_ptrs, ldm, rows, cols, num_ops);

	gpuErrchk( cudaGetLastError() );
}

void batch_rand_blocks(float** M_ptrs, int ldm, int rows, int cols, int num_ops, GPUBlasHandle& handle)
{
	batch_rand_blocks_template<float>(M_ptrs, ldm, rows, cols, num_ops, handle);
}

void batch_rand_blocks(double** M_ptrs, int ldm, int rows, int cols, int num_ops, GPUBlasHandle& handle)
{
	batch_rand_blocks_template<double>(M_ptrs, ldm, rows, cols, num_ops, handle);
}
