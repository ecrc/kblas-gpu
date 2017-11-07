#ifndef __BATCH_BLOCK_COPY_H__
#define __BATCH_BLOCK_COPY_H__

#ifdef __cplusplus
extern "C" {
#endif
// Strided interface
int kblasDcopyBlock_batch_strided(
	kblasHandle_t handle, int rows, int cols, 
	double* dest_array, int row_offset_dest, int col_offset_dest, int ld_dest, int stride_dest,
	double* src_array, int row_offset_src, int col_offset_src, int ld_src, int stride_src, int ops
);

int kblasScopyBlock_batch_strided(
	kblasHandle_t handle, int rows, int cols, 	
	float* dest_array, int row_offset_dest, int col_offset_dest, int ld_dest, int stride_dest,
	float* src_array, int row_offset_src, int col_offset_src, int ld_src, int stride_src, int ops
);

// Array of pointers interface
int kblasDcopyBlock_batch(
	kblasHandle_t handle, int rows, int cols, 
	double** dest_array, int row_offset_dest, int col_offset_dest, int ld_dest,
	double** src_array, int row_offset_src, int col_offset_src, int ld_src, int ops
);

int kblasScopyBlock_batch(
	kblasHandle_t handle, int rows, int cols, 	
	float** dest_array, int row_offset_dest, int col_offset_dest, int ld_dest,
	float** src_array, int row_offset_src, int col_offset_src, int ld_src, int ops
);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
// Strided interface
inline int kblas_copyBlock_batch(
	kblasHandle_t handle, int rows, int cols, 	
	double* dest_array, int row_offset_dest, int col_offset_dest, int ld_dest, int stride_dest, 
	double* src_array, int row_offset_src, int col_offset_src, int ld_src, int stride_src, int ops
)
{
	return kblasDcopyBlock_batch_strided(
		handle, rows, cols, 
		dest_array, row_offset_dest, col_offset_dest, ld_dest, stride_dest,
		src_array, row_offset_src, col_offset_src, ld_src, stride_src, ops
	);
}
inline int kblas_copyBlock_batch(
	kblasHandle_t handle, int rows, int cols, 	
	float* dest_array, int row_offset_dest, int col_offset_dest, int ld_dest, int stride_dest, 
	float* src_array, int row_offset_src, int col_offset_src, int ld_src, int stride_src, int ops
)
{
	return kblasScopyBlock_batch_strided(
		handle, rows, cols, 
		dest_array, row_offset_dest, col_offset_dest, ld_dest, stride_dest,
		src_array, row_offset_src, col_offset_src, ld_src, stride_src, ops
	);
}

// Array of pointers interface
inline int kblas_copyBlock_batch(
	kblasHandle_t handle, int rows, int cols, 
	double** dest_array, int row_offset_dest, int col_offset_dest, int ld_dest,
	double** src_array, int row_offset_src, int col_offset_src, int ld_src, int ops
)
{
	return kblasDcopyBlock_batch(
		handle, rows, cols, 
		dest_array, row_offset_dest, col_offset_dest, ld_dest, 
		src_array, row_offset_src, col_offset_src, ld_src, ops
	);	
}

inline int kblas_copyBlock_batch(
	kblasHandle_t handle, int rows, int cols, 	
	float** dest_array, int row_offset_dest, int col_offset_dest, int ld_dest,
	float** src_array, int row_offset_src, int col_offset_src, int ld_src, int ops
)
{
	return kblasScopyBlock_batch(
		handle, rows, cols, 
		dest_array, row_offset_dest, col_offset_dest, ld_dest, 
		src_array, row_offset_src, col_offset_src, ld_src, ops
	);
}

// Common interface for array of pointers with dummy strides
template<class T>
inline int kblas_copyBlock_batch(
	kblasHandle_t handle, int rows, int cols, 	
	T** dest_array, int row_offset_dest, int col_offset_dest, int ld_dest, int stride_dest, 
	T** src_array, int row_offset_src, int col_offset_src, int ld_src, int stride_src, int ops
)
{
	return kblas_copyBlock_batch(
		handle, rows, cols, 
		dest_array, row_offset_dest, col_offset_dest, ld_dest,
		src_array, row_offset_src, col_offset_src, ld_src, ops
	);
}
#endif

#endif //__BATCH_BLOCK_COPY_H__
