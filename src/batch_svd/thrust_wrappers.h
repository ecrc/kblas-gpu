#ifndef __THRUST_WRAPPERS_H__
#define __THRUST_WRAPPERS_H__

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Some hmatrix related stuff
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void basisProjectionOffsetScan(int* new_rows, int* old_rows, int* offsets, int n);
void getNewRankFromErrorThreshold(double* s, int stride_s, int* new_rank, int current_rank, double eps, int relative, int num_ops);
void getNewRankFromErrorThreshold(float* s, int stride_s, int* new_rank, int current_rank, double eps, int relative, int num_ops);
void generateLevelProjectionPointers(
    int* leaf_node_index, int* basis_tree_nodes, float* basis_tree_data, float** array_of_tree_nodes, 
    int increment, int coupling_start, int tree_node_offset, int num_arrays
);
void generateLevelProjectionPointers(
    int* leaf_node_index, int* basis_tree_nodes, double* basis_tree_data, double** array_of_tree_nodes, 
    int increment, int coupling_start, int tree_node_offset, int num_arrays
);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Some array utility functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void exclusiveScan(int* array, int num_entries, int* result, int init = 0, cudaStream_t stream = 0);
void inclusiveScan(int* array, int num_entries, int* result, cudaStream_t stream = 0);

void fillArray(float* array, int num_entries, float val, cudaStream_t stream = 0);
void fillArray(double* array, int num_entries, double val, cudaStream_t stream = 0);

int getMaxElement(int* a, int elements, cudaStream_t stream = 0);
float getMaxElement(float* a, int elements, cudaStream_t stream = 0);
double getMaxElement(double* a, int elements, cudaStream_t stream = 0);

double reduceSum(double* a, int elements, cudaStream_t stream = 0);
float reduceSum(float* a, int elements, cudaStream_t stream = 0);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generating array of pointers from either a strided array or another array of pointers
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void generateArrayOfPointers(double* original_array, double** array_of_arrays, int stride, int offset, int num_arrays, cudaStream_t stream = 0);
void generateArrayOfPointers(double* original_array, double** array_of_arrays, int stride, int num_arrays, cudaStream_t stream = 0);
void generateArrayOfPointers(double** original_array, double** array_of_arrays, int stride, int num_arrays, cudaStream_t stream = 0);
void generateArrayOfPointers(double** original_array, double** array_of_arrays, int stride, int offset, int num_arrays, cudaStream_t stream = 0);

void generateArrayOfPointers(float* original_array, float** array_of_arrays, int stride, int offset, int num_arrays, cudaStream_t stream = 0);
void generateArrayOfPointers(float* original_array, float** array_of_arrays, int stride, int num_arrays, cudaStream_t stream = 0);
void generateArrayOfPointers(float** original_array, float** array_of_arrays, int stride, int num_arrays, cudaStream_t stream = 0);
void generateArrayOfPointers(float** original_array, float** array_of_arrays, int stride, int offset, int num_arrays, cudaStream_t stream = 0);

void generateArrayOfPointers(int* original_array, int** array_of_arrays, int stride, int offset, int num_arrays, cudaStream_t stream = 0);
void generateArrayOfPointers(int* original_array, int** array_of_arrays, int stride, int num_arrays, cudaStream_t stream = 0);
void generateArrayOfPointers(int** original_array, int** array_of_arrays, int stride, int num_arrays, cudaStream_t stream = 0);
void generateArrayOfPointers(int** original_array, int** array_of_arrays, int stride, int offset, int num_arrays, cudaStream_t stream = 0);

#endif
