/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_svd/thrust_wrappers.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Wajih Halim Boukaram
 * @date 2018-11-14
 **/

#ifndef __THRUST_WRAPPERS_H__
#define __THRUST_WRAPPERS_H__

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Some array utility functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void exclusiveScan(int* array, int num_entries, int* result, int init = 0, cudaStream_t stream = 0);
void inclusiveScan(int* array, int num_entries, int* result, cudaStream_t stream = 0);

void fillArray(float* array, int num_entries, float val, cudaStream_t stream = 0);
void fillArray(double* array, int num_entries, double val, cudaStream_t stream = 0);
void fillArray(int* array, int num_entries, int val, cudaStream_t stream = 0);

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
