/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xhelper_funcs.ch

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef __XHELPER_FUNCS__
#define __XHELPER_FUNCS__

#include "kblas_prec_def.h"


//==============================================================================================
int iset_value_diff_1(int* output_array, const int* input_array1, int* input_array2,
                      long batchCount, cudaStream_t cuda_stream);

int Xset_value_1( TYPE *output_array, TYPE input,
                  long batchCount, cudaStream_t cuda_stream);

// input, output_array: host pointer to array of device pointers to device buffers
int Xset_pointer_1( TYPE **output_array, TYPE **input, int offset_r, int offset_c, int* lda,
                    long batchCount, cudaStream_t cuda_stream);

// inputX, output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_2( TYPE **output_array1, const TYPE **input1, int offset_r1, int offset_c1, int lda1,
                    TYPE **output_array2, const TYPE **input2, int offset_r2, int offset_c2, int lda2,
                    long batchCount, cudaStream_t cuda_stream);

int Xset_pointer_3( TYPE **output_array1, const TYPE **input1, int offset_r1, int offset_c1, int lda1,
                    TYPE **output_array2, const TYPE **input2, int offset_r2, int offset_c2, int lda2,
                    TYPE **output_array3, const TYPE **input3, int offset_r3, int offset_c3, int lda3,
                    long batchCount, cudaStream_t cuda_stream);

// input: host pointer to device buffers
// output_array: host pointer to array of device pointers to device buffers
int Xset_pointer_1( TYPE **output_array, const TYPE *input, int lda, long batch_offset,
                    long batchCount, cudaStream_t cuda_stream);

// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_2( TYPE **output_array1, const TYPE *input1, int ld1, long batch_offset1,
                    TYPE **output_array2, const TYPE *input2, int ld2, long batch_offset2,
                    long batchCount, cudaStream_t cuda_stream);

// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_3( TYPE **output_array1, const TYPE *input1, int ld1, long batch_offset1,
                    TYPE **output_array2, const TYPE *input2, int ld2, long batch_offset2,
                    TYPE **output_array3, const TYPE *input3, int ld3, long batch_offset3,
                    long batchCount, cudaStream_t cuda_stream);

// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_4( TYPE **output_array1, const TYPE *input1, int ld1, long batch_offset1,
                    TYPE **output_array2, const TYPE *input2, int ld2, long batch_offset2,
                    TYPE **output_array3, const TYPE *input3, int ld3, long batch_offset3,
                    TYPE **output_array4, const TYPE *input4, int ld4, long batch_offset4,
                    long batchCount, cudaStream_t cuda_stream);

// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_5( TYPE **output_array1, const TYPE *input1, int ld1, long batch_offset1,
                    TYPE **output_array2, const TYPE *input2, int ld2, long batch_offset2,
                    TYPE **output_array3, const TYPE *input3, int ld3, long batch_offset3,
                    TYPE **output_array4, const TYPE *input4, int ld4, long batch_offset4,
                    TYPE **output_array5, const TYPE *input5, int ld5, long batch_offset5,
                    long batchCount, cudaStream_t cuda_stream);

// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_7( TYPE **output_array1, const TYPE *input1, int ld1, long batch_offset1,
                    TYPE **output_array2, const TYPE *input2, int ld2, long batch_offset2,
                    TYPE **output_array3, const TYPE *input3, int ld3, long batch_offset3,
                    TYPE **output_array4, const TYPE *input4, int ld4, long batch_offset4,
                    TYPE **output_array5, const TYPE *input5, int ld5, long batch_offset5,
                    TYPE **output_array6, const TYPE *input6, int ld6, long batch_offset6,
                    TYPE **output_array7, const TYPE *input7, int ld7, long batch_offset7,
                    long batchCount, cudaStream_t cuda_stream);

// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_9( TYPE **output_array1, const TYPE *input1, int ld1, long batch_offset1,
                    TYPE **output_array2, const TYPE *input2, int ld2, long batch_offset2,
                    TYPE **output_array3, const TYPE *input3, int ld3, long batch_offset3,
                    TYPE **output_array4, const TYPE *input4, int ld4, long batch_offset4,
                    TYPE **output_array5, const TYPE *input5, int ld5, long batch_offset5,
                    TYPE **output_array6, const TYPE *input6, int ld6, long batch_offset6,
                    TYPE **output_array7, const TYPE *input7, int ld7, long batch_offset7,
                    TYPE **output_array8, const TYPE *input8, int ld8, long batch_offset8,
                    TYPE **output_array9, const TYPE *input9, int ld9, long batch_offset9,
                    long batchCount, cudaStream_t cuda_stream);

int Xset_pointer_2d_1( TYPE **output_array, const TYPE *input, int lda, long mt, long nt, long mb, long nb,
                        cudaStream_t cuda_stream);

#endif //__XHELPER_FUNCS__
