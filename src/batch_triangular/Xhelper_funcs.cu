/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xhelper_funcs.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#include <stdlib.h>
#include <stdio.h>
#include <set>
#include <sys/time.h>
#include <hipblas.h>
#include "kblas.h"
#include "kblas_operators.h"
// #include "Xtr_common.ch"

#include "kblas_error.h"
#include "kblas_prec_def.h"

#include "Xhelper_funcs.cuh"

//==============================================================================================
int Xset_value_1( TYPE *output_array, TYPE input,
                  long batchCount, hipStream_t hip_stream)
{
  return Xset_value_1_core<TYPE>(output_array, input, batchCount, hip_stream);
}
//==============================================================================================
// input, output_array: host pointer to array of device pointers to device buffers
int Xset_pointer_1( TYPE **output_array, TYPE **input, int offset_r, int offset_c, int* lda,
                    long batchCount, hipStream_t hip_stream)
{
  return Xset_pointer_1_core<TYPE>( output_array, input, offset_r, offset_c, lda,
                                    batchCount, hip_stream);
}

//==============================================================================================
// inputX, output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_2( TYPE **output_array1, const TYPE **input1, int offset_r1, int offset_c1, int lda1,
                    TYPE **output_array2, const TYPE **input2, int offset_r2, int offset_c2, int lda2,
                    long batchCount, hipStream_t hip_stream)
{
  return Xset_pointer_2_core<TYPE>( output_array1, input1, offset_r1, offset_c1, lda1,
                                    output_array2, input2, offset_r2, offset_c2, lda2,
                                    batchCount, hip_stream);
}
//==============================================================================================
// inputX, output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_3( TYPE **output_array1, const TYPE **input1, int offset_r1, int offset_c1, int lda1,
                    TYPE **output_array2, const TYPE **input2, int offset_r2, int offset_c2, int lda2,
                    TYPE **output_array3, const TYPE **input3, int offset_r3, int offset_c3, int lda3,
                    long batchCount, hipStream_t hip_stream)
{
  return Xset_pointer_3_core<TYPE>( output_array1, input1, offset_r1, offset_c1, lda1,
                                    output_array2, input2, offset_r2, offset_c2, lda2,
                                    output_array3, input3, offset_r3, offset_c3, lda3,
                                    batchCount, hip_stream);
}

//==============================================================================================
// input: host pointer to device buffers
// output_array: host pointer to array of device pointers to device buffers
int Xset_pointer_1( TYPE **output_array, const TYPE *input, int lda, long batch_offset,
                    long batchCount, hipStream_t hip_stream)
{
  return Xset_pointer_1_core<TYPE>( output_array, input, lda, batch_offset,
                                    batchCount, hip_stream);
}

//==============================================================================================
// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_2( TYPE **output_array1, const TYPE *input1, int ld1, long batch_offset1,
                    TYPE **output_array2, const TYPE *input2, int ld2, long batch_offset2,
                    long batchCount, hipStream_t hip_stream)
{
  return Xset_pointer_2_core<TYPE>( output_array1, input1, ld1, batch_offset1,
                                    output_array2, input2, ld2, batch_offset2,
                                    batchCount, hip_stream);
}

//==============================================================================================
// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_3( TYPE **output_array1, const TYPE *input1, int ld1, long batch_offset1,
                    TYPE **output_array2, const TYPE *input2, int ld2, long batch_offset2,
                    TYPE **output_array3, const TYPE *input3, int ld3, long batch_offset3,
                    long batchCount, hipStream_t hip_stream)
{
  return Xset_pointer_3_core<TYPE>( output_array1, input1, ld1, batch_offset1,
                                    output_array2, input2, ld2, batch_offset2,
                                    output_array3, input3, ld3, batch_offset3,
                                    batchCount, hip_stream);
}

//==============================================================================================
// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_4( TYPE **output_array1, const TYPE *input1, int ld1, long batch_offset1,
                    TYPE **output_array2, const TYPE *input2, int ld2, long batch_offset2,
                    TYPE **output_array3, const TYPE *input3, int ld3, long batch_offset3,
                    TYPE **output_array4, const TYPE *input4, int ld4, long batch_offset4,
                    long batchCount, hipStream_t hip_stream)
{
  return Xset_pointer_4_core<TYPE>( output_array1, input1, ld1, batch_offset1,
                                    output_array2, input2, ld2, batch_offset2,
                                    output_array3, input3, ld3, batch_offset3,
                                    output_array4, input4, ld4, batch_offset4,
                                    batchCount, hip_stream);
}

//==============================================================================================
// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_5( TYPE **output_array1, const TYPE *input1, int ld1, long batch_offset1,
                    TYPE **output_array2, const TYPE *input2, int ld2, long batch_offset2,
                    TYPE **output_array3, const TYPE *input3, int ld3, long batch_offset3,
                    TYPE **output_array4, const TYPE *input4, int ld4, long batch_offset4,
                    TYPE **output_array5, const TYPE *input5, int ld5, long batch_offset5,
                    long batchCount, hipStream_t hip_stream)
{
  return Xset_pointer_5_core<TYPE>( output_array1, input1, ld1, batch_offset1,
                                    output_array2, input2, ld2, batch_offset2,
                                    output_array3, input3, ld3, batch_offset3,
                                    output_array4, input4, ld4, batch_offset4,
                                    output_array5, input5, ld5, batch_offset5,
                                    batchCount, hip_stream);
}

//==============================================================================================
// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_7( TYPE **output_array1, const TYPE *input1, int ld1, long batch_offset1,
                    TYPE **output_array2, const TYPE *input2, int ld2, long batch_offset2,
                    TYPE **output_array3, const TYPE *input3, int ld3, long batch_offset3,
                    TYPE **output_array4, const TYPE *input4, int ld4, long batch_offset4,
                    TYPE **output_array5, const TYPE *input5, int ld5, long batch_offset5,
                    TYPE **output_array6, const TYPE *input6, int ld6, long batch_offset6,
                    TYPE **output_array7, const TYPE *input7, int ld7, long batch_offset7,
                    long batchCount, hipStream_t hip_stream)
{
  return Xset_pointer_7_core<TYPE>( output_array1, input1, ld1, batch_offset1,
                                    output_array2, input2, ld2, batch_offset2,
                                    output_array3, input3, ld3, batch_offset3,
                                    output_array4, input4, ld4, batch_offset4,
                                    output_array5, input5, ld5, batch_offset5,
                                    output_array6, input6, ld6, batch_offset6,
                                    output_array7, input7, ld7, batch_offset7,
                                    batchCount, hip_stream);
}

//==============================================================================================
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
                    long batchCount, hipStream_t hip_stream)
{
  return Xset_pointer_9_core<TYPE>( output_array1, input1, ld1, batch_offset1,
                                    output_array2, input2, ld2, batch_offset2,
                                    output_array3, input3, ld3, batch_offset3,
                                    output_array4, input4, ld4, batch_offset4,
                                    output_array5, input5, ld5, batch_offset5,
                                    output_array6, input6, ld6, batch_offset6,
                                    output_array7, input7, ld7, batch_offset7,
                                    output_array8, input8, ld8, batch_offset8,
                                    output_array9, input9, ld9, batch_offset9,
                                    batchCount, hip_stream);
}

//==============================================================================================
int Xset_pointer_2d_1( TYPE **output_array, const TYPE *input, int lda, long mt, long nt, long mb, long nb,
                        hipStream_t hip_stream)
{
    return Xset_pointer_2d_1_core<TYPE>( output_array, input, lda, mt, nt, mb, nb,
                                        hip_stream);
}
