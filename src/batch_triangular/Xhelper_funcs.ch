/**
  - -* (C) Copyright 2013 King Abdullah University of Science and Technology
  Authors:
  Ali Charara (ali.charara@kaust.edu.sa)
  David Keyes (david.keyes@kaust.edu.sa)
  Hatem Ltaief (hatem.ltaief@kaust.edu.sa)

  Redistribution  and  use  in  source and binary forms, with or without
  modification,  are  permitted  provided  that the following conditions
  are met:

  * Redistributions  of  source  code  must  retain  the above copyright
  * notice,  this  list  of  conditions  and  the  following  disclaimer.
  * Redistributions  in  binary  form must reproduce the above copyright
  * notice,  this list of conditions and the following disclaimer in the
  * documentation  and/or other materials provided with the distribution.
  * Neither  the  name of the King Abdullah University of Science and
  * Technology nor the names of its contributors may be used to endorse
  * or promote products derived from this software without specific prior
  * written permission.
  *
  THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
  LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**/

#ifndef __XHELPER_FUNCS__
#define __XHELPER_FUNCS__

#include "kblas_prec_def.h"


//==============================================================================================
int Xset_value_diff_1(int* output_array, const int* input_array1, int* input_array2, long batchCount, cudaStream_t cuda_stream);

int Xset_value_1(
  TYPE *output_array, TYPE input,
  long batchCount, cudaStream_t cuda_stream);

// input, output_array: host pointer to array of device pointers to device buffers
int Xset_pointer_1(
  TYPE **output_array, TYPE **input, int offset_r, int offset_c, int* lda,
  long batchCount, cudaStream_t cuda_stream);

// inputX, output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_2(
  TYPE **output_array1, const TYPE **input1, int offset_r1, int offset_c1, int lda1,
  TYPE **output_array2, const TYPE **input2, int offset_r2, int offset_c2, int lda2,
  long batchCount, cudaStream_t cuda_stream);

int Xset_pointer_3(
  TYPE **output_array1, const TYPE **input1, int offset_r1, int offset_c1, int lda1,
  TYPE **output_array2, const TYPE **input2, int offset_r2, int offset_c2, int lda2,
  TYPE **output_array3, const TYPE **input3, int offset_r3, int offset_c3, int lda3,
  long batchCount, cudaStream_t cuda_stream);

// input: host pointer to device buffers
// output_array: host pointer to array of device pointers to device buffers
int Xset_pointer_1(
  TYPE **output_array, const TYPE *input, int lda, long batch_offset,
  long batchCount, cudaStream_t cuda_stream);

// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_2(
  TYPE **output_array1, const TYPE *input1, int ld1, long batch_offset1,
  TYPE **output_array2, const TYPE *input2, int ld2, long batch_offset2,
  long batchCount, cudaStream_t cuda_stream);

// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_3(
  TYPE **output_array1, const TYPE *input1, int ld1, long batch_offset1,
  TYPE **output_array2, const TYPE *input2, int ld2, long batch_offset2,
  TYPE **output_array3, const TYPE *input3, int ld3, long batch_offset3,
  long batchCount, cudaStream_t cuda_stream);

// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_4(
  TYPE **output_array1, const TYPE *input1, int ld1, long batch_offset1,
  TYPE **output_array2, const TYPE *input2, int ld2, long batch_offset2,
  TYPE **output_array3, const TYPE *input3, int ld3, long batch_offset3,
  TYPE **output_array4, const TYPE *input4, int ld4, long batch_offset4,
  long batchCount, cudaStream_t cuda_stream);

// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_5(
  TYPE **output_array1, const TYPE *input1, int ld1, long batch_offset1,
  TYPE **output_array2, const TYPE *input2, int ld2, long batch_offset2,
  TYPE **output_array3, const TYPE *input3, int ld3, long batch_offset3,
  TYPE **output_array4, const TYPE *input4, int ld4, long batch_offset4,
  TYPE **output_array5, const TYPE *input5, int ld5, long batch_offset5,
  long batchCount, cudaStream_t cuda_stream);

#endif //__XHELPER_FUNCS__