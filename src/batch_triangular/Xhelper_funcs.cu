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
#include <stdlib.h>
#include <stdio.h>
#include <set>
#include <sys/time.h>
#include <cublas_v2.h>
#include "kblas.h"
#include "operators.h"
// #include "Xtr_common.ch"

#include "kblas_error.h"
#include "kblas_prec_def.h"



//==============================================================================================
template<typename T>
__global__ void kernel_set_value_1(T *output_array, T input, long count){
  long ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind < count)
    output_array[ind] = (T)input;
}
int Xset_value_1(
  TYPE *output_array, TYPE input,
  long batchCount, cudaStream_t cuda_stream){
  kernel_set_value_1<TYPE><<< batchCount / 256 + (batchCount % 256 > 0) , 256, 0, cuda_stream>>>(
    output_array, input, batchCount);
  check_error_ret( cudaGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}
//==============================================================================================
template<typename T>
__global__ void kernel_set_pointer_1(
  T **output_array, T **input, int offset_r, int offset_c, int* lda,
  long count){
  int ind = blockIdx.x;// * blockDim.x + threadIdx.x;
  //if(ind < count){
    T* ptr = input[ind];
    output_array[ind] = &ptr[offset_r + offset_c * lda[ind]];
  //}
}
// input, output_array: host pointer to array of device pointers to device buffers
int Xset_pointer_1(
  TYPE **output_array, TYPE **input, int offset_r, int offset_c, int* lda,
  long batchCount, cudaStream_t cuda_stream){
  dim3 block(1,1);
  dim3 grid(batchCount);/* / block.x + ((batchCount % block.x) > 0),1);*/
  kernel_set_pointer_1<TYPE><<< grid, block, 0, cuda_stream>>>(
    output_array, input, offset_r, offset_c, lda,
    batchCount);
  check_error_ret( cudaGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

//==============================================================================================
template<typename T>
__global__ void kernel_set_pointer_2(
  T **output_array1, const T **input1, int offset_r1, int offset_c1, int lda1,
  T **output_array2, const T **input2, int offset_r2, int offset_c2, int lda2,
  long count){
  int ind = blockIdx.x;// * blockDim.x + threadIdx.x;
  //if(ind < count){
    const T* ptr1 = input1[ind];
    output_array1[ind] = (T*)&ptr1[offset_r1 + offset_c1 * lda1];
    const T* ptr2 = input2[ind];
    output_array2[ind] = (T*)&ptr2[offset_r2 + offset_c2 * lda2];
  //}
}
// inputX, output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_2(
  TYPE **output_array1, const TYPE **input1, int offset_r1, int offset_c1, int lda1,
  TYPE **output_array2, const TYPE **input2, int offset_r2, int offset_c2, int lda2,
  long batchCount, cudaStream_t cuda_stream){
  dim3 block(1,1);
  dim3 grid(batchCount);/* / block.x + ((batchCount % block.x) > 0),1);*/
  kernel_set_pointer_2<TYPE><<< grid, block, 0, cuda_stream>>>(
    output_array1, input1, offset_r1, offset_c1, lda1,
    output_array2, input2, offset_r2, offset_c2, lda2,
    batchCount);
  check_error_ret( cudaGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}
//==============================================================================================
template<typename T>
__global__ void kernel_set_pointer_3(
  T **output_array1, const T **input1, int offset_r1, int offset_c1, int lda1,
  T **output_array2, const T **input2, int offset_r2, int offset_c2, int lda2,
  T **output_array3, const T **input3, int offset_r3, int offset_c3, int lda3,
  long count){
  int ind = blockIdx.x;// * blockDim.x + threadIdx.x;
  //if(ind < count){
    const T* ptr1 = input1[ind];
    output_array1[ind] = (T*)&ptr1[offset_r1 + offset_c1 * lda1];
    const T* ptr2 = input2[ind];
    output_array2[ind] = (T*)&ptr2[offset_r2 + offset_c2 * lda2];
    const T* ptr3 = input3[ind];
    output_array3[ind] = (T*)&ptr3[offset_r3 + offset_c3 * lda3];
  //}
}
// inputX, output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_3(
  TYPE **output_array1, const TYPE **input1, int offset_r1, int offset_c1, int lda1,
  TYPE **output_array2, const TYPE **input2, int offset_r2, int offset_c2, int lda2,
  TYPE **output_array3, const TYPE **input3, int offset_r3, int offset_c3, int lda3,
  long batchCount, cudaStream_t cuda_stream){
  dim3 block(1,1);
  dim3 grid(batchCount);/* / block.x + ((batchCount % block.x) > 0),1);*/
  kernel_set_pointer_3<TYPE><<< grid, block, 0, cuda_stream>>>(
    output_array1, input1, offset_r1, offset_c1, lda1,
    output_array2, input2, offset_r2, offset_c2, lda2,
    output_array3, input3, offset_r3, offset_c3, lda3,
    batchCount);
  check_error_ret( cudaGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

//==============================================================================================
template<typename T>
__global__ void kernel_set_pointer_1(
  T **output_array, const T *input, int lda, long batch_offset){
  output_array[blockIdx.x] = (T*)&input[blockIdx.x * batch_offset];
}
// input: host pointer to device buffers
// output_array: host pointer to array of device pointers to device buffers
int Xset_pointer_1(
  TYPE **output_array, const TYPE *input, int lda, long batch_offset,
  long batchCount, cudaStream_t cuda_stream){
  kernel_set_pointer_1<TYPE><<< batchCount, 1, 0, cuda_stream>>>(
    output_array, input, lda, batch_offset);
  check_error_ret( cudaGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

//==============================================================================================
template<typename T>
__global__ void kernel_set_pointer_2(
  T **output_array1, const T *input1, int ld1, long batch_offset1,
  T **output_array2, const T *input2, int ld2, long batch_offset2){
  output_array1[blockIdx.x] = (T*)input1 + blockIdx.x * batch_offset1;
  output_array2[blockIdx.x] = (T*)input2 + blockIdx.x * batch_offset2;
}
// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_2(
  TYPE **output_array1, const TYPE *input1, int ld1, long batch_offset1,
  TYPE **output_array2, const TYPE *input2, int ld2, long batch_offset2,
  long batchCount, cudaStream_t cuda_stream){
  kernel_set_pointer_2<TYPE><<< batchCount, 1, 0, cuda_stream>>>(
    output_array1, input1, ld1, batch_offset1,
    output_array2, input2, ld2, batch_offset2
  );
  check_error_ret( cudaGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

//==============================================================================================
template<typename T>
__global__ void kernel_set_pointer_3(
  T **output_array1, const T *input1, int ld1, long batch_offset1,
  T **output_array2, const T *input2, int ld2, long batch_offset2,
  T **output_array3, const T *input3, int ld3, long batch_offset3){
  output_array1[blockIdx.x] = (T*)input1 + blockIdx.x * batch_offset1;
  output_array2[blockIdx.x] = (T*)input2 + blockIdx.x * batch_offset2;
  output_array3[blockIdx.x] = (T*)input3 + blockIdx.x * batch_offset3;
}
// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_3(
  TYPE **output_array1, const TYPE *input1, int ld1, long batch_offset1,
  TYPE **output_array2, const TYPE *input2, int ld2, long batch_offset2,
  TYPE **output_array3, const TYPE *input3, int ld3, long batch_offset3,
  long batchCount, cudaStream_t cuda_stream){
  kernel_set_pointer_3<TYPE><<< batchCount, 1, 0, cuda_stream>>>(
    output_array1, input1, ld1, batch_offset1,
    output_array2, input2, ld2, batch_offset2,
    output_array3, input3, ld3, batch_offset3
  );
  check_error_ret( cudaGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

//==============================================================================================
template<typename T>
__global__ void kernel_set_pointer_4(
  T **output_array1, const T *input1, int ld1, long batch_offset1,
  T **output_array2, const T *input2, int ld2, long batch_offset2,
  T **output_array3, const T *input3, int ld3, long batch_offset3,
  T **output_array4, const T *input4, int ld4, long batch_offset4){
  output_array1[blockIdx.x] = (T*)input1 + blockIdx.x * batch_offset1;
  output_array2[blockIdx.x] = (T*)input2 + blockIdx.x * batch_offset2;
  output_array3[blockIdx.x] = (T*)input3 + blockIdx.x * batch_offset3;
  output_array4[blockIdx.x] = (T*)input4 + blockIdx.x * batch_offset4;
}
// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_4(
  TYPE **output_array1, const TYPE *input1, int ld1, long batch_offset1,
  TYPE **output_array2, const TYPE *input2, int ld2, long batch_offset2,
  TYPE **output_array3, const TYPE *input3, int ld3, long batch_offset3,
  TYPE **output_array4, const TYPE *input4, int ld4, long batch_offset4,
  long batchCount, cudaStream_t cuda_stream){
  kernel_set_pointer_4<TYPE><<< batchCount, 1, 0, cuda_stream>>>(
    output_array1, input1, ld1, batch_offset1,
    output_array2, input2, ld2, batch_offset2,
    output_array3, input3, ld3, batch_offset3,
    output_array4, input4, ld4, batch_offset4
  );
  check_error_ret( cudaGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

//==============================================================================================
template<typename T>
__global__ void kernel_set_pointer_5(
  T **output_array1, const T *input1, int ld1, long batch_offset1,
  T **output_array2, const T *input2, int ld2, long batch_offset2,
  T **output_array3, const T *input3, int ld3, long batch_offset3,
  T **output_array4, const T *input4, int ld4, long batch_offset4,
  T **output_array5, const T *input5, int ld5, long batch_offset5){
  output_array1[blockIdx.x] = (T*)input1 + blockIdx.x * batch_offset1;
  output_array2[blockIdx.x] = (T*)input2 + blockIdx.x * batch_offset2;
  output_array3[blockIdx.x] = (T*)input3 + blockIdx.x * batch_offset3;
  output_array4[blockIdx.x] = (T*)input4 + blockIdx.x * batch_offset4;
  output_array5[blockIdx.x] = (T*)input5 + blockIdx.x * batch_offset5;
}
// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
int Xset_pointer_5(
  TYPE **output_array1, const TYPE *input1, int ld1, long batch_offset1,
  TYPE **output_array2, const TYPE *input2, int ld2, long batch_offset2,
  TYPE **output_array3, const TYPE *input3, int ld3, long batch_offset3,
  TYPE **output_array4, const TYPE *input4, int ld4, long batch_offset4,
  TYPE **output_array5, const TYPE *input5, int ld5, long batch_offset5,
  long batchCount, cudaStream_t cuda_stream){
  kernel_set_pointer_5<TYPE><<< batchCount, 1, 0, cuda_stream>>>(
    output_array1, input1, ld1, batch_offset1,
    output_array2, input2, ld2, batch_offset2,
    output_array3, input3, ld3, batch_offset3,
    output_array4, input4, ld4, batch_offset4,
    output_array5, input5, ld5, batch_offset5
  );
  check_error_ret( cudaGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}
