#include "hip/hip_runtime.h"
/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xhelper_funcs.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef __XHELPER_FUNCS_CORE_H__
#define __XHELPER_FUNCS_CORE_H__


//==============================================================================================
template<typename T>
__global__
void kernel_set_value_1(T *output_array, T input, long count)
{
  long ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind < count)
    output_array[ind] = (T)input;
}
template<typename T>
inline
int Xset_value_1_core(T *output_array, T input,
                      long batchCount, hipStream_t hip_stream)
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_set_value_1<T>), dim3(batchCount / 256 + (batchCount % 256 > 0) ), dim3(256), 0, hip_stream, 
                        output_array, input, batchCount);
  check_error_ret( hipGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}
//==============================================================================================
template<typename T>
__global__
void kernel_set_value_2(T *output_array1, T input1,
                        T *output_array2, T input2,
                        long count)
{
  long ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind < count){
    output_array1[ind] = (T)input1;
    output_array2[ind] = (T)input2;
  }
}
template<typename T>
inline
int Xset_value_2_core(T *output_array1, T input1,
                      T *output_array2, T input2,
                      long batchCount, hipStream_t hip_stream)
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_set_value_2<T>), dim3(batchCount / 256 + (batchCount % 256 > 0) ), dim3(256), 0, hip_stream, 
                        output_array1, input1,
                        output_array2, input2,
                        batchCount);
  check_error_ret( hipGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}
//==============================================================================================
template<typename T>
__global__
void kernel_set_value_4(T *output_array1, T input1,
                        T *output_array2, T input2,
                        T *output_array3, T input3,
                        T *output_array4, T input4,
                        long count)
{
  long ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind < count){
    output_array1[ind] = (T)input1;
    output_array2[ind] = (T)input2;
    output_array3[ind] = (T)input3;
    output_array4[ind] = (T)input4;
  }
}
template<typename T>
inline
int Xset_value_4_core(T *output_array1, T input1,
                      T *output_array2, T input2,
                      T *output_array3, T input3,
                      T *output_array4, T input4,
                      long batchCount, hipStream_t hip_stream)
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_set_value_4<T>), dim3(batchCount / 256 + (batchCount % 256 > 0) ), dim3(256), 0, hip_stream, 
                        output_array1, input1,
                        output_array2, input2,
                        output_array3, input3,
                        output_array4, input4,
                        batchCount);
  check_error_ret( hipGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}
//==============================================================================================
template<typename T>
__global__
void kernel_set_value_5(T *output_array1, T input1,
                        T *output_array2, T input2,
                        T *output_array3, T input3,
                        T *output_array4, T input4,
                        T *output_array5, T input5,
                        long count)
{
  long ind = blockIdx.x * blockDim.x + threadIdx.x;
  if(ind < count){
    output_array1[ind] = (T)input1;
    output_array2[ind] = (T)input2;
    output_array3[ind] = (T)input3;
    output_array4[ind] = (T)input4;
    output_array5[ind] = (T)input5;
  }
}
template<typename T>
inline
int Xset_value_5_core(T *output_array1, T input1,
                      T *output_array2, T input2,
                      T *output_array3, T input3,
                      T *output_array4, T input4,
                      T *output_array5, T input5,
                      long batchCount, hipStream_t hip_stream)
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_set_value_5<T>), dim3(batchCount / 256 + (batchCount % 256 > 0) ), dim3(256), 0, hip_stream, 
                        output_array1, input1,
                        output_array2, input2,
                        output_array3, input3,
                        output_array4, input4,
                        output_array5, input5,
                        batchCount);
  check_error_ret( hipGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}
//==============================================================================================
template<typename T>
__global__
void kernel_set_pointer_1( T **output_array, T **input, int offset_r, int offset_c, int* lda,
                          long count)
{
  int ind = blockIdx.x;// * blockDim.x + threadIdx.x;
  //if(ind < count){
    T* ptr = input[ind];
    output_array[ind] = &ptr[offset_r + offset_c * lda[ind]];
  //}
}
// input, output_array: host pointer to array of device pointers to device buffers
template<typename T>
inline
int Xset_pointer_1_core(T **output_array, T **input, int offset_r, int offset_c, int* lda,
                        long batchCount, hipStream_t hip_stream)
{
  dim3 block(1,1);
  dim3 grid(batchCount);/* / block.x + ((batchCount % block.x) > 0),1);*/
  hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_set_pointer_1<T>), dim3(grid), dim3(block), 0, hip_stream, 
                          output_array, input, offset_r, offset_c, lda,
                          batchCount);
  check_error_ret( hipGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

//==============================================================================================
template<typename T>
__global__
void kernel_set_pointer_2(T **output_array1, const T **input1, int offset_r1, int offset_c1, int lda1,
                          T **output_array2, const T **input2, int offset_r2, int offset_c2, int lda2,
                          long count)
{
  int ind = blockIdx.x;// * blockDim.x + threadIdx.x;
  //if(ind < count){
    const T* ptr1 = input1[ind];
    output_array1[ind] = (T*)&ptr1[offset_r1 + offset_c1 * lda1];
    const T* ptr2 = input2[ind];
    output_array2[ind] = (T*)&ptr2[offset_r2 + offset_c2 * lda2];
  //}
}
// inputX, output_arrayX: host pointer to array of device pointers to device buffers
template<typename T>
inline
int Xset_pointer_2_core(T **output_array1, const T **input1, int offset_r1, int offset_c1, int lda1,
                        T **output_array2, const T **input2, int offset_r2, int offset_c2, int lda2,
                        long batchCount, hipStream_t hip_stream)
{
  dim3 block(1,1);
  dim3 grid(batchCount);/* / block.x + ((batchCount % block.x) > 0),1);*/
  hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_set_pointer_2<T>), dim3(grid), dim3(block), 0, hip_stream, 
                          output_array1, input1, offset_r1, offset_c1, lda1,
                          output_array2, input2, offset_r2, offset_c2, lda2,
                          batchCount);
  check_error_ret( hipGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}
//==============================================================================================
template<typename T>
__global__
void kernel_set_pointer_3(T **output_array1, const T **input1, int offset_r1, int offset_c1, int lda1,
                          T **output_array2, const T **input2, int offset_r2, int offset_c2, int lda2,
                          T **output_array3, const T **input3, int offset_r3, int offset_c3, int lda3,
                          long count)
{
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
template<typename T>
inline
int Xset_pointer_3_core(T **output_array1, const T **input1, int offset_r1, int offset_c1, int lda1,
                        T **output_array2, const T **input2, int offset_r2, int offset_c2, int lda2,
                        T **output_array3, const T **input3, int offset_r3, int offset_c3, int lda3,
                        long batchCount, hipStream_t hip_stream)
{
  dim3 block(1,1);
  dim3 grid(batchCount);/* / block.x + ((batchCount % block.x) > 0),1);*/
  hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_set_pointer_3<T>), dim3(grid), dim3(block), 0, hip_stream, 
                          output_array1, input1, offset_r1, offset_c1, lda1,
                          output_array2, input2, offset_r2, offset_c2, lda2,
                          output_array3, input3, offset_r3, offset_c3, lda3,
                          batchCount);
  check_error_ret( hipGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

//==============================================================================================
template<typename T>
__global__
void kernel_set_pointer_1(T **output_array, const T *input, int lda, long batch_offset)
{
  output_array[blockIdx.x] = (T*)&input[blockIdx.x * batch_offset];
}
// input: host pointer to device buffers
// output_array: host pointer to array of device pointers to device buffers
template<typename T>
inline
int Xset_pointer_1_core(T **output_array, const T *input, int lda, long batch_offset,
                        long batchCount, hipStream_t hip_stream)
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_set_pointer_1<T>), dim3(batchCount), dim3(1), 0, hip_stream, 
                          output_array, input, lda, batch_offset);
  check_error_ret( hipGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

//==============================================================================================
template<typename T>
__global__
void kernel_set_pointer_2(T **output_array1, const T *input1, int ld1, long batch_offset1,
                          T **output_array2, const T *input2, int ld2, long batch_offset2)
{
  output_array1[blockIdx.x] = (T*)input1 + blockIdx.x * batch_offset1;
  output_array2[blockIdx.x] = (T*)input2 + blockIdx.x * batch_offset2;
}
// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
template<typename T>
inline
int Xset_pointer_2_core(T **output_array1, const T *input1, int ld1, long batch_offset1,
                        T **output_array2, const T *input2, int ld2, long batch_offset2,
                        long batchCount, hipStream_t hip_stream)
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_set_pointer_2<T>), dim3(batchCount), dim3(1), 0, hip_stream, 
                          output_array1, input1, ld1, batch_offset1,
                          output_array2, input2, ld2, batch_offset2);
  check_error_ret( hipGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

//==============================================================================================
template<typename T>
__global__
void kernel_set_pointer_3(T **output_array1, const T *input1, int ld1, long batch_offset1,
                          T **output_array2, const T *input2, int ld2, long batch_offset2,
                          T **output_array3, const T *input3, int ld3, long batch_offset3)
{
  output_array1[blockIdx.x] = (T*)input1 + blockIdx.x * batch_offset1;
  output_array2[blockIdx.x] = (T*)input2 + blockIdx.x * batch_offset2;
  output_array3[blockIdx.x] = (T*)input3 + blockIdx.x * batch_offset3;
}
// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
template<typename T>
inline
int Xset_pointer_3_core(T **output_array1, const T *input1, int ld1, long batch_offset1,
                        T **output_array2, const T *input2, int ld2, long batch_offset2,
                        T **output_array3, const T *input3, int ld3, long batch_offset3,
                        long batchCount, hipStream_t hip_stream)
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_set_pointer_3<T>), dim3(batchCount), dim3(1), 0, hip_stream, 
                          output_array1, input1, ld1, batch_offset1,
                          output_array2, input2, ld2, batch_offset2,
                          output_array3, input3, ld3, batch_offset3);
  check_error_ret( hipGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

//==============================================================================================
template<typename T>
__global__
void kernel_set_pointer_4(T **output_array1, const T *input1, int ld1, long batch_offset1,
                          T **output_array2, const T *input2, int ld2, long batch_offset2,
                          T **output_array3, const T *input3, int ld3, long batch_offset3,
                          T **output_array4, const T *input4, int ld4, long batch_offset4)
{
  output_array1[blockIdx.x] = (T*)input1 + blockIdx.x * batch_offset1;
  output_array2[blockIdx.x] = (T*)input2 + blockIdx.x * batch_offset2;
  output_array3[blockIdx.x] = (T*)input3 + blockIdx.x * batch_offset3;
  output_array4[blockIdx.x] = (T*)input4 + blockIdx.x * batch_offset4;
}
// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
template<typename T>
inline
int Xset_pointer_4_core(T **output_array1, const T *input1, int ld1, long batch_offset1,
                        T **output_array2, const T *input2, int ld2, long batch_offset2,
                        T **output_array3, const T *input3, int ld3, long batch_offset3,
                        T **output_array4, const T *input4, int ld4, long batch_offset4,
                        long batchCount, hipStream_t hip_stream)
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_set_pointer_4<T>), dim3(batchCount), dim3(1), 0, hip_stream, 
                          output_array1, input1, ld1, batch_offset1,
                          output_array2, input2, ld2, batch_offset2,
                          output_array3, input3, ld3, batch_offset3,
                          output_array4, input4, ld4, batch_offset4);
  check_error_ret( hipGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

//==============================================================================================
template<typename T>
__global__
void kernel_set_pointer_5(T **output_array1, const T *input1, int ld1, long batch_offset1,
                          T **output_array2, const T *input2, int ld2, long batch_offset2,
                          T **output_array3, const T *input3, int ld3, long batch_offset3,
                          T **output_array4, const T *input4, int ld4, long batch_offset4,
                          T **output_array5, const T *input5, int ld5, long batch_offset5)
{
  output_array1[blockIdx.x] = (T*)input1 + blockIdx.x * batch_offset1;
  output_array2[blockIdx.x] = (T*)input2 + blockIdx.x * batch_offset2;
  output_array3[blockIdx.x] = (T*)input3 + blockIdx.x * batch_offset3;
  output_array4[blockIdx.x] = (T*)input4 + blockIdx.x * batch_offset4;
  output_array5[blockIdx.x] = (T*)input5 + blockIdx.x * batch_offset5;
}
// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
template<typename T>
inline
int Xset_pointer_5_core(T **output_array1, const T *input1, int ld1, long batch_offset1,
                        T **output_array2, const T *input2, int ld2, long batch_offset2,
                        T **output_array3, const T *input3, int ld3, long batch_offset3,
                        T **output_array4, const T *input4, int ld4, long batch_offset4,
                        T **output_array5, const T *input5, int ld5, long batch_offset5,
                        long batchCount, hipStream_t hip_stream)
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_set_pointer_5<T>), dim3(batchCount), dim3(1), 0, hip_stream, 
                          output_array1, input1, ld1, batch_offset1,
                          output_array2, input2, ld2, batch_offset2,
                          output_array3, input3, ld3, batch_offset3,
                          output_array4, input4, ld4, batch_offset4,
                          output_array5, input5, ld5, batch_offset5);
  check_error_ret( hipGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

//==============================================================================================
template<typename T>
__global__
void kernel_set_pointer_7(T **output_array1, const T *input1, int ld1, long batch_offset1,
                          T **output_array2, const T *input2, int ld2, long batch_offset2,
                          T **output_array3, const T *input3, int ld3, long batch_offset3,
                          T **output_array4, const T *input4, int ld4, long batch_offset4,
                          T **output_array5, const T *input5, int ld5, long batch_offset5,
                          T **output_array6, const T *input6, int ld6, long batch_offset6,
                          T **output_array7, const T *input7, int ld7, long batch_offset7)
{
  output_array1[blockIdx.x] = (T*)input1 + blockIdx.x * batch_offset1;
  output_array2[blockIdx.x] = (T*)input2 + blockIdx.x * batch_offset2;
  output_array3[blockIdx.x] = (T*)input3 + blockIdx.x * batch_offset3;
  output_array4[blockIdx.x] = (T*)input4 + blockIdx.x * batch_offset4;
  output_array5[blockIdx.x] = (T*)input5 + blockIdx.x * batch_offset5;
  output_array6[blockIdx.x] = (T*)input6 + blockIdx.x * batch_offset6;
  output_array7[blockIdx.x] = (T*)input7 + blockIdx.x * batch_offset7;
}
// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
template<typename T>
inline
int Xset_pointer_7_core(T **output_array1, const T *input1, int ld1, long batch_offset1,
                        T **output_array2, const T *input2, int ld2, long batch_offset2,
                        T **output_array3, const T *input3, int ld3, long batch_offset3,
                        T **output_array4, const T *input4, int ld4, long batch_offset4,
                        T **output_array5, const T *input5, int ld5, long batch_offset5,
                        T **output_array6, const T *input6, int ld6, long batch_offset6,
                        T **output_array7, const T *input7, int ld7, long batch_offset7,
                        long batchCount, hipStream_t hip_stream)
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_set_pointer_7<T>), dim3(batchCount), dim3(1), 0, hip_stream, 
                          output_array1, input1, ld1, batch_offset1,
                          output_array2, input2, ld2, batch_offset2,
                          output_array3, input3, ld3, batch_offset3,
                          output_array4, input4, ld4, batch_offset4,
                          output_array5, input5, ld5, batch_offset5,
                          output_array6, input6, ld6, batch_offset6,
                          output_array7, input7, ld7, batch_offset7);
  check_error_ret( hipGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

//==============================================================================================
template<typename T>
__global__
void kernel_set_pointer_9(T **output_array1, const T *input1, int ld1, long batch_offset1,
                          T **output_array2, const T *input2, int ld2, long batch_offset2,
                          T **output_array3, const T *input3, int ld3, long batch_offset3,
                          T **output_array4, const T *input4, int ld4, long batch_offset4,
                          T **output_array5, const T *input5, int ld5, long batch_offset5,
                          T **output_array6, const T *input6, int ld6, long batch_offset6,
                          T **output_array7, const T *input7, int ld7, long batch_offset7,
                          T **output_array8, const T *input8, int ld8, long batch_offset8,
                          T **output_array9, const T *input9, int ld9, long batch_offset9)
{
  output_array1[blockIdx.x] = (T*)input1 + blockIdx.x * batch_offset1;
  output_array2[blockIdx.x] = (T*)input2 + blockIdx.x * batch_offset2;
  output_array3[blockIdx.x] = (T*)input3 + blockIdx.x * batch_offset3;
  output_array4[blockIdx.x] = (T*)input4 + blockIdx.x * batch_offset4;
  output_array5[blockIdx.x] = (T*)input5 + blockIdx.x * batch_offset5;
  output_array6[blockIdx.x] = (T*)input6 + blockIdx.x * batch_offset6;
  output_array7[blockIdx.x] = (T*)input7 + blockIdx.x * batch_offset7;
  output_array8[blockIdx.x] = (T*)input8 + blockIdx.x * batch_offset8;
  output_array9[blockIdx.x] = (T*)input9 + blockIdx.x * batch_offset9;
}
// inputX: host pointer to device buffers
// output_arrayX: host pointer to array of device pointers to device buffers
template<typename T>
inline
int Xset_pointer_9_core(T **output_array1, const T *input1, int ld1, long batch_offset1,
                        T **output_array2, const T *input2, int ld2, long batch_offset2,
                        T **output_array3, const T *input3, int ld3, long batch_offset3,
                        T **output_array4, const T *input4, int ld4, long batch_offset4,
                        T **output_array5, const T *input5, int ld5, long batch_offset5,
                        T **output_array6, const T *input6, int ld6, long batch_offset6,
                        T **output_array7, const T *input7, int ld7, long batch_offset7,
                        T **output_array8, const T *input8, int ld8, long batch_offset8,
                        T **output_array9, const T *input9, int ld9, long batch_offset9,
                        long batchCount, hipStream_t hip_stream)
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_set_pointer_9<T>), dim3(batchCount), dim3(1), 0, hip_stream, 
                          output_array1, input1, ld1, batch_offset1,
                          output_array2, input2, ld2, batch_offset2,
                          output_array3, input3, ld3, batch_offset3,
                          output_array4, input4, ld4, batch_offset4,
                          output_array5, input5, ld5, batch_offset5,
                          output_array6, input6, ld6, batch_offset6,
                          output_array7, input7, ld7, batch_offset7,
                          output_array8, input8, ld8, batch_offset8,
                          output_array9, input9, ld9, batch_offset9);
  check_error_ret( hipGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

//==============================================================================================
template<typename T>
__global__
void kernel_set_pointer_2d_1(T **output_array, const T *input, int lda, long mb, long nb)
{
  output_array[blockIdx.x + blockIdx.y * gridDim.x] = (T*)&input[blockIdx.x * mb + blockIdx.y * nb * lda];
}
// input: host pointer to device buffers
// output_array: host pointer to array of device pointers to device buffers
template<typename T>
inline
int Xset_pointer_2d_1_core( T **output_array, const T *input, int lda, long mt, long nt, long mb, long nb,
                            hipStream_t hip_stream)
{
  dim3 block(1,1);
  dim3 grid(mt,nt);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_set_pointer_2d_1<T>), dim3(grid), dim3(block), 0, hip_stream, 
                          output_array, input, lda, mb, nb);
  check_error_ret( hipGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}
#endif //__XHELPER_FUNCS_CORE_H__
