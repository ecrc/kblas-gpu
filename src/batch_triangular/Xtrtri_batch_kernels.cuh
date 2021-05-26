/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xtrtri_batch_kernels.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ali Charara
 * @date 2020-12-10
 **/

#ifndef __XTRTRI_BATCH_KERNELS_H__
#define __XTRTRI_BATCH_KERNELS_H__


//==============================================================================================
#define WARP 32
#define WARP1 33
#define tx threadIdx.x
#define ty threadIdx.y
//==============================================================================================
//Naming convention <dev/kernel>_<KernelName>_<Non/Uniform>_<Right/Left><Lower/Upper><Non/Transpose><Non/Diag>_<variants>
//==============================================================================================
#ifndef TARGET_SM
  #error "TARGET_SM is not defined"
#elif (TARGET_SM >= 30)


//==============================================================================================
template<typename T, int TX>
__device__ inline void
dev_trtri_U_registers_Nfix(int n, T* A, int lda)
{
  T rA[TX], s, a, zero = make_zero<T>(), one = make_one<T>();
  //copy needed data from global to registers
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    //if(tx >= i)
      rA[ i ] = __ldg(&(A[ tx + i * lda ]));
  }

  //perform inverting on registers
  #pragma unroll
  for(int j = TX-1; j >= 0; j--)
  {
    s = zero;
    #pragma unroll
    for(int i = 0; i < TX; i++){
      a = shfl(rA[j], i, TX);
      if(j < i && i <= tx)
        s = FMA( rA[i], a, s);
    }
    a = shfl(rA[j], j, TX);
    if(tx == j)
      rA[j] = one / a;
    else
    if(tx > j)
      rA[j] = -s / a;
    //rA[j] = (tx == j ? one : -s ) / a;
  }

  //copy data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx >= i)
      A[ tx + i * lda ] = rA[ i ];
  }
}

//--------------------------------------------------------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int TX>
__global__ void  //__launch_bounds__(256)
kernel_trtri_U_registers_Nfix(const int n, int batchCount,
                              T_PTR A_array, int A_row_off, int A_col_off, int lda, long strideA,
                              int* info)
{
  //are we within bounds
  if(blockIdx.x * blockDim.y + ty >= batchCount) return;

  //n should be a multiple of TX, for processing many diagonal blocks in parallel
  if(n % TX != 0){ info[blockIdx.x * blockDim.y + ty] = -1; return; }

  T *A;
  if(STRIDED == true){
    A = (T*)A_array + (blockIdx.x * blockDim.y + ty) * strideA;
  }else{
    A = ((T**)A_array)[blockIdx.x * blockDim.y + ty];
  }
  A += A_row_off + A_col_off * lda + blockIdx.y * TX * (1 + lda);

  dev_trtri_U_registers_Nfix<T, TX>(n, A, lda);

}

//==============================================================================================
template<typename T, int TX>
__device__ inline void
dev_trtri_U_registers_Nvar(int n, T* A, int lda)
{
  T rA[TX], s, a, zero = make_zero<T>(), one = make_one<T>();
  bool lastBlock = (blockIdx.y == (gridDim.y - 1)) && (n % TX != 0);
  int N = lastBlock ? (n - blockIdx.y * TX) : TX;

  //copy needed data from global to registers
  if(!lastBlock){
    #pragma unroll
    for(int i = 0; i < TX; i++){
      rA[ i ] = __ldg(&(A[ tx + i * lda ]));
    }
  }else{
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(tx < N && i < N)
        rA[ i ] = __ldg(&(A[ tx + i * lda ]));
    }
  }

  //perform inverting on registers
  if(!lastBlock){
    #pragma unroll
    for(int j = TX-1; j >= 0; j--){
      s = zero;
      #pragma unroll
      for(int i = 0; i < TX; i++){
        a = shfl(rA[j], i, TX);
        if(j < i && i <= tx)
          s = FMA( rA[i], a, s);
      }
      a = shfl(rA[j], j, TX);
      if(tx == j)
        rA[j] = one / a;
      else
      if(tx > j)
        rA[j] = -s / a;
    }
  }else{
    #pragma unroll
    for(int j = TX-1; j >= 0; j--){
      if(j < N){
        s = zero;
        #pragma unroll
        for(int i = 0; i < TX; i++){
          a = shfl(rA[j], i, TX);
          if(i < N && j < i && i <= tx)
            s = FMA( rA[i], a, s);
        }
        a = shfl(rA[j], j, TX);
        if(tx == j)
          rA[j] = one / a;
        else
        if(tx > j)
          rA[j] = -s / a;
      }
    }
  }

  //copy data back to global mem
  if(!lastBlock){
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(tx >= i)
        A[ tx + i * lda ] = rA[ i ];
    }
  }else{
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(tx >= i && i < N && tx < N)
        A[ tx + i * lda ] = rA[ i ];
    }
  }
}

//--------------------------------------------------------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int TX>
__global__ void  //__launch_bounds__(256)
kernel_trtri_U_registers_Nvar(const int n, int batchCount,
                              T_PTR A_array, int A_row_off, int A_col_off, int lda, long strideA,
                              int* info)
{
  //are we within bounds
  if(blockIdx.x * blockDim.y + ty >= batchCount) return;

  T *A;
  if(STRIDED == true){
    A = (T*)A_array + (blockIdx.x * blockDim.y + ty) * strideA;
  }else{
    A = ((T**)A_array)[blockIdx.x * blockDim.y + ty];
  }
  A += A_row_off + A_col_off * lda + blockIdx.y * TX * (1 + lda);

  dev_trtri_U_registers_Nvar<T, TX>(n, A, lda);

}
//==============================================================================================
#else
  #error "Pre-Kepler architechture is not supported in KBLAS batch TRTRI"
#endif

#endif //__XTRTRI_BATCH_KERNELS_H__
