/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xpotri_batch_kernels.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 2.0.0
 * @author Ali Charara
 * @date 2017-11-13
 **/

#ifndef __XPOTRI_BATCH_KERNELS_H__
#define __XPOTRI_BATCH_KERNELS_H__


//==============================================================================================
#define WARP 32
#define WARP1 33
#define tx threadIdx.x
#define ty threadIdx.y
//==============================================================================================
//Naming convention <dev/kernel>_<KernelName>_<Non/Uniform>_<Right/Left><Lower/Upper><Non/Transpose><Non/Diag>_<variants>
//==============================================================================================
#ifndef SM
  #error "SM is not defined"
#elif (SM >= 30)

//==============================================================================================
template<typename T, int TX>
__device__ inline void
dev_potri_U_L_reg_shared_Nfix(int n, T* A, int lda)
{
  const int TX1 = TX + 1;
  //setup shared memory
  extern __shared__ __align__(sizeof(T)) unsigned char sh_data[];
  T* sdata = reinterpret_cast<T *>(sh_data) + ty * TX * TX1;

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

  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ tx + i * TX1 ] = rA[ i ];
  }
  #pragma unroll
  for(int i = 0; i < TX; i++){
    rA[ i ] = sdata[ i + tx * TX1 ];//TODO handle bank conflicts
  }
  #pragma unroll
  for(int j = 0; j < TX; j++)
  {
    s = zero;
    #pragma unroll
    for(int i = 0; i < TX; i++){
      a = shfl(rA[i], j, TX);
      if(j <= i && j >= tx)
        s = FMA( rA[i], a, s);
    }
    rA[j] = s;
  }
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ i + tx * TX1 ] = rA[ i ];
  }

  //copy data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx >= i)
      A[ tx + i * lda ] = sdata[ tx + i * TX1 ];
  }
}

//==============================================================================================
template<typename T, typename T_PTR, bool STRIDED, int TX>
__global__ void  //__launch_bounds__(256)
kernel_potri_U_L_reg_shared_Nfix( const int n, int batchCount,
                                  T_PTR A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                  int* info)
{
  //are we within bounds
  if(blockIdx.x * blockDim.y + ty >= batchCount) return;

  if(n != TX){ info[blockIdx.x * blockDim.y + ty] = -1; return; }

  T *A;
  if(STRIDED == true){
    A = (T*)A_array + (blockIdx.x * blockDim.y + ty) * strideA;
  }else{
    A = ((T**)A_array)[blockIdx.x * blockDim.y + ty];
  }
  A += A_row_off + A_col_off * lda;

  dev_potri_U_L_reg_shared_Nfix<T, TX>(n, A, lda);

}


//==============================================================================================
template<typename T, int TX>
__device__ inline void
dev_potri_U_L_reg_shared_Nvar(int n, T* A, int lda)
{
  const int TX1 = TX + 1;
  //setup shared memory
  extern __shared__ __align__(sizeof(T)) unsigned char sh_data[];
  T* sdata = reinterpret_cast<T *>(sh_data) + ty * TX * TX1;

  T rA[TX], s, a, zero = make_zero<T>(), one = make_one<T>();
  //copy needed data from global to registers
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx < n && i < n)
      rA[ i ] = __ldg(&(A[ tx + i * lda ]));
  }

  //perform inverting on registers
  #pragma unroll
  for(int j = TX-1; j >= 0; j--){
    if(j < n){
      s = zero;
      #pragma unroll
      for(int i = 0; i < TX; i++){
        a = shfl(rA[j], i, TX);
        if(i < n && j < i && i <= tx)
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

  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ tx + i * TX1 ] = rA[ i ];
  }
  #pragma unroll
  for(int i = 0; i < TX; i++){
    rA[ i ] = sdata[ i + tx * TX1 ];//TODO handle bank conflicts
  }
  #pragma unroll
  for(int j = 0; j < TX; j++)
  {
    s = zero;
    #pragma unroll
    for(int i = 0; i < TX; i++){
      a = shfl(rA[i], j, TX);
      if(j <= i && j >= tx)
        s = FMA( rA[i], a, s);
    }
    rA[j] = s;
  }
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ i + tx * TX1 ] = rA[ i ];
  }

  //copy data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx >= i && i < n && tx < n)
      A[ tx + i * lda ] = sdata[ tx + i * TX1 ];
  }
}

//==============================================================================================
template<typename T, typename T_PTR, bool STRIDED, int TX>
__global__ void  //__launch_bounds__(256)
kernel_potri_U_L_reg_shared_Nvar( const int n, int batchCount,
                                  T_PTR A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                  int* info)
{
  //are we within bounds
  if(blockIdx.x * blockDim.y + ty >= batchCount) return;

  if(n > TX){ info[blockIdx.x * blockDim.y + ty] = -1; return; }

  T *A;
  if(STRIDED == true){
    A = (T*)A_array + (blockIdx.x * blockDim.y + ty) * strideA;
  }else{
    A = ((T**)A_array)[blockIdx.x * blockDim.y + ty];
  }
  A += A_row_off + A_col_off * lda;

  dev_potri_U_L_reg_shared_Nvar<T, TX>(n, A, lda);

}
//==============================================================================================
#else
  #error "Pre-Kepler architechture is not supported in KBLAS batch POTRI"
#endif

#endif //__XPOTRI_BATCH_KERNELS_H__
