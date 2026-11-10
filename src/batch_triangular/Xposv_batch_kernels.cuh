/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xposv_batch_kernels.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ali Charara
 * @date 2020-12-10
 **/

#ifndef __XPOSV_BATCH_KERNELS_H__
#define __XPOSV_BATCH_KERNELS_H__


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
dev_posv_U_RL_registers_Nfix_Mvar(const int m, const int n,
                                  T* A, int lda,
                                  T* B, int ldb)
{
  T rA[TX], rB[TX], s;
  int ind0, b;
  int mb = m / TX;

  //copy needed data from global to registers
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    //if(tx >= i)
    rA[ i ] = __ldg(&(A[ tx + i * lda ]));
  }

  //perform factorization on registers
  #pragma unroll
  for(int j = 0; j < TX; j++)
  {
    s = sqrt( shfl(rA[j], j, TX) );
    if(tx == j)
      rA[j] = s;
    if(tx > j)
      rA[j] /= s;

    #pragma unroll
    for(int i = 0; i < TX; i++){
      s = -shfl(rA[j], i, TX);
      if(j < i && i <= tx)
        rA[i] = FMA( rA[j], s, rA[i]);
    }
  }

  //copy data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx >= i)
      A[ tx + i * lda ] = rA[ i ];
  }

  for(b = 0; b < mb; b++){
    ind0 = tx + TX * b;
    //copy needed data from global to registers
    #pragma unroll
    for(int i = 0; i < TX; i++)
    {
      rB[ i ] = __ldg(&(B[ ind0 + i * ldb ]));
    }

    #pragma unroll
    for(int j = 0; j < TX; j++)
    {
      s = shfl(rA[j], j, TX);
      rB[j] /= s;

      #pragma unroll
      for(int i = 0; i < TX; i++){
        s = -shfl(rA[j], i, TX);
        if(j < i)
          rB[i] = FMA( rB[j], s, rB[i]);
      }

    }

    #pragma unroll
    for(int j = TX-1; j >= 0; j--)
    {
      #pragma unroll
      for(int i = 0; i < TX; i++){
        s = -shfl(rA[j], i, TX);
        if(j < i)
          rB[j] = FMA( rB[i], s, rB[j]);
      }

      s = shfl(rA[j], j, TX);
      rB[j] /= s;
    }

    //copy data back to global mem
    #pragma unroll
    for(int i = 0; i < TX; i++)
    {
      B[ ind0 + i * ldb ] = rB[ i ];
    }
  }
  if(m % TX != 0){
    ind0 = tx + TX * b;
    //copy needed data from global to registers
    if(ind0 < m){
      #pragma unroll
      for(int i = 0; i < TX; i++)
        rB[ i ] = __ldg(&(B[ ind0 + i * ldb ]));
    }

    #pragma unroll
    for(int j = 0; j < TX; j++)
    {
      s = shfl(rA[j], j, TX);
      rB[j] /= s;

      #pragma unroll
      for(int i = 0; i < TX; i++){
        s = -shfl(rA[j], i, TX);
        if(j < i)
          rB[i] = FMA( rB[j], s, rB[i]);
      }
    }

    #pragma unroll
    for(int j = TX-1; j >= 0; j--)
    {
      #pragma unroll
      for(int i = 0; i < TX; i++){
        s = -shfl(rA[j], i, TX);
        if(j < i)
          rB[j] = FMA( rB[i], s, rB[j]);
      }

      s = shfl(rA[j], j, TX);
      rB[j] /= s;
    }

    //copy data back to global mem
    if(ind0 < m){
      #pragma unroll
      for(int i = 0; i < TX; i++)
        B[ ind0 + i * ldb ] = rB[ i ];
    }
  }
}

//--------------------------------------------------------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int TX>
__global__ void  //__launch_bounds__(256)
kernel_posv_U_RL_registers_Nfix_Mvar(const int m, const int n, int batchCount,
                                     T_PTR A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                     T_PTR B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( TX != n ) return;//necessary condition

  //are we within bounds
  if(blockIdx.x * blockDim.y + ty >= batchCount) return;

  T *A;
  T *B;
  if(STRIDED == true){
    A = (T*)A_array + (blockIdx.x * blockDim.y + ty) * strideA;
    B = (T*)B_array + (blockIdx.x * blockDim.y + ty) * strideB;
  }else{
    A = ((T**)A_array)[blockIdx.x * blockDim.y + ty];
    B = ((T**)B_array)[blockIdx.x * blockDim.y + ty];
  }
  A += A_row_off + A_col_off * lda;
  B += B_row_off + B_col_off * ldb;

  dev_posv_U_RL_registers_Nfix_Mvar<T, TX>(m, n,
                                            A, lda,
                                            B, ldb);
}

//==============================================================================================
template<typename T, int TX>
__device__ inline void
dev_posv_U_RL_registers_NMvar(const int m, const int n,
                              T* A, int lda,
                              T* B, int ldb)
{
  T rA[TX], rB[TX], s, zero = make_zero<T>();
  int ind0, b;
  int mb = m / TX;

  //copy needed data from global to registers
  if(tx < n){
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(i < n)
        rA[ i ] = __ldg(&(A[ tx + i * lda ]));
      //else
      //  rA[ i ] = zero;
    }
  }
  //perform factorization on registers
  #pragma unroll
  for(int j = 0; j < TX; j++)
  {
    s = sqrt( shfl(rA[j], j, TX) );
    if(j < n){
      // if(tx == j)
      //   rA[j] = s;
      // if(tx > j)
        rA[j] /= s;
    }
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(j < i && i < n){
        s = -shfl(rA[j], i, TX);
        if(i <= tx)
          rA[i] = FMA( rA[j], s, rA[i]);
      }
    }
  }

  //copy data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx >= i && i < n && tx < n)
      A[ tx + i * lda ] = rA[ i ];
  }

  for(b = 0; b < mb; b++){
    ind0 = tx + TX * b;
    //copy needed data from global to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(i < n)
        rB[ i ] = __ldg(&(B[ ind0 + i * ldb ]));
      //else
      //  rB[ i ] = zero;
    }

    #pragma unroll
    for(int j = 0; j < TX; j++)
    {
      s = shfl(rA[j], j, TX);
      if(s != zero)
        rB[j] /= s;

      #pragma unroll
      for(int i = 0; i < TX; i++){
        s = -shfl(rA[j], i, TX);
        if(j < i)
          rB[i] = FMA( rB[j], s, rB[i]);
      }
    }

    #pragma unroll
    for(int j = TX-1; j >= 0; j--)
    {
      #pragma unroll
      for(int i = 0; i < TX; i++){
        s = -shfl(rA[j], i, TX);
        if(j < i)
          rB[j] = FMA( rB[i], s, rB[j]);
      }

      s = shfl(rA[j], j, TX);
      if(s != zero)
        rB[j] /= s;
    }

    //copy data back to global mem
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(i < n)
        B[ ind0 + i * ldb ] = rB[ i ];
    }
  }
  if(m % TX != 0){
    ind0 = tx + TX * b;
    //copy needed data from global to registers
    if(ind0 < m){
      #pragma unroll
      for(int i = 0; i < TX; i++)
        if(i < n)
          rB[ i ] = __ldg(&(B[ ind0 + i * ldb ]));
    }

    #pragma unroll
    for(int j = 0; j < TX; j++)
    {
      s = shfl(rA[j], j, TX);
      if(s != zero)
        rB[j] /= s;

      #pragma unroll
      for(int i = 0; i < TX; i++){
        s = -shfl(rA[j], i, TX);
        if(j < i)
          rB[i] = FMA( rB[j], s, rB[i]);
      }
    }

    #pragma unroll
    for(int j = TX-1; j >= 0; j--)
    {
      #pragma unroll
      for(int i = 0; i < TX; i++){
        s = -shfl(rA[j], i, TX);
        if(j < i)
          rB[j] = FMA( rB[i], s, rB[j]);
      }

      s = shfl(rA[j], j, TX);
      if(s != zero)
        rB[j] /= s;
    }

    //copy data back to global mem
    if(ind0 < m){
      #pragma unroll
      for(int i = 0; i < TX; i++)
        if(i < n)
          B[ ind0 + i * ldb ] = rB[ i ];
    }
  }
}
//--------------------------------------------------------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int TX>
__global__ void  //__launch_bounds__(256)
kernel_posv_U_RL_registers_NMvar(const int m, const int n, int batchCount,
                                 T_PTR A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                 T_PTR B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( TX < n ) return;//necessary condition

  //are we within bounds
  if(blockIdx.x * blockDim.y + ty >= batchCount) return;

  T *A;
  T *B;
  if(STRIDED == true){
    A = (T*)A_array + (blockIdx.x * blockDim.y + ty) * strideA;
    B = (T*)B_array + (blockIdx.x * blockDim.y + ty) * strideB;
  }else{
    A = ((T**)A_array)[blockIdx.x * blockDim.y + ty];
    B = ((T**)B_array)[blockIdx.x * blockDim.y + ty];
  }
  A += A_row_off + A_col_off * lda;
  B += B_row_off + B_col_off * ldb;

  dev_posv_U_RL_registers_NMvar<T, TX>(m, n,
                                        A, lda,
                                        B, ldb);
}
//==============================================================================================
#else
  #error "Pre-Kepler architechture is not supported in KBLAS batch POSV"
#endif

#endif //__XPOSV_BATCH_KERNELS_H__
