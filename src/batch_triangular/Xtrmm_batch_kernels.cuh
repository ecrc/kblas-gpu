#include "hip/hip_runtime.h"
/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xtrmm_batch_kernels.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef __XTRMM_BATCH_KERNELS_H__
#define __XTRMM_BATCH_KERNELS_H__


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
template<typename T, bool TRANS, int TX, int TY>
__device__ inline void
dev_trmm_U_LLXN_reg_shared_Mfix_Nvar(const int m, const int n,
                                     const T alpha,
                                     const T* __restrict__ A, int lda,
                                                        T* B, int ldb)
{
  if( TX != m ) return;//necessary condition
  const int TX1 = TX + 2;
  //setup shared memory
  extern __shared__ __align__(sizeof(T)) unsigned char sh_data[];
  T* sdata = reinterpret_cast<T *>(sh_data) + ty * TX * TX1;

  //TODO better grid layout can be devised here
  int Bn_start = TY * blockIdx.y;
  int Bn_end = (n > (TY * (blockIdx.y+1))) ? TY * (blockIdx.y+1) : n;

  T rA[TX], rB[TX], s, a, zero = make_zero<T>();
  int ind0, bl = 0;
  int nb = (Bn_end - Bn_start) / TX;

  //copy needed data from global to registers
  #pragma unroll
  for(int i = 0; i < TX; i++){
    //if(tx >= i)
    rA[ i ] = __ldg(&(A[ tx + i * lda ]));
  }
  for(bl = 0; bl < nb; bl++)
  {
    ind0 = tx + TX * bl * ldb;
    //copy needed data from global to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      sdata[ tx + i*TX1 ] = __ldg(&(B[ ind0 + i * ldb ]));
    }
    //transpose data from shared to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      rB[ i ] = alpha * sdata[ i + tx * TX1 ];//TODO handle bank conflicts
    }
    if(TRANS){
      #pragma unroll
      for(int j = 0; j < TX; j++){
        s = zero;
        #pragma unroll
        for(int i = 0; i < TX; i++){
          if(i >= j){
            a = shfl(rA[j], i, TX);
            s = FMA( rB[i], a, s);
          }
        }
        rB[j] = s;
      }
    }else{
      #pragma unroll
      for(int j = TX-1; j >= 0; j--){
        s = zero;
        #pragma unroll
        for(int i = TX-1; i >=0; i--){
          if(i <= j){
            a = shfl(rA[i], j, TX);
            s = FMA( rB[i], a, s);
          }
        }
        rB[j] = s;
      }
    }

    //transpose data from registers to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      sdata[ i + tx * TX1 ] = rB[ i ];
    }
    //copy data back to global mem
    #pragma unroll
    for(int i = 0; i < TX; i++)
    {
      B[ ind0 + i * ldb ] = sdata[ tx + i*TX1 ];
      //B[ ind0 + i ] = alpha * rB[ i ];
    }
  }

  if((Bn_end - Bn_start) % TX != 0){

    ind0 = tx + TX * bl * ldb;
    //copy needed data from global to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      sdata[ tx + i*TX1 ] = ( (Bn_start + i + TX * bl) < Bn_end ) ? __ldg(&(B[ ind0 + i * ldb ])) : zero;
    }
    //transpose data from shared to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      rB[ i ] = alpha * sdata[ i + tx * TX1 ];
    }
    if(TRANS){
      #pragma unroll
      for(int j = 0; j < TX; j++){
        s = zero;
        #pragma unroll
        for(int i = 0; i < TX; i++){
          if(i >= j){
            a = shfl(rA[j], i, TX);
            s = FMA( rB[i], a, s);
          }
        }
        rB[j] = s;
      }
    }else{
      #pragma unroll
      for(int j = TX-1; j >= 0; j--){
        s = zero;
        #pragma unroll
        for(int i = TX-1; i >=0; i--){
          if(i <= j){
            a = shfl(rA[i], j, TX);
            s = FMA( rB[i], a, s);
          }
        }
        rB[j] = s;
      }
    }

    //transpose data from registers to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      sdata[ i + tx * TX1 ] = rB[ i ];
    }
    //copy data back to global mem
    #pragma unroll
    for(int i = 0; i < TX; i++)
    {
      if((Bn_start + i + TX * bl) < Bn_end)
        B[ ind0 + i * ldb ] = sdata[ tx + i*TX1 ];
    }

  }
}
//--------------------------------------------------------------------------------------------
template<typename T, typename T_PTR, bool TRANS, int TX, int TY>
__global__ void  //__launch_bounds__(256)
kernel_trmm_U_LLXN_reg_shared_Mfix_Nvar(const int m, const int n, int batchCount,
                                        const T alpha, T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                                                    T_PTR B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( TX != m ) return;//necessary condition

  //are we within bounds
  if(blockIdx.x * blockDim.y + ty >= batchCount) return;

  //TODO better grid layout can be devised here
  int Bn_start = TY * blockIdx.y;
  const T *A = getOperationPtr(A_array, blockIdx.x * blockDim.y + ty, strideA);
        T *B = getOperationPtr(B_array, blockIdx.x * blockDim.y + ty, strideB);
  // if(STRIDED == true){
  //   A = (const T*)A_array + (blockIdx.x * blockDim.y + ty) * strideA;
  //   B =       (T*)B_array + (blockIdx.x * blockDim.y + ty) * strideB;
  // }else{
  //   A = ((const T**)A_array)[blockIdx.x * blockDim.y + ty];
  //   B =       ((T**)B_array)[blockIdx.x * blockDim.y + ty];
  // }
  A += A_row_off + A_col_off * lda;
  B += B_row_off + B_col_off * ldb + Bn_start * ldb;

  dev_trmm_U_LLXN_reg_shared_Mfix_Nvar<T, TRANS, TX, TY>(m, n,
                                                         alpha, A, lda,
                                                                B, ldb);
}
//==============================================================================================
template<typename T, bool TRANS, int TX, int TY>
__device__ inline void
dev_trmm_U_LLXN_reg_shared_MNvar(const int m, const int n,
                                 const T alpha,
                                 const T* __restrict__ A, int lda,
                                                    T* B, int ldb)
{
  const int TX1 = TX + 2;
  //setup shared memory
  extern __shared__ __align__(sizeof(T)) unsigned char sh_data[];
  T* sdata = reinterpret_cast<T *>(sh_data) + ty * TX * TX1;

  //TODO better grid layout can be devised here
  int Bn_start = TY * blockIdx.y;
  int Bn_end = (n > (TY * (blockIdx.y+1))) ? TY * (blockIdx.y+1) : n;

  T rA[TX], rB[TX], s, a, zero = make_zero<T>();
  int ind0, bl = 0;
  int nb = (Bn_end - Bn_start) / TX;

  //copy needed data from global to registers

  if(tx < m){
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(i < m)
        rA[ i ] = __ldg(&(A[ tx + i * lda ]));
    }
  }
  for(bl = 0; bl < nb; bl++)
  {
    ind0 = tx + TX * bl * ldb;
    //copy needed data from global to shared
    if(tx < m){
      #pragma unroll
      for(int i = 0; i < TX; i++){
        sdata[ tx + i*TX1 ] = __ldg(&(B[ ind0 + i * ldb ]));
      }
    }
    //transpose data from shared to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(i < m)
        rB[ i ] = alpha * sdata[ i + tx * TX1 ];//TODO handle bank conflicts
    }
    if(TRANS){
      #pragma unroll
      for(int j = 0; j < TX; j++){
        if(j < m){
          s = zero;
          #pragma unroll
          for(int i = 0; i < TX; i++){
            if(i >= j && i < m){
              a = shfl(rA[j], i, TX);
              s = FMA( rB[i], a, s);
            }
          }
          rB[j] = s;
        }
      }
    }else{
      #pragma unroll
      for(int j = TX-1; j >= 0; j--){
        if(j < m){
          s = zero;
          #pragma unroll
          for(int i = TX-1; i >=0; i--){
            if(i <= j){
              a = shfl(rA[i], j, TX);
              s = FMA( rB[i], a, s);
            }
          }
          rB[j] = s;
        }
      }
    }

    //transpose data from registers to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(i < m)
        sdata[ i + tx * TX1 ] = rB[ i ];
    }
    //copy data back to global mem
    if(tx < m){
      #pragma unroll
      for(int i = 0; i < TX; i++){
        B[ ind0 + i * ldb ] = sdata[ tx + i*TX1 ];
      }
    }
  }

  if((Bn_end - Bn_start) % TX != 0){

    ind0 = tx + TX * bl * ldb;
    //copy needed data from global to shared
    if(tx < m){
      #pragma unroll
      for(int i = 0; i < TX; i++){
        sdata[ tx + i*TX1 ] = ( (Bn_start + i + TX * bl) < Bn_end ) ? __ldg(&(B[ ind0 + i * ldb ])) : zero;
      }
    }
    //transpose data from shared to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(i < m)
        rB[ i ] = alpha * sdata[ i + tx * TX1 ];
    }

    if(TRANS){
      #pragma unroll
      for(int j = 0; j < TX; j++){
        if(j < m){
          s = zero;
          #pragma unroll
          for(int i = 0; i < TX; i++){
            if(i >= j && i < m){
              a = shfl(rA[j], i, TX);
              s = FMA( rB[i], a, s);
            }
          }
          rB[j] = s;
        }
      }
    }else{
      #pragma unroll
      for(int j = TX-1; j >= 0; j--){
        if(j < m){
          s = zero;
          #pragma unroll
          for(int i = TX-1; i >=0; i--){
            if(i <= j){
              a = shfl(rA[i], j, TX);
              s = FMA( rB[i], a, s);
            }
          }
          rB[j] = s;
        }
      }
    }

    //transpose data from registers to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(i < m)
        sdata[ i + tx * TX1 ] = rB[ i ];
    }
    //copy data back to global mem
    if(tx < m){
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if((Bn_start + i + TX * bl) < Bn_end)
          B[ ind0 + i * ldb ] = sdata[ tx + i*TX1 ];
      }
    }
  }
}
//--------------------------------------------------------------------------------------------
template<typename T, typename T_PTR, bool TRANS, int TX, int TY>
__global__ void  //__launch_bounds__(256)
kernel_trmm_U_LLXN_reg_shared_MNvar(const int m, const int n, int batchCount,
                                    const T alpha, T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                                                T_PTR B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( TX < m ) return;//necessary condition

  //are we within bounds
  if(blockIdx.x * blockDim.y + ty >= batchCount) return;

  //TODO better grid layout can be devised here
  int Bn_start = TY * blockIdx.y;
  const T *A = getOperationPtr(A_array, blockIdx.x * blockDim.y + ty, strideA);
        T *B = getOperationPtr(B_array, blockIdx.x * blockDim.y + ty, strideB);
  // if(STRIDED == true){
  //   A = (const T*)A_array + (blockIdx.x * blockDim.y + ty) * strideA;
  //   B =       (T*)B_array + (blockIdx.x * blockDim.y + ty) * strideB;
  // }else{
  //   A = ((const T**)A_array)[blockIdx.x * blockDim.y + ty];
  //   B =       ((T**)B_array)[blockIdx.x * blockDim.y + ty];
  // }
  A += A_row_off + A_col_off * lda;
  B += B_row_off + B_col_off * ldb + Bn_start * ldb;

  dev_trmm_U_LLXN_reg_shared_MNvar<T, TRANS, TX, TY>(m, n,
                                                     alpha, A, lda,
                                                            B, ldb);
}

//==============================================================================================
#else
  #error "Pre-Kepler architechture is not supported in KBLAS batch SYRK"
#endif

#endif //__XTRSM_BATCH_KERNELS_H__
