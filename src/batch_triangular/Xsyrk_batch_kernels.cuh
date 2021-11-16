#include "hip/hip_runtime.h"
/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xsyrk_batch_kernels.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef __XSYRK_BATCH_KERNELS_H__
#define __XSYRK_BATCH_KERNELS_H__


//==============================================================================================
#define WARP 32
#define WARP1 33
#define tx threadIdx.x
#define ty threadIdx.y
//==============================================================================================
//Naming convention <dev/kernel>_<KernelName>_<Non/Uniform><Non/Strided>_<Lower/Upper><Non/Transpose>_<variants>
//==============================================================================================
#ifndef TARGET_SM
#error "TARGET_SM is not defined"
#endif

//shuffle intrinsic is not supported before KEPLER
#if (TARGET_SM >= 30)

//==============================================================================================
template<typename T, int B_ROWS, int A_COLS_PTY>
__device__ inline void
dev_syrk_U_LN_registers_Mfix_Nmul( const int m, const int n,
                                   const T alpha, const T* __restrict__ A, int lda,
                                   const T beta, T* B, int ldb)
{
  T rA0[A_COLS_PTY], s, rB0[B_ROWS], zero = make_zero<T>();//, *B0;
  int blockCount = n / A_COLS_PTY, ind;

  //copy needed data from global to registers of block A[0,0]
  //A0 = A;
  #pragma unroll
  for(int i = 0; i < B_ROWS; i++){
    rB0[ i ] = __ldg(&(B[ tx + i * ldb ])) * beta ;
  }

  //#pragma unroll
  for(int b = 0; b < blockCount; b++){
    //int b = 0;
    //  A0 = A + BS * b * lda;
    ind = tx  + A_COLS_PTY * b * lda;
    #pragma unroll
    for(int i = 0; i < A_COLS_PTY; i++){
      rA0[ i ] = __ldg(&(A[ ind + i * lda ]));
    }

    //1. syrk on B[0] by A[b]
    {
      #pragma unroll
      for(int j = 0; j < B_ROWS; j++){
        s = zero;
        #pragma unroll
        for(int i = 0; i < A_COLS_PTY; i++){
          s = FMA(rA0[i], shfl(rA0[i], j, B_ROWS), s);
        }
        rB0[j] = FMA( alpha, s, rB0[j] );
      }
    }
  }
  //copy B[0] data back to global mem
  #pragma unroll
  for(int i = 0; i < B_ROWS; i++)
  {
    if(tx >= i)
      B[ tx + i * ldb ] = rB0[ i ];
  }
}
//----------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int B_ROWS, int A_COLS_PTY>
__global__ void  //__launch_bounds__(256)
kernel_syrk_U_LN_registers_Mfix_Nmul( const int m, const int n, int batchCount,
                                      const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                      const T beta,        T_PTR              B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( m != B_ROWS ) return;//necessary condition
  if( n % A_COLS_PTY ) return;//necessary condition
  unsigned int ind = blockIdx.x * blockDim.y + ty;
  //are we within bounds
  if(ind >= batchCount) return;
  const T *A;
        T *B;
  if(STRIDED == true){
    A = (const T*)A_array + ind * strideA;
    B =       (T*)B_array + ind * strideB;
  }else{
    A = ((const T**)A_array)[ind];
    B =       ((T**)B_array)[ind];
    A += A_row_off + A_col_off * lda;
    B += B_row_off + B_col_off * ldb;
  }
  dev_syrk_U_LN_registers_Mfix_Nmul<T, B_ROWS, A_COLS_PTY>(
                                    m, n,
                                    alpha, A, lda,
                                    beta, B, ldb);
}

//==============================================================================================;
template<typename T, int B_ROWS, int A_COLS_PTY>
__device__ inline void
dev_syrk_U_LN_registers_Mfix_Nvar(const int m, const int n,
                                  const T alpha, const T* __restrict__ A, int lda,
                                  const T beta, T* B, int ldb)
{
  T rA0[A_COLS_PTY], s, rB0[B_ROWS];
  int blockCount = n / A_COLS_PTY, ind, b;

  //copy needed data from global to registers
  #pragma unroll
  for(int i = 0; i < B_ROWS; i++){
    rB0[ i ] = __ldg(&(B[ tx + i * ldb ])) * beta ;
  }

  for(b = 0; b < blockCount; b++){

    ind = tx  + A_COLS_PTY * b * lda;
    #pragma unroll
    for(int i = 0; i < A_COLS_PTY; i++){
      rA0[ i ] = __ldg(&(A[ ind + i * lda ]));
    }

    //1. syrk on B[0] by A[b]
    {
      #pragma unroll
      for(int j = 0; j < B_ROWS; j++){
        s = make_zero<T>();
        #pragma unroll
        for(int i = 0; i < A_COLS_PTY; i++){
          s = FMA(rA0[i], shfl(rA0[i], j, B_ROWS), s);
        }
        rB0[j] = FMA( alpha, s, rB0[j] );
      }
    }
  }
  if(n % A_COLS_PTY){

    ind = tx  + A_COLS_PTY * b * lda;
    int acols = n - A_COLS_PTY * b;
    #pragma unroll
    for(int i = 0; i < A_COLS_PTY; i++){
      if( i < acols )
        rA0[ i ] = __ldg(&(A[ ind + i * lda ]));
    }

    //1. syrk on B[0] by A[b]
    {
      #pragma unroll
      for(int j = 0; j < B_ROWS; j++){
        s = make_zero<T>();
        #pragma unroll
        for(int i = 0; i < A_COLS_PTY; i++){
          if(i < acols)
            s = FMA(rA0[i], shfl(rA0[i], j, B_ROWS), s);
        }
        rB0[j] = FMA( alpha, s, rB0[j] );
      }
    }
  }

  //copy B[0] data back to global mem
  #pragma unroll
  for(int i = 0; i < B_ROWS; i++)
  {
    if(tx >= i)
      B[ tx + i * ldb ] = rB0[ i ];
  }
}
//----------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int B_ROWS, int A_COLS_PTY>
__global__ void  //__launch_bounds__(256)
kernel_syrk_U_LN_registers_Mfix_Nvar( const int m, const int n, int batchCount,
                                      const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                      const T beta,        T_PTR              B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( m != B_ROWS ) return;//necessary condition
  unsigned int ind = blockIdx.x * blockDim.y + ty;
  //are we within bounds
  if(ind >= batchCount) return;

  const T *A;
        T *B;
  if(STRIDED == true){
    A = (const T*)A_array + ind * strideA;
    B =       (T*)B_array + ind * strideB;
  }else{
    A = ((const T**)A_array)[ind];
    B =       ((T**)B_array)[ind];
    A += A_row_off + A_col_off * lda;
    B += B_row_off + B_col_off * ldb;
  }
  dev_syrk_U_LN_registers_Mfix_Nvar<T, B_ROWS, A_COLS_PTY>(
                                    m, n,
                                    alpha, A, lda,
                                    beta, B, ldb);
}

//==============================================================================================;
template<typename T, int B_ROWS, int A_COLS_PTY>
__device__ inline void
dev_syrk_U_LN_registers_MNvar(const int m, const int n,
                              const T alpha, const T* __restrict__ A, int lda,
                              const T beta, T* B, int ldb)
{
  T rA0[A_COLS_PTY], s, rB0[B_ROWS], zero = make_zero<T>();
  int blockCount = n / A_COLS_PTY, ind, b;

  //initialize our variables
  #pragma unroll
  for(int i = 0; i < A_COLS_PTY; i++)
    rA0[ i ] = zero;
  #pragma unroll
  for(int i = 0; i < B_ROWS; i++)
    rB0[ i ] = zero;

  //copy needed data from global to registers
  if(tx < m){
    #pragma unroll
    for(int i = 0; i < B_ROWS; i++){
      if(i < m)
        rB0[ i ] = __ldg(&(B[ tx + i * ldb ])) * beta ;
    }
  }
  for(b = 0; b < blockCount; b++){

    ind = tx + A_COLS_PTY * b * lda;
    if(tx < m){
      #pragma unroll
      for(int i = 0; i < A_COLS_PTY; i++){
        rA0[ i ] = __ldg(&(A[ ind + i * lda ]));
      }
    }
    //1. syrk on B[0] by A[b]
    {
      #pragma unroll
      for(int j = 0; j < B_ROWS; j++){
        s = make_zero<T>();
        #pragma unroll
        for(int i = 0; i < A_COLS_PTY; i++){
          s = FMA(rA0[i], shfl(rA0[i], j, B_ROWS), s);
        }
        rB0[j] = FMA( alpha, s, rB0[j] );
      }
    }
  }
  if(n % A_COLS_PTY){

    ind = tx + A_COLS_PTY * b * lda;
    int acols = n - A_COLS_PTY * b;
    if(tx < m){
      #pragma unroll
      for(int i = 0; i < A_COLS_PTY; i++){
        if( i < acols )
          rA0[ i ] = __ldg(&(A[ ind + i * lda ]));
      }
    }

    //1. syrk on B[0] by A[b]
    {
      #pragma unroll
      for(int j = 0; j < B_ROWS; j++){
        s = make_zero<T>();
        #pragma unroll
        for(int i = 0; i < A_COLS_PTY; i++){
          if(i < acols)
            s = FMA(rA0[i], shfl(rA0[i], j, B_ROWS), s);
        }
        rB0[j] = FMA( alpha, s, rB0[j] );
      }
    }
  }

  //copy B[0] data back to global mem
  if(tx < m){
    #pragma unroll
    for(int i = 0; i < B_ROWS; i++)
    {
      if(tx >= i && i < m)
        B[ tx + i * ldb ] = rB0[ i ];
    }
  }
}
//----------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int B_ROWS, int A_COLS_PTY>
__global__ void  //__launch_bounds__(256)
kernel_syrk_U_LN_registers_MNvar( const int m, const int n, int batchCount,
                                  const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                  const T beta,        T_PTR              B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if(B_ROWS < m) return;
  unsigned int ind = blockIdx.x * blockDim.y + ty;
  //are we within bounds
  if(ind >= batchCount) return;

  const T *A;
        T *B;
  if(STRIDED == true){
    A = (const T*)A_array + ind * strideA;
    B =       (T*)B_array + ind * strideB;
  }else{
    A = ((const T**)A_array)[ind];
    B =       ((T**)B_array)[ind];
    A += A_row_off + A_col_off * lda;
    B += B_row_off + B_col_off * ldb;
  }
  dev_syrk_U_LN_registers_MNvar<T, B_ROWS, A_COLS_PTY>(
                                m, n,
                                alpha, A, lda,
                                beta, B, ldb);
}

//==============================================================================================;
template<typename T, int TX, int A_COLS_PTY>
__device__ inline void
dev_syrk_U_LN_registers_Mblock2_Nmul( const int m, const int n,
                                      const T alpha, const T* __restrict__ A, int lda,
                                      const T beta, T* B, int ldb)
{
  T rA0[A_COLS_PTY], rA1[A_COLS_PTY], rB00[TX], rB10[TX], rB11[TX];
  T s0, s1, s2;
  int blockCount = n / A_COLS_PTY, ind0, ind1, ind2;

  ind0 = tx;
  ind1 = tx + TX;
  ind2 = tx + TX * (1+ldb);

  #pragma unroll
  for(int i = 0; i < TX; i++){
    rB00[ i ] = __ldg(&(B[ ind0 + i * ldb ])) * beta ;
    rB10[ i ] = __ldg(&(B[ ind1 + i * ldb ])) * beta ;
    rB11[ i ] = __ldg(&(B[ ind2 + i * ldb ])) * beta ;
  }

  //#pragma unroll
  for(int b = 0; b < blockCount; b++){

    ind0 = tx  + A_COLS_PTY * b * lda;
    ind1 = tx  + A_COLS_PTY * b * lda + TX;
    #pragma unroll
    for(int i = 0; i < A_COLS_PTY; i++){
      rA0[ i ] = __ldg(&(A[ ind0 + i * lda ]));
      rA1[ i ] = __ldg(&(A[ ind1 + i * lda ]));
    }

    {
      #pragma unroll
      for(int j = 0; j < TX; j++){
        s0 = s1 = s2 = make_zero<T>();
        #pragma unroll
        for(int i = 0; i < A_COLS_PTY; i++){
          s0 = FMA(rA0[i], shfl(rA0[i], j, TX), s0);
          s1 = FMA(rA1[i], shfl(rA0[i], j, TX), s1);
          s2 = FMA(rA1[i], shfl(rA1[i], j, TX), s2);
        }
        rB00[j] = FMA( alpha, s0, rB00[j] );
        rB10[j] = FMA( alpha, s1, rB10[j] );
        rB11[j] = FMA( alpha, s2, rB11[j] );
      }
    }
  }


  ind0 = tx;
  ind1 = tx + TX;
  ind2 = tx + TX * (1+ldb);
  //copy B[0] data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx >= i){
      B[ ind0 + i * ldb ] = rB00[ i ];
      B[ ind2 + i * ldb ] = rB11[ i ];
    }
    B[ ind1 + i * ldb ] = rB10[ i ];
  }
}
//----------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int B_ROWS, int A_COLS_PTY>
__global__ void
kernel_syrk_U_LN_registers_Mblock2_Nmul(const int m, const int n, int batchCount,
                                        const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                        const T beta,        T_PTR              B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( m % (B_ROWS * 2) ) return;//necessary condition
  if( n % A_COLS_PTY != 0 ) return;//necessary condition
  unsigned int ind = blockIdx.x * blockDim.y + ty;
  //are we within bounds
  if(ind >= batchCount) return;
  const T *A;
        T *B;
  if(STRIDED == true){
    A = (const T*)A_array + (ind) * strideA + blockIdx.y * 2 * B_ROWS;
    B =       (T*)B_array + (ind) * strideB + blockIdx.y * 2 * B_ROWS * (1 + ldb);
  }else{
    A = ((const T**)A_array)[ind] + blockIdx.y * 2 * B_ROWS;
    B =       ((T**)B_array)[ind] + blockIdx.y * 2 * B_ROWS * (1 + ldb);
    A += A_row_off + A_col_off * lda;
    B += B_row_off + B_col_off * ldb;
  }
  dev_syrk_U_LN_registers_Mblock2_Nmul<T, B_ROWS, A_COLS_PTY>(
                                        m, n,
                                        alpha, A, lda,
                                        beta, B, ldb);
}

//==============================================================================================;
template<typename T, int TX, int A_COLS_PTY>
__device__ inline void
dev_syrk_U_LN_registers_Mblock2_Nvar( const int m, const int n,
                                      const T alpha, const T* __restrict__ A, int lda,
                                      const T beta, T* B, int ldb)
{
  T rA0[A_COLS_PTY], rA1[A_COLS_PTY], rB00[TX], rB10[TX], rB11[TX];
  T s0, s1, s2;
  int blockCount = n / A_COLS_PTY, ind0, ind1, ind2, b;

  ind0 = tx;
  ind1 = tx + TX;
  ind2 = tx + TX * (1+ldb);

  #pragma unroll
  for(int i = 0; i < TX; i++){
    rB00[ i ] = __ldg(&(B[ ind0 + i * ldb ])) * beta ;
    rB10[ i ] = __ldg(&(B[ ind1 + i * ldb ])) * beta ;
    rB11[ i ] = __ldg(&(B[ ind2 + i * ldb ])) * beta ;
  }

  for(b = 0; b < blockCount; b++){

    ind0 = tx  + A_COLS_PTY * b * lda;
    ind1 = tx  + A_COLS_PTY * b * lda + TX;
    #pragma unroll
    for(int i = 0; i < A_COLS_PTY; i++){
      rA0[ i ] = __ldg(&(A[ ind0 + i * lda ]));
      rA1[ i ] = __ldg(&(A[ ind1 + i * lda ]));
    }

    {
      #pragma unroll
      for(int j = 0; j < TX; j++){
        s0 = s1 = s2 = make_zero<T>();
        #pragma unroll
        for(int i = 0; i < A_COLS_PTY; i++){
          s0 = FMA(rA0[i], shfl(rA0[i], j, TX), s0);
          s1 = FMA(rA1[i], shfl(rA0[i], j, TX), s1);
          s2 = FMA(rA1[i], shfl(rA1[i], j, TX), s2);
        }
        rB00[j] = FMA( alpha, s0, rB00[j] );
        rB10[j] = FMA( alpha, s1, rB10[j] );
        rB11[j] = FMA( alpha, s2, rB11[j] );
      }
    }
  }
  if(n % A_COLS_PTY){

    ind0 = tx + A_COLS_PTY * b * lda;
    ind1 = tx + A_COLS_PTY * b * lda + TX;
    int acols = n - A_COLS_PTY * b;
    #pragma unroll
    for(int i = 0; i < A_COLS_PTY; i++){
      if( i < acols ){
        rA0[ i ] = __ldg(&(A[ ind0 + i * lda ]));
        rA1[ i ] = __ldg(&(A[ ind1 + i * lda ]));
      }
    }

    {
      #pragma unroll
      for(int j = 0; j < TX; j++){
        s0 = s1 = s2 = make_zero<T>();
        #pragma unroll
        for(int i = 0; i < A_COLS_PTY; i++){
          if( i < acols ){
            s0 = FMA(rA0[i], shfl(rA0[i], j, TX), s0);
            s1 = FMA(rA1[i], shfl(rA0[i], j, TX), s1);
            s2 = FMA(rA1[i], shfl(rA1[i], j, TX), s2);
          }
        }
        rB00[j] = FMA( alpha, s0, rB00[j] );
        rB10[j] = FMA( alpha, s1, rB10[j] );
        rB11[j] = FMA( alpha, s2, rB11[j] );
      }
    }
  }

  ind0 = tx;
  ind1 = tx + TX;
  ind2 = tx + TX * (1+ldb);
  //copy B[0] data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx >= i){
      B[ ind0 + i * ldb ] = rB00[ i ];
      B[ ind2 + i * ldb ] = rB11[ i ];
    }
    B[ ind1 + i * ldb ] = rB10[ i ];
  }
}
//----------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int B_ROWS, int A_COLS_PTY>
__global__ void  //__launch_bounds__(256)
kernel_syrk_U_LN_registers_Mblock2_Nvar(const int m, const int n, int batchCount,
                                        const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                        const T beta,        T_PTR              B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( m % (B_ROWS * 2) ) return;//necessary condition
  unsigned int ind = blockIdx.x * blockDim.y + ty;
  //are we within bounds
  if(ind >= batchCount) return;

  const T *A;
        T *B;
  if(STRIDED == true){
    A = (const T*)A_array + (ind) * strideA + blockIdx.y * 2 * B_ROWS;
    B =       (T*)B_array + (ind) * strideB + blockIdx.y * 2 * B_ROWS * (1 + ldb);
  }else{
    A = ((const T**)A_array)[ind] + blockIdx.y * 2 * B_ROWS;
    B =       ((T**)B_array)[ind] + blockIdx.y * 2 * B_ROWS * (1 + ldb);
    A += A_row_off + A_col_off * lda;
    B += B_row_off + B_col_off * ldb;
  }
  dev_syrk_U_LN_registers_Mblock2_Nvar<T, B_ROWS, A_COLS_PTY>(
                                        m, n,
                                        alpha, A, lda,
                                        beta, B, ldb);
}

//==============================================================================================;
template<typename T, int TX, int A_COLS_PTY>
__device__ inline void
dev_syrk_U_LN_registers_NMblock2var(const int m, const int n,
                                    const T alpha, const T* __restrict__ A, int lda,
                                    const T beta, T* B, int ldb)
{
  T rA0[A_COLS_PTY], rA1[A_COLS_PTY], rB00[TX], rB10[TX], rB11[TX];
  T s0, s1, s2, zero = make_zero<T>();
  int blockCount = n / A_COLS_PTY, ind0, ind1, ind2, b;
  bool last = (blockIdx.y == (gridDim.y - 1)) && (m % (2*TX) != 0);
  int M = last ? (m - blockIdx.y * 2 * TX) : (2 * TX);
  bool tx2_act = (M > TX ? ((tx+TX) < M) : 0);
  //initialize our variables
  #pragma unroll
  for(int i = 0; i < A_COLS_PTY; i++){
    rA0[ i ] = zero;
    rA1[ i ] = zero;
  }
  #pragma unroll
  for(int i = 0; i < TX; i++){
    rB00[ i ] = zero;
    rB10[ i ] = zero;
    rB11[ i ] = zero;
  }
  ind0 = tx;
  ind1 = tx + TX;
  ind2 = tx + TX * (1+ldb);

  if(!last){
    #pragma unroll
    for(int i = 0; i < TX; i++){
      rB00[ i ] = __ldg(&(B[ ind0 + i * ldb ])) * beta ;
      rB10[ i ] = __ldg(&(B[ ind1 + i * ldb ])) * beta ;
      rB11[ i ] = __ldg(&(B[ ind2 + i * ldb ])) * beta ;
    }
  }else{
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(tx < M && i < M)
        rB00[ i ] = __ldg(&(B[ ind0 + i * ldb ])) * beta ;
      if(tx2_act){
        rB10[ i ] = __ldg(&(B[ ind1 + i * ldb ])) * beta ;
        if(i < M-TX)
          rB11[ i ] = __ldg(&(B[ ind2 + i * ldb ])) * beta ;
      }
    }
  }

  for(b = 0; b < blockCount; b++){

    ind0 = tx  + A_COLS_PTY * b * lda;
    ind1 = tx  + A_COLS_PTY * b * lda + TX;

    if(!last){
      #pragma unroll
      for(int i = 0; i < A_COLS_PTY; i++){
        rA0[ i ] = __ldg(&(A[ ind0 + i * lda ]));
        rA1[ i ] = __ldg(&(A[ ind1 + i * lda ]));
      }
    }else{
      #pragma unroll
      for(int i = 0; i < A_COLS_PTY; i++){
        if(tx < M)
          rA0[ i ] = __ldg(&(A[ ind0 + i * lda ]));
        if(tx2_act)
          rA1[ i ] = __ldg(&(A[ ind1 + i * lda ]));
      }
    }

    {
      #pragma unroll
      for(int j = 0; j < TX; j++){
        s0 = s1 = s2 = make_zero<T>();
        #pragma unroll
        for(int i = 0; i < A_COLS_PTY; i++){
          s0 = FMA(rA0[i], shfl(rA0[i], j, TX), s0);
          s1 = FMA(rA1[i], shfl(rA0[i], j, TX), s1);
          s2 = FMA(rA1[i], shfl(rA1[i], j, TX), s2);
        }
        rB00[j] = FMA( alpha, s0, rB00[j] );
        rB10[j] = FMA( alpha, s1, rB10[j] );
        rB11[j] = FMA( alpha, s2, rB11[j] );
      }
    }
  }
  if(n % A_COLS_PTY){

    ind0 = tx + A_COLS_PTY * b * lda;
    ind1 = tx + A_COLS_PTY * b * lda + TX;
    int acols = n - A_COLS_PTY * b;
    if(!last){
      #pragma unroll
      for(int i = 0; i < A_COLS_PTY; i++){
        if( i < acols ){
          rA0[ i ] = __ldg(&(A[ ind0 + i * lda ]));
          rA1[ i ] = __ldg(&(A[ ind1 + i * lda ]));
        }
      }
    }else{
      #pragma unroll
      for(int i = 0; i < A_COLS_PTY; i++){
        if( i < acols ){
          if(tx < M)
            rA0[ i ] = __ldg(&(A[ ind0 + i * lda ]));
          if(tx2_act)
            rA1[ i ] = __ldg(&(A[ ind1 + i * lda ]));
        }
      }
    }

    {
      #pragma unroll
      for(int j = 0; j < TX; j++){
        s0 = s1 = s2 = make_zero<T>();
        #pragma unroll
        for(int i = 0; i < A_COLS_PTY; i++){
          if( i < acols ){
            s0 = FMA(rA0[i], shfl(rA0[i], j, TX), s0);
            s1 = FMA(rA1[i], shfl(rA0[i], j, TX), s1);
            s2 = FMA(rA1[i], shfl(rA1[i], j, TX), s2);
          }
        }
        rB00[j] = FMA( alpha, s0, rB00[j] );
        rB10[j] = FMA( alpha, s1, rB10[j] );
        rB11[j] = FMA( alpha, s2, rB11[j] );
      }
    }
  }

  ind0 = tx;
  ind1 = tx + TX;
  ind2 = tx + TX * (1+ldb);
  /*/copy B[0] data back to global mem
  if(!last){TODO
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(tx >= i){
        B[ ind0 + i * ldb ] = rB00[ i ];
        B[ ind2 + i * ldb ] = rB11[ i ];
      }
      B[ ind1 + i * ldb ] = rB10[ i ];
    }
  }else*/{
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if( (tx >= i) && (tx < M) && (i < M) )   B[ ind0 + i * ldb ] = rB00[ i ];
      if( tx2_act )                            B[ ind1 + i * ldb ] = rB10[ i ];
      if( (tx >= i) && tx2_act && (i < M-TX) ) B[ ind2 + i * ldb ] = rB11[ i ];
    }
    /*#pragma unroll
    for(int i = 0; i < TX; i++){
      if(tx >= i){
        if(tx < M && i < M)
          B[ ind0 + i * ldb ] = rB00[ i ];
        if((tx < M-TX) && (i < M-TX))
          B[ ind2 + i * ldb ] = rB11[ i ];
      }
      if(tx < (M-TX))
        B[ ind1 + i * ldb ] = rB10[ i ];
    }*/
  }
}
//----------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int B_ROWS, int A_COLS_PTY>
__global__ void  //__launch_bounds__(256)
kernel_syrk_U_LN_registers_NMblock2var( const int m, const int n, int batchCount,
                                        const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                        const T beta,        T_PTR              B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  unsigned int ind = blockIdx.x * blockDim.y + ty;
  //are we within bounds
  if(ind >= batchCount) return;
  const T *A;
        T *B;
  if(STRIDED == true){
    A = (const T*)A_array + (ind) * strideA + blockIdx.y * 2 * B_ROWS;
    B =       (T*)B_array + (ind) * strideB + blockIdx.y * 2 * B_ROWS * (1 + ldb);
  }else{
    A = ((const T**)A_array)[ind] + blockIdx.y * 2 * B_ROWS;
    B =       ((T**)B_array)[ind] + blockIdx.y * 2 * B_ROWS * (1 + ldb);
    A += A_row_off + A_col_off * lda;
    B += B_row_off + B_col_off * ldb;
  }
  dev_syrk_U_LN_registers_NMblock2var<T, B_ROWS, A_COLS_PTY>(
                                      m, n,
                                      alpha, A, lda,
                                      beta, B, ldb);
}
//==============================================================================================;
template<typename T, int TX, int A_ROWS_PTY>
__device__ inline void
dev_syrk_U_LT_reg_shared_Mfix_Nmul( const int m, const int n,
                                    const T alpha, const T* __restrict__ A, int lda,
                                    const T beta, T* B, int ldb)
{
  //setup shared memory
  extern __shared__ __align__(sizeof(T)) unsigned char sh_data[];
  //extern __shared__ T sh_data[];
  T* sdata = reinterpret_cast<T *>(sh_data) + ty * TX * TX;

  T rA0[TX], s, rB0[TX], zero = make_zero<T>();//, *B0;
  int blockCount = n / TX, ind;

  //copy needed data from global to shared
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ tx + i*TX ] = __ldg(&(B[ tx + i * ldb ]));
  }
  //transpose data from shared to registers
  #pragma unroll
  for(int i = 0; i < TX; i++){
    rB0[ i ] = sdata[ i + tx * TX ] * beta;//TODO handle bank conflicts ;
  }

  for(int b = 0; b < blockCount; b++){

    ind = tx + TX * b;
    //copy needed data from global to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      sdata[ tx + i*TX ] = __ldg(&(A[ ind + i * lda ]));
    }
    //transpose data from shared to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      rA0[ i ] = sdata[ i + tx * TX ];//TODO handle bank conflicts
    }

    //1. syrk on B[0] by A[b]
    #pragma unroll
    for(int j = 0; j < TX; j++){
      s = zero;
      #pragma unroll
      for(int i = 0; i < TX; i++){
        s = FMA(rA0[i], shfl(rA0[i], j, TX), s);
      }
      rB0[j] = FMA( alpha, s, rB0[j] );
    }
  }

  //transpose data from registers to shared
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ i + tx * TX ] = rB0[ i ];
  }
  //copy B[0] data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx >= i)
      B[ tx + i * ldb ] = sdata[ tx + i * TX ];
  }
}
//----------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int B_ROWS, int A_ROWS_PTY>
__global__ void  //__launch_bounds__(256)
kernel_syrk_U_LT_reg_shared_Mfix_Nmul(const int m, const int n, int batchCount,
                                      const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                      const T beta,        T_PTR              B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( m != B_ROWS ) return;//necessary condition
  if( n % B_ROWS ) return;//necessary condition
  unsigned int ind = blockIdx.x * blockDim.y + ty;
  //are we within bounds
  if(ind >= batchCount) return;
  const T *A;
        T *B;
  if(STRIDED == true){
    A = (const T*)A_array + ind * strideA;
    B =       (T*)B_array + ind * strideB;
  }else{
    A = ((const T**)A_array)[ind];
    B =       ((T**)B_array)[ind];
    A += A_row_off + A_col_off * lda;
    B += B_row_off + B_col_off * ldb;
  }
  dev_syrk_U_LT_reg_shared_Mfix_Nmul<T, B_ROWS, A_ROWS_PTY>(
                                      m, n,
                                      alpha, A, lda,
                                      beta, B, ldb);
}

//==============================================================================================;
template<typename T, int TX, int A_ROWS_PTY>
__device__ inline void
dev_syrk_U_LT_reg_shared_Mfix_Nvar( const int m, const int n,
                                    const T alpha, const T* __restrict__ A, int lda,
                                    const T beta, T* B, int ldb)
{
  //setup shared memory
  extern __shared__ __align__(sizeof(T)) unsigned char sh_data[];
  //extern __shared__ T sh_data[];
  T* sdata = reinterpret_cast<T *>(sh_data) + ty * TX * TX;

  T rA0[TX], s, rB0[TX], zero = make_zero<T>();
  int blockCount = n / TX, ind, b = 0;

  //copy needed data from global to shared
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ tx + i*TX ] = __ldg(&(B[ tx + i * ldb ]));
  }
  //transpose data from shared to registers
  #pragma unroll
  for(int i = 0; i < TX; i++){
    rB0[ i ] = sdata[ i + tx * TX ] * beta;//TODO handle bank conflicts ;
  }

  for(b = 0; b < blockCount; b++){

    ind = tx  + TX * b;
    //copy needed data from global to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      sdata[ tx + i*TX ] = __ldg(&(A[ ind + i * lda ]));
    }
    //transpose data from shared to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      rA0[ i ] = sdata[ i + tx * TX ];//TODO handle bank conflicts
    }

    //1. syrk on B[0] by A[b]
    #pragma unroll
    for(int j = 0; j < TX; j++){
      s = zero;
      #pragma unroll
      for(int i = 0; i < TX; i++){
        s = FMA(rA0[i], shfl(rA0[i], j, TX), s);
      }
      rB0[j] = FMA( alpha, s, rB0[j] );
    }
  }
  if( (n % TX) != 0 ){

    ind = tx + TX * b;
    int arows = n - TX * b;
    if( tx < arows ){
      //copy needed data from global to shared
      #pragma unroll
      for(int i = 0; i < TX; i++){
        sdata[ tx + i*TX ] = __ldg(&(A[ ind + i * lda ]));
      }
    }
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(i < arows)
        rA0[ i ] = sdata[ i + tx * TX ];//TODO handle bank conflicts
    }

    //1. syrk on B[0] by A[b]
    #pragma unroll
    for(int j = 0; j < TX; j++){
      s = zero;
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if(i < arows)
          s = FMA(rA0[i], shfl(rA0[i], j, TX), s);
      }
      rB0[j] = FMA( alpha, s, rB0[j] );
    }
  }

  //transpose data from registers to shared
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ i + tx * TX ] = rB0[ i ];
  }
  //copy B[0] data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx >= i)
      B[ tx + i * ldb ] = sdata[ tx + i * TX ];
  }
}
//----------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int B_ROWS, int A_ROWS_PTY>
__global__ void  //__launch_bounds__(256)
kernel_syrk_U_LT_reg_shared_Mfix_Nvar(const int m, const int n, int batchCount,
                                      const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                      const T beta,        T_PTR              B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( m != B_ROWS ) return;//necessary condition
  unsigned int ind = blockIdx.x * blockDim.y + ty;
  //are we within bounds
  if(ind >= batchCount) return;
  const T *A;
        T *B;
  if(STRIDED == true){
    A = (const T*)A_array + ind * strideA;
    B =       (T*)B_array + ind * strideB;
  }else{
    A = ((const T**)A_array)[ind];
    B =       ((T**)B_array)[ind];
    A += A_row_off + A_col_off * lda;
    B += B_row_off + B_col_off * ldb;
  }
  dev_syrk_U_LT_reg_shared_Mfix_Nvar<T, B_ROWS, A_ROWS_PTY>(
                                      m, n,
                                      alpha, A, lda,
                                      beta, B, ldb);
}

//==============================================================================================;
template<typename T, int TX, int A_ROWS_PTY>
__device__ inline void
dev_syrk_U_LT_reg_shared_MNvar( const int m, const int n,
                                const T alpha, const T* __restrict__ A, int lda,
                                const T beta, T* B, int ldb)
{
  //setup shared memory
  extern __shared__ __align__(sizeof(T)) unsigned char sh_data[];
  //extern __shared__ T sh_data[];
  T* sdata = reinterpret_cast<T *>(sh_data) + ty * TX * TX;

  T rA0[TX], s, rB0[TX], zero = make_zero<T>();
  int blockCount = n / TX, ind, b;

  //initialize our variables
  #pragma unroll
  for(int i = 0; i < TX; i++)
    rA0[ i ] = zero;
  #pragma unroll
  for(int i = 0; i < TX; i++)
    rB0[ i ] = zero;

  //copy needed data from global to shared
  if(tx < m){
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(i < m)
        sdata[ tx + i*TX ] = __ldg(&(B[ tx + i * ldb ]));
    }
  }
  //transpose data from shared to registers
  #pragma unroll
  for(int i = 0; i < TX; i++){
    if(i < m)
      rB0[ i ] = sdata[ i + tx * TX ] * beta;//TODO handle bank conflicts ;
  }

  for(b = 0; b < blockCount; b++){

    ind = tx + TX * b;
    //copy needed data from global to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(i < m)
        sdata[ tx + i*TX ] = __ldg(&(A[ ind + i * lda ]));
    }
    //transpose data from shared to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      rA0[ i ] = sdata[ i + tx * TX ];//TODO handle bank conflicts ;
    }
    //1. syrk on B[0] by A[b]
    #pragma unroll
    for(int j = 0; j < TX; j++){
      //if(j < m) TODO is it faster to compute more or to add more predicates?
      {
        s = zero;
        #pragma unroll
        for(int i = 0; i < TX; i++){
          s = FMA(rA0[i], shfl(rA0[i], j, TX), s);
        }
        rB0[j] = FMA( alpha, s, rB0[j] );
      }
    }
  }
  if( (n % TX) != 0 ){

    ind = tx + TX * b;
    int arows = n - TX * b;
    if(tx < arows){
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if(i < m)
          sdata[ tx + i*TX ] = __ldg(&(A[ ind + i * lda ]));
      }
    }
    //transpose data from shared to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(i < arows)
        rA0[ i ] = sdata[ i + tx * TX ];//TODO handle bank conflicts ;
    }

    //1. syrk on B[0] by A[b]
    #pragma unroll
    for(int j = 0; j < TX; j++){
      //if(j < m) TODO is it faster to compute more or to add more predicates?
      {
        s = zero;
        #pragma unroll
        for(int i = 0; i < TX; i++){
          if(i < arows)
            s = FMA(rA0[i], shfl(rA0[i], j, TX), s);
        }
        rB0[j] = FMA( alpha, s, rB0[j] );
      }
    }
  }

  //transpose data from registers to shared
  #pragma unroll
  for(int i = 0; i < TX; i++){
    if(i < m)
      sdata[ i + tx * TX ] = rB0[ i ];
  }

  //copy B[0] data back to global mem
  if(tx < m){
    #pragma unroll
    for(int i = 0; i < TX; i++)
    {
      if(tx >= i && i < m)
        B[ tx + i * ldb ] = sdata[ tx + i * TX ];
    }
  }
}
//----------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int B_ROWS, int A_ROWS_PTY>
__global__ void  //__launch_bounds__(256)
kernel_syrk_U_LT_reg_shared_MNvar(const int m, const int n, int batchCount,
                                  const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                  const T beta,        T_PTR              B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if(B_ROWS < m) return;
  unsigned int ind = blockIdx.x * blockDim.y + ty;
  //are we within bounds
  if(ind >= batchCount) return;

  const T *A;
        T *B;
  if(STRIDED == true){
    A = (const T*)A_array + ind * strideA;
    B =       (T*)B_array + ind * strideB;
  }else{
    A = ((const T**)A_array)[ind];
    B =       ((T**)B_array)[ind];
    A += A_row_off + A_col_off * lda;
    B += B_row_off + B_col_off * ldb;
  }
  dev_syrk_U_LT_reg_shared_MNvar<T, B_ROWS, A_ROWS_PTY>(
                                  m, n,
                                  alpha, A, lda,
                                  beta, B, ldb);
}

//==============================================================================================;
template<typename T, int TX, int A_COLS_PTY>
__device__ inline void
dev_syrk_U_LT_reg_shared_Mblock2_Nmul(const int m, const int n,
                                      const T alpha, const T* __restrict__ A, int lda,
                                      const T beta, T* B, int ldb)
{
  //setup shared memory
  extern __shared__ __align__(sizeof(T)) unsigned char sh_data[];
  //extern __shared__ T sh_data[];
  T* sdata = reinterpret_cast<T *>(sh_data) + ty * TX * TX;

  T rA0[TX], rA1[TX], rB00[TX], rB10[TX], rB11[TX];
  T s0, s1, s2, zero = make_zero<T>();
  int blockCount = n / TX, ind;


  ind = tx;
  //copy needed data from global to shared
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ tx + i*TX ] = __ldg(&(B[ ind + i * ldb ]));
  }
  //transpose data from shared to registers
  #pragma unroll
  for(int i = 0; i < TX; i++){
    rB00[ i ] = sdata[ i + tx * TX ] * beta;//TODO handle bank conflicts ;
  }
  ind = tx + TX;
  //copy needed data from global to shared
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ tx + i*TX ] = __ldg(&(B[ ind + i * ldb ]));
  }
  //transpose data from shared to registers
  #pragma unroll
  for(int i = 0; i < TX; i++){
    rB10[ i ] = sdata[ i + tx * TX ] * beta;//TODO handle bank conflicts ;
  }
  ind = tx + TX * (1+ldb);
  //copy needed data from global to shared
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ tx + i*TX ] = __ldg(&(B[ ind + i * ldb ]));
  }
  //transpose data from shared to registers
  #pragma unroll
  for(int i = 0; i < TX; i++){
    rB11[ i ] = sdata[ i + tx * TX ] * beta;//TODO handle bank conflicts ;
  }

  for(int b = 0; b < blockCount; b++){

    ind = tx + TX * b;
    //copy needed data from global to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      sdata[ tx + i*TX ] = __ldg(&(A[ ind + i * lda ]));
    }
    //transpose data from shared to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      rA0[ i ] = sdata[ i + tx * TX ];//TODO handle bank conflicts
    }
    ind = tx + TX * b + TX * lda;
    //copy needed data from global to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      sdata[ tx + i*TX ] = __ldg(&(A[ ind + i * lda ]));
    }
    //transpose data from shared to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      rA1[ i ] = sdata[ i + tx * TX ];//TODO handle bank conflicts
    }

    #pragma unroll
    for(int j = 0; j < TX; j++){
      s0 = zero;
      s1 = zero;
      s2 = zero;
      #pragma unroll
      for(int i = 0; i < TX; i++){
        s0 = FMA(rA0[i], shfl(rA0[i], j, TX), s0);
        s1 = FMA(rA0[i], shfl(rA1[i], j, TX), s1);
        s2 = FMA(rA1[i], shfl(rA1[i], j, TX), s2);
      }
      rB00[j] = FMA( alpha, s0, rB00[j] );
      rB10[j] = FMA( alpha, s1, rB10[j] );
      rB11[j] = FMA( alpha, s2, rB11[j] );
    }
  }


  //transpose data from registers to shared
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ i + tx * TX ] = rB00[ i ];
  }
  ind = tx;
  //copy B[00] data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx >= i)
      B[ ind + i * ldb ] = sdata[ tx + i * TX ];
  }
  //transpose data from registers to shared
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ i + tx * TX ] = rB10[ i ];
  }
  ind = tx + TX;
  //copy B[10] data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    B[ ind + i * ldb ] = sdata[ tx + i * TX ];
  }
  //transpose data from registers to shared
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ i + tx * TX ] = rB11[ i ];
  }
  ind = tx + TX * (1+ldb);
  //copy B[11] data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx >= i)
      B[ ind + i * ldb ] = sdata[ tx + i * TX ];
  }
}
//----------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int B_ROWS, int A_COLS_PTY>
__global__ void  //__launch_bounds__(256)
kernel_syrk_U_LT_reg_shared_Mblock2_Nmul( const int m, const int n, int batchCount,
                                          const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                          const T beta,        T_PTR              B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( m % (B_ROWS * 2) ) return;//necessary condition
  if( n % B_ROWS != 0 ) return;//necessary condition
  unsigned int ind = blockIdx.x * blockDim.y + ty;
  //are we within bounds
  if(ind >= batchCount) return;
  const T *A;
        T *B;
  if(STRIDED == true){
    A = (const T*)A_array + (ind) * strideA + blockIdx.y * 2 * B_ROWS;
    B =       (T*)B_array + (ind) * strideB + blockIdx.y * 2 * B_ROWS * (1 + ldb);
  }else{
    A = ((const T**)A_array)[ind] + blockIdx.y * 2 * B_ROWS;
    B =       ((T**)B_array)[ind] + blockIdx.y * 2 * B_ROWS * (1 + ldb);
    A += A_row_off + A_col_off * lda;
    B += B_row_off + B_col_off * ldb;
  }
  dev_syrk_U_LT_reg_shared_Mblock2_Nmul<T, B_ROWS, A_COLS_PTY>(
                                        m, n,
                                        alpha, A, lda,
                                        beta, B, ldb);
}

//==============================================================================================;
template<typename T, int TX, int A_COLS_PTY>
__device__ inline void
dev_syrk_U_LT_reg_shared_Mblock2_Nvar(const int m, const int n,
                                      const T alpha, const T* __restrict__ A, int lda,
                                      const T beta, T* B, int ldb)
{
  //setup shared memory
  extern __shared__ __align__(sizeof(T)) unsigned char sh_data[];
  //extern __shared__ T sh_data[];
  T* sdata = reinterpret_cast<T *>(sh_data) + ty * TX * TX;

  T rA0[TX], rA1[TX], rB00[TX], rB10[TX], rB11[TX];
  T s0, s1, s2, zero = make_zero<T>();
  int blockCount = n / TX, ind, b;

  ind = tx;
  //copy needed data from global to shared
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ tx + i*TX ] = __ldg(&(B[ ind + i * ldb ]));
  }
  //transpose data from shared to registers
  #pragma unroll
  for(int i = 0; i < TX; i++){
    rB00[ i ] = sdata[ i + tx * TX ] * beta;//TODO handle bank conflicts ;
  }
  ind = tx + TX;
  //copy needed data from global to shared
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ tx + i*TX ] = __ldg(&(B[ ind + i * ldb ]));
  }
  //transpose data from shared to registers
  #pragma unroll
  for(int i = 0; i < TX; i++){
    rB10[ i ] = sdata[ i + tx * TX ] * beta;//TODO handle bank conflicts ;
  }
  ind = tx + TX * (1+ldb);
  //copy needed data from global to shared
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ tx + i*TX ] = __ldg(&(B[ ind + i * ldb ]));
  }
  //transpose data from shared to registers
  #pragma unroll
  for(int i = 0; i < TX; i++){
    rB11[ i ] = sdata[ i + tx * TX ] * beta;//TODO handle bank conflicts ;
  }

  for( b = 0; b < blockCount; b++){

    ind = tx + TX * b;
    //copy needed data from global to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      sdata[ tx + i*TX ] = __ldg(&(A[ ind + i * lda ]));
    }
    //transpose data from shared to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      rA0[ i ] = sdata[ i + tx * TX ];//TODO handle bank conflicts
    }
    ind = tx + TX * b + TX * lda;
    //copy needed data from global to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      sdata[ tx + i*TX ] = __ldg(&(A[ ind + i * lda ]));
    }
    //transpose data from shared to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      rA1[ i ] = sdata[ i + tx * TX ];//TODO handle bank conflicts
    }

    #pragma unroll
    for(int j = 0; j < TX; j++){
      s0 = zero;
      s1 = zero;
      s2 = zero;
      #pragma unroll
      for(int i = 0; i < TX; i++){
        s0 = FMA(rA0[i], shfl(rA0[i], j, TX), s0);
        s1 = FMA(rA0[i], shfl(rA1[i], j, TX), s1);
        s2 = FMA(rA1[i], shfl(rA1[i], j, TX), s2);
      }
      rB00[j] = FMA( alpha, s0, rB00[j] );
      rB10[j] = FMA( alpha, s1, rB10[j] );
      rB11[j] = FMA( alpha, s2, rB11[j] );
    }
  }
  if( (n % TX) != 0 ){

    ind = tx + TX * b;
    int arows = n - TX * b;
    if( tx < arows ){
      //copy needed data from global to shared
      #pragma unroll
      for(int i = 0; i < TX; i++){
        sdata[ tx + i*TX ] = __ldg(&(A[ ind + i * lda ]));
      }
    }
    //transpose data from shared to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(i < arows)
        rA0[ i ] = sdata[ i + tx * TX ];//TODO handle bank conflicts
    }
    ind = tx + TX * b + TX * lda;
    if( tx < arows ){
      //copy needed data from global to shared
      #pragma unroll
      for(int i = 0; i < TX; i++){
        sdata[ tx + i*TX ] = __ldg(&(A[ ind + i * lda ]));
      }
    }
    //transpose data from shared to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(i < arows)
        rA1[ i ] = sdata[ i + tx * TX ];//TODO handle bank conflicts
    }

    #pragma unroll
    for(int j = 0; j < TX; j++){
      s0 = zero;
      s1 = zero;
      s2 = zero;
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if( i < arows ){
          s0 = FMA(rA0[i], shfl(rA0[i], j, TX), s0);
          s1 = FMA(rA0[i], shfl(rA1[i], j, TX), s1);
          s2 = FMA(rA1[i], shfl(rA1[i], j, TX), s2);
        }
      }
      rB00[j] = FMA( alpha, s0, rB00[j] );
      rB10[j] = FMA( alpha, s1, rB10[j] );
      rB11[j] = FMA( alpha, s2, rB11[j] );
    }
  }

  //transpose data from registers to shared
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ i + tx * TX ] = rB00[ i ];
  }
  ind = tx;
  //copy B[00] data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx >= i)
      B[ ind + i * ldb ] = sdata[ tx + i * TX ];
  }
  //transpose data from registers to shared
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ i + tx * TX ] = rB10[ i ];
  }
  ind = tx + TX;
  //copy B[10] data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    B[ ind + i * ldb ] = sdata[ tx + i * TX ];
  }
  //transpose data from registers to shared
  #pragma unroll
  for(int i = 0; i < TX; i++){
    sdata[ i + tx * TX ] = rB11[ i ];
  }
  ind = tx + TX * (1+ldb);
  //copy B[11] data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx >= i)
      B[ ind + i * ldb ] = sdata[ tx + i * TX ];
  }
}
//----------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int B_ROWS, int A_COLS_PTY>
__global__ void  //__launch_bounds__(256)
kernel_syrk_U_LT_reg_shared_Mblock2_Nvar( const int m, const int n, int batchCount,
                                          const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                          const T beta,        T_PTR              B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( m % (B_ROWS * 2) ) return;//necessary condition
  unsigned int ind = blockIdx.x * blockDim.y + ty;
  //are we within bounds
  if(ind >= batchCount) return;

  const T *A;
        T *B;
  if(STRIDED == true){
    A = (const T*)A_array + (ind) * strideA + blockIdx.y * 2 * B_ROWS * lda;
    B =       (T*)B_array + (ind) * strideB + blockIdx.y * 2 * B_ROWS * (1 + ldb);
  }else{
    A = ((const T**)A_array)[ind] + blockIdx.y * 2 * B_ROWS * lda;
    B =       ((T**)B_array)[ind] + blockIdx.y * 2 * B_ROWS * (1 + ldb);
    A += A_row_off + A_col_off * lda;
    B += B_row_off + B_col_off * ldb;
  }
  dev_syrk_U_LT_reg_shared_Mblock2_Nvar<T, B_ROWS, A_COLS_PTY>(
                                        m, n,
                                        alpha, A, lda,
                                        beta, B, ldb);
}

//==============================================================================================;
template<typename T, int TX, int A_COLS_PTY>
__device__ inline void
dev_syrk_U_LT_reg_shared_NMblock2var( const int m, const int n,
                                      const T alpha, const T* __restrict__ A, int lda,
                                      const T beta, T* B, int ldb)
{
  //setup shared memory
  extern __shared__ __align__(sizeof(T)) unsigned char sh_data[];
  //extern __shared__ T sh_data[];
  T* sdata = reinterpret_cast<T *>(sh_data) + ty * TX * TX;

  T rA0[A_COLS_PTY], rA1[A_COLS_PTY], rB00[TX], rB10[TX], rB11[TX];
  T s0, s1, s2, zero = make_zero<T>();
  int blockCount = n / A_COLS_PTY, ind, b;
  bool last = (blockIdx.y == (gridDim.y - 1)) && (m % (2*TX) != 0);
  int M = last ? (m - blockIdx.y * 2 * TX) : (2 * TX);
  bool tx2_act = (M > TX ? ((tx+TX) < M) : 0);

  //initialize our variables
  #pragma unroll
  for(int i = 0; i < A_COLS_PTY; i++){
    rA0[ i ] = zero;
    rA1[ i ] = zero;
  }
  #pragma unroll
  for(int i = 0; i < TX; i++){
    rB00[ i ] = zero;
    rB10[ i ] = zero;
    rB11[ i ] = zero;
  }

  if(!last){
    ind = tx;
    //copy needed data from global to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      sdata[ tx + i*TX ] = __ldg(&(B[ ind + i * ldb ]));
    }
    //transpose data from shared to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      rB00[ i ] = sdata[ i + tx * TX ] * beta;//TODO handle bank conflicts ;
    }
    ind = tx + TX;
    //copy needed data from global to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      sdata[ tx + i*TX ] = __ldg(&(B[ ind + i * ldb ]));
    }
    //transpose data from shared to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      rB10[ i ] = sdata[ i + tx * TX ] * beta;//TODO handle bank conflicts ;
    }
    ind = tx + TX * (1+ldb);
    //copy needed data from global to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      sdata[ tx + i*TX ] = __ldg(&(B[ ind + i * ldb ]));
    }
    //transpose data from shared to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      rB11[ i ] = sdata[ i + tx * TX ] * beta;//TODO handle bank conflicts ;
    }
  }else{
    ind = tx;
    if(tx < M){
      //copy needed data from global to shared
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if(i < M)
          sdata[ tx + i*TX ] = __ldg(&(B[ ind + i * ldb ]));
      }
    }
    //transpose data from shared to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(i < M)
        rB00[ i ] = sdata[ i + tx * TX ] * beta;//TODO handle bank conflicts ;
    }
    ind = tx + TX;
    if(tx2_act){
      //copy needed data from global to shared
      #pragma unroll
      for(int i = 0; i < TX; i++){
        sdata[ tx + i*TX ] = __ldg(&(B[ ind + i * ldb ]));
      }
    }
    if(M > TX){
      //transpose data from shared to registers
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if(i < M-TX)
          rB10[ i ] = sdata[ i + tx * TX ] * beta;//TODO handle bank conflicts ;
      }
    }
    ind = tx + TX * (1+ldb);
    if(tx2_act){
      //copy needed data from global to shared
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if(i < M-TX)
          sdata[ tx + i*TX ] = __ldg(&(B[ ind + i * ldb ]));
      }
    }
    if(M > TX){
      //transpose data from shared to registers
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if(i < M-TX)
          rB11[ i ] = sdata[ i + tx * TX ] * beta;//TODO handle bank conflicts ;
      }
    }
  }

  for(b = 0; b < blockCount; b++){

    if(!last){
      ind = tx + TX * b;
      //copy needed data from global to shared
      #pragma unroll
      for(int i = 0; i < TX; i++){
        sdata[ tx + i*TX ] = __ldg(&(A[ ind + i * lda ]));
      }
      //transpose data from shared to registers
      #pragma unroll
      for(int i = 0; i < TX; i++){
        rA0[ i ] = sdata[ i + tx * TX ];//TODO handle bank conflicts
      }
      ind = tx + TX * b + TX * lda;
      //copy needed data from global to shared
      #pragma unroll
      for(int i = 0; i < TX; i++){
        sdata[ tx + i*TX ] = __ldg(&(A[ ind + i * lda ]));
      }
      //transpose data from shared to registers
      #pragma unroll
      for(int i = 0; i < TX; i++){
        rA1[ i ] = sdata[ i + tx * TX ];//TODO handle bank conflicts
      }

    }else{
      ind = tx + TX * b;
      //copy needed data from global to shared
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if(i < M)
          sdata[ tx + i*TX ] = __ldg(&(A[ ind + i * lda ]));
      }
      //transpose data from shared to registers
      #pragma unroll
      for(int i = 0; i < TX; i++){
        rA0[ i ] = sdata[ i + tx * TX ];//TODO handle bank conflicts
      }
      ind = tx + TX * b + TX * lda;
      if(M > TX){
        //copy needed data from global to shared
        #pragma unroll
        for(int i = 0; i < TX; i++){
          if(i < M-TX)
            sdata[ tx + i*TX ] = __ldg(&(A[ ind + i * lda ]));
        }
        //transpose data from shared to registers
        #pragma unroll
        for(int i = 0; i < TX; i++){
          rA1[ i ] = sdata[ i + tx * TX ];//TODO handle bank conflicts
        }
      }
    }

    #pragma unroll
    for(int j = 0; j < TX; j++){
      s0 = zero;
      s1 = zero;
      s2 = zero;
      #pragma unroll
      for(int i = 0; i < TX; i++){
        s0 = FMA(rA0[i], shfl(rA0[i], j, TX), s0);
        s1 = FMA(rA0[i], shfl(rA1[i], j, TX), s1);
        s2 = FMA(rA1[i], shfl(rA1[i], j, TX), s2);
      }
      rB00[j] = FMA( alpha, s0, rB00[j] );
      rB10[j] = FMA( alpha, s1, rB10[j] );
      rB11[j] = FMA( alpha, s2, rB11[j] );
    }
  }
  if(n % A_COLS_PTY){

    int arows = n - TX * b;
    if(!last){
      ind = tx + TX * b;
      if( tx < arows ){
        #pragma unroll
        for(int i = 0; i < TX; i++){
          sdata[ tx + i*TX ] = __ldg(&(A[ ind + i * lda ]));
        }
      }
      //transpose data from shared to registers
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if(i < arows)
          rA0[ i ] = sdata[ i + tx * TX ];//TODO handle bank conflicts ;
      }
      ind = tx + TX * b + TX * lda;
      if( tx < arows ){
        #pragma unroll
        for(int i = 0; i < TX; i++){
          sdata[ tx + i*TX ] = __ldg(&(A[ ind + i * lda ]));
        }
      }
      //transpose data from shared to registers
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if(i < arows)
          rA1[ i ] = sdata[ i + tx * TX ];//TODO handle bank conflicts ;
      }
    }else{
      ind = tx + TX * b;
      if(tx < arows){
        #pragma unroll
        for(int i = 0; i < TX; i++){
          if(i < M)
            sdata[ tx + i*TX ] = __ldg(&(A[ ind + i * lda ]));
        }
      }
      //transpose data from shared to registers
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if(i < arows)
          rA0[ i ] = sdata[ i + tx * TX ];//TODO handle bank conflicts ;
      }
      if(M > TX){
        ind = tx + TX * b + TX * lda;
        if(tx < arows){
          #pragma unroll
          for(int i = 0; i < TX; i++){
            if(i < M-TX)
              sdata[ tx + i*TX ] = __ldg(&(A[ ind + i * lda ]));
          }
        }
        //transpose data from shared to registers
        #pragma unroll
        for(int i = 0; i < TX; i++){
          if(i < arows)
            rA1[ i ] = sdata[ i + tx * TX ];//TODO handle bank conflicts ;
        }
      }
    }

    #pragma unroll
    for(int j = 0; j < TX; j++){
      s0 = zero;
      s1 = zero;
      s2 = zero;
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if( i < arows ){
          s0 = FMA(rA0[i], shfl(rA0[i], j, TX), s0);
          s1 = FMA(rA0[i], shfl(rA1[i], j, TX), s1);
          s2 = FMA(rA1[i], shfl(rA1[i], j, TX), s2);
        }
      }
      rB00[j] = FMA( alpha, s0, rB00[j] );
      rB10[j] = FMA( alpha, s1, rB10[j] );
      rB11[j] = FMA( alpha, s2, rB11[j] );
    }
  }

  if(!last){
    //transpose data from registers to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      sdata[ i + tx * TX ] = rB00[ i ];
    }
    ind = tx;
    //copy B[00] data back to global mem
    #pragma unroll
    for(int i = 0; i < TX; i++)
    {
      if(tx >= i)
        B[ ind + i * ldb ] = sdata[ tx + i * TX ];
    }
    //transpose data from registers to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      sdata[ i + tx * TX ] = rB10[ i ];
    }
    ind = tx + TX;
    //copy B[10] data back to global mem
    #pragma unroll
    for(int i = 0; i < TX; i++)
    {
      B[ ind + i * ldb ] = sdata[ tx + i * TX ];
    }
    //transpose data from registers to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      sdata[ i + tx * TX ] = rB11[ i ];
    }
    ind = tx + TX * (1+ldb);
    //copy B[11] data back to global mem
    #pragma unroll
    for(int i = 0; i < TX; i++)
    {
      if(tx >= i)
        B[ ind + i * ldb ] = sdata[ tx + i * TX ];
    }
  }else{
    //transpose data from registers to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(i < M)
        sdata[ i + tx * TX ] = rB00[ i ];
    }
    ind = tx;
    //copy B[00] data back to global mem
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if((tx >= i) && (tx < M) && (i < M))
        B[ ind + i * ldb ] = sdata[ tx + i * TX ];
    }
    if(M > TX){
      //transpose data from registers to shared
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if(i < M-TX)
          sdata[ i + tx * TX ] = rB10[ i ];
      }
    }
    if(tx2_act){
      ind = tx + TX;
      //copy B[10] data back to global mem
      #pragma unroll
      for(int i = 0; i < TX; i++){
        B[ ind + i * ldb ] = sdata[ tx + i * TX ];
      }
    }
    if(M > TX){
      //transpose data from registers to shared
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if(i < M-TX)
          sdata[ i + tx * TX ] = rB11[ i ];
      }
    }
    if(tx2_act){
      ind = tx + TX * (1+ldb);
      //copy B[11] data back to global mem
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if(tx >= i && i < M-TX)
          B[ ind + i * ldb ] = sdata[ tx + i * TX ];
      }
    }
  }
}
//----------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int B_ROWS, int A_COLS_PTY>
__global__ void  //__launch_bounds__(256)
kernel_syrk_U_LT_reg_shared_NMblock2var(const int m, const int n, int batchCount,
                                        const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                        const T beta,        T_PTR              B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  unsigned int ind = blockIdx.x * blockDim.y + ty;
  //are we within bounds
  if(ind >= batchCount) return;

  const T *A;
        T *B;
  if(STRIDED == true){
    A = (const T*)A_array + (ind) * strideA + blockIdx.y * 2 * B_ROWS * lda;
    B =       (T*)B_array + (ind) * strideB + blockIdx.y * 2 * B_ROWS * (1 + ldb);
  }else{
    A = ((const T**)A_array)[ind] + blockIdx.y * 2 * B_ROWS * lda;
    B =       ((T**)B_array)[ind] + blockIdx.y * 2 * B_ROWS * (1 + ldb);
    A += A_row_off + A_col_off * lda;
    B += B_row_off + B_col_off * ldb;
  }
  dev_syrk_U_LT_reg_shared_NMblock2var<T, B_ROWS, A_COLS_PTY>(
                                        m, n,
                                        alpha, A, lda,
                                        beta, B, ldb);
}

//==============================================================================================;
template<typename T, int TX, int A_COLS_PTY>
__device__ inline void
dev_syrk_U_LN_registers_Mfix_Nvar_DB( const int m, const int n,
                                      const T alpha, const T* __restrict__ A, int lda,
                                      const T beta, T* B, int ldb)
{
  const int TX1 = TX+1;
  //setup shared memory
  extern __shared__ __align__(sizeof(T)) unsigned char sh_data[];
  //extern __shared__ T sh_data[];
  T* sdata = reinterpret_cast<T *>(sh_data) + ty * A_COLS_PTY * TX1;

  T rA0[A_COLS_PTY], s, rB0[TX];
  int blockCount = n / A_COLS_PTY, ind, b;
  //copy needed data from global to registers
  #pragma unroll
  for(int i = 0; i < TX; i++){
    rB0[ i ] = __ldg(&(B[ tx + i * ldb ])) * beta ;
  }

  b = 0;
  if(b < blockCount){
    ind = tx  + A_COLS_PTY * b * lda;
    #pragma unroll
    for(int i = 0; i < A_COLS_PTY; i++){
      rA0[ i ] = __ldg(&(A[ ind + i * lda ]));
    }
  }

  for(; b < blockCount; b++){

    if(b < blockCount-1){
      ind = tx  + A_COLS_PTY * (b+1) * lda;
      //copy needed data from global to shared
      #pragma unroll
      for(int i = 0; i < A_COLS_PTY; i++){
        sdata[ tx + i*TX1 ] = __ldg(&(A[ ind + i * lda ]));
      }
    }

    //1. syrk on B[0] by A[b]
    {
      #pragma unroll
      for(int j = 0; j < TX; j++){
        s = make_zero<T>();
        #pragma unroll
        for(int i = 0; i < A_COLS_PTY; i++){
          s = FMA(rA0[i], shfl(rA0[i], j, TX), s);
        }
        rB0[j] = FMA( alpha, s, rB0[j] );
      }
    }

    if(b < blockCount-1){
      //ind = tx  + A_COLS_PTY * b * lda;
      #pragma unroll
      for(int i = 0; i < A_COLS_PTY; i++){
        rA0[ i ] = sdata[ tx + i * TX1 ];
      }
    }
  }
  if(n % A_COLS_PTY){

    ind = tx  + A_COLS_PTY * b * lda;
    int acols = n - A_COLS_PTY * b;
    #pragma unroll
    for(int i = 0; i < A_COLS_PTY; i++){
      if( i < acols )
        rA0[ i ] = __ldg(&(A[ ind + i * lda ]));
    }

    //1. syrk on B[0] by A[b]
    {
      #pragma unroll
      for(int j = 0; j < TX; j++){
        s = make_zero<T>();
        #pragma unroll
        for(int i = 0; i < A_COLS_PTY; i++){
          if(i < acols)
            s = FMA(rA0[i], shfl(rA0[i], j, TX), s);
        }
        rB0[j] = FMA( alpha, s, rB0[j] );
      }
    }
  }

  //copy B[0] data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx >= i)
      B[ tx + i * ldb ] = rB0[ i ];
  }
}
//----------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int B_ROWS, int A_COLS_PTY>
__global__ void  //__launch_bounds__(256)
kernel_syrk_U_LN_registers_Mfix_Nvar_DB(const int m, const int n, int batchCount,
                                        const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                        const T beta,        T_PTR              B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( m != B_ROWS ) return;//necessary condition
  unsigned int ind = blockIdx.x * blockDim.y + ty;
  //are we within bounds
  if(ind >= batchCount) return;

  const T *A;
        T *B;
  if(STRIDED == true){
    A = (const T*)A_array + ind * strideA;
    B =       (T*)B_array + ind * strideB;
  }else{
    A = ((const T**)A_array)[ind];
    B =       ((T**)B_array)[ind];
    A += A_row_off + A_col_off * lda;
    B += B_row_off + B_col_off * ldb;
  }
  dev_syrk_U_LN_registers_Mfix_Nvar_DB<T, B_ROWS, A_COLS_PTY>(
                                        m, n,
                                        alpha, A, lda,
                                        beta, B, ldb);
}

//==============================================================================================;
template<typename T, int TX, int A_COLS_PTY>
__device__ inline void
dev_syrk_U_LN_registers_Mblock2_Nvar_DB( const int m, const int n,
                                         const T alpha, const T* __restrict__ A, int lda,
                                         const T beta, T* B, int ldb)
{
  const int TX1 = TX+1;
  //setup shared memory
  extern __shared__ __align__(sizeof(T)) unsigned char sh_data[];
  //extern __shared__ T sh_data[];
  T* sdata0 = reinterpret_cast<T *>(sh_data) + ty * TX * TX1;
  T* sdata1 = reinterpret_cast<T *>(sh_data) + ty * TX * TX1 + 4*TX1;

  T rA0[A_COLS_PTY], rA1[A_COLS_PTY], rB00[TX], rB10[TX], rB11[TX];
  T s0, s1, s2;
  int blockCount = n / A_COLS_PTY, ind0, ind1, ind2, b;

  ind0 = tx;
  ind1 = tx + TX;
  ind2 = tx + TX * (1+ldb);

  #pragma unroll
  for(int i = 0; i < TX; i++){
    rB00[ i ] = __ldg(&(B[ ind0 + i * ldb ])) * beta ;
    rB10[ i ] = __ldg(&(B[ ind1 + i * ldb ])) * beta ;
    rB11[ i ] = __ldg(&(B[ ind2 + i * ldb ])) * beta ;
  }

  b = 0;
  if(b < blockCount){

    ind0 = tx  + A_COLS_PTY * b * lda;
    ind1 = tx  + A_COLS_PTY * b * lda + TX;
    #pragma unroll
    for(int i = 0; i < A_COLS_PTY; i++){
      rA0[ i ] = __ldg(&(A[ ind0 + i * lda ]));
      rA1[ i ] = __ldg(&(A[ ind1 + i * lda ]));
    }
  }
  for(b = 0; b < blockCount; b++){

    if(b < blockCount-1){
      ind0 = tx  + A_COLS_PTY * (b+1) * lda;
      ind1 = tx  + A_COLS_PTY * (b+1) * lda + TX;
      //copy needed data from global to shared
      #pragma unroll
      for(int i = 0; i < A_COLS_PTY; i++){
        sdata0[ tx + i*TX1 ] = __ldg(&(A[ ind0 + i * lda ]));
        sdata1[ tx + i*TX1 ] = __ldg(&(A[ ind1 + i * lda ]));
      }
    }

    #pragma unroll
    for(int j = 0; j < TX; j++){
      s0 = s1 = s2 = make_zero<T>();
      #pragma unroll
      for(int i = 0; i < A_COLS_PTY; i++){
        s0 = FMA(rA0[i], shfl(rA0[i], j, TX), s0);
        s1 = FMA(rA1[i], shfl(rA0[i], j, TX), s1);
        s2 = FMA(rA1[i], shfl(rA1[i], j, TX), s2);
      }
      rB00[j] = FMA( alpha, s0, rB00[j] );
      rB10[j] = FMA( alpha, s1, rB10[j] );
      rB11[j] = FMA( alpha, s2, rB11[j] );
    }

    if(b < blockCount-1){
      //ind = tx  + A_COLS_PTY * b * lda;
      #pragma unroll
      for(int i = 0; i < A_COLS_PTY; i++){
        rA0[ i ] = sdata0[ tx + i * TX1 ];
        rA1[ i ] = sdata1[ tx + i * TX1 ];
      }
    }
  }
  if(n % A_COLS_PTY){

    ind0 = tx + A_COLS_PTY * b * lda;
    ind1 = tx + A_COLS_PTY * b * lda + TX;
    int acols = n - A_COLS_PTY * b;
    #pragma unroll
    for(int i = 0; i < A_COLS_PTY; i++){
      if( i < acols ){
        rA0[ i ] = __ldg(&(A[ ind0 + i * lda ]));
        rA1[ i ] = __ldg(&(A[ ind1 + i * lda ]));
      }
    }

    {
      #pragma unroll
      for(int j = 0; j < TX; j++){
        s0 = s1 = s2 = make_zero<T>();
        #pragma unroll
        for(int i = 0; i < A_COLS_PTY; i++){
          if( i < acols ){
            s0 = FMA(rA0[i], shfl(rA0[i], j, TX), s0);
            s1 = FMA(rA1[i], shfl(rA0[i], j, TX), s1);
            s2 = FMA(rA1[i], shfl(rA1[i], j, TX), s2);
          }
        }
        rB00[j] = FMA( alpha, s0, rB00[j] );
        rB10[j] = FMA( alpha, s1, rB10[j] );
        rB11[j] = FMA( alpha, s2, rB11[j] );
      }
    }
  }

  ind0 = tx;
  ind1 = tx + TX;
  ind2 = tx + TX * (1+ldb);
  //copy B[0] data back to global mem
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx >= i){
      B[ ind0 + i * ldb ] = rB00[ i ];
      B[ ind2 + i * ldb ] = rB11[ i ];
    }
    B[ ind1 + i * ldb ] = rB10[ i ];
  }
}
//----------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, int B_ROWS, int A_COLS_PTY>
__global__ void  //__launch_bounds__(256)
kernel_syrk_U_LN_registers_Mblock2_Nvar_DB( const int m, const int n, int batchCount,
                                            const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                            const T beta,        T_PTR              B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( m % (B_ROWS * 2) ) return;//necessary condition

  unsigned int ind = blockIdx.x * blockDim.y + ty;

  //are we within bounds
  if(ind >= batchCount) return;

  const T *A;
        T *B;
  if(STRIDED == true){
    A = (const T*)A_array + (ind) * strideA + blockIdx.y * 2 * B_ROWS;
    B =       (T*)B_array + (ind) * strideB + blockIdx.y * 2 * B_ROWS * (1 + ldb);
  }else{
    A = ((const T**)A_array)[ind] + blockIdx.y * 2 * B_ROWS;
    B =       ((T**)B_array)[ind] + blockIdx.y * 2 * B_ROWS * (1 + ldb);
    A += A_row_off + A_col_off * lda;
    B += B_row_off + B_col_off * ldb;
  }
  dev_syrk_U_LN_registers_Mblock2_Nvar_DB<T, B_ROWS, A_COLS_PTY>(
                                           m, n,
                                           alpha, A, lda,
                                           beta, B, ldb);
}
//==============================================================================================
#else
#error "Pre-Kepler architechture is not supported in KBLAS batch SYRK"
#endif

#endif //__XSYRK_BATCH_KERNELS_H__
