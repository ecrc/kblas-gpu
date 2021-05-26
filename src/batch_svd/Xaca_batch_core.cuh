/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_svd/Xaca_batch_core.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ali Charara
 * @date 2020-12-10
 **/

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include "kblas.h"
#include "kblas_operators.h"

#include "kblas_struct.h"
#include "kblas_prec_def.h"
#include "kblas_gpu_util.ch"

#include "workspace_queries.ch"

//==============================================================================================

#define kmin(a,b) ((a)>(b)?(b):(a))
#define kmax(a,b) ((a)<(b)?(b):(a))
#define DEBUG_MSG //printf("Line %d\n", __LINE__);fflush( stdout );
// #define PARALLEL

template<class T>
__host__ __device__
void _printMatrix(int m, int n, T* A, int lda){
  for(int r = 0; r < m; r++){
    for(int c = 0; c < n; c++){
      printf("%.7e ", A[r + c*lda]);
    }
    printf("\n");
  }
  printf("\n");
}

template<class T>
int getMax(int m, int n,
          T* A, int lda,
          int *max_i, int *max_j, T* max_v)
{
  T maxVal = absolute(A[0]), v;
  int maxInd_i = 0, maxInd_j = 0;

  #ifdef PARALLEL
  #pragma omp parallel for
  #endif
  for (int r = 0; r < m; ++r)
  {
    for (int c = 0; c < n; ++c)
    {
      v = absolute(A[r + c * lda]);
      if(v > maxVal)
      {
        maxVal = v;
        maxInd_i = r;
        maxInd_j = c;
      }
    }
  }
  *max_i = maxInd_i;
  *max_j = maxInd_j;
  *max_v = A[maxInd_i + maxInd_j * lda];

  return 1;
}

template<class T>
int getMax1(int m, int n,
          T* A, int lda,
          int *mask_r, int *mask_c,
          int *max_i, int *max_j, T* max_v)
{
  T maxVal = absolute(A[0]), v;
  int maxInd_i = 0, maxInd_j = 0;

  for (int r = 0; r < m; ++r)
  {
    for (int c = 0; c < n; ++c)
    {
      v = absolute(A[mask_r[r] + mask_c[c] * lda]);
      if(v > maxVal)
      {
        maxVal = v;
        maxInd_i = mask_r[r];
        maxInd_j = mask_c[c];
      }
    }
  }
  *max_i = maxInd_i;
  *max_j = maxInd_j;
  *max_v = A[maxInd_i + maxInd_j * lda];

  return 1;
}


template<class T>
int ACAf( int m, int n,
          T* A, int lda,
          T* U, int ldu,
          T* V, int ldv,
          T* S,
          double maxacc, int maxrk,
          double* acc, int* rk)
{
  bool converged = false;

  int k = 0, maxInd_i = 0, maxInd_j = 0;
  T maxVal = A[0];

  //find index of maximum of value of A
  // [~,I]=max(abs(R(:)));
  // d = R(i,j);
  // getMax( m, n,
  //         A, lda,
  //         &maxInd_i, &maxInd_j, &maxVal);
  // printf("Line %d, maxInd_i(%d), maxInd_j(%d), maxVal(%f)\n", __LINE__, maxInd_i, maxInd_j, maxVal);fflush( stdout );

  while(!converged){

    getMax( m, n,
            A, lda,
            &maxInd_i, &maxInd_j, &maxVal);
    // printf("Line %d, maxInd_i(%d), maxInd_j(%d), maxVal(%e)\n", __LINE__, maxInd_i, maxInd_j, maxVal);fflush( stdout );
    // S(k) = d;
    S[k] = maxVal;

    // U(:,k) = R(:,j);
    #ifdef PARALLEL
    #pragma omp parallel for
    #endif
    for (int r = 0; r < m; ++r)
    {
      U[r + k * ldu] = A[r + maxInd_j * lda];
    }
    // memcpy(U + k * ldu, A + maxInd_j * lda, m * sizeof(T) );
    // printf("Line %d\n", __LINE__);fflush( stdout );
    // printf("A %p, V %p\n", A, V);fflush( stdout );
    // V(:,k) = R(i,:)'/d;
    // dlacpy( &chA,//"A",
    //         &rows, &cols,
    //         A + maxInd_i, &lda,
    //         V /*+ k * ldv*/, &one);
    #ifdef PARALLEL
    #pragma omp parallel for
    #endif
    for (int c = 0; c < n; ++c)
    {
      V[c + k*ldv] = A[maxInd_i + c * lda] / maxVal;
    }
    // DEBUG_MSG

    // for (int c = 0; c < n; ++c)
    // {
    //   V[c + k * ldv] /= maxVal;
    // }
    // alpha = 1./maxVal;
    // dscal(&n, &alpha, V + k * ldv, &one);
    DEBUG_MSG


    // R = R - U(:,k)*V(:,k)';
    // dgemm("N", "T",
    //       &m, &n, &one,
    //       &moneT, U + k * ldu, &ldu,
    //               V + k * ldv, &one,
    //       &oneT, A, &lda);

    #ifdef PARALLEL
    #pragma omp parallel for
    #endif
    for (int r = 0; r < m; ++r)
    {
      for (int c = 0; c < n; ++c)
      {
        A[r + c * lda] -= U[r + k * ldu] * V[c + k * ldv];
      }
    }
    DEBUG_MSG
    k = k+1;

    // acc=max(max(abs(R)));
    if(maxacc > 0 && absolute(maxVal) < maxacc)
        converged = true;
    if(maxrk > 0 && k >= maxrk)
        converged = true;    

    //find index of maximum value of A
    // getMax( m, n,
    //         A, lda,
    //         &maxInd_i, &maxInd_j, &maxVal);
    DEBUG_MSG

    // printf("Line %d\n", __LINE__);fflush( stdout );
    if( k >= kmin(m,n) )
      break;
  }
    DEBUG_MSG

  *rk = k;
  *acc = absolute(maxVal);
  return 1;
}

#define HASHED

template<class T>
int ACAf1( int m, int n,
          T* A, int lda,
          T* U, int ldu,
          T* V, int ldv,
          T* S,
          double maxacc, int maxrk,
          double* acc, int* rk)
{
  bool converged = false;

  int k = 0, maxInd_i = 0, maxInd_j = 0;
  T maxVal = A[0];
  int rows, cols;

  int mask_r[m], mask_c[n];
  for (int r = 0; r < m; ++r)
    mask_r[r] = r;
  for (int c = 0; c < n; ++c)
    mask_c[c] = c;

  while(!converged){

    //find index of maximum value of A
    getMax1(
        #ifdef HASHED
            m-k, n-k,
        #else
            m, n,
        #endif
            A, lda,
            mask_r, mask_c,
            &maxInd_i, &maxInd_j, &maxVal);

    S[k] = maxVal;

    // U(:,k) = R(:,j);
    for (int r = 0; r < m; ++r)
    {
      U[r + k * ldu] = A[r + maxInd_j * lda];
    }

    for (int c = 0; c < n; ++c)
    {
      V[c + k*ldv] = A[maxInd_i + c * lda] / maxVal;
    }

    #ifndef HASHED
      rows = m; cols = n;
    #else
      rows = m-k; cols = n-k;
    #endif
    for (int r = 0; r < rows; ++r)
    {
      for (int c = 0; c < cols; ++c)
      {
        // A[r + c * lda] -= U[r + k * ldu] * V[c + k * ldv];
        A[mask_r[r] + mask_c[c] * lda] -= U[mask_r[r] + k * ldu] * V[mask_c[c] + k * ldv];
      }
    }

    #ifdef HASHED
    mask_r[maxInd_i] = mask_r[m-k-1];
    mask_c[maxInd_j] = mask_c[n-k-1];
    #endif
    k = k+1;

    if(maxacc > 0 && absolute(maxVal) < maxacc)
        converged = true;
    if(maxrk > 0 && k >= maxrk)
        converged = true; 
        
    // printf("Line %d\n", __LINE__);fflush( stdout );
    if( k >= kmin(m,n) )
      break;
  }
    DEBUG_MSG

  *rk = k;
  *acc = absolute(maxVal);
  return 1;
}

//--------------------------------------------------------------------------------------------
template<class T>
__device__
int getMax(int m, int n,
          T* A, int lda,
          int *max_i, int *max_j, T* max_v,
          int tx, int* sdata, int nWarps)
{
  T maxVal = absolute(A[tx<m?tx:0]), v;
  int maxInd_i = 0, maxInd_j = 0;

  __syncthreads();

  #pragma unroll
  for (int c = 1; c < n; ++c)
  {
    if(tx < m){
      v = absolute(A[tx + c * lda]);
      if(v > maxVal)
      {
        maxVal = v;
        // maxInd_i = r;
        maxInd_j = c;
      }
    }
  }

  if(tx < m)
    sdata[tx] = maxInd_j;

  __syncthreads();

  if(tx < nWarps){
    maxVal = absolute(A[tx + sdata[tx] * lda]);
    maxInd_i = tx;
    for (int r = nWarps; r < m; r+=nWarps)
    {
      if(tx+r < m){
        v = absolute(A[tx+r + sdata[tx+r] * lda]);
        if(v > maxVal)
        {
          maxVal = v;
          maxInd_i = tx+r;
        }
      }
    }
    sdata[tx+m] = maxInd_i;
  }

  __syncthreads();

  if (tx == 0)
  {
    maxInd_i = sdata[m];
    maxVal = absolute(A[maxInd_i + sdata[maxInd_i] * lda]);
    for (int r = 1; r < nWarps; r++)
    {
      v = absolute(A[sdata[r+m] + sdata[sdata[r+m]] * lda]);
      if(v > maxVal)
      {
        maxVal = v;
        maxInd_i = sdata[r+m];
      }
    }
    maxInd_j = sdata[maxInd_i];
    *max_i = maxInd_i;
    *max_j = maxInd_j;
    *max_v = A[maxInd_i + maxInd_j * lda];
  }

  return 1;
}

//--------------------------------------------------------------------------------------------
template<class T>
__global__
void kernel_ACAf( int m, int n,
                  T* A, int lda,
                  T* U, int ldu,
                  T* V, int ldv,
                  T* S,
                  double maxacc, int maxrk,
                  double* acc, T* rk)
{
  int tx = threadIdx.x, mWarps = (m+31) / 32, nWarps = (n+31) / 32;

  // sdata size: m + nWarps + 2;
  extern __shared__ int sdata[];

  bool converged = false;

  int k = 0;
  int *maxInd_i = sdata+m+mWarps, *maxInd_j = sdata+m+mWarps+1;
  T maxVal = make_zero<T>();

  // printf("tx %d, A[tx] %.7e\n", tx, A[tx]);

  while(!converged){

    //find index of maximum of value of A
    getMax( m, n,
            A, lda,
            maxInd_i, maxInd_j, &maxVal,
            threadIdx.x, sdata, nWarps);

    // printf("Line %d, maxInd_i(%d), maxInd_j(%d), maxVal(%e)\n", __LINE__, maxInd_i, maxInd_j, maxVal);fflush( stdout );
    // S(k) = d;
    if(0 == tx)
      S[k] = maxVal;

    __syncthreads();

    maxVal = S[k];

    // U(:,k) = R(:,j);
    if(tx < m)
      U[tx + k * ldu] = A[tx + (*maxInd_j) * lda];

    // V(:,k) = R(i,:)'/d;
    if(tx < n)
      V[tx + k * ldv] = A[(*maxInd_i) + tx * lda] / maxVal;

    __syncthreads();

    if(tx < m){
      for (int c = 0; c < n; ++c)
      {
        A[tx + c * lda] -= U[tx + k * ldu] * V[c + k * ldv];
      }
    }
    k = k+1;

    // acc=max(max(abs(R)));
    if(maxacc > 0 && absolute(maxVal) < maxacc)
        converged = true;
    if(maxrk > 0 && k >= maxrk)
        converged = true; 

    //find index of maximum value of A
    // getMax( m, n,
    //         A, lda,
    //         &maxInd_i, &maxInd_j, &maxVal);
    DEBUG_MSG

    //*/
    // printf("Line %d\n", __LINE__);fflush( stdout );
    if( k >= kmin(m,n) )
      break;
  }

  if(0 == tx){
    // printf("ACA rk=%d\n", k);
    *rk = (T)k;
    *acc = absolute(maxVal);
  }
}


//--------------------------------------------------------------------------------------------
template<class T>
__device__ inline
void dev_ACAf_batch(int m, int n,
                    T* A, int lda,
                    T* U, int ldu,
                    T* V, int ldv,
                    T* S, int lds,
                    double maxacc, int maxrk,
                    double* acc, int* rk)
{

  int tx = threadIdx.x, mWarps = (m+31) / 32, nWarps = (n+31) / 32;

  // sdata size: m + nWarps + 2;
  extern __shared__ int sdata[];

  bool converged = false;

  int k = 0;
  int *maxInd_i = sdata+m+mWarps, *maxInd_j = sdata+m+mWarps+1;
  T maxVal = make_zero<T>();

  // printf("tx %d, A[tx] %.7e\n", tx, A[tx]);

  while(!converged){

    //find index of maximum of value of A
    getMax( m, n,
            A, lda,
            maxInd_i, maxInd_j, &maxVal,
            threadIdx.x, sdata, nWarps);

    // printf("Line %d, maxInd_i(%d), maxInd_j(%d), maxVal(%e)\n", __LINE__, maxInd_i, maxInd_j, maxVal);fflush( stdout );
    // S(k) = d;
    if(0 == tx)
      S[k*lds] = maxVal;

    __syncthreads();

    maxVal = S[k*lds];

    // U(:,k) = R(:,j);
    if(tx < m)
      U[tx + k * ldu] = A[tx + (*maxInd_j) * lda];

    // V(:,k) = R(i,:)'/d;
    if(tx < n)
      V[tx + k * ldv] = A[(*maxInd_i) + tx * lda] / maxVal;

    __syncthreads();

    if(tx < m){
      for (int c = 0; c < n; ++c)
      {
        A[tx + c * lda] -= U[tx + k * ldu] * V[c + k * ldv];
      }
    }
    k = k+1;

    // acc=max(max(abs(R)));
    if(maxacc > 0 && absolute(maxVal) < maxacc)
        converged = true;
    if(maxrk > 0 && k >= maxrk)
        converged = true; 

    //find index of maximum value of A
    // getMax( m, n,
    //         A, lda,
    //         &maxInd_i, &maxInd_j, &maxVal);
    DEBUG_MSG

    //*/
    // printf("Line %d\n", __LINE__);fflush( stdout );
    if( k >= kmin(m,n) )
      break;
  }

  if(0 == tx){
    // printf("ACA rk=%d\n", k);
    if(rk != NULL)
      *rk = (T)k;
    if(acc != NULL)
      *acc = absolute(maxVal);
  }
}


//--------------------------------------------------------------------------------------------
template<typename T, typename T_PTR>
__global__ void
kernel_ACAf_batch_U(const int m, const int n,
                    T_PTR A_array, int lda, long strideA,
                    T_PTR U_array, int ldu, long strideU,
                    T_PTR V_array, int ldv, long strideV,
                    T_PTR S_array, int lds, long strideS,
                    double maxacc, int maxrk,
                    double* acc_array, int* rk_array)
{
  //TODO validate input
  double *acc = NULL;
  int *rk = NULL;

  T *A = getOperationPtr(A_array, blockIdx.x, strideA),
    *U = getOperationPtr(U_array, blockIdx.x, strideU),
    *V = getOperationPtr(V_array, blockIdx.x, strideV),
    *S = getOperationPtr(S_array, blockIdx.x, strideS);


  // if(STRIDED == true){
  //   A = (T*)A_array + blockIdx.x * strideA;
  //   U = (T*)U_array + blockIdx.x * strideU;
  //   V = (T*)V_array + blockIdx.x * strideV;
  //   S = (T*)S_array + blockIdx.x * strideS;
  // }else{
  //   A = ((T**)A_array)[blockIdx.x];
  //   U = ((T**)U_array)[blockIdx.x];
  //   V = ((T**)V_array)[blockIdx.x];
  //   S = ((T**)S_array)[blockIdx.x];
  // }

  if(rk_array != NULL)
    rk = (int*)rk_array + blockIdx.x;
  if(acc_array != NULL)
    acc = (double*)acc_array + blockIdx.x;

  dev_ACAf_batch<T>(m, n,
                    A, lda,
                    U, ldu,
                    V, ldv,
                    S, lds,
                    maxacc, maxrk,
                    acc, rk);
}

//--------------------------------------------------------------------------------------------
template<typename T, typename T_PTR, typename LDT>
__global__ void
kernel_ACAf_batch_N(int* m_array, int* n_array,
                    T_PTR A_array, LDT lda_array, long strideA,
                    T_PTR U_array, LDT ldu_array, long strideU,
                    T_PTR V_array, LDT ldv_array, long strideV,
                    T_PTR S_array, LDT lds_array, long strideS,
                    double maxacc, int maxrk,
                    double* acc_array, int* rk_array)
{
  //TODO validate input
  double *acc = NULL;
  int *rk = NULL;
  int m, n,
      lda = getOperationVal((LDT)lda_array, blockIdx.x),
      ldu = getOperationVal((LDT)ldu_array, blockIdx.x),
      ldv = getOperationVal((LDT)ldv_array, blockIdx.x),
      lds = getOperationVal((LDT)lds_array, blockIdx.x);

  T *A = getOperationPtr(A_array, blockIdx.x, strideA),
    *U = getOperationPtr(U_array, blockIdx.x, strideU),
    *V = getOperationPtr(V_array, blockIdx.x, strideV),
    *S = getOperationPtr(S_array, blockIdx.x, strideS);

  // if(STRIDED == true){
  //   A = (T*)A_array + blockIdx.x * strideA;
  //   U = (T*)U_array + blockIdx.x * strideU;
  //   V = (T*)V_array + blockIdx.x * strideV;
  //   S = (T*)S_array + blockIdx.x * strideS;
  //   lda = (int)lda_array;
  //   ldu = (int)ldu_array;
  //   ldv = (int)ldv_array;
  //   lds = (int)lds_array;
  // }else{
  //   A = ((T**)A_array)[blockIdx.x];
  //   U = ((T**)U_array)[blockIdx.x];
  //   V = ((T**)V_array)[blockIdx.x];
  //   S = ((T**)S_array)[blockIdx.x];
  //   lda = ((int*)lda_array)[blockIdx.x];
  //   ldu = ((int*)ldu_array)[blockIdx.x];
  //   ldv = ((int*)ldv_array)[blockIdx.x];
  //   lds = ((int*)lds_array)[blockIdx.x];
  // }

  m = m_array[blockIdx.x];
  n = n_array[blockIdx.x];

  if(rk_array != NULL)
    rk = (int*)rk_array + blockIdx.x;
  if(acc_array != NULL)
    acc = (double*)acc_array + blockIdx.x;

  dev_ACAf_batch<T>(m, n,
                    A, lda,
                    U, ldu,
                    V, ldv,
                    S, lds,
                    maxacc, maxrk,
                    acc, rk);
}
