/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/lapack/Xpotrf_core.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ali Charara
 * @author Wajih Halim Boukaram
 * @date 2020-12-10
 **/

#ifndef __XPOTRF_CORE_CUH__
#define __XPOTRF_CORE_CUH__

#define KBLAS_POTRF_NB 32

//==============================================================================================
template<class T>
__device__ inline void chol_sh_unpack_ex32_2d_fgs(int dim, T* smem, int tx, int ty, int* s_info, int bar = 1)
{
  const int n = 32;
  T s, sx, sy;
  
  for(int j = 0; j < dim && ty >= j; j++)
  {
    if(j < dim-1)
      asm volatile("bar.sync %0, %1;" : : "r"(bar), "r"(n*(n-j)) : "memory" );
    s = smem[j]; sx = smem[tx]; sy = smem[ty];
    
    if(s <= 0)
    {
        if(tx == 0 && ty == j)
            *s_info = j;
        return;
    }
    
    if(ty == j)
    {
      s = sqrt(s);
      smem[tx] = tx == j ? s : sx / s;
    }
    else
      //if(ty > j)
      smem[tx + (ty-j)*n] -= sx * sy / s;

    smem += n;
  }
}

template<class T>
__global__ void kernel_potrf_ex32_unpack_2d_fgs(int N, T* x, int incx, int col_start, int* info)
{
  const int n = 32;
  int tx = threadIdx.x, ty = threadIdx.y, ti = threadIdx.x + threadIdx.y * blockDim.x;
  
  //setup shared memory
  __shared__ T sdata[n*n];
  __shared__ int s_info;
  //__shared__ volatile int sflag[n];

  //if(tx == 0)
  //  sflag[ty] = 0;
  
  // Make sure to set the info to 0 on the first block 
  if(info && ti == 0)
  {
    if(col_start == 0)
        *info = 0;
    s_info = 0;
  }
    
  //copy needed data from global to shared mem unpacked format
  if(tx < N && ty < N && tx >= ty)
    sdata[ti] = x[tx + ty * incx];

  //perform factorization on shared mem
  chol_sh_unpack_ex32_2d_fgs(N, sdata, tx, ty, &s_info);
  
  __syncthreads();
  // If the factorization failed, set the info to the column where it stopped 
  if(info && s_info != 0 && *info == 0 && ti == 0)
    *info = -(col_start + s_info);

  //copy data back to global mem
  if(tx < N && ty < N && tx >= ty)
    x[tx + ty * incx] = sdata[ti];
}

//==============================================================================================
template<class T>
int Xpotrf_rec(kblasHandle_t handle,
                char uplo, int n,
                T *A, int lda, int col_start,
                int* info)
{
  if(n <= 0)
    return KBLAS_Success;

  int status;
  
  if(n <= 32){
    dim3 dimBlock(32,32);
    kernel_potrf_ex32_unpack_2d_fgs<T><<< 1, dimBlock, 0, handle->stream>>> (n, A, lda, col_start, info);
    check_error_ret( cudaGetLastError(), KBLAS_UnknownError);

    return KBLAS_Success;
  }
  else
  {
    int n1, n2;
    
    if(REG_SIZE(n))
      n1 = n2 = n/2;
    else{
      n1 = CLOSEST_REG_SIZE(n);
      n2 = n-n1;
    }

    T one     =  make_one<T>();
    T neg_one = -one;

    status = Xpotrf_rec(handle, uplo, n1, A, lda, col_start, info);
    check_error_ret( status, status );

    status = kblasXtrsm( handle,
                         KBLAS_Right, KBLAS_Lower, KBLAS_Trans, KBLAS_NonUnit,
                         n2, n1,
                         one, A, lda,
                              A + n1, lda);
    check_error_ret( status, status );

    status = kblasXsyrk( handle,
                         KBLAS_Lower, KBLAS_NoTrans,
                         n2, n1,
                         neg_one, A + n1, lda,
                         one,     A + n1*(lda+1), lda);
    check_error_ret( status, status );

    status = Xpotrf_rec(handle, uplo, n2, A + n1*(lda+1), lda, col_start + n1, info);
    check_error_ret( status, status );

  }
  // KBLAS_UNUSED(status);
  return KBLAS_Success;
}

//==============================================================================================
template<class T>
int Xpotrf_core(kblasHandle_t handle, char uplo, int n, T *A, int lda, int col_start, int* info)
{
  if(n <= 0)
    return KBLAS_Success;

  int status;
  T one     =  make_one<T>();
  T neg_one = -one;

  int nb = min(KBLAS_POTRF_NB, n);

  status = Xpotrf_rec(handle, uplo, nb, A, lda, col_start, info);
  check_error_ret( status, status );

  if(n - nb > 0){

    status = kblasXtrsm( handle,
                         KBLAS_Right, KBLAS_Lower, KBLAS_Trans, KBLAS_NonUnit,
                         n-nb, nb,
                         one, A, lda,
                              A + nb, lda);
    check_error_ret( status, status );

    status = kblasXsyrk( handle,
                         KBLAS_Lower, KBLAS_NoTrans,
                         n-nb, nb,
                         neg_one, A + nb, lda,
                         one,     A + nb*(lda+1), lda);
    check_error_ret( status, status );

    status = Xpotrf_core(handle, uplo, n-nb, A + nb*(lda+1), lda, col_start + nb, info);
    check_error_ret( status, status );
  }

  return KBLAS_Success;
}

template<class T>
int Xpotrf(kblasHandle_t handle, char uplo, int n, T *A, int lda, int* info)
{
  return Xpotrf_core(handle, uplo, n, A, lda, 0, info);
}

#undef KBLAS_POTRF_NB

#endif //__XPOTRF_CORE_CUH__