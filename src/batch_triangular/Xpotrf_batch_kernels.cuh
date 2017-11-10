/**
  --* (C) Copyright 2013 King Abdullah University of Science and Technology
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

#ifndef __XPOTRF_BATCH_KERNELS_H__
#define __XPOTRF_BATCH_KERNELS_H__


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

template<typename T, int N>
__device__ inline void
dev_potrf_U_registers_fixN(int n, T* A, int lda, int* info)
{
  T rA[N], s;
  //copy needed data from global to registers
  #pragma unroll
  for(int i = 0; i < N; i++)
  {
    //if(tx >= i)
      rA[ i ] = __ldg(&(A[ tx + i * lda ]));
  }

  //perform factorization on registers
  #pragma unroll
  for(int j = 0; j < N; j++)
  {
    s = sqrt( shfl(rA[j], j, N) );
    //if(tx == j)
      //rA[j] = s;
    //if(tx >= j)
      rA[j] /= s;

    #pragma unroll
    for(int i = 0; i < N; i++){
      if(j < i){
        s = -shfl(rA[j], i, N);
        if(i <= tx)
          rA[i] = FMA( rA[j], s, rA[i]);
        //rA[i] -= rA[j] * s;
      }
    }
  }

  //copy data back to global mem
  #pragma unroll
  for(int i = 0; i < N; i++)
  {
    if(tx >= i)
      A[ tx + i * lda ] = rA[ i ];
  }
}

template<typename T, typename T_PTR, bool STRIDED, int N>
__global__ void  //__launch_bounds__(256)
kernel_potrf_U_registers_fixN(const int n, int batchCount,
                              T_PTR A_array, int A_row_off, int A_col_off, int lda, long strideA,
                              int* info)
{
  //are we within bounds
  if(blockIdx.x * blockDim.y + ty >= batchCount) return;

  if(n != N){ info[blockIdx.x * blockDim.y + ty] = -1; return; }

  T *A;
  if(STRIDED == true){
    A = (T*)A_array + (blockIdx.x * blockDim.y + ty) * strideA;
  }else{
    A = ((T**)A_array)[blockIdx.x * blockDim.y + ty];
  }
  A += A_row_off + A_col_off * lda;

  dev_potrf_U_registers_fixN<T, N>(n, A, lda, info);
}

//==============================================================================================
template<typename T, int TX>
__device__ inline void
dev_potrf_U_registers_varN(int n, T* A, int lda, int* info)
{
  T rA[TX], s, zero = make_zero<T>();
  //copy needed data from global to registers
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    if(tx < n && i < n)
      rA[ i ] = __ldg(&(A[ tx + i * lda ]));
  }

  //perform factorization on registers
  #pragma unroll
  for(int j = 0; j < TX; j++)
  {
    s = sqrt( shfl(rA[j], j, TX) );
    if(j < n){
      //if(s == zero){
      //  info[blockIdx.x * blockDim.y + ty] = j; return;
      //}
      //if(tx == j)
        //rA[j] = s;
      //if(tx > j)
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
}

template<typename T, typename T_PTR, bool STRIDED, int TX>
__global__ void  //__launch_bounds__(256)
kernel_potrf_U_registers_varN(int n, int batchCount,
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

  dev_potrf_U_registers_varN<T, TX>(n, A, lda, info);
}

//==============================================================================================
//viable BS: 4,8,16,32 with blockDim.y is multiple of: 8,4,2,1 respectively
//TODO may use double buffering through shared memory to speedup
template<typename T, int N, int BS>
__device__ inline void
dev_potrf_U_registers_fixN_blocked_2(int n, T* A, int lda, int* info)
{
  T rA0[BS], s, rA1[BS], *A0;

  //copy needed data from global to registers of block A[0,0]
  A0 = A;
  #pragma unroll
  for(int i = 0; i < BS; i++){
    //if(tx >= i)
      rA0[ i ] = __ldg(&(A0[ tx + i * lda ]));
  }

  //1. potrf A[0,0]
  {
    //perform factorization on registers
    #pragma unroll
    for(int j = 0; j < BS; j++)
    {
      s = sqrt( shfl(rA0[j], j, BS) );
      //if(tx == j)
        //rA0[j] = s;
      //if(tx > j)
        rA0[j] /= s;

      #pragma unroll
      for(int i = 0; i < BS; i++){
        s = -shfl(rA0[j], i, BS);
        if(j < i)// && i <= tx)
          rA0[i] = FMA( rA0[j], s, rA0[i]);
          //rA[i] -= rA[j] * s;
      }
    }

    //copy A[0,0] data back to global mem
    #pragma unroll
    for(int i = 0; i < BS; i++)
    {
      if(tx >= i)
        A[ tx + i * lda ] = rA0[ i ];
    }
  }

  //2. RLT trsm A[0,0]->A[1,0]
  {
    //fetch A[1,0] into registers
    A0 = A + BS;
    #pragma unroll
    for(int i = 0; i < BS; i++){
      rA1[ i ] = __ldg(&(A0[ tx + i * lda ]));
    }

    //trsm on registers
    #pragma unroll
    for(int k = 0; k < BS; k++){
      //if(non-unit)
      rA1[k] /= shfl(rA0[k], k, BS);
      #pragma unroll
      for(int j = 0; j < BS; j++){
        if(j > k){
          s = -shfl(rA0[k], j, BS);
          rA1[j] = FMA( s, rA1[k], rA1[j]);
        }
      }
    }

    //copy A[1,0] data back to global memory
    #pragma unroll
    for(int i = 0; i < BS; i++){
      A0[ tx + i * lda ] = rA1[ i ];
    }

  }

  //3. syrk on A[1,1] by A[1,0]
  {
    //fetch A[1,1] into registers
    A0 = A + (1 + lda) * BS;
    #pragma unroll
    for(int i = 0; i < BS; i++){
      rA0[ i ] = __ldg(&(A0[ tx + i * lda ]));
    }
    #pragma unroll
    for(int j = 0; j < BS; j++){
      s = make_zero<T>();
      #pragma unroll
      for(int i = 0; i < BS; i++){
        s = FMA(rA1[i], shfl(rA1[i], j, BS), s);
      }
      //if(j <= tx)
      rA0[j] -= s;
    }
  }

  //4. potrf A[1,1]
  {
    //perform factorization on registers
    #pragma unroll
    for(int j = 0; j < BS; j++)
    {
      s = sqrt( shfl(rA0[j], j, BS) );
      //if(tx == j)
        //rA0[j] = s;
      //if(tx > j)
        rA0[j] /= s;

      #pragma unroll
      for(int i = 0; i < BS; i++){
        s = -shfl(rA0[j], i, BS);
        if(j < i)// && i <= tx)//TODO
          rA0[i] = FMA( rA0[j], s, rA0[i]);
          //rA[i] -= rA[j] * s;
      }
    }

    //copy A[1,1] data back to global mem
    #pragma unroll
    for(int i = 0; i < BS; i++)
    {
      if(tx >= i)
        A0[ tx + i * lda ] = rA0[ i ];
    }
  }
}

//viable BS: 4,8,16,32 with blockDim.y is multiple of: 8,4,2,1 respectively
//TODO may use double buffering through shared memory to speedup
template<typename T, typename T_PTR, bool STRIDED, int N, int BS>
__global__ void  //__launch_bounds__(256)
kernel_potrf_U_registers_fixN_blocked_2(int n, int batchCount,
                                        T_PTR A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                        int* info)
{
  //are we within bounds
  if(blockIdx.x * blockDim.y + ty >= batchCount) return;

  if(n != N){ info[blockIdx.x * blockDim.y + ty] = -1; return; }

  T *A;
  if(STRIDED == true){
    A = (T*)A_array + (blockIdx.x * blockDim.y + ty) * strideA;
  }else{
    A = ((T**)A_array)[blockIdx.x * blockDim.y + ty];
  }
  A += A_row_off + A_col_off * lda;

  dev_potrf_U_registers_fixN_blocked_2<T, N, BS>(n, A, lda, info);

}
//==============================================================================================
//viable BS: 4,8,16,32 with blockDim.y is multiple of: 8,4,2,1 respectively
template<typename T, int TX, int BS>
__device__ inline void
dev_potrf_U_registers_varN_blocked_2(int n, T* A, int lda, int* info)
{
  T rA0[BS], s, rA1[BS], *A0;

  bool tx2_act = (n > BS ? ((tx+BS) < n) : 0);

  //copy needed data from global to registers of block A[0,0]
  //A0 = A + BS;
  #pragma unroll
  for(int i = 0; i < BS; i++){
    //if(tx >= i)
    rA0[ i ] = __ldg(&(A [ tx + i * lda ]));
    //if(tx2_act)
      //rA1[ i ] = __ldg(&(A0[ tx + i * lda ]));
  }

  //1. potrf A[0,0]
  {
    //perform factorization on registers
    #pragma unroll
    for(int j = 0; j < BS; j++)
    {
      s = sqrt( shfl(rA0[j], j, BS) );
      //if(tx == j)
        //rA0[j] = s;
      //if(tx > j)
        rA0[j] /= s;

      #pragma unroll
      for(int i = 0; i < BS; i++){
        s = -shfl(rA0[j], i, BS);
        if(j < i)// && i <= tx)
          rA0[i] = FMA( rA0[j], s, rA0[i]);
          //rA[i] -= rA[j] * s;
      }
    }

    //copy A[0,0] data back to global mem
    #pragma unroll
    for(int i = 0; i < BS; i++)
    {
      if(tx >= i)
        A[ tx + i * lda ] = rA0[ i ];
    }
  }

  //2. RLT trsm A[0,0]->A[1,0]
  {
    //fetch A[1,0] into registers
    A0 = A + BS;
    #pragma unroll
    for(int i = 0; i < BS; i++){
      if(tx2_act)
        rA1[ i ] = __ldg(&(A0[ tx + i * lda ]));
    }

    //trsm on registers
    #pragma unroll
    for(int k = 0; k < BS; k++){
      //if(non-unit)
      s = shfl(rA0[k], k, BS);
      rA1[k] /= s;
      #pragma unroll
      for(int j = 0; j < BS; j++){
        if(j > k){
          s = -shfl(rA0[k], j, BS);
          rA1[j] = FMA( s, rA1[k], rA1[j]);
        }
      }
    }

    //copy A[1,0] data back to global memory
    #pragma unroll
    for(int i = 0; i < BS; i++){
      if(tx2_act)
        A0[ tx + i * lda ] = rA1[ i ];
    }

  }

  //3. syrk on A[1,1] by A[1,0]
  {//TODO we can start pulling data earlier
    //fetch A[1,1] into registers
    A0 = A + (1 + lda) * BS;
    #pragma unroll
    for(int i = 0; i < BS; i++){
      if(tx2_act && (i+BS) < n)
        rA0[ i ] = __ldg(&(A0[ tx + i * lda ]));
    }
    #pragma unroll
    for(int j = 0; j < BS; j++){
      s = make_zero<T>();
      if(j+BS < n){
        #pragma unroll
        for(int i = 0; i < BS; i++){
          s = FMA(rA1[i], shfl(rA1[i], j, BS), s);
        }
        //if(j <= tx)
        rA0[j] -= s;
      }
    }
  }

  //4. potrf A[1,1]
  {
    //perform factorization on registers
    #pragma unroll
    for(int j = 0; j < BS; j++)
    {
      s = sqrt( shfl(rA0[j], j, BS) );
      if(j+BS < n){
        //if(tx == j)
          //rA0[j] = s;
        //if(tx > j)
          rA0[j] /= s;
      }
      #pragma unroll
      for(int i = 0; i < BS; i++){
        s = -shfl(rA0[j], i, BS);
        if(i+BS < n && j < i)// && i <= tx)//TODO
          rA0[i] = FMA( rA0[j], s, rA0[i]);
          //rA[i] -= rA[j] * s;
      }
    }

    //copy A[1,1] data back to global mem
    #pragma unroll
    for(int i = 0; i < BS; i++)
    {
      if(tx2_act && (i+BS) < n && tx >= i)
        A0[ tx + i * lda ] = rA0[ i ];
    }
  }
}
//===============================
//viable BS: 4,8,16,32 with blockDim.y is multiple of: 8,4,2,1 respectively
template<typename T, typename T_PTR, bool STRIDED, int TX, int BS>
__global__ void  //__launch_bounds__(256)
kernel_potrf_U_registers_varN_blocked_2(int n, int batchCount,
                                        T_PTR A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                        int* info)
{
  //are we within bounds
  if(blockIdx.x * blockDim.y + ty >= batchCount) return;

  if(BS >= n || n > 2*BS){ info[blockIdx.x * blockDim.y + ty] = -1; return; }

  T *A;
  if(STRIDED == true){
    A = (T*)A_array + (blockIdx.x * blockDim.y + ty) * strideA;
  }else{
    A = ((T**)A_array)[blockIdx.x * blockDim.y + ty];
  }
  A += A_row_off + A_col_off * lda;

  dev_potrf_U_registers_varN_blocked_2<T, TX, BS>(n, A, lda, info);
}
//==============================================================================================
#else
  #error "Pre-Kepler architechture is not supported in KBLAS batch POTRF"
#endif

#endif //__XPOTRF_BATCH_KERNELS_H__