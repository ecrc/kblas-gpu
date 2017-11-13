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
#ifndef SM
  #error "SM is not defined"
#elif (SM >= 30)


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