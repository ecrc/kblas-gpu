/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xtrsm_batch_kernels.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ali Charara
 * @date 2020-12-10
 **/

#ifndef __XTRSM_BATCH_KERNELS_H__
#define __XTRSM_BATCH_KERNELS_H__


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
dev_trsm_U_RLXN_registers_fixN_mulM(const int m, const int n,
                                    const T alpha, const T* __restrict__ A, int lda,
                                                                      T* B, int ldb)
{
  int Bm_start = TY * blockIdx.y,
      Bm_end = (m > (TY * (blockIdx.y+1))) ? TY * (blockIdx.y+1) : m;

  T rA[TX], rB[TX], s;
  int ind0;
  int mb = (Bm_end - Bm_start) / TX;

  //copy needed data from global to registers
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    //if(tx >= i)
    rA[ i ] = __ldg(&(A[ tx + i * lda ]));
  }
  for(int b = 0; b < mb; b++)
  {
    ind0 = tx + TX * b;
    //copy needed data from global to registers
    #pragma unroll
    for(int i = 0; i < TX; i++)
    {
      if(TRANS)
        rB[ i ] = __ldg(&(B[ ind0 + i * ldb ]));
      else
        rB[ i ] = alpha * __ldg(&(B[ ind0 + i * ldb ]));
    }

    #pragma unroll
    for(int j = (TRANS ? 0 : TX-1); (TRANS && j < TX) || (!TRANS && j >= 0); j+= (TRANS ? 1 : -1))
    {
      if(TRANS){
        s = shfl(rA[j], j, TX);
        rB[j] /= s;
      }

      #pragma unroll
      for(int i = 0; i < TX; i++){
        s = -shfl(rA[j], i, TX);
        if(j < i)
          if(TRANS)
            rB[i] = FMA( rB[j], s, rB[i]);
          else
            rB[j] = FMA( rB[i], s, rB[j]);
      }

      if(!TRANS){
        s = shfl(rA[j], j, TX);
        rB[j] /= s;
      }
    }

    //copy data back to global mem
    #pragma unroll
    for(int i = 0; i < TX; i++)
    {
      if(TRANS)
        B[ ind0 + i * ldb ] = alpha * rB[ i ];
      else
        B[ ind0 + i * ldb ] = rB[ i ];
    }
  }
}
//--------------------------------------------------------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, bool TRANS, int TX, int TY>
__global__ void  //__launch_bounds__(256)
kernel_trsm_U_RLXN_registers_fixN_mulM( const int m, const int n, int batchCount,
                                        const T alpha, T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                                                    T_PTR B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( (TX != n) || (m % TX) ) return;//necessary condition

  unsigned int ind = blockIdx.x * blockDim.y + ty;
  //are we within bounds
  if(ind >= batchCount) return;

  int Bm_start = TY * blockIdx.y;
  const T *A;
        T *B;
  if(STRIDED == true){
    A = (const T*)A_array + (ind) * strideA;
    B =       (T*)B_array + (ind) * strideB;
  }else{
    A = ((const T**)A_array)[ind];
    B =       ((T**)B_array)[ind];
  }
  A += A_row_off + A_col_off * lda;
  B += B_row_off + B_col_off * ldb + Bm_start;

  dev_trsm_U_RLXN_registers_fixN_mulM<T, TRANS, TX, TY>(m, n,
                                                        alpha, A, lda,
                                                               B, ldb);
}

//==============================================================================================
template<typename T, bool TRANS, int TX, int BY>
__device__ inline void
dev_trsm_U_RLXN_registers_fixN_varM(const int m, const int n,
                                    const T alpha, const T* __restrict__ A, int lda,
                                                                      T* B, int ldb)
{
  //if( (TX != n) ) return;//necessary condition
  int Bm_start = BY * blockIdx.y, Bm_end = (m > BY * (blockIdx.y+1)) ? BY * (blockIdx.y+1) : m;

  T rA[TX], rB[TX], s;
  int ind0, b;
  int mb = (Bm_end - Bm_start) / TX;// + ((m / gridDim.y) % TX != 0);

  //copy needed data from global to registers
  #pragma unroll
  for(int i = 0; i < TX; i++)
  {
    //if(tx >= i)
    rA[ i ] = __ldg(&(A[ tx + i * lda ]));
  }
  for(b = 0; b < mb; b++)
  {
    ind0 = tx + TX * b;
    //copy needed data from global to registers
    #pragma unroll
    for(int i = 0; i < TX; i++)
    {
      if(TRANS)
        rB[ i ] = __ldg(&(B[ ind0 + i * ldb ]));
      else
        rB[ i ] = alpha * __ldg(&(B[ ind0 + i * ldb ]));
    }

    #pragma unroll
    for(int j = (TRANS ? 0 : TX-1); (TRANS && j < TX) || (!TRANS && j >= 0); j+= (TRANS ? 1 : -1))
    {
      if(TRANS){
        s = shfl(rA[j], j, TX);
        rB[j] /= s;
      }

      #pragma unroll
      for(int i = 0; i < TX; i++){
        s = -shfl(rA[j], i, TX);
        if(j < i)
          if(TRANS)
            rB[i] = FMA( rB[j], s, rB[i]);
          else
            rB[j] = FMA( rB[i], s, rB[j]);
      }

      if(!TRANS){
        s = shfl(rA[j], j, TX);
        rB[j] /= s;
      }
    }

    //copy data back to global mem
    #pragma unroll
    for(int i = 0; i < TX; i++)
    {
      if(TRANS)
        B[ ind0 + i * ldb ] = alpha * rB[ i ];
      else
        B[ ind0 + i * ldb ] = rB[ i ];
    }
  }
  if((Bm_end - Bm_start) % TX != 0)
  {
    ind0 = tx + TX * b;
    //copy needed data from global to registers
    if(ind0 < m){
      #pragma unroll
      for(int i = 0; i < TX; i++)
      {
        if(TRANS)
          rB[ i ] = __ldg(&(B[ ind0 + i * ldb ]));
        else
          rB[ i ] = alpha * __ldg(&(B[ ind0 + i * ldb ]));
      }
    }

    #pragma unroll
    for(int j = (TRANS ? 0 : TX-1); (TRANS && j < TX) || (!TRANS && j >= 0); j+= (TRANS ? 1 : -1))
    {
      if(TRANS){
        s = shfl(rA[j], j, TX);
        rB[j] /= s;
      }

      #pragma unroll
      for(int i = 0; i < TX; i++){
        s = -shfl(rA[j], i, TX);
        if(j < i)
          if(TRANS)
            rB[i] = FMA( rB[j], s, rB[i]);
          else
            rB[j] = FMA( rB[i], s, rB[j]);
      }

      if(!TRANS){
        s = shfl(rA[j], j, TX);
        rB[j] /= s;
      }
    }

    //copy data back to global mem
    if(ind0 < m){
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if(TRANS)
          B[ ind0 + i * ldb ] = alpha * rB[ i ];
        else
          B[ ind0 + i * ldb ] = rB[ i ];
      }
    }
  }
}
//--------------------------------------------------------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, bool TRANS, int TX, int TY>
__global__ void  //__launch_bounds__(256)
kernel_trsm_U_RLXN_registers_fixN_varM( const int m, const int n, int batchCount,
                                        const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                                                          T_PTR B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  unsigned int ind = blockIdx.x * blockDim.y + ty;
  //are we within bounds
  if(ind >= batchCount) return;

  int Bm_start = TY * blockIdx.y;
  const T *A;
        T *B;
  if(STRIDED){
    A = (const T*)A_array + (ind) * strideA;
    B =       (T*)B_array + (ind) * strideB;
  }else{
    A = ((const T**)A_array)[ind];
    B =       ((T**)B_array)[ind];
  }
  A += A_row_off + A_col_off * lda;
  B += B_row_off + B_col_off * ldb + Bm_start;
  dev_trsm_U_RLXN_registers_fixN_varM<T, TRANS, TX, TY>(m, n,
                                                        alpha, A, lda,
                                                               B, ldb);
}

//==============================================================================================
template<typename T, bool TRANS, int TX, int TY>
__device__ inline void
dev_trsm_U_RLXN_registers_varN_varM(const int m, const int n,
                                    const T alpha, const T* __restrict__ A, int lda,
                                                                      T* B, int ldb)
{
  int Bm_start = TY * blockIdx.y, Bm_end = (m > (TY * (blockIdx.y+1))) ? TY * (blockIdx.y+1) : m;

  T rA[TX], rB[TX], s, zero = make_zero<T>();
  int ind0, b;
  int mb = (Bm_end - Bm_start) / TX;

  //initialize our variables
  #pragma unroll
  for(int i = 0; i < TX; i++){
    rA[ i ] = zero;
    rB[ i ] = zero;
  }
  //copy needed data from global to registers
  if(tx < n){
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(i < n)
        rA[ i ] = __ldg(&(A[ tx + i * lda ]));
    }
  }
  for(b = 0; b < mb; b++)
  {
    ind0 = tx + TX * b;
    //copy needed data from global to registers
    #pragma unroll
    for(int i = 0; i < TX; i++)
    {
      if(i < n){
        if(TRANS)
          rB[ i ] = __ldg(&(B[ ind0 + i * ldb ]));
        else
          rB[ i ] = alpha * __ldg(&(B[ ind0 + i * ldb ]));
      }
    }

    #pragma unroll
    for(int j = (TRANS ? 0 : TX-1); (TRANS && j < TX) || (!TRANS && j >= 0); j+= (TRANS ? 1 : -1))
    {
      if(TRANS){
        s = shfl(rA[j], j, TX);
        if(s != zero)
          rB[j] /= s;
      }

      #pragma unroll
      for(int i = 0; i < TX; i++){
        s = -shfl(rA[j], i, TX);
        if(j < i)
          if(TRANS)
            rB[i] = FMA( rB[j], s, rB[i]);
          else
            rB[j] = FMA( rB[i], s, rB[j]);
      }

      if(!TRANS){
        s = shfl(rA[j], j, TX);
        if(s != zero)
          rB[j] /= s;
      }
    }

    //copy data back to global mem
    #pragma unroll
    for(int i = 0; i < TX; i++)
    {
      if(i < n){
        if(TRANS)
          B[ ind0 + i * ldb ] = alpha * rB[ i ];
        else
          B[ ind0 + i * ldb ] = rB[ i ];
      }
    }
  }
  if((Bm_end - Bm_start) % TX != 0)
  {
    ind0 = tx + TX * b;
    //copy needed data from global to registers
    if(ind0 + Bm_start < m){
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if(i < n){
          if(TRANS)
            rB[ i ] = __ldg(&(B[ ind0 + i * ldb ]));
          else
            rB[ i ] = alpha * __ldg(&(B[ ind0 + i * ldb ]));
        }
      }
    }

    #pragma unroll
    for(int j = (TRANS ? 0 : TX-1); (TRANS && j < TX) || (!TRANS && j >= 0); j+= (TRANS ? 1 : -1))
    {
      if(TRANS){
        s = shfl(rA[j], j, TX);
        if(s != zero)
          rB[j] /= s;
      }

      #pragma unroll
      for(int i = 0; i < TX; i++){
        s = -shfl(rA[j], i, TX);
        if(j < i)
          if(TRANS)
            rB[i] = FMA( rB[j], s, rB[i]);
          else
            rB[j] = FMA( rB[i], s, rB[j]);
      }

      if(!TRANS){
        s = shfl(rA[j], j, TX);
        if(s != zero)
          rB[j] /= s;
      }
    }

    //copy data back to global mem
    if(ind0 + Bm_start < m){
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if(i < n){
          if(TRANS)
            B[ ind0 + i * ldb ] = alpha * rB[ i ];
          else
            B[ ind0 + i * ldb ] = rB[ i ];
        }
      }
    }
  }
}
//--------------------------------------------------------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, bool TRANS, int TX, int TY>
__global__ void  //__launch_bounds__(256)
kernel_trsm_U_RLXN_registers_varN_varM( const int m, const int n, int batchCount,
                                        const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                                                          T_PTR B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( (TX < n) ) return;//necessary condition

  unsigned int ind = blockIdx.x * blockDim.y + ty;
  //are we within bounds
  if(ind >= batchCount) return;

  int Bm_start = TY * blockIdx.y;
  const T *A;
        T *B;
  if(STRIDED){
    A = (const T*)A_array + (ind) * strideA;
    B =       (T*)B_array + (ind) * strideB;
  }else{
    A = ((const T**)A_array)[ind];
    B =       ((T**)B_array)[ind];
  }
  A += A_row_off + A_col_off * lda;
  B += B_row_off + B_col_off * ldb + Bm_start;
  dev_trsm_U_RLXN_registers_varN_varM<T, TRANS, TX, TY>(m, n,
                                                        alpha, A, lda,
                                                               B, ldb);
}

//==============================================================================================
template<typename T, bool TRANS, int TX, int TY>
__device__ inline void
dev_trsm_U_LLXN_registers_Mfix_Nmul(const int m, const int n,
                                    const T alpha, const T* __restrict__ A, int lda,
                                                                      T* B, int ldb)
{
  //TODO better grid layout can be devised here
  int Bn_start = TY * blockIdx.y, Bn_end = (n > (TY * (blockIdx.y+1))) ? TY * (blockIdx.y+1) : n;

  T rA[TX], rB[TX], s, a;
  int ind0;
  int nb = (Bn_end - Bn_start) / TX;

  //copy needed data from global to registers
  #pragma unroll
  for(int i = 0; i < TX; i++){
    //if(tx >= i)
    rA[ i ] = __ldg(&(A[ tx + i * lda ]));
  }
  for(int b = 0; b < nb; b++)
  {
    ind0 = tx + TX * b * ldb;
    //copy needed data from global to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(TRANS)
        rB[ i ] = __ldg(&(B[ ind0 + i * ldb ]));
      else
        rB[ i ] = alpha * __ldg(&(B[ ind0 + i * ldb ]));
    }

    #pragma unroll
    for(int j = (!TRANS ? 0 : TX-1); (!TRANS && j < TX) || (TRANS && j >= 0); j+= (!TRANS ? 1 : -1))
    {
      /*if(!TRANS){
        s = shfl(rA[j], j, TX);
        rB[j] /= s;
      }*/

      #pragma unroll
      for(int i = 0; i < TX; i++){
        a = shfl(rA[i],i,TX);
        if(tx == i){
          rB[j] /= a;
        }
        s = -shfl(rB[j], i, TX);
        if(tx > i)
          if(TRANS)
            rB[j] = FMA( rB[i], s, rB[j]);
          else
            rB[j] = FMA( rA[i], s, rB[j]);
      }

      /*if(TRANS){
        s = shfl(rA[j], j, TX);
        rB[j] /= s;
      }*/
    }

    //copy data back to global mem
    #pragma unroll
    for(int i = 0; i < TX; i++)
    {
      if(TRANS)
        B[ ind0 + i * ldb ] = alpha * rB[ i ];
      else
        B[ ind0 + i * ldb ] = rB[ i ];
    }
  }
}
//--------------------------------------------------------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, bool TRANS, int TX, int TY>
__global__ void  //__launch_bounds__(256)
kernel_trsm_U_LLXN_registers_Mfix_Nmul( const int m, const int n, int batchCount,
                                        const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                                                          T_PTR B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( (TX != m) || (n % TX) ) return;//necessary condition

  unsigned int ind = blockIdx.x * blockDim.y + ty;
  //are we within bounds
  if(ind >= batchCount) return;

  //TODO better grid layout can be devised here
  int Bn_start = TY * blockIdx.y;
  const T *A;
        T *B;
  if(STRIDED){
    A = (const T*)A_array + (ind) * strideA;
    B =       (T*)B_array + (ind) * strideB;
  }else{
    A = ((const T**)A_array)[ind];
    B =       ((T**)B_array)[ind];
  }
  A += A_row_off + A_col_off * lda;
  B += B_row_off + B_col_off * ldb + Bn_start;
  dev_trsm_U_LLXN_registers_Mfix_Nmul<T, TRANS, TX, TY>(m, n,
                                                        alpha, A, lda,
                                                               B, ldb);
}

//==============================================================================================
template<typename T, bool TRANS, int TX, int TY>
__device__ inline void
dev_trsm_U_LLXN_registers_Mfix_Nvar(const int m, const int n,
                                    const T alpha, const T* __restrict__ A, int lda,
                                                                      T* B, int ldb)
{
  const int TX1 = TX+1;
  //setup shared memory
  extern __shared__ __align__(sizeof(T)) unsigned char sh_data[];
  //extern __shared__ T sh_data[];
  T* sdata = reinterpret_cast<T *>(sh_data) + ty * TX * TX1;

  //TODO better grid layout can be devised here
  int Bn_start = TY * blockIdx.y;
  int Bn_end = (n > (TY * (blockIdx.y+1))) ? TY * (blockIdx.y+1) : n;

  T rA[TX], rB[TX], s;
  int ind0, b = 0;
  int nb = (Bn_end - Bn_start) / TX;

  //copy needed data from global to registers
  #pragma unroll
  for(int i = 0; i < TX; i++){
    //if(tx >= i)
    //rA[ i ] = A[ tx + i * lda ];
    rA[ i ] = __ldg(&(A[ tx + i * lda ]));
  }
  for(b = 0; b < nb; b++)
  {
    ind0 = (tx + TX * b) * ldb;//TODO use shared memory to avoid non-coalesced memory read
    //copy needed data from global to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      //if(TRANS)
      //  rB[ i ] = __ldg(&(B[ ind0 + i ]));
      //else
        //rB[ i ] = alpha * B[ ind0 + i ];
        rB[ i ] = alpha * __ldg(&(B[ ind0 + i ]));
    }
    if(TRANS){
      #pragma unroll
      for(int j = TX-1; j >= 0; j--){
        #pragma unroll
        for(int i = 0; i < TX; i++){
          if(i > j){
            s = -shfl(rA[j], i, TX);
            rB[j] = FMA( rB[i], s, rB[j]);
          }
        }

        s = shfl(rA[j], j, TX);
        rB[j] /= s;
      }
    }else{
      #pragma unroll
      for(int j = 0; j < TX; j++){
        #pragma unroll
        for(int i = 0; i < TX; i++){
          if(i < j){
            s = -shfl(rA[i], j, TX);
            rB[j] = FMA( s, rB[i], rB[j]);
          }
        }

        s = shfl(rA[j], j, TX);
        rB[j] /= s;
      }
    }
    //transpose data from registers to shared
    #pragma unroll
    for(int i = 0; i < TX; i++){
      sdata[ i + tx * TX1 ] = rB[ i ];//TODO handle bank conflicts ;
    }
    ind0 = tx + TX * b * ldb;
    // ind0 = tx;
    //copy rB data back to global mem
    #pragma unroll
    for(int i = 0; i < TX; i++)
    {
      B[ ind0 + i * ldb ] = sdata[ tx + i * TX1 ];
    }

    /*/
    //copy data back to global mem
    #pragma unroll
    for(int i = 0; i < TX; i++)
    {
      B[ ind0 + i ] = rB[ i ];
    }//*/
  }

  if((Bn_end - Bn_start) % TX != 0){

    ind0 = (tx + TX * b) * ldb;
    //copy needed data from global to registers
    if( (Bn_start + tx + TX * b) < Bn_end ){
      #pragma unroll
      for(int i = 0; i < TX; i++){
        //rB[ i ] = alpha * B[ ind0 + i ];
        rB[ i ] = alpha * __ldg(&(B[ ind0 + i ]));
      }
    }

    if(TRANS){
      #pragma unroll
      for(int j = TX-1; j >= 0; j--){
        #pragma unroll
        for(int i = 0; i < TX; i++){
          if(i > j){
            s = -shfl(rA[j], i, TX);
            rB[j] = FMA( rB[i], s, rB[j]);
          }
        }

        s = shfl(rA[j], j, TX);
        rB[j] /= s;
      }
    }else{
      #pragma unroll
      for(int j = 0; j < TX; j++){
        #pragma unroll
        for(int i = 0; i < TX; i++){
          if(i < j){
            s = -shfl(rA[i], j, TX);
            rB[j] = FMA( s, rB[i], rB[j]);
          }
        }

        s = shfl(rA[j], j, TX);
        rB[j] /= s;
      }
    }

    //copy data back to global mem
    if( (Bn_start + tx + TX * b) < Bn_end){
      #pragma unroll
      for(int i = 0; i < TX; i++)
      {
        B[ ind0 + i ] = rB[ i ];
      }
    }
  }
}
//--------------------------------------------------------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, bool TRANS, int TX, int TY>
__global__ void  //__launch_bounds__(256)
kernel_trsm_U_LLXN_registers_Mfix_Nvar( const int m, const int n, int batchCount,
                                        const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                                                          T_PTR B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( TX != m ) return;//necessary condition

  unsigned int ind = blockIdx.x * blockDim.y + ty;
  //are we within bounds
  if(ind >= batchCount) return;

  //TODO better grid layout can be devised here
  int Bn_start = TY * blockIdx.y;
  const T *A;
        T *B;
  if(STRIDED){
    A = (const T*)A_array + (ind) * strideA;
    B =       (T*)B_array + (ind) * strideB;
  }else{
    A = ((const T**)A_array)[ind];
    B =       ((T**)B_array)[ind];
  }
  A += A_row_off + A_col_off * lda;
  B += B_row_off + B_col_off * ldb + Bn_start * ldb;
  dev_trsm_U_LLXN_registers_Mfix_Nvar<T, TRANS, TX, TY>(m, n,
                                                        alpha, A, lda,
                                                               B, ldb);
}

//==============================================================================================
template<typename T, bool TRANS, int TX, int TY>
__device__ inline void
dev_trsm_U_LLXN_registers_MNvar(const int m, const int n,
                                const T alpha, const T* __restrict__ A, int lda,
                                                                  T* B, int ldb)
{
  //TODO better grid layout can be devised here
  int Bn_start = TY * blockIdx.y;
  int Bn_end = (n > (TY * (blockIdx.y+1))) ? TY * (blockIdx.y+1) : n;

  T rA[TX], rB[TX], s;
  int ind0, b = 0;
  int nb = (Bn_end - Bn_start) / TX;

  //copy needed data from global to registers

  if(tx < m){
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(i < m)
        rA[ i ] = __ldg(&(A[ tx + i * lda ]));
    }
  }
  for(b = 0; b < nb; b++)
  {
    ind0 = (tx + TX * b) * ldb;
    //copy needed data from global to registers
    #pragma unroll
    for(int i = 0; i < TX; i++){
      if(i < m)
        rB[ i ] = alpha * __ldg(&(B[ ind0 + i ]));
    }
    if(TRANS){
      #pragma unroll
      for(int j = TX-1; j >= 0; j--){
        if(j < m){
          #pragma unroll
          for(int i = 0; i < TX; i++){
            if(j < i && i < m){
              s = -shfl(rA[j], i, TX);
              rB[j] = FMA( rB[i], s, rB[j]);
            }
          }

          s = shfl(rA[j], j, TX);
          rB[j] /= s;
        }
      }
    }else{
      #pragma unroll
      for(int j = 0; j < TX; j++){
        if(j < m){
          #pragma unroll
          for(int i = 0; i < TX; i++){
            if(i < j){
              s = -shfl(rA[i], j, TX);
              rB[j] = FMA( s, rB[i], rB[j]);
            }
          }

          s = shfl(rA[j], j, TX);
          rB[j] /= s;
        }
      }
    }
    //TODO avoid non-coalesced memory write by using shared memory
    /*#pragma unroll
    for(int i = 0; i < TX; i++){
      rBT[i] = shfl(rB[tx], i, TX);
    }*/

    //copy data back to global mem
    #pragma unroll
    for(int i = 0; i < TX; i++)
    {
        /*#pragma unroll
        for(int j = 0; j < TX; j++){
          T a = shfl(rB[j], i, TX);
          if(j == tx) s = a;
        }

        B[ tx + TX * b * ldb + i * ldb ] = s;*/
      if(i < m)
        B[ ind0 + i ] = rB[ i ];
    }
  }

  if((Bn_end - Bn_start) % TX != 0){

    ind0 = (tx + TX * b) * ldb;
    //copy needed data from global to registers
    if( (Bn_start + tx + TX * b) < Bn_end ){
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if(i < m)
          rB[ i ] = alpha * __ldg(&(B[ ind0 + i ]));
      }
    }

    if(TRANS){
      #pragma unroll
      for(int j = TX-1; j >= 0; j--){
        if(j < m){
          #pragma unroll
          for(int i = 0; i < TX; i++){
            if(j < i && i < m){
              s = -shfl(rA[j], i, TX);
              rB[j] = FMA( rB[i], s, rB[j]);
            }
          }

          s = shfl(rA[j], j, TX);
          rB[j] /= s;
        }
      }
    }else{
      #pragma unroll
      for(int j = 0; j < TX; j++){
        if(j < m){
          #pragma unroll
          for(int i = 0; i < TX; i++){
            if(i < j){
              s = -shfl(rA[i], j, TX);
              rB[j] = FMA( s, rB[i], rB[j]);
            }
          }

          s = shfl(rA[j], j, TX);
          rB[j] /= s;
        }
      }
    }

    //copy data back to global mem
    if( (Bn_start + tx + TX * b) < Bn_end){
      #pragma unroll
      for(int i = 0; i < TX; i++){
        if(i < m)
          B[ ind0 + i ] = rB[ i ];
      }
    }
  }
}
//--------------------------------------------------------------------------------------------
template<typename T, typename T_PTR, bool STRIDED, bool TRANS, int TX, int TY>
__global__ void  //__launch_bounds__(256)
kernel_trsm_U_LLXN_registers_MNvar( const int m, const int n, int batchCount,
                                    const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                                                      T_PTR B_array, int B_row_off, int B_col_off, int ldb, long strideB)
{
  if( TX < m ) return;//necessary condition

  unsigned int ind = blockIdx.x * blockDim.y + ty;
  //are we within bounds
  if(ind >= batchCount) return;

  //TODO better grid layout can be devised here
  int Bn_start = TY * blockIdx.y;
  const T *A;
        T *B;
  if(STRIDED){
    A = (const T*)A_array + (ind) * strideA;
    B =       (T*)B_array + (ind) * strideB;
  }else{
    A = ((const T**)A_array)[ind];
    B =       ((T**)B_array)[ind];
  }
  A += A_row_off + A_col_off * lda;
  B += B_row_off + B_col_off * ldb + Bn_start * ldb;
  dev_trsm_U_LLXN_registers_MNvar<T, TRANS, TX, TY>(m, n,
                                                    alpha, A, lda,
                                                           B, ldb);
}

//==============================================================================================
#else
  #error "Pre-Kepler architechture is not supported in KBLAS batch TRSM"
#endif

#endif //__XTRSM_BATCH_KERNELS_H__
