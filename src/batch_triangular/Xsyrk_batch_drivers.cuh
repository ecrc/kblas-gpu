/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/Xsyrk_batch_drivers.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef __XSYRK_BATCH_DRIVERS_H__
#define __XSYRK_BATCH_DRIVERS_H__


#include "Xsyrk_batch_kernels.cuh"

//==============================================================================================
//TODO tuning variable
#define A_COLS_PTY 8

// invoke gemm kernel on off diagonal blocks (strided case)
// workspace needed: device pointers
// A, B: host pointer to device buffers
template<class T>
int Xsyrk_gemm_rec_flat_strided(kblasHandle_t handle,
                                char uplo, char trans,
                                const int m, const int n,
                                const T alpha,
                                const T* A, int lda, long strideA,
                                const T beta,
                                      T* B, int ldb, long strideB,
                                int batchCount)
{
  //these gemm calls can run in parallel, through streams

  if(m <= 16){
    return KBLAS_Success;
  }

  int depth = 0, s = 16;
  while(s < m){
    s = s << 1;
    depth++;
  }

  char transA = trans,
       transB = (transA == KBLAS_NoTrans ? KBLAS_Trans : KBLAS_NoTrans);

  kblasWorkspace_t ws_current = &(handle->work_space);

  T **A_work,
    **B_work,
    **C_work;
  A_work = (T**)(ws_current->d_ptrs);
  B_work = A_work + (1 << (depth-1) ) * batchCount;
  C_work = B_work + (1 << (depth-1) ) * batchCount;

  int d = 0,
      M = m,
      row = CLOSEST_REG_SIZE(m),
      mm = M - row,
      nn = row,
      kk = n;

  while(d < depth){
    int cur_batchCount = 0;
    int b = 0;
    int uniform_batches = (d == 0) ? 1 : (m / row);

    while((b < uniform_batches) && m >= (mm+row+2*b*row) ){
      //append this batch call
      Xset_pointer_3(A_work+b*batchCount, A + (row+2*b*row)*(trans == KBLAS_NoTrans ? 1 : lda), lda, strideA,
                     B_work+b*batchCount, A + (2*b*row)*(trans == KBLAS_NoTrans ? 1 : lda), lda, strideA,
                     C_work+b*batchCount, B + row * (uplo == KBLAS_Lower ? 1 : ldb)+2*b*row*(1+ldb), ldb, strideB,
                     batchCount, handle->stream);
      cur_batchCount += batchCount;
      b++;
    }
    //issue one batch call
    if(cur_batchCount > 0){
      kblas_gemm_batch( handle,
                        transA, transB,
                        mm, nn, kk,
                        alpha, (const T**)A_work, lda,
                               (const T**)B_work, lda,
                        beta,             C_work, ldb,
                        cur_batchCount);
    }
    if((d > 0) && ((m % row) != 0) && m > (row+2*b*row) ){
      //one block is remaining that is not regular size
      Xset_pointer_3(A_work, A + (row+2*b*row)*(trans == KBLAS_NoTrans ? 1 : lda), lda, strideA,
                     B_work, A + (2*b*row)*(trans == KBLAS_NoTrans ? 1 : lda), lda, strideA,
                     C_work, B + row * (uplo == KBLAS_Lower ? 1 : ldb)+2*b*row*(1+ldb), ldb, strideB,
                     batchCount, handle->stream);
      //issue one batch call
      kblas_gemm_batch( handle,
                        transA, transB,
                        m - (row+2*b*row), nn, kk,
                        alpha, (const T**)A_work, lda,
                               (const T**)B_work, lda,
                        beta,             C_work, ldb,
                        batchCount);
    }


    d++;
    row /= 2;
    mm = nn = row;
  }
  return KBLAS_Success;
}

//-------------------------------------------------------------------
// batch strided SYRK core routine
// workspace needed: device pointers
// A, B: host pointer to device buffers
template<typename T, typename T_PTR>
int Xsyrk_batch_strided_core( kblasHandle_t handle,
                              char uplo, char trans,
                              const int m, const int n,
                              const T alpha, const T_PTR A, int lda, long strideA,
                              const T beta,        T_PTR B, int ldb, long strideB,
                              int batchCount)
{
  if( uplo == KBLAS_Upper ){
    printf("Upper SYRK_BATCH is not implemented yet\n");
    return KBLAS_NotImplemented;
  }
  int status;

  //invoke gemm for off diagonal blocks
  if(m > 16){

    // do we have enough workspace allocated
    KBlasWorkspaceState ws_needed;
    syrk_batch_wsquery_core( m, batchCount, (kblasWorkspaceState_t)&ws_needed);

    if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) ){
      return KBLAS_InsufficientWorkspace;
    }

    check_error_ret(status = Xsyrk_gemm_rec_flat_strided(handle,
                                                         uplo, trans,
                                                         m, n,
                                                         alpha, A, lda, strideA,
                                                         beta,  B, ldb, strideB,
                                                         batchCount), status);
  }


  // handle diagonal blocks with custom CUDA kernels
  // process all diagonal tiles in one CUDA kernel launch with 2D grid

  int2 dims[7] = {
    { 8, 4},//1 warps
    { 8, 8},//2 warps
    { 8,12},//3 warps
    { 8,16},//4 warps
    {16, 2},//1 warps
    {16, 4},//2 warps
    {16, 8} //4 warps
  };
  int func_idx = 0;
  int dim_idx = 0;

  typedef void (*syrk_kernels_type)(const int m, const int n, int batchCount,
                                    const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                    const T beta,        T_PTR              B_array, int B_row_off, int B_col_off, int ldb, long strideB);

  syrk_kernels_type syrk_kernels[] = {
    kernel_syrk_U_LN_registers_Mfix_Nmul    <T, T_PTR, true, 8, A_COLS_PTY>,
    kernel_syrk_U_LN_registers_Mfix_Nvar    <T, T_PTR, true, 8, A_COLS_PTY>,
    kernel_syrk_U_LN_registers_MNvar        <T, T_PTR, true, 8, A_COLS_PTY>,
    kernel_syrk_U_LN_registers_MNvar        <T, T_PTR, true, 8, A_COLS_PTY>,
    kernel_syrk_U_LN_registers_Mblock2_Nmul <T, T_PTR, true, 8, A_COLS_PTY>,
    kernel_syrk_U_LN_registers_Mblock2_Nvar <T, T_PTR, true, 8, A_COLS_PTY>,
    kernel_syrk_U_LN_registers_NMblock2var  <T, T_PTR, true, 8, A_COLS_PTY>,
    kernel_syrk_U_LN_registers_NMblock2var  <T, T_PTR, true, 8, A_COLS_PTY>,
    kernel_syrk_U_LT_reg_shared_Mfix_Nmul   <T, T_PTR, true, 8, A_COLS_PTY>,
    kernel_syrk_U_LT_reg_shared_Mfix_Nvar   <T, T_PTR, true, 8, A_COLS_PTY>,
    kernel_syrk_U_LT_reg_shared_MNvar       <T, T_PTR, true, 8, A_COLS_PTY>,
    kernel_syrk_U_LT_reg_shared_MNvar       <T, T_PTR, true, 8, A_COLS_PTY>,
    kernel_syrk_U_LT_reg_shared_Mblock2_Nmul<T, T_PTR, true, 8, A_COLS_PTY>,
    kernel_syrk_U_LT_reg_shared_Mblock2_Nvar<T, T_PTR, true, 8, A_COLS_PTY>,
    kernel_syrk_U_LT_reg_shared_NMblock2var <T, T_PTR, true, 8, A_COLS_PTY>,
    kernel_syrk_U_LT_reg_shared_NMblock2var <T, T_PTR, true, 8, A_COLS_PTY>,
    kernel_syrk_U_LN_registers_Mfix_Nvar_DB <T, T_PTR, true, 8, 4>,
    //kernel_syrk_LN_registers_Mblock2_Nvar<T, T_PTR, true, 8, 4>,
    kernel_syrk_U_LN_registers_Mblock2_Nvar_DB<T, T_PTR, true, 8, 4>
  };

  // determine which kernel to launch
  int mvar = (m < 8 || (8 < m && m < 16) || m > 16) && (m % 16 != 0), nvar = 1;//(n % 8) != 0;//(m != 8) && (m != 16);
  func_idx = 8 * (trans == KBLAS_Trans) + 4 * (m > 8) + nvar + 2*mvar;//(n % A_COLS_PTY != 0);
  dim_idx = (m <= 8);
  // printf("func_idx(%d), dim_idx(%d)\n", func_idx, dim_idx);fflush( stdout );

  dim3 blockDim( dims[dim_idx].x, dims[dim_idx].y );
  dim3 gridDim( batchCount / blockDim.y + (batchCount % blockDim.y != 0), (m / 16) + (m % 16 != 0));
  // printf("blockDim(%d,%d), gridDim(%d,%d)\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);fflush( stdout );

  // set dynamic shared memory requirement for each kernel
  long sh_mem = blockDim.x*(blockDim.x)*blockDim.y*sizeof(T);
  long syrk_kernels_sharedMem[] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    sh_mem, sh_mem, sh_mem, sh_mem, sh_mem, sh_mem, sh_mem, sh_mem,
    4*(blockDim.x+1)*blockDim.y*sizeof(T),
    8*(blockDim.x+1)*blockDim.y*sizeof(T)
  };

  //invoke the syrk kernel on diagonal blocks
  syrk_kernels[func_idx]<<< gridDim, blockDim, syrk_kernels_sharedMem[func_idx], handle->stream>>>
                         (m, n, batchCount,
                          alpha, A, 0, 0, lda, strideA,
                          beta,  B, 0, 0, ldb, strideB);

  check_error_ret( hipGetLastError(), KBLAS_UnknownError);

  return KBLAS_Success;
}

//==============================================================================================
// invoke gemm kernel on off diagonal blocks
// workspace needed: device pointers
// A, B: host pointer to array of device pointers to device buffers
template<class T, class T_PTR>
int Xsyrk_gemm_rec_flat(kblasHandle_t handle,
                        char uplo, char trans,
                        const int m, const int n,
                        const T alpha,
                        const T_PTR A, int A_row_off, int A_col_off, int lda,
                        const T beta,
                              T_PTR B, int B_row_off, int B_col_off, int ldb,
                        int batchCount)
{
  //these gemm calls can run in parallel, through streams or merged batch call

  if(m <= 16){
    return KBLAS_Success;
  }

  // determine the depth of the recursion
  int depth = 0, s = 16;
  while(s < m){
    s = s << 1;
    depth++;
  }

  char transA = trans,
       transB = (transA == KBLAS_NoTrans ? KBLAS_Trans : KBLAS_NoTrans);

  // use pre-allocated workspace buffers to host pointers to off-diagonal blocks
  kblasWorkspace_t ws_current = &(handle->work_space);

  T **A_work,
    **B_work,
    **C_work;
  A_work = (T**)(ws_current->d_ptrs);
  B_work = A_work + (1 << (depth-1) ) * batchCount;
  C_work = B_work + (1 << (depth-1) ) * batchCount;

  int d = 0,
      M = m,
      row = CLOSEST_REG_SIZE(m),
      mm = M - row,
      nn = row,
      kk = n;

  // combine calls to batch gemm at the same recursion level into one batch call
  while(d < depth){
    int cur_batchCount = 0;
    int b = 0;
    int uniform_batches = (d == 0) ? 1 : (m / row);

    //TODO merge into one call
    while((b < uniform_batches) && m >= (mm+row+2*b*row) ){
      //append this batch call
      Xset_pointer_3(A_work+b*batchCount, (const T**)(A), A_row_off + (trans == KBLAS_NoTrans)*(row+2*b*row), A_col_off + (trans == KBLAS_Trans)*(row+2*b*row), lda,
                     B_work+b*batchCount, (const T**)(A), A_row_off + (trans == KBLAS_NoTrans)*(2*b*row),     A_col_off + (trans == KBLAS_Trans)*(2*b*row), lda,
                     C_work+b*batchCount, (const T**)(B), B_row_off + (uplo == KBLAS_Lower)*row+2*b*row,      B_col_off + (uplo == KBLAS_Upper)*row+2*b*row, ldb,
                     batchCount, handle->stream);
      cur_batchCount += batchCount;
      b++;
    }
    //issue one batch call
    if(cur_batchCount > 0){
      kblas_gemm_batch( handle,
                        transA, transB,
                        mm, nn, kk,
                        alpha, (const T**)A_work, lda,
                               (const T**)B_work, lda,
                        beta,             C_work, ldb,
                        cur_batchCount);
    }
    if((d > 0) && ((m % row) != 0) && m > (row+2*b*row) ){
      //one block is remaining that is not regular size
      Xset_pointer_3(A_work, (const T**)(A), A_row_off + (trans == KBLAS_NoTrans)*(row+2*b*row), A_col_off + (trans == KBLAS_Trans)*(row+2*b*row), lda,
                     B_work, (const T**)(A), A_row_off + (trans == KBLAS_NoTrans)*(2*b*row),     A_col_off + (trans == KBLAS_Trans)*(2*b*row), lda,
                     C_work, (const T**)(B), B_row_off + (uplo == KBLAS_Lower)*row+2*b*row,      B_col_off + (uplo == KBLAS_Upper)*row+2*b*row, ldb,
                     batchCount, handle->stream);
      //issue one batch call
      kblas_gemm_batch( handle,
                        transA, transB,
                        m - (row+2*b*row), nn, kk,
                        alpha, (const T**)A_work, lda,
                               (const T**)B_work, lda,
                        beta,             C_work, ldb,
                        batchCount);
    }
    d++;
    row /= 2;
    mm = nn = row;
  }
  return KBLAS_Success;
}

//-------------------------------------------------------------------
// batch SYRK core routine
// workspace needed: device pointers
// A, B: host pointer to array of device pointers to device buffers
template<typename T, typename T_PTR>
int Xsyrk_batch_core( kblasHandle_t handle,
                      char uplo, char trans,
                      const int m, const int n,
                      const T alpha,  T_PTR A, int A_row_off, int A_col_off, int lda,
                      const T beta,   T_PTR B, int B_row_off, int B_col_off, int ldb,
                      int batchCount)
{
  //printf("Xsyrk_batch_strided_core\n");
  if( uplo == KBLAS_Upper ){
    printf("Upper SYRK_BATCH is not implemented yet\n");
    return KBLAS_NotImplemented;
  }
  int status;

  //invoke gemm for off diagonal blocks
  if(m > 16){

    // do we have enough workspace allocated
    KBlasWorkspaceState ws_needed;
    syrk_batch_wsquery_core( m, batchCount, (kblasWorkspaceState_t)&ws_needed);

    if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) ){
      return KBLAS_InsufficientWorkspace;
    }

    check_error_ret( status = Xsyrk_gemm_rec_flat( handle,
                                                   uplo, trans,
                                                   m, n,
                                                   alpha, (T_PTR)A, A_row_off, A_col_off, lda,
                                                   beta,  (T_PTR)B, B_row_off, B_col_off, ldb,
                                                   batchCount), status);
  }

  // handle diagonal blocks with custom CUDA kernels
  // process all diagonal tiles in one CUDA kernel launch with 2D grid

  int2 dims[7] = {
    { 8, 4},//1 warps
    { 8, 8},//2 warps
    { 8,12},//3 warps
    { 8,16},//4 warps
    {16, 2},//1 warps
    {16, 4},//2 warps
    {16, 8} //4 warps
  };
  int func_idx = 0;
  int dim_idx = 0;


  typedef void (*syrk_kernels_type)(const int m, const int n, int batchCount,
                                    const T alpha, const T_PTR __restrict__ A_array, int A_row_off, int A_col_off, int lda, long strideA,
                                    const T beta,        T_PTR              B_array, int B_row_off, int B_col_off, int ldb, long strideB);

  syrk_kernels_type syrk_kernels[] = {
    kernel_syrk_U_LN_registers_Mfix_Nmul    <T, T_PTR, false, 8, A_COLS_PTY>,
    kernel_syrk_U_LN_registers_Mfix_Nvar    <T, T_PTR, false, 8, A_COLS_PTY>,
    kernel_syrk_U_LN_registers_MNvar        <T, T_PTR, false, 8, A_COLS_PTY>,
    kernel_syrk_U_LN_registers_MNvar        <T, T_PTR, false, 8, A_COLS_PTY>,
    kernel_syrk_U_LN_registers_Mblock2_Nmul <T, T_PTR, false, 8, A_COLS_PTY>,
    kernel_syrk_U_LN_registers_Mblock2_Nvar <T, T_PTR, false, 8, A_COLS_PTY>,
    kernel_syrk_U_LN_registers_NMblock2var  <T, T_PTR, false, 8, A_COLS_PTY>,
    kernel_syrk_U_LN_registers_NMblock2var  <T, T_PTR, false, 8, A_COLS_PTY>,
    kernel_syrk_U_LT_reg_shared_Mfix_Nmul   <T, T_PTR, false, 8, A_COLS_PTY>,
    kernel_syrk_U_LT_reg_shared_Mfix_Nvar   <T, T_PTR, false, 8, A_COLS_PTY>,
    kernel_syrk_U_LT_reg_shared_MNvar       <T, T_PTR, false, 8, A_COLS_PTY>,
    kernel_syrk_U_LT_reg_shared_MNvar       <T, T_PTR, false, 8, A_COLS_PTY>,
    kernel_syrk_U_LT_reg_shared_Mblock2_Nmul<T, T_PTR, false, 8, A_COLS_PTY>,
    kernel_syrk_U_LT_reg_shared_Mblock2_Nvar<T, T_PTR, false, 8, A_COLS_PTY>,
    kernel_syrk_U_LT_reg_shared_NMblock2var <T, T_PTR, false, 8, A_COLS_PTY>,
    kernel_syrk_U_LT_reg_shared_NMblock2var <T, T_PTR, false, 8, A_COLS_PTY>,
    kernel_syrk_U_LN_registers_Mfix_Nvar_DB <T, T_PTR, false, 8, 4>,
    //kernel_syrk_LN_registers_Mblock2_Nvar<T, T_PTR, false, 8, 4>,
    kernel_syrk_U_LN_registers_Mblock2_Nvar_DB<T, T_PTR, false, 8, 4>
  };

  // determine which kernel to launch
  int mvar = (m < 8 || (8 < m && m < 16) || m > 16) && (m % 16 != 0), nvar = 1;//(n % 8) != 0;//(m != 8) && (m != 16);
  func_idx = 8 * (trans == KBLAS_Trans) + 4 * (m > 8) + nvar + 2*mvar;//(n % A_COLS_PTY != 0);
  dim_idx = (m <= 8);
  // printf("func_idx(%d), dim_idx(%d)\n", func_idx, dim_idx);fflush( stdout );

  dim3 blockDim( dims[dim_idx].x, dims[dim_idx].y );
  dim3 gridDim( batchCount / blockDim.y + (batchCount % blockDim.y != 0), (m / 16) + (m % 16 != 0));
  // printf("blockDim(%d,%d), gridDim(%d,%d)\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);fflush( stdout );

  // set dynamic shared memory requirement for each kernel
  long sh_mem = blockDim.x*(blockDim.x)*blockDim.y*sizeof(T);
  long syrk_kernels_sharedMem[] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    sh_mem, sh_mem, sh_mem, sh_mem, sh_mem, sh_mem, sh_mem, sh_mem,
    4*(blockDim.x+1)*blockDim.y*sizeof(T),
    8*(blockDim.x+1)*blockDim.y*sizeof(T)
  };

  //invoke the syrk kernel on diagonal blocks
  syrk_kernels[func_idx]<<< gridDim, blockDim, syrk_kernels_sharedMem[func_idx], handle->stream>>>
                       (m, n, batchCount,
                        alpha, A, A_row_off, A_col_off, lda, 0,
                        beta,  B, B_row_off, B_col_off, ldb, 0);

  check_error_ret( hipGetLastError(), KBLAS_UnknownError);

  return KBLAS_Success;
}

//==============================================================================================
template<class T>
int Xsyrk_batch_nonuniform_core(kblasHandle_t handle,
                                char uplo, char trans,
                                int *m, int *n,
                                T alpha, T** A, int *lda,
                                T beta,  T** B, int *ldb,
                                int max_m, int max_n,
                                int batchCount)
{
  if(handle->use_magma){
  #ifdef USE_MAGMA

    //TODO: it might be better to look up the maximum per 65k chunck, except that synchromizations will be forced
    // if(batchCount > 65535) return KBLAS_Error_WrongInput;
    KBlasWorkspaceState ws_needed;
    syrk_batch_nonuniform_wsquery_core((kblasWorkspaceState_t)&ws_needed);

    if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) ){
      return KBLAS_InsufficientWorkspace;
    }

    int h_max_mn[2];
    kblasWorkspace_t ws_current = &(handle->work_space);
    int* d_max_mn = (int*)(ws_current->d_data);

    //take care of batch size limitation with magma
    int batch_increment = 65535;
    int batch_start = 0;
    if(max_m > 0 || max_n > 0){
      h_max_mn[0] = max_m;
      h_max_mn[1] = max_n;
    }

    while(batch_start != batchCount)
    {
      int batch_size = kmin(batch_increment, batchCount - batch_start);

      if((batchCount > batch_increment) || (max_m <= 0 && max_n <= 0)){
        // compute the max. dimensions
        kblas_imax_size_2(handle, m, n, *d_max_mn, *(d_max_mn+1), batch_size);
        check_error_ret( cublasGetVectorAsync( 2, sizeof(int), d_max_mn, 1, h_max_mn, 1, handle->stream ), KBLAS_cuBLAS_Error);
        check_error_ret( hipStreamSynchronize(handle->stream), KBLAS_CUDA_Error );
      }
      magmablas_Xsyrk_vbatched_max_nocheck(
                  (magma_uplo_t)(uplo == KBLAS_Lower ? MagmaLower : MagmaUpper),
                  (magma_trans_t)(trans == KBLAS_Trans ? MagmaTrans : MagmaNoTrans),
                  m, n,
                  alpha, A, lda,
                  beta,  B, ldb,
                  batch_size,
                  h_max_mn[0], h_max_mn[1], handle->magma_queue);

      A += batch_size;
      B += batch_size;
      m += batch_size;
      n += batch_size;
      lda += batch_size;
      ldb += batch_size;

      batch_start += batch_size;
      check_error_ret( hipGetLastError(), KBLAS_MAGMA_Error);
    }
  #else
    printf("Configuration error at %s in file %s at line %d, MAGMA required but KBLAS not compiled with it!\n", __func__, __FILE__, __LINE__ );
    return KBLAS_WrongConfig;
  #endif
  }
  else
  if(!handle->use_magma){
    printf("Configuration error at %s in file %s at line %d, MAGMA required but not enabled!\n", __func__, __FILE__, __LINE__ );
    return KBLAS_WrongConfig;
  }

  return KBLAS_Success;
}

#endif //__XSYRK_BATCH_DRIVERS_H__
