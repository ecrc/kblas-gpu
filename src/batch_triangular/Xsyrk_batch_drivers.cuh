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
 * @version 2.0.0
 * @author Ali Charara
 * @date 2017-11-13
 **/

#ifndef __XSYRK_BATCH_DRIVERS_H__
#define __XSYRK_BATCH_DRIVERS_H__


#include "Xsyrk_batch_kernels.cuh"

//==============================================================================================
//TODO tuning variable
#define A_COLS_PTY 8

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
  //these gemm calls can run in parallel, through merged batch call

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
                        m - mm*b, nn, kk,
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
// workspace needed: device pointers
// A, B: host pointer to device buffers
template<class T>
int Xsyrk_batch_strided_core( kblasHandle_t handle,
                              char uplo, char trans,
                              const int m, const int n,
                              const T alpha, const T* A, int lda, long strideA,
                              const T beta,        T* B, int ldb, long strideB,
                              int batchCount)
{
  if( uplo == KBLAS_Upper ){
    printf("Upper SYRK_BATCH is not implemented yet\n");
    return KBLAS_NotImplemented;
  }
  int status;

  if(m > 16){
    KBlasWorkspaceState ws_needed;
    syrk_batch_wsquery_core( m, batchCount, (kblasWorkspaceState_t)&ws_needed);

    bool suffWorkspace = (ws_needed.d_ptrs_bytes <= handle->work_space.allocated_ws_state.d_ptrs_bytes);

    if(!suffWorkspace){
      return KBLAS_InsufficientWorkspace;
    }

    // if(0){// &&(m % 16 != 0)){
    //   check_error_ret(status = Xsyrk_gemm_rec_strided( handle,
    //                                                    uplo, trans,
    //                                                    m, n,
    //                                                    alpha, A, lda, strideA,
    //                                                    beta,  B, ldb, strideB,
    //                                                    batchCount), status);
    //   printf(" rec ");
    // }
    // else
      check_error_ret(status = Xsyrk_gemm_rec_flat_strided(handle,
                                                           uplo, trans,
                                                           m, n,
                                                           alpha, A, lda, strideA,
                                                           beta,  B, ldb, strideB,
                                                           batchCount), status);
  }

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

  //if( m % 8 == 0 && m <= 16)
  if(1)
  {

    typedef void (*syrk_kernels_type)( const int m, const int n, int batchCount,
                                       const T alpha, const T* __restrict__ A_array, int lda, long strideA,
                                       const T beta, T* B_array, int ldb, long strideB);

    syrk_kernels_type syrk_kernels[] = {
      kernel_syrk_US_LN_registers_Mfix_Nmul   <T, 8, A_COLS_PTY>,
      kernel_syrk_US_LN_registers_Mfix_Nvar   <T, 8, A_COLS_PTY>,
      kernel_syrk_US_LN_registers_MNvar       <T, 8, A_COLS_PTY>,
      kernel_syrk_US_LN_registers_MNvar       <T, 8, A_COLS_PTY>,
      kernel_syrk_US_LN_registers_Mblock2_Nmul<T, 8, A_COLS_PTY>,
      kernel_syrk_US_LN_registers_Mblock2_Nvar<T, 8, A_COLS_PTY>,
      kernel_syrk_US_LN_registers_NMblock2var <T, 8, A_COLS_PTY>,
      kernel_syrk_US_LN_registers_NMblock2var <T, 8, A_COLS_PTY>,
      kernel_syrk_US_LT_reg_shared_Mfix_Nmul   <T, 8, A_COLS_PTY>,
      kernel_syrk_US_LT_reg_shared_Mfix_Nvar   <T, 8, A_COLS_PTY>,
      kernel_syrk_US_LT_reg_shared_MNvar       <T, 8, A_COLS_PTY>,
      kernel_syrk_US_LT_reg_shared_MNvar       <T, 8, A_COLS_PTY>,
      kernel_syrk_US_LT_reg_shared_Mblock2_Nmul<T, 8, A_COLS_PTY>,
      kernel_syrk_US_LT_reg_shared_Mblock2_Nvar<T, 8, A_COLS_PTY>,
      kernel_syrk_US_LT_reg_shared_NMblock2var <T, 8, A_COLS_PTY>,
      kernel_syrk_US_LT_reg_shared_NMblock2var <T, 8, A_COLS_PTY>,
      kernel_syrk_US_LN_registers_Mfix_Nvar_DB <T, 8, 4>,
      //kernel_syrk_LN_registers_Mblock2_Nvar<T, 8, 4>,
      kernel_syrk_US_LN_registers_Mblock2_Nvar_DB<T, 8, 4>
    };
    int mvar = (m < 8 || (8 < m && m < 16) || m > 16) && (m % 16 != 0), nvar = 1;//(n % 8) != 0;//(m != 8) && (m != 16);
    func_idx = 8 * (trans == KBLAS_Trans) + 4 * (m > 8) + nvar + 2*mvar;//(n % A_COLS_PTY != 0);
    dim_idx = (m <= 8);
    // printf("func_idx(%d), dim_idx(%d)\n", func_idx, dim_idx);fflush( stdout );
    dim3 blockDim( dims[dim_idx].x, dims[dim_idx].y );
    dim3 gridDim( batchCount / blockDim.y + (batchCount % blockDim.y != 0), (m / 16) + (m % 16 != 0));
    // printf("blockDim(%d,%d), gridDim(%d,%d)\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);fflush( stdout );
    long sh_mem = blockDim.x*(blockDim.x)*blockDim.y*sizeof(T);
    long syrk_kernels_sharedMem[] = {
      0, 0, 0, 0, 0, 0, 0, 0,
      sh_mem, sh_mem, sh_mem, sh_mem, sh_mem, sh_mem, sh_mem, sh_mem,
      4*(blockDim.x+1)*blockDim.y*sizeof(T),
      8*(blockDim.x+1)*blockDim.y*sizeof(T)
    };

    syrk_kernels[func_idx]<<< gridDim, blockDim, syrk_kernels_sharedMem[func_idx], handle->stream>>>
                          (m, n, batchCount, alpha, A, lda, strideA, beta, B, ldb, strideB);

    check_error_ret( cudaGetLastError(), KBLAS_UnknownError);
  }else
  {
    return KBLAS_NotImplemented;
  }
  return KBLAS_Success;
}

//==============================================================================================

// workspace needed: device pointers
// d_A, d_B: host pointer to array of device pointers to device buffers
template<class T>
int Xsyrk_gemm_rec_flat(kblasHandle_t handle,
                        char uplo, char trans,
                        const int m, const int n,
                        const T alpha,
                        const T** A, int A_row_off, int A_col_off, int lda,
                        const T beta,
                              T** B, int B_row_off, int B_col_off, int ldb,
                        int batchCount)
{
  //these gemm calls can run in parallel, through streams or merged batch call

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
                        m - mm*b, nn, kk,
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
// workspace needed: device pointers
// A, B: host pointer to array of device pointers to device buffers
template<class T>
int Xsyrk_batch_core( kblasHandle_t handle,
                      char uplo, char trans,
                      const int m, const int n,
                      const T alpha, const T** A, int A_row_off, int A_col_off, int lda,
                      const T beta,        T** B, int B_row_off, int B_col_off, int ldb,
                      int batchCount)
{
  //printf("Xsyrk_batch_strided_core\n");
  if( uplo == KBLAS_Upper ){
    printf("Upper SYRK_BATCH is not implemented yet\n");
    return KBLAS_NotImplemented;
  }
  int status;

  if(m > 16){

    KBlasWorkspaceState ws_needed;
    syrk_batch_wsquery_core( m, batchCount, (kblasWorkspaceState_t)&ws_needed);

    bool suffWorkspace = (ws_needed.d_ptrs_bytes <= handle->work_space.allocated_ws_state.d_ptrs_bytes);

    if(!suffWorkspace){
      return KBLAS_InsufficientWorkspace;
    }

    // if(0 && !REG_SIZE(m)){
    //   check_error_ret(status = Xsyrk_gemm_rec( handle,
    //                                            uplo, trans,
    //                                            m, n,
    //                                            alpha, A, lda,
    //                                            beta,  B, ldb,
    //                                            batchCount), status);
    // }
    // else
      check_error_ret( status = Xsyrk_gemm_rec_flat( handle,
                                                     uplo, trans,
                                                     m, n,
                                                     alpha, A, A_row_off, A_col_off, lda,
                                                     beta,  B, B_row_off, B_col_off, ldb,
                                                     batchCount), status);
  }


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

  //if( m % 8 == 0 && m <= 16)
  if(1)
  {

    typedef void (*syrk_kernels_type)( const int m, const int n, int batchCount,
                                       const T alpha, const T** __restrict__ A_array, int A_row_off, int A_col_off, int lda,
                                       const T beta,                     T** B_array, int B_row_off, int B_col_off, int ldb);

    syrk_kernels_type syrk_kernels[] = {
      kernel_syrk_UN_LN_registers_Mfix_Nmul   <T, 8, A_COLS_PTY>,
      kernel_syrk_UN_LN_registers_Mfix_Nvar   <T, 8, A_COLS_PTY>,
      kernel_syrk_UN_LN_registers_MNvar       <T, 8, A_COLS_PTY>,
      kernel_syrk_UN_LN_registers_MNvar       <T, 8, A_COLS_PTY>,
      kernel_syrk_UN_LN_registers_Mblock2_Nmul<T, 8, A_COLS_PTY>,
      kernel_syrk_UN_LN_registers_Mblock2_Nvar<T, 8, A_COLS_PTY>,
      kernel_syrk_UN_LN_registers_NMblock2var <T, 8, A_COLS_PTY>,
      kernel_syrk_UN_LN_registers_NMblock2var <T, 8, A_COLS_PTY>,
      kernel_syrk_UN_LT_reg_shared_Mfix_Nmul   <T, 8, A_COLS_PTY>,
      kernel_syrk_UN_LT_reg_shared_Mfix_Nvar   <T, 8, A_COLS_PTY>,
      kernel_syrk_UN_LT_reg_shared_MNvar       <T, 8, A_COLS_PTY>,
      kernel_syrk_UN_LT_reg_shared_MNvar       <T, 8, A_COLS_PTY>,
      kernel_syrk_UN_LT_reg_shared_Mblock2_Nmul<T, 8, A_COLS_PTY>,
      kernel_syrk_UN_LT_reg_shared_Mblock2_Nvar<T, 8, A_COLS_PTY>,
      kernel_syrk_UN_LT_reg_shared_NMblock2var <T, 8, A_COLS_PTY>,
      kernel_syrk_UN_LT_reg_shared_NMblock2var <T, 8, A_COLS_PTY>,
      kernel_syrk_UN_LN_registers_Mfix_Nvar_DB <T, 8, 4>,
      //kernel_syrk_LN_registers_Mblock2_Nvar<T, 8, 4>,
      kernel_syrk_UN_LN_registers_Mblock2_Nvar_DB<T, 8, 4>
    };
    int mvar = (m < 8 || (8 < m && m < 16) || m > 16) && (m % 16 != 0), nvar = 1;//(n % 8) != 0;//(m != 8) && (m != 16);
    func_idx = 8 * (trans == KBLAS_Trans) + 4 * (m > 8) + nvar + 2*mvar;//(n % A_COLS_PTY != 0);
    dim_idx = (m <= 8);
    // printf("func_idx(%d), dim_idx(%d)\n", func_idx, dim_idx);fflush( stdout );

    dim3 blockDim( dims[dim_idx].x, dims[dim_idx].y );
    dim3 gridDim( batchCount / blockDim.y + (batchCount % blockDim.y != 0), (m / 16) + (m % 16 != 0));

    // printf("blockDim(%d,%d), gridDim(%d,%d)\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);fflush( stdout );
    long sh_mem = blockDim.x*(blockDim.x)*blockDim.y*sizeof(T);
    long syrk_kernels_sharedMem[] = {
      0, 0, 0, 0, 0, 0, 0, 0,
      sh_mem, sh_mem, sh_mem, sh_mem, sh_mem, sh_mem, sh_mem, sh_mem,
      4*(blockDim.x+1)*blockDim.y*sizeof(T),
      8*(blockDim.x+1)*blockDim.y*sizeof(T)
    };

    syrk_kernels[func_idx]<<< gridDim, blockDim, syrk_kernels_sharedMem[func_idx], handle->stream>>>
                         (m, n, batchCount,
                          alpha, A, A_row_off, A_col_off, lda,
                          beta,  B, B_row_off, B_col_off, ldb);

    check_error_ret( cudaGetLastError(), KBLAS_UnknownError);
  }else
  {
    return KBLAS_NotImplemented;
  }
  return KBLAS_Success;
}

#endif //__XSYRK_BATCH_DRIVERS_H__
