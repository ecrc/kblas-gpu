/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_svd/Xsvd_full_batch_core.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ali Charara
 * @date 2020-12-10
 **/

#ifndef __XSVD_FULL_CORE__
#define __XSVD_FULL_CORE__


//==============================================================================================
// #define DBG_MSG
//==============================================================================================
// forward declaration, this is an ugly hack
// template<class T>
// void batch_tall_svd_workspace(int rows, int cols, int num_ops, KBlasWorkspaceState& requested_ws, int top_level);
// // template<>
// // void batch_tall_svd_workspace<TYPE>(int rows, int cols, int num_ops, KBlasWorkspaceState& requested_ws, int top_level);
// template<class T>
// void batch_svd_osbj_workspace(int rows, int cols, int num_ops, KBlasWorkspaceState& requested_ws, int top_level);
// template<>
// void batch_svd_osbj_workspace<TYPE>(int rows, int cols, int num_ops, KBlasWorkspaceState& requested_ws, int top_level);



//==============================================================================================
// workspace needed: @see svd_full_batch_wsquery_core()
// on input:  A: host pointer to device buffer, contains the input strided matrices
//            S,U,V: host pointers to device buffers, preallocated
//            rank: if > 0, the rank required
//            variant: which variant of svd to run, may be overiden based on matrix size
//            m,n: matrix dimention should conform to limitation of variant selected
// on output: S: contains singular values
//            U: contains right singular vectors (up to rank if rank > 0)
//            V: contains left singular vectors scaled by S (up to rank if rank > 0)
//            A: not modified
//TODO redundant, use the templatized one below
template<class T>
int Xsvd_full_batch_core( kblasHandle_t handle,
                          int m, int n, int rank,
                          T* d_A, int lda, int stride_a,
                          T* d_S, int stride_s,
                          T* d_U, int ldu, int stride_u,
                          T* d_V, int ldv, int stride_v,
                          SVD_method variant,
                          kblasRandState_t rand_state,
                          int batchCount)
{
  T zero = make_zero<T>(),
    one = make_one<T>();

  //TODO verify input
  if((int)variant > 2 || (rank <= 0)) {
    return KBLAS_NotImplemented;
  }
  int trim_rank = 0;
  if(rank > 0){
    trim_rank = rank;
  }else{
    //TODO pick the rank
    //may be we need to choose accuracy
    return KBLAS_NotImplemented;
  }

  //verify enough workspace is available
  KBlasWorkspaceState ws_needed;
  svd_full_batch_wsquery_core<T, true, true>( m, n, rank, batchCount, variant,
                                        (kblasWorkspaceState_t)&ws_needed);
  if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) )
    return KBLAS_InsufficientWorkspace;


  //copy A onto temporary buffer, original A is needed later
  int lda_cached = kblas_roundup(m, align);
  int stride_a_cached = lda_cached * n;
  size_t d_ws_data = (ws_needed.d_data_bytes) / sizeof(T) - stride_a_cached * size_t(batchCount);
  T* d_A_cached = (T*)handle->work_space.d_data + d_ws_data;

  check_ret_error( kblas_copyBlock_batch( handle,
                                          m, n,
                                          d_A_cached, 0, 0, lda_cached, stride_a_cached,
                                          d_A,        0, 0, lda, stride_a,
                                          batchCount));

  //perform batch svd to generate U & S
  if(variant == SVD_Jacobi){
    check_ret_error( kblas_gesvj_batch( handle,
                                        m, n,
                                        d_A_cached, lda_cached, stride_a_cached,
                                        d_S, stride_s,
                                        batchCount));
  }else
  if(variant == SVD_Jacobi_gram){
    check_ret_error( kblas_gesvj_gram_batch(handle,
                                            m, n,
                                            d_A_cached, lda_cached, stride_a_cached,
                                            d_S, stride_s,
                                            batchCount));
  }else
  if(variant == SVD_random){
    check_ret_error( kblas_rsvd_batch(handle,
                                      m, n, rank,
                                      d_A_cached, lda_cached, stride_a_cached,
                                      d_S, stride_s,
                                      rand_state,
                                      batchCount));
  }
  else{
    return KBLAS_NotImplemented;
  }

  //copy U from A_cached onto d_U
  // d_U buffer should have enough space to fit U
  check_ret_error( kblas_copyBlock_batch( handle,
                                          m, trim_rank,
                                          d_U,      0, 0, ldu, stride_u,
                                          d_A_cached, 0, 0, lda_cached, stride_a_cached,
                                          batchCount));

  // printf("here %d\n", __LINE__);
  //perform gemm to regenerate V
  //A = U * S * V**T
  //U**T * A = S * V**T = V'**T
  //we want (V'**T) stored in transpose form
  //AT * U = V'
  //which is a gemm
  check_ret_error( kblas_gemm_batch(handle,
                                    KBLAS_Trans, KBLAS_NoTrans,
                                    n, trim_rank, m,
                                    one,  d_A, lda, stride_a,
                                          d_U, ldu, stride_u,
                                    zero, d_V, ldv, stride_v,
                                    batchCount));

  return KBLAS_Success;
}

//==============================================================================================
// workspace needed: @see svd_full_batch_wsquery_core()
// on input:  A: host pointer to device buffer, contains the input strided matrices
//            S,U,V: host pointers to device buffers, preallocated
//            rank: if > 0, the rank required
//            variant: which variant of svd to run, may be overiden based on matrix size
//            m,n: matrix dimention should conform to limitation of variant selected
// on output: S: contains singular values
//            U: contains right singular vectors (up to rank if rank > 0)
//            V: contains left singular vectors scaled by S (up to rank if rank > 0)
//            A: not modified
template<class T, class T_PTR, bool STRIDED>
int Xsvd_full_batch_core( kblasHandle_t handle,
                          int m, int n, int rank,
                          T_PTR d_A, int lda, int stride_a,
                          T_PTR d_S, int stride_s,
                          T_PTR d_U, int ldu, int stride_u,
                          T_PTR d_V, int ldv, int stride_v,
                          SVD_method variant,
                          kblasRandState_t rand_state,
                          int batchCount)
{
  T zero = make_zero<T>(),
    one = make_one<T>();

  //TODO verify input
  if((int)variant > 2 || (rank <= 0)) {
    return KBLAS_NotImplemented;
  }
  int trim_rank = 0;
  if(rank > 0){
    trim_rank = rank;
  }else{
    //TODO pick the rank
    //may be we need to choose accuracy
    return KBLAS_NotImplemented;
  }

  //verify enough workspace is available
  KBlasWorkspaceState ws_needed;
  svd_full_batch_wsquery_core<T, STRIDED, true>(m, n, rank, batchCount, variant,
                                                (kblasWorkspaceState_t)&ws_needed);
  if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) )
    return KBLAS_InsufficientWorkspace;


  //copy A onto temporary buffer, original A is needed later
  int lda_cached = m;
  int stride_a_cached = lda_cached * n;
  size_t d_ws_data = (ws_needed.d_data_bytes) / sizeof(T) - stride_a_cached * size_t(batchCount);

  T* work = (T*)handle->work_space.d_data + d_ws_data;
  T **work_ptrs;

  if(!STRIDED){
    size_t d_ws_ptrs = (ws_needed.d_ptrs_bytes) / sizeof(T*) - size_t(batchCount);
    work_ptrs = (T**)handle->work_space.d_ptrs + d_ws_ptrs;
    check_ret_error( Xset_pointer_1(work_ptrs, work, lda, stride_a_cached,
                                    batchCount, handle->stream) );
  }

  T_PTR d_A_cached;
  if(STRIDED){
    d_A_cached = (T_PTR)work;
  }else{
    d_A_cached = (T_PTR)work_ptrs;
  }

  check_ret_error( kblas_copyBlock_batch( handle,
                                          m, n,
                                          d_A_cached, 0, 0, lda_cached, stride_a_cached,
                                          d_A,        0, 0, lda, stride_a,
                                          batchCount));

  //perform batch svd to generate U & S
  if(variant == SVD_Jacobi){
    check_ret_error( kblas_gesvj_batch( handle,
                                        m, n,
                                        d_A_cached, lda_cached, stride_a_cached,
                                        d_S, stride_s,
                                        batchCount));
  }else
  if(variant == SVD_Jacobi_gram){
    check_ret_error( kblas_gesvj_gram_batch(handle,
                                            m, n,
                                            d_A_cached, lda_cached, stride_a_cached,
                                            d_S, stride_s,
                                            batchCount));
  }else
  if(variant == SVD_random){
    check_ret_error( kblas_rsvd_batch(handle,
                                      m, n, rank,
                                      d_A_cached, lda_cached, stride_a_cached,
                                      d_S, stride_s,
                                      rand_state,
                                      batchCount));
  }
  else{
    return KBLAS_NotImplemented;
  }

  //copy U from A_cached onto d_U
  // d_U buffer should have enough space to fit U
  check_ret_error( kblas_copyBlock_batch( handle,
                                          m, trim_rank,
                                          d_U,      0, 0, ldu, stride_u,
                                          d_A_cached, 0, 0, lda_cached, stride_a_cached,
                                          batchCount));

  // printf("here %d\n", __LINE__);
  //perform gemm to regenerate V
  //A = U * S * V**T
  //U**T * A = S * V**T = V'**T
  //we want (V'**T) stored in transpose form
  //AT * U = V'
  //which is a gemm
  check_ret_error( Xgemm_batch( handle,
                                KBLAS_Trans, KBLAS_NoTrans,
                                n, trim_rank, m,
                                one,  d_A, lda, stride_a,
                                      d_U, ldu, stride_u,
                                zero, d_V, ldv, stride_v,
                                batchCount));

  return KBLAS_Success;
}

//==============================================================================================
//TODO optimize
template<typename T>
__global__
void kernel_get_rank(int* output_ranks, const T** sigma, double tol, int max_rank)
{
  int rank = 0;
  while(rank < max_rank && sigma[blockIdx.x][rank] >= tol)
    rank++;
  output_ranks[blockIdx.x] = rank;
}

template<typename T>
inline
int Xget_ranks( int* output_ranks, const T** sigma, double tol, int max_rank,
                long batchCount, cudaStream_t cuda_stream)
{
  kernel_get_rank<T><<< batchCount, 1, 0, cuda_stream>>>(
                  output_ranks, sigma, tol, max_rank);
  check_error_ret( cudaGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

//==============================================================================================
template<class T>
int Xsvd_full_batch_nonuniform_core(kblasHandle_t handle,
                                    int m, int n,
                                    int* m_array, int* n_array,
                                    T** d_A, int lda, int* lda_array,
                                    T** d_S, //int stride_s,
                                    T** d_U, int ldu, int* ldu_array,
                                    T** d_V, int ldv, int* ldv_array,
                                    SVD_method variant,
                                    kblasRandState_t rand_state,
                                    int batchCount,
                                    int max_rank, double tolerance,
                                    int* ranks_array)
{
  const T zero = make_zero<T>(),
           one = make_one<T>();

  //TODO verify input
  if((int)variant > 2 || (max_rank <= 0) || (tolerance <= 0)){
    return KBLAS_Error_WrongInput;
  }


  //verify enough workspace is available
  KBlasWorkspaceState ws_full, ws_child;
  svd_full_batch_wsquery_core<T, false, false>( m, n, max_rank, batchCount, variant,
                                              (kblasWorkspaceState_t)&ws_full,
                                              (kblasWorkspaceState_t)&ws_child);
  if( !ws_full.isSufficient( &(handle->work_space.allocated_ws_state) ) )
    return KBLAS_InsufficientWorkspace;

  size_t d_data_bytes = 0;

  //copy A onto temporary buffer, original A is needed later
  int lda_cached = kblas_roundup(m, align),
      stride_a_cached = lda_cached * n;

  char* d_ws_data = (char*)(handle->work_space.d_data);
  d_data_bytes += kblas_roundup_s(ws_child.d_data_bytes, sizeof(T));

  // int d_ws_data = (ws_child.d_data_bytes) / sizeof(T) - (stride_a_cached * batchCount); //ECHO_I(d_ws_data);
  T* d_A_cached = (T*)((char*)d_ws_data + d_data_bytes);

  size_t d_child_ptrs = (ws_child.d_ptrs_bytes) / sizeof(T*);
  T** d_A_ptrs = (T**)handle->work_space.d_ptrs + d_child_ptrs;

  check_ret_error( Xset_pointer_1(d_A_ptrs, d_A_cached, lda_cached, stride_a_cached,
                                  batchCount, handle->stream) );

  check_ret_error( kblas_copyBlock_batch( handle,
                                          m, n,
                                          d_A_ptrs, 0, 0, lda_cached,
                                          d_A,      0, 0, lda,
                                          batchCount));

  //perform batch svd to generate U & S
  if(variant == SVD_Jacobi){
    check_ret_error( kblas_gesvj_batch( handle,
                                        m, n,
                                        d_A_ptrs, lda_cached,
                                        d_S,
                                        batchCount));
  }else
  if(variant == SVD_Jacobi_gram){
    check_ret_error( kblas_gesvj_gram_batch(handle,
                                            m, n,
                                            d_A_ptrs, lda_cached,
                                            d_S,
                                            batchCount));
  }else
  if(variant == SVD_random){
    check_ret_error( kblas_rsvd_batch(handle,
                                      m, n, max_rank,
                                      d_A_ptrs, lda_cached,
                                      d_S,
                                      rand_state,
                                      batchCount));
  }
  else{
    return KBLAS_NotImplemented;
  }

  //find ranks based on tolerance
  check_ret_error( Xget_ranks(ranks_array, (const T**)d_S, tolerance, max_rank,
                              batchCount, handle->stream) );

  //copy U from A_cached onto d_U
  // d_U buffer should have enough space to fit U
  check_ret_error( kblas_copyBlock_batch( handle,
                                          m, max_rank,
                                          d_U,      0, 0, ldu,
                                          d_A_ptrs, 0, 0, lda_cached,
                                          batchCount));


  // printf("here %d\n", __LINE__);
  //perform gemm to regenerate V
  //A = U * S * V**T
  //U**T * A = S * V**T = V'**T
  //we want (V'**T) stored in transpose form
  //AT * U = V'
  //which is a gemm
  check_ret_error( kblas_gemm_batch(handle,
                                    KBLAS_Trans, KBLAS_NoTrans,
                                    n_array, ranks_array, m_array,
                                    n, max_rank, m,
                                    one,  (const T**)d_A, lda_array,
                                          (const T**)d_U, ldu_array,
                                    zero,       (T**)d_V, ldv_array,
                                    batchCount ) );

  return KBLAS_Success;
}

#endif //__XSVD_FULL_CORE__
