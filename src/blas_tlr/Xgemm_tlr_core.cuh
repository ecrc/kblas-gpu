/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_tlr/Xgemm_tlr_core.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef __XGEMM_TLR_CORE__
#define __XGEMM_TLR_CORE__

//==============================================================================================
// workspace needed: device data
// Au, Av, Bu, Bv, C: host pointers to device buffers
// handles the non-transpose A & B case,
// for transposition you need to swap Au/Av or Bu/Bv, or call @Xgemm_lr
template<class T>
int Xgemm_LR_core(kblasHandle_t handle,
                  const int M, const int N, const int K,
                  const T alpha,
                  const T* Au, int ldAu, const T* Av, int ldAv, int kA,
                  const T* Bu, int ldBu, const T* Bv, int ldBv, int kB,
                  const T beta,
                        T* C, int ldC)
{
  //TODO input validation
  T zero = make_zero<T>();
  T one = make_one<T>();
  //T mone = zero - one;
  int ldW = kblas_roundup(kA, align);

  int work1_size = ldW * kB;

  KBlasWorkspaceState ws_needed;
  gemm_lr_wsquery_core<T>(N, kA, kB,
                          (kblasWorkspaceState_t)&ws_needed);

  if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) )
    return KBLAS_InsufficientWorkspace;

  T  *work1 = (T*)handle->work_space.d_data,
     *work2 = work1 + work1_size;
  int status;

  check_error_ret(status = kblasXgemm(handle,
                                      KBLAS_Trans, KBLAS_NoTrans,
                                      kA, kB, K,
                                      one, (const T*) Av, ldAv,
                                           (const T*) Bu, ldBu,
                                      zero,           work1, ldW), status);
  check_error_ret(status = kblasXgemm(handle,
                                      KBLAS_NoTrans, KBLAS_Trans,
                                      kA, N, kB,
                                      one, (const T*) work1, ldW,
                                           (const T*) Bv, ldBv,
                                      zero,           work2, ldW), status);
  check_error_ret(status = kblasXgemm(handle,
                                      KBLAS_NoTrans, KBLAS_NoTrans,
                                      M, N, kA,
                                      alpha, (const T*) Au, ldAu,
                                             (const T*) work2, ldW,
                                      beta,             C, ldC), status);
  return KBLAS_Success;
}

//==============================================================================================
// workspace needed: device data & pointers
// Au, Av, Bu, Bv, C: host pointers to device buffers
template<class T, class T_PTR, bool STRIDED>
int Xgemm_LR_batch_core( kblasHandle_t handle,
                          const int M, const int N, const int K,
                          const T alpha,
                          const T_PTR Au, int ldAu, long strideAu,
                          const T_PTR Av, int ldAv, long strideAv, int kA,
                          const T_PTR Bu, int ldBu, long strideBu,
                          const T_PTR Bv, int ldBv, long strideBv, int kB,
                          const T beta,
                                T_PTR C,  int ldC,  long strideC,
                          int batchCount)
{
  //TODO input validation
  if(batchCount < 1)
    return KBLAS_Error_WrongInput;

  T zero = make_zero<T>();
  T one = make_one<T>();

  int ldW = kblas_roundup(kA, align),
      strideW1 = ldW * kB,
      strideW2 = ldW * N;

  KBlasWorkspaceState ws_needed;
  gemm_lr_batch_wsquery_core<T, STRIDED>( N, kA, kB, batchCount,
                                          (kblasWorkspaceState_t)&ws_needed);

  if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) )
    return KBLAS_InsufficientWorkspace;

  T *work1 = (T*)handle->work_space.d_data,
    *work2 = work1 + strideW1 * size_t(batchCount);

  T **work1_ptrs, **work2_ptrs;

  if(!STRIDED){
    work1_ptrs = (T**)handle->work_space.d_ptrs;
    work2_ptrs = work1_ptrs + batchCount;
    check_error_ret( Xset_pointer_2(work1_ptrs, work1, ldW, strideW1,
                                    work2_ptrs, work2, ldW, strideW2,
                                    batchCount, handle->stream), KBLAS_UnknownError);
  }

  T_PTR W1;
  T_PTR W2;
  if(STRIDED){
    W1 = (T_PTR)work1;
    W2 = (T_PTR)work2;
  }else{
    W1 = (T_PTR)work1_ptrs;
    W2 = (T_PTR)work2_ptrs;
  }

  check_ret_error( Xgemm_batch( handle,
                                KBLAS_Trans, KBLAS_NoTrans,
                                kA, kB, K,
                                one,  (T_PTR) Av, ldAv, strideAv,
                                      (T_PTR) Bu, ldBu, strideBu,
                                zero, (T_PTR) W1, ldW , strideW1,
                                batchCount) );

  check_ret_error( Xgemm_batch( handle,
                                KBLAS_NoTrans, KBLAS_Trans,
                                kA, N, kB,
                                one,  (T_PTR) W1, ldW , strideW1,
                                      (T_PTR) Bv, ldBv, strideBv,
                                zero, (T_PTR) W2, ldW , strideW2,
                                batchCount) );

  check_ret_error( Xgemm_batch( handle,
                                KBLAS_NoTrans, KBLAS_NoTrans,
                                M, N, kA,
                                alpha, (T_PTR) Au, ldAu, strideAu,
                                       (T_PTR) W2, ldW , strideW2,
                                beta,  (T_PTR) C,  ldC , strideC,
                                batchCount) );

  return KBLAS_Success;
}


//==============================================================================================
// computes C = A * B
template<class T>
int Xgemm_LR_core( kblasHandle_t handle,
                    const int M, const int N, const int K,
                    const T alpha,
                    const T* Au, int ldAu, const T* Av, int ldAv, int kA,
                    const T* Bu, int ldBu, const T* Bv, int ldBv, int kB,
                    const T beta,
                          T* Cu, int ldCu,       T* Cv, int ldCv, int& kC,
                    int max_rk, double max_acc,
                    T* d_workspace)
{
  cudaStream_t cuda_stream[3];
  cuda_stream[0] = handle->stream;
  cuda_stream[1] = handle->stream;//TODO
  cuda_stream[2] = handle->stream;//TODO

  T zero = make_zero<T>();
  T one = make_one<T>();

  //----------------------------
  int CUV_ncols = kC + kA;

  T* AvtBu = d_workspace;
  int ldAvtBu = kblas_roundup(kA, align);
  d_workspace += ldAvtBu * kB;

  T* qrtauA = d_workspace;
  d_workspace += CUV_ncols;//kmin(M, CUV_ncols);contrary to standard, KBLAS requires this

  T* qrtauB = d_workspace;
  d_workspace += CUV_ncols;//kmin(N, CUV_ncols);contrary to standard, KBLAS requires this

  T* rA = d_workspace;
  int rA_cols = CUV_ncols;
  int ld_rA  = kblas_roundup(rA_cols, align);
  d_workspace += ld_rA * rA_cols;

  T* rB;
  int rB_cols = CUV_ncols;
  int ld_rB = kblas_roundup(rB_cols, align);
  if(N < CUV_ncols){
    rB = d_workspace;
    d_workspace += ld_rB * rB_cols;
  }

  T* Tu = d_workspace;
  int ld_Tu = kblas_roundup(CUV_ncols, align);
  d_workspace += kmin(CUV_ncols, max_rk) * ld_Tu;

  T* Tv = d_workspace;
  int ld_Tv = kblas_roundup(CUV_ncols, align);
  d_workspace += kmin(CUV_ncols, max_rk) * ld_Tv;

  T *d_sigma  = d_workspace;
  d_workspace += CUV_ncols;

  int finalrank = -1;

  //----------------------------
  T* _Cu = Cu; int _ldCu = ldCu;
  T* _Cv = Cv; int _ldCv = ldCv;

  bool clone_CUV = (CUV_ncols > max_rk);

  if(clone_CUV){

    _Cu = d_workspace;
    _ldCu = kblas_roundup(M, align);
    d_workspace += _ldCu * CUV_ncols;

    _Cv = d_workspace;
    _ldCv = kblas_roundup(N, align);
    d_workspace += _ldCv * CUV_ncols;

    check_error_ret( cudaMemcpy2DAsync ((void*)_Cu, (size_t)(_ldCu * sizeof(T)),
                                        (const void*)Cu, (size_t)(ldCu * sizeof(T)),
                                        (size_t)(M * sizeof(T)), (size_t)(kC),
                                        cudaMemcpyDeviceToDevice,
                                        cuda_stream[0]), KBLAS_CUDA_Error);

    check_error_ret( cudaMemcpy2DAsync ((void*)_Cv, (size_t)(_ldCv * sizeof(T)),
                                        (const void*)Cv, (size_t)(ldCv * sizeof(T)),
                                        (size_t)(N * sizeof(T)), (size_t)(kC),
                                        cudaMemcpyDeviceToDevice,
                                        cuda_stream[1]), KBLAS_CUDA_Error);
  }

  //----------------------------
  handle->SetStream(cuda_stream[1]);{
    // P = AvtBu = Av^T * Bu
    check_ret_error( kblasXgemm(handle,
                                KBLAS_Trans, KBLAS_NoTrans,
                                kA, kB, K,
                                alpha, (const T*) Av, ldAv,
                                       (const T*) Bu, ldBu,
                                zero,             AvtBu, ldAvtBu));

    //G = P * Bv^T <=> G^T = Bv * P^T
    //_Cv = _Cv | G^T
    check_ret_error( kblasXgemm(handle,
                                KBLAS_NoTrans, KBLAS_Trans,
                                N, kA, kB,
                                one, (const T*) Bv, ldBv,
                                     (const T*) AvtBu, ldAvtBu,
                                zero,           &_Cv[kC*_ldCv], _ldCv));
  }

  //----------------------------
  // QR A
  {
    //concat _Cu+Av
    handle->SetStream(cuda_stream[0]);

    check_error_ret( cudaMemcpy2DAsync ((void*)&_Cu[kC*_ldCu], (size_t)(_ldCu * sizeof(T)),
                                        (const void*)Au, (size_t)(ldAu * sizeof(T)),
                                        (size_t)(M * sizeof(T)), (size_t)(kA),
                                        cudaMemcpyDeviceToDevice,
                                        cuda_stream[0]), KBLAS_CUDA_Error);

    check_ret_error( kblas_geqrf_batch( handle,
                                        M, CUV_ncols,
                                        _Cu, _ldCu, _ldCu*CUV_ncols,
                                        qrtauA, CUV_ncols,
                                        1));
    if(M < CUV_ncols){
      check_error_ret( cudaMemsetAsync( rA,
                                        0,
                                        (size_t)(ld_rA * rA_cols * sizeof(T)),
                                        cuda_stream[0]), KBLAS_CUDA_Error);
    }
    //copy rA from CU
    check_ret_error( kblas_copy_upper_batch(handle,
                                            M, CUV_ncols,
                                            _Cu, _ldCu, _ldCu*CUV_ncols,
                                            rA, ld_rA, ld_rA*rA_cols,
                                            1));
  }

  //----------------------------
  // QR B
  {
    handle->SetStream(cuda_stream[1]);

    if(beta != one){
      check_ret_error( kblasXgeam(handle,
                                  KBLAS_NoTrans, KBLAS_NoTrans,
                                  N, kC,
                                  beta, (const T*)_Cv, _ldCv,
                                  zero, (const T*)_Cu, _ldCu,
                                              (T*)_Cv, _ldCv));
    }

    check_ret_error( kblas_geqrf_batch( handle,
                                        N, CUV_ncols,
                                        _Cv, _ldCv, _ldCv*CUV_ncols,
                                        qrtauB, CUV_ncols,
                                        1));
    if(N < CUV_ncols){
      check_error_ret( cudaMemsetAsync( rB,
                                        0,
                                        (size_t)(ld_rB * rB_cols * sizeof(T)),
                                        cuda_stream[1]), KBLAS_CUDA_Error);
      //copy rB from CV
      check_ret_error( kblas_copy_upper_batch(handle,
                                              N, CUV_ncols,
                                              _Cv, _ldCv, _ldCv*CUV_ncols,
                                              rB, ld_rB, ld_rB*rB_cols,
                                              1));
    }else{
      rB = _Cv;
      ld_rB  = _ldCv;
    }
    // rA = rA * rB^T
    check_ret_error( kblasXtrmm(handle,
                                KBLAS_Right, KBLAS_Upper, KBLAS_Trans, KBLAS_NonUnit,
                                rA_cols, rA_cols,
                                // CUV_ncols, CUV_ncols,
                                one, rB, ld_rB,
                                     rA, ld_rA));
  }

  //----------------------------
  //SVD
  {
    handle->SetStream(cuda_stream[2]);

    T* pfinalacc = d_workspace;//device memory
    d_workspace += 1;
    #ifdef HCUDA_GEMM_ASYNC_CPU
      T* pfinalrank = d_Crk;//device memory
    #else
      T* pfinalrank = pfinalacc+1;//device memory
      d_workspace += 1;
    #endif
    /*
    kblasRandState_t rand_state;
    kblasInitRandState(handle, &rand_state, CUV_ncols*2, 0);
    int svd_rank = max_acc > 0 ? 0 : max_rk;
    check_ret_error( kblas_svd_full_batch_strided(handle,
                                                  CUV_ncols, CUV_ncols, svd_rank,
                                                  rA, ld_rA, ld_rA * CUV_ncols,
                                                  d_sigma, CUV_ncols,
                                                  Tu, ld_Tu, CUV_ncols * ld_Tu,
                                                  Tv, ld_Tv, CUV_ncols * ld_Tv,
                                                  SVD_random,
                                                  rand_state,
                                                  1));
    kblasDestroyRandState(rand_state);
    /*/
    check_ret_error( kblas_acaf_gpu(handle,
                                    CUV_ncols, CUV_ncols,
                                    rA, ld_rA,
                                    Tu, ld_Tu,
                                    Tv, ld_Tv,
                                    d_sigma,
                                    max_acc, max_acc > 0 ? 0 : max_rk,
                                    (double*)pfinalacc, pfinalrank));//TODO IMP not correct due to data mis-alignment
                                    //*/
    #ifndef HCUDA_GEMM_ASYNC_CPU
      if(max_acc > 0){
        T dfinal[2];
        check_error_ret( cudaMemcpyAsync(&dfinal[0],
                                         pfinalacc,
                                         sizeof(T)*2,
                                         cudaMemcpyDeviceToHost,
                                         cuda_stream[2]), KBLAS_CUDA_Error );
        cudaStreamSynchronize( cuda_stream[2] );
        finalrank = (int)dfinal[1];
        //T finalacc = dfinal[0];
      }else{
        finalrank = kmin(max_rk, CUV_ncols);
      }
    #else
        // cudaStat = cudaMemcpyAsync(d_Crk, pfinalrank, sizeof(int), cudaMemcpyDeviceToDevice, cuda_stream[2]);
        // cudaStat = cudaStreamSynchronize( cuda_stream[2] );
        finalrank = CUV_ncols;
    #endif
    finalrank = kmin(max_rk, finalrank);
    finalrank = kmin(kmin(M,N), finalrank);
    kC = finalrank;
  }

  //----------------------------
  // ORGQR A
  {
    handle->SetStream(cuda_stream[0]);
    check_ret_error( kblas_orgqr_batch( handle,
                                        M, CUV_ncols,
                                        _Cu, _ldCu, _ldCu*CUV_ncols,
                                        qrtauA, CUV_ncols,
                                        1) );
  }

  //----------------------------
  // ORGQR B
  {
    handle->SetStream(cuda_stream[1]);
    check_ret_error( kblas_orgqr_batch( handle,
                                        N, CUV_ncols,
                                        _Cv, _ldCv, _ldCv*CUV_ncols,
                                        qrtauB, CUV_ncols,
                                        1) );
  }

  T* newUV;
  int ld_newUV;
  //----------------------------
  // construct final U
  {
    if(!clone_CUV){
      newUV = d_workspace;
      ld_newUV = kblas_roundup(kmax(M, N), align);
      d_workspace += ld_newUV * finalrank;
    }else{
      newUV = Cu;
      ld_newUV = ldCu;
    }
    handle->SetStream(cuda_stream[0]);
    check_ret_error( kblasXgemm(handle,
                                KBLAS_NoTrans, KBLAS_NoTrans,
                                M, finalrank, CUV_ncols,
                                one, (const T*) _Cu, _ldCu,
                                     (const T*) Tu, ld_Tu,
                                zero,           newUV, ld_newUV));

    if(!clone_CUV){
      check_error_ret( cudaMemcpy2DAsync ((void*)Cu, (size_t)(ldCu * sizeof(T)),
                                          (const void*)newUV, (size_t)(ld_newUV * sizeof(T)),
                                          (size_t)(M * sizeof(T)), (size_t)(finalrank),
                                          cudaMemcpyDeviceToDevice,
                                          cuda_stream[0]), KBLAS_CUDA_Error);
    }
  }

  //----------------------------
  // construct final U
  {
    if(clone_CUV){
      newUV = Cv;
      ld_newUV = ldCv;
    }
    check_ret_error( kblasXgemm(handle,
                                KBLAS_NoTrans, KBLAS_NoTrans,
                                N, finalrank, CUV_ncols,
                                one, (const T*) _Cv, _ldCv,
                                     (const T*) Tv, ld_Tv,
                                zero,           newUV, ld_newUV));

    if(!clone_CUV){
      check_error_ret( cudaMemcpy2DAsync ((void*)Cv, (size_t)(ldCv * sizeof(T)),
                                          (const void*)newUV, (size_t)(ld_newUV * sizeof(T)),
                                          (size_t)(N * sizeof(T)), (size_t)(finalrank),
                                          cudaMemcpyDeviceToDevice,
                                          cuda_stream[0]), KBLAS_CUDA_Error);
    }
  }

  return KBLAS_Success;
}

//==============================================================================================
// computes C[i] = A[i] * B[i]
// assumes uniform input sizes (M,N,K) and uniform input ranks (kA,kB,kC)
//
template<class T, class T_PTR, bool STRIDED>
int Xgemm_LR_batch_uniform_core( kblasHandle_t handle,
                                  const int M, const int N, const int K,
                                  const T alpha,
                                  const T_PTR Au, int ldAu, long strideAu,
                                  const T_PTR Av, int ldAv, long strideAv, int kA,
                                  const T_PTR Bu, int ldBu, long strideBu,
                                  const T_PTR Bv, int ldBv, long strideBv, int kB,
                                  const T beta,
                                        T_PTR Cu, int ldCu, long strideCu,
                                        T_PTR Cv, int ldCv, long strideCv, int& kC,
                                  int max_rk, double max_acc, int batchCount)
{
  if(max_rk <= 0 || max_acc > 0)
    return KBLAS_NotImplemented;

  //----------------------------
  //validate & prepare workspace
  KBlasWorkspaceState ws_needed, ws_child;
  gemm_lr_batch_wsquery_core<T, STRIDED>(M, N,
                                          kA, kB, kC, max_rk,
                                          batchCount,
                                          (kblasWorkspaceState_t)&ws_needed,
                                          (kblasWorkspaceState_t)&ws_child);

  if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) ){
    printf("supplied(%zu,%zu,%zu,%zu) needed(%zu,%zu,%zu,%zu) \n",
            handle->work_space.allocated_ws_state.h_data_bytes,
            handle->work_space.allocated_ws_state.h_ptrs_bytes,
            handle->work_space.allocated_ws_state.d_data_bytes,
            handle->work_space.allocated_ws_state.d_ptrs_bytes,
            ws_needed.h_data_bytes,
            ws_needed.h_ptrs_bytes,
            ws_needed.d_data_bytes,
            ws_needed.d_ptrs_bytes);
    return KBLAS_InsufficientWorkspace;
  }

  char* d_ws_data = (char*)(handle->work_space.d_data);
  size_t d_ws_data_offset = kblas_roundup_s(ws_child.d_data_bytes, sizeof(T));

  T** d_ws_ptrs;
  size_t d_ws_ptrs_offset = 0;
  if(!STRIDED){
    d_ws_ptrs_offset = (ws_child.d_ptrs_bytes) / sizeof(void*);
    d_ws_ptrs = (T**)(handle->work_space.d_ptrs);
  }

  cudaStream_t cuda_stream[3];
  if(handle->nStreams >= 2 && !handle->use_magma){
  cuda_stream[0] = handle->stream;
    cuda_stream[1] = handle->streams[0];
    cuda_stream[2] = handle->streams[1];
  }else{
    cuda_stream[0] = handle->stream;
    cuda_stream[1] = handle->stream;
    cuda_stream[2] = handle->stream;
  }
  cudaEvent_t event_exec[4];
  for (int i = 0; i < 4; ++i)
  {
    check_error_ret( cudaEventCreateWithFlags(&event_exec[i], cudaEventDisableTiming), KBLAS_CUDA_Error);
  }

  T zero = make_zero<T>();
  T one = make_one<T>();

  //----------------------------
  size_t batchBytes = batchCount * sizeof(T);
  int CUV_ncols = kC + kA;
  //TODO set pointer and adjust workspace alignment
    T_PTR AvtBu;
    size_t wsdo_AvtBu = d_ws_data_offset;
    size_t wspo_AvtBu = d_ws_ptrs_offset;
    int ld_AvtBu = kblas_roundup(kA, align),
        stride_AvtBu = ld_AvtBu * kB;
    d_ws_data_offset += stride_AvtBu * batchBytes;
    d_ws_ptrs_offset += batchCount;
    if(STRIDED){
      AvtBu = (T_PTR)((char*)d_ws_data + wsdo_AvtBu);
    }else{
      AvtBu = (T_PTR)d_ws_ptrs + wspo_AvtBu;
    }

    T_PTR qrtauA;
    size_t wsdo_qrtauA = d_ws_data_offset;
    size_t wspo_qrtauA = d_ws_ptrs_offset;
    int stride_qrtauA = CUV_ncols;//kmin(M, CUV_ncols);contrary to standard, KBLAS requires this
    d_ws_data_offset += stride_qrtauA * batchBytes;
    d_ws_ptrs_offset += batchCount;
    if(STRIDED){
      qrtauA = (T_PTR)((char*)d_ws_data + wsdo_qrtauA);
    }else{
      qrtauA = (T_PTR)d_ws_ptrs + wspo_qrtauA;
    }

    T_PTR qrtauB;
    size_t wsdo_qrtauB = d_ws_data_offset;
    size_t wspo_qrtauB = d_ws_ptrs_offset;
    int stride_qrtauB = CUV_ncols;//kmin(N, CUV_ncols);contrary to standard, KBLAS requires this
    d_ws_data_offset += stride_qrtauB * batchBytes;
    d_ws_ptrs_offset += batchCount;
    if(STRIDED){
      qrtauB = (T_PTR)((char*)d_ws_data + wsdo_qrtauB);
    }else{
      qrtauB = (T_PTR)d_ws_ptrs + wspo_qrtauB;
    }

    T_PTR rA;
    int rA_cols = CUV_ncols;
    size_t wsdo_rA = d_ws_data_offset;
    size_t wspo_rA = d_ws_ptrs_offset;
    int ld_rA  = kblas_roundup(rA_cols, align),
        stride_rA = ld_rA * rA_cols;
    d_ws_data_offset += stride_rA * batchBytes;
    d_ws_ptrs_offset += batchCount;
    if(STRIDED){
      rA = (T_PTR)((char*)d_ws_data + wsdo_rA);
    }else{
      rA = (T_PTR)d_ws_ptrs + wspo_rA;
    }

    T_PTR rB;
    int rB_cols = CUV_ncols;
    int ld_rB  = kblas_roundup(rB_cols, align);
    long stride_rB = ld_rB * rB_cols;
    size_t wsdo_rB = 0,
         wspo_rB = 0;
    if(N < CUV_ncols){
      wsdo_rB = d_ws_data_offset;
      wspo_rB = d_ws_ptrs_offset;
      d_ws_data_offset += stride_rB * batchBytes;
      d_ws_ptrs_offset += batchCount;
      if(STRIDED){
        rB = (T_PTR)((char*)d_ws_data + wsdo_rB);
      }else{
        rB = (T_PTR)d_ws_ptrs + wspo_rB;
      }
    }

    T_PTR Tu;
    size_t wsdo_Tu = d_ws_data_offset;
    size_t wspo_Tu = d_ws_ptrs_offset;
    int ld_Tu = kblas_roundup(CUV_ncols, align),
        stride_Tu = kmin(CUV_ncols, max_rk) * ld_Tu;
    d_ws_data_offset += stride_Tu * batchBytes;
    d_ws_ptrs_offset += batchCount;
    if(STRIDED){
      Tu = (T_PTR)((char*)d_ws_data + wsdo_Tu);
    }else{
      Tu = (T_PTR)d_ws_ptrs + wspo_Tu;
    }

    T_PTR Tv;
    size_t wsdo_Tv = d_ws_data_offset;
    size_t wspo_Tv = d_ws_ptrs_offset;
    int ld_Tv = kblas_roundup(CUV_ncols, align),
        stride_Tv = kmin(CUV_ncols, max_rk) * ld_Tv;
    d_ws_data_offset += stride_Tv * batchBytes;
    d_ws_ptrs_offset += batchCount;
    if(STRIDED){
      Tv = (T_PTR)((char*)d_ws_data + wsdo_Tv);
    }else{
      Tv = (T_PTR)d_ws_ptrs + wspo_Tv;
    }

    T_PTR d_sigma;
    size_t wsdo_sigma = d_ws_data_offset;
    size_t wspo_sigma = d_ws_ptrs_offset;
    int ld_sigma = 1,
        stride_sigma = ld_sigma * CUV_ncols;
    d_ws_data_offset += stride_sigma * batchBytes;
    d_ws_ptrs_offset += batchCount;
    if(STRIDED){
      d_sigma = (T_PTR)((char*)d_ws_data + wsdo_sigma);
    }else{
      d_sigma = (T_PTR)d_ws_ptrs + wspo_sigma;
    }

    if(!STRIDED){
      check_ret_error( Xset_pointer_7( d_ws_ptrs + wspo_AvtBu, (T*)((char*)d_ws_data + wsdo_AvtBu), ld_AvtBu, stride_AvtBu,
                                       d_ws_ptrs + wspo_qrtauA, (T*)((char*)d_ws_data + wsdo_qrtauA), 1, stride_qrtauA,
                                       d_ws_ptrs + wspo_qrtauB, (T*)((char*)d_ws_data + wsdo_qrtauB), 1, stride_qrtauB,
                                       d_ws_ptrs + wspo_rA, (T*)((char*)d_ws_data + wsdo_rA), ld_rA, stride_rA,
                                       d_ws_ptrs + wspo_Tu, (T*)((char*)d_ws_data + wsdo_Tu), ld_Tu, stride_Tu,
                                       d_ws_ptrs + wspo_Tv, (T*)((char*)d_ws_data + wsdo_Tv), ld_Tv, stride_Tv,
                                       d_ws_ptrs + wspo_sigma, (T*)((char*)d_ws_data + wsdo_sigma), ld_sigma, stride_sigma,
                                       batchCount, cuda_stream[0]) );
      if(N < CUV_ncols){
        check_ret_error( Xset_pointer_1(d_ws_ptrs + wspo_rB, (T*)((char*)d_ws_data + wsdo_rB), ld_rB, stride_rB,
                                        batchCount, cuda_stream[0]) );
      }
    }

    int finalrank = -1;

    bool clone_CUV = CUV_ncols > max_rk;

    T_PTR newUV;
    size_t wsdo_newUV = d_ws_data_offset;
    size_t wspo_newUV = d_ws_ptrs_offset;
    int ld_newUV, stride_newUV;
    if(!clone_CUV){
      ld_newUV = kblas_roundup(kmax(M, N), align);
      stride_newUV = ld_newUV * kmin(max_rk, CUV_ncols);
      d_ws_data_offset += stride_newUV * batchBytes;
      d_ws_ptrs_offset += batchCount;
      if(STRIDED){
        newUV = (T_PTR)((char*)d_ws_data + wsdo_newUV);
      }else{
        newUV = (T_PTR)d_ws_ptrs + wspo_newUV;
        check_ret_error( Xset_pointer_1( d_ws_ptrs + wspo_newUV, (T*)((char*)d_ws_data + wsdo_newUV), ld_newUV, stride_newUV,
                                         batchCount, cuda_stream[0]) );
      }
    }else{
      newUV = Cu;
      ld_newUV = ldCu;
      stride_newUV = strideCu;
    }
  //----------------------------
  T_PTR _Cu = Cu; int _ldCu = ldCu; int _strideCu = strideCu;
  T_PTR _Cv = Cv; int _ldCv = ldCv; int _strideCv = strideCv;


  if(clone_CUV){

    // _Cu = (T*)((char*)d_ws_data + d_ws_data_offset);
    size_t wsdo_Cu = d_ws_data_offset;
    size_t wspo_Cu = d_ws_ptrs_offset;
    _ldCu = kblas_roundup(M, align);
    _strideCu = _ldCu * CUV_ncols;
    d_ws_data_offset += _strideCu * batchBytes;
    d_ws_ptrs_offset += batchCount;
    if(STRIDED){
      _Cu = (T_PTR)((char*)d_ws_data + wsdo_Cu);
    }else{
      _Cu = (T_PTR)d_ws_ptrs + wspo_Cu;
    }

    // _Cv = (T*)((char*)d_ws_data + d_ws_data_offset);
    size_t wsdo_Cv = d_ws_data_offset;
    size_t wspo_Cv = d_ws_ptrs_offset;
    _ldCv = kblas_roundup(N, align);
    _strideCv = _ldCv * CUV_ncols;
    d_ws_data_offset += _strideCv * batchBytes;
    d_ws_ptrs_offset += batchCount;
    if(STRIDED){
      _Cv = (T_PTR)((char*)d_ws_data + wsdo_Cv);
    }else{
      _Cv = (T_PTR)d_ws_ptrs + wspo_Cv;
    }

    if(!STRIDED){
      check_ret_error( Xset_pointer_2( d_ws_ptrs + wspo_Cu, (T*)((char*)d_ws_data + wsdo_Cu), _ldCu, _strideCu,
                                       d_ws_ptrs + wspo_Cv, (T*)((char*)d_ws_data + wsdo_Cv), _ldCv, _strideCv,
                                       batchCount, cuda_stream[0]) );
    }

    check_ret_error( kblas_copyBlock_batch( handle,
                                            M, kC,
                                            (T_PTR)_Cu, 0, 0, _ldCu, _strideCu,
                                            (T_PTR) Cu, 0, 0,  ldCu,  strideCu,
                                            batchCount) );

    check_ret_error( kblas_copyBlock_batch( handle,
                                            N, kC,
                                            (T_PTR)_Cv, 0, 0, _ldCv, _strideCv,
                                            (T_PTR) Cv, 0, 0,  ldCv,  strideCv,
                                            batchCount) );
  }
  {
    //force other streams to wait for main one
    cudaEvent_t event_start;
    cudaEventCreateWithFlags(&event_start, cudaEventDisableTiming);
    cudaEventRecord(event_start, cuda_stream[0]);
    cudaStreamWaitEvent(cuda_stream[1], event_start, 0);
    cudaStreamWaitEvent(cuda_stream[2], event_start, 0);
    cudaEventDestroy(event_start);
  }

  //----------------------------
  handle->SetStream(cuda_stream[1]);{
    // P = AvtBu = Av^T * Bu
    check_ret_error( Xgemm_batch( handle,
                                  KBLAS_Trans, KBLAS_NoTrans,
                                  kA, kB, K,
                                  alpha, (T_PTR) Av, ldAv, strideAv,
                                         (T_PTR) Bu, ldBu, strideBu,
                                  zero,          AvtBu, ld_AvtBu, stride_AvtBu,
                                  batchCount));

    //G = P * Bv^T <=> G^T = Bv * P^T
    //_Cv = _Cv | G^T
    check_ret_error( Xgemm_batch( handle,
                                  KBLAS_NoTrans, KBLAS_Trans,
                                  N, kA, kB,
                                  one, (T_PTR) Bv,    0,  0, ldBv,     strideBv,
                                       (T_PTR) AvtBu, 0,  0, ld_AvtBu, stride_AvtBu,
                                  zero,(T_PTR) _Cv,   0, kC, _ldCv,   _strideCv,
                                  batchCount));
  }

  //----------------------------
  // QR A
  {
    //concat _Cu+Av
    handle->SetStream(cuda_stream[0]);

    check_ret_error( kblas_copyBlock_batch( handle,
                                            M, kA,
                                            (T_PTR)_Cu, 0, kC, _ldCu, _strideCu,
                                            (T_PTR) Au, 0,  0,  ldAu,  strideAu,
                                            batchCount) );
    // ECHO_I(M); ECHO_I(CUV_ncols); ECHO_I(_ldCu); ECHO_I(_strideCu); ECHO_I(stride_qrtauA); ECHO_I(batchCount); ECHO_LN;
    check_ret_error( kblas_geqrf_batch( handle,
                                        M, CUV_ncols,
                                        (T_PTR)_Cu, _ldCu, _strideCu,
                                        qrtauA, stride_qrtauA,
                                        batchCount));
    //TODO
    // if(M < CUV_ncols){
    //   check_error_ret( cudaMemsetAsync( rA,
    //                                     0,
    //                                     (size_t)(ld_rA * rA_cols * sizeof(T)),
    //                                     cuda_stream[0]), KBLAS_CUDA_Error);
    // }

    //copy rA from CU
    check_ret_error( kblas_copy_upper_batch(handle,
                                            M, CUV_ncols,
                                            (T_PTR)_Cu, _ldCu, _strideCu,
                                            rA, ld_rA, stride_rA,
                                            batchCount));
    cudaEventRecord(event_exec[0], cuda_stream[0]);
  }

  //----------------------------
  // QR B
  {
    handle->SetStream(cuda_stream[1]);

    if(beta != one){
      // check_ret_error( kblasXgeam(handle,
      //                             KBLAS_NoTrans, KBLAS_NoTrans,
      //                             N, kC,
      //                             beta, (const T*)_Cv, _ldCv,
      //                             zero, (const T*)_Cu, _ldCu,
      //                                         (T*)_Cv, _ldCv));

      //TODO find a better performing way
      check_ret_error( Xgemm_batch( handle,
                                    KBLAS_NoTrans, KBLAS_NoTrans,
                                    N, kC, 1,
                                    zero, (T_PTR) Av, ldAv, strideAv,//dummy
                                          (T_PTR) Bu, ldBu, strideBu,//dummy
                                    beta,        _Cv, _ldCv, _strideCv,
                                    batchCount));
    }

    check_ret_error( kblas_geqrf_batch( handle,
                                        N, CUV_ncols,
                                        (T_PTR)_Cv, _ldCv, _strideCv,
                                        qrtauB, stride_qrtauB,
                                        batchCount));

    if(N < CUV_ncols){
      //TODO
      // check_error_ret( cudaMemsetAsync( rB,
      //                                   0,
      //                                   (size_t)(ld_rB * rB_cols * sizeof(T)),
      //                                   cuda_stream[1]), KBLAS_CUDA_Error);
      //copy rB from CV
      check_ret_error( kblas_copy_upper_batch(handle,
                                              N, CUV_ncols,
                                              (T_PTR)_Cv, _ldCv, _strideCv,
                                              rB, ld_rB, stride_rB,
                                              batchCount));
    }else{
      rB = _Cv;
      ld_rB  = _ldCv,
      stride_rB = _strideCv;
    }
    cudaStreamWaitEvent(cuda_stream[1], event_exec[0], 0);
    // rA = rA * rB^T
    check_ret_error( Xtrmm_batch( handle,
                                  KBLAS_Right, KBLAS_Upper, KBLAS_Trans, KBLAS_NonUnit,
                                  rA_cols, rA_cols,
                                  // CUV_ncols, CUV_ncols,
                                  one, (T_PTR)rB, 0, 0, ld_rB, stride_rB,
                                       (T_PTR)rA, 0, 0, ld_rA, stride_rA,
                                  batchCount));
    cudaEventRecord(event_exec[1], cuda_stream[1]);
  }

  //----------------------------
  //SVD
  {
    handle->SetStream(cuda_stream[2]);

    d_ws_data_offset = kblas_roundup_s(d_ws_data_offset, sizeof(double));
    double* pfinalacc = (double*)((char*)d_ws_data + d_ws_data_offset);//device memory
    d_ws_data_offset += size_t(batchCount) * sizeof(double);
    #ifdef HCUDA_GEMM_ASYNC_CPU
      int* pfinalrank = d_Crk;//device memory
    #else
      d_ws_data_offset = kblas_roundup_s(d_ws_data_offset, sizeof(int));
      int* pfinalrank = (int*)((char*)d_ws_data + d_ws_data_offset);//device memory
      d_ws_data_offset += size_t(batchCount) * sizeof(int);
    #endif
    /*
    kblasRandState_t rand_state;
    kblasInitRandState(handle, &rand_state, CUV_ncols*2, 0);
    int svd_rank = max_acc > 0 ? 0 : max_rk;
    check_ret_error( kblas_svd_full_batch_strided(handle,
                                                  CUV_ncols, CUV_ncols, svd_rank,
                                                  rA, ld_rA, ld_rA * CUV_ncols,
                                                  d_sigma, CUV_ncols,
                                                  Tu, ld_Tu, CUV_ncols * ld_Tu,
                                                  Tv, ld_Tv, CUV_ncols * ld_Tv,
                                                  SVD_random,
                                                  rand_state,
                                                  1));
    kblasDestroyRandState(rand_state);
    //*/
    cudaStreamWaitEvent(cuda_stream[2], event_exec[1], 0);
    check_ret_error( Xacaf_batch( handle,
                                  CUV_ncols, CUV_ncols,
                                  (T_PTR)rA, ld_rA, (long)stride_rA,
                                  (T_PTR)Tu, ld_Tu, (long)stride_Tu,
                                  (T_PTR)Tv, ld_Tv, (long)stride_Tv,
                                  (T_PTR)d_sigma, ld_sigma, (long)CUV_ncols,
                                  (double)max_acc, (max_acc > 0 ? 0 : max_rk),
                                  (double*)pfinalacc, pfinalrank,
                                  batchCount));
    #ifndef HCUDA_GEMM_ASYNC_CPU
      // if(max_acc > 0){
      //   int dfinal[2];
      //   check_error_ret( cudaMemcpyAsync(&dfinal[0],
      //                                    pfinalacc,
      //                                    sizeof(T)*2,
      //                                    cudaMemcpyDeviceToHost,
      //                                    cuda_stream[2]), KBLAS_CUDA_Error );
      //   cudaStreamSynchronize( cuda_stream[2] );
      //   finalrank = (int)dfinal[1];
      //   T finalacc = dfinal[0];
      // }else
      {
        finalrank = kmin(max_rk, CUV_ncols);
      }
    #else
        // cudaStat = cudaMemcpyAsync(d_Crk, pfinalrank, sizeof(int), cudaMemcpyDeviceToDevice, cuda_stream[2]);
        // cudaStat = cudaStreamSynchronize( cuda_stream[2] );
        finalrank = CUV_ncols;
    #endif
    finalrank = kmin(max_rk, finalrank);
    finalrank = kmin(kmin(M,N), finalrank);
    kC = finalrank;
    cudaEventRecord(event_exec[2], cuda_stream[2]);
  }

  //----------------------------
  // ORGQR A
  {
    handle->SetStream(cuda_stream[0]);
    check_ret_error( kblas_orgqr_batch( handle,
                                        M, CUV_ncols,//finalrank,//
                                        (T_PTR)_Cu, _ldCu, _strideCu,
                                        qrtauA, stride_qrtauA,
                                        batchCount) );
  }

  //----------------------------
  // ORGQR B
  {
    handle->SetStream(cuda_stream[1]);
    check_ret_error( kblas_orgqr_batch( handle,
                                        N, CUV_ncols,//finalrank,//
                                        (T_PTR)_Cv, _ldCv, _strideCv,
                                        qrtauB, stride_qrtauB,
                                        batchCount) );
    cudaEventRecord(event_exec[3], cuda_stream[1]);
  }

  //----------------------------
  // construct final U
  {
    handle->SetStream(cuda_stream[0]);
    cudaStreamWaitEvent(cuda_stream[0], event_exec[2], 0);
    check_ret_error( Xgemm_batch( handle,
                                  KBLAS_NoTrans, KBLAS_NoTrans,
                                  M, finalrank, CUV_ncols,//finalrank,
                                  one, (T_PTR) _Cu, _ldCu, _strideCu,
                                       (T_PTR)  Tu, ld_Tu, stride_Tu,
                                  zero,      newUV, ld_newUV, stride_newUV,
                                  batchCount));

    if(!clone_CUV){
      check_ret_error( kblas_copyBlock_batch( handle,
                                              M, finalrank,
                                              (T_PTR)Cu,    0, 0, ldCu,     strideCu,
                                              (T_PTR)newUV, 0, 0, ld_newUV, stride_newUV,
                                              batchCount) );
    }
  }

  //----------------------------
  // construct final V
  {
    if(clone_CUV){
      newUV = Cv;
      ld_newUV = ldCv;
    }
    cudaStreamWaitEvent(cuda_stream[0], event_exec[3], 0);
    check_ret_error( Xgemm_batch( handle,
                                  KBLAS_NoTrans, KBLAS_NoTrans,
                                  N, finalrank, CUV_ncols,//finalrank,
                                  one, (T_PTR) _Cv, _ldCv, _strideCv,
                                       (T_PTR)  Tv, ld_Tv, stride_Tv,
                                  zero,      newUV, ld_newUV, stride_newUV,
                                  batchCount));

    if(!clone_CUV){
      check_ret_error( kblas_copyBlock_batch( handle,
                                              N, finalrank,
                                              (T_PTR)Cv,    0, 0, ldCv,     strideCv,
                                              (T_PTR)newUV, 0, 0, ld_newUV, stride_newUV,
                                              batchCount) );
    }
  }

  for (int i = 0; i < 4; ++i){
    cudaEventDestroy(event_exec[i]);
  }
  return KBLAS_Success;
}

//==============================================================================================
//==============================================================================================
template<typename T>
__global__
void kernel_collect_pointers_2d_5(T **output_Au, T **input_Au, T **output_Av, T **input_Av, int ld_Aptrs,
                                  T **output_Bu, T **input_Bu, T **output_Bv, T **input_Bv, int ld_Bptrs,
                                  T **output_C, const T  *input_C, int ld_C, long mb, long nb,
                                  bool transA, bool transB,
                                  int kTile)
{
  int ind = blockIdx.x + blockIdx.y * gridDim.x;
  output_Au[ind] = input_Au[transA ? kTile + blockIdx.x * ld_Aptrs : blockIdx.x + kTile * ld_Aptrs];
  output_Av[ind] = input_Av[transA ? kTile + blockIdx.x * ld_Aptrs : blockIdx.x + kTile * ld_Aptrs];
  output_Bu[ind] = input_Bu[transB ? blockIdx.y + kTile * ld_Bptrs : kTile + blockIdx.y * ld_Bptrs];
  output_Bv[ind] = input_Bv[transB ? blockIdx.y + kTile * ld_Bptrs : kTile + blockIdx.y * ld_Bptrs];
  output_C[ind]  = (T*)&input_C[blockIdx.x * mb + blockIdx.y * nb * ld_C];
}
template<typename T>
inline
int Xcollect_pointers_2d_5( T **output_Au, const T **input_Au, T **output_Av, const T **input_Av, int ld_Aptrs,
                            T **output_Bu, const T **input_Bu, T **output_Bv, const T **input_Bv, int ld_Bptrs,
                            T **output_C, const T  *input_C, int ld_C, long mb, long nb,
                            long mt, long nt, int kTile,
                            char transA, char transB,
                            cudaStream_t cuda_stream)
{
  dim3 block(1,1);
  dim3 grid(mt, nt);
  kernel_collect_pointers_2d_5<T><<< grid, block, 0, cuda_stream>>>(
                                  output_Au, (T**)input_Au, output_Av, (T**)input_Av, ld_Aptrs,
                                  output_Bu, (T**)input_Bu, output_Bv, (T**)input_Bv, ld_Bptrs,
                                  output_C, input_C, ld_C, mb, nb,
                                  transA == KBLAS_Trans, transB == KBLAS_Trans,
                                  kTile);
  check_error_ret( cudaGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

//==============================================================================================
// workspace needed: ?? device data + pointers + host pointers ??
// Au_array, Av_array, Bu_array, Bv_array, C_array: array of host pointers to device buffers
template<class T>
int Xgemm_TLR_core( kblasHandle_t handle,
                    char transA, char transB,
                    const int MTiles, const int NTiles, const int KTiles,
                    const int mb, const int nb, const int kb,
                    const T alpha,
                    const T** d_Au, int ldAu,
                    const T** d_Av, int ldAv, int ld_Aptrs, int kA,
                    const T** d_Bu, int ldBu,
                    const T** d_Bv, int ldBv, int ld_Bptrs, int kB,
                    const T beta,
                          T* C, int ldC)
{
  //TODO input validation
  // if(M % mb || N % nb || K % kb){
  //   printf("please make sure gemm_plr_tied dimensions(%d,%d,%d) are multiple of tile dimensions(%d,%d,%d)\n",
  //     M,N,K,mb,nb,kb);
  //   return KBLAS_Error_WrongInput;
  // }

  // int MTiles = M/mb;
  // int NTiles = N/nb;
  // int KTiles = K/kb;

  // if( (MTiles * NTiles * KTiles) == 1 )
  //   return KBLAS_Error_WrongInput;

  T one = make_one<T>(), zbeta;



  KBlasWorkspaceState ws_needed;
  //TODO
  gemm_tlr_wsquery_core<T, false, true>(MTiles, NTiles, kA, kB,
                                        mb, nb,
                                        (kblasWorkspaceState_t)&ws_needed);

  if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) )
    return KBLAS_InsufficientWorkspace;


  int status = KBLAS_Success;

  int batchCount = MTiles * NTiles;

  ws_needed.reset();
  gemm_lr_batch_wsquery_core<T, false>( nb, kA, kB, batchCount,
                              (kblasWorkspaceState_t)&ws_needed);
  size_t d_batch_ptrs = ws_needed.d_ptrs_bytes / sizeof(void*);

  T **d_Au_array = (T**)(handle->work_space.d_ptrs) + d_batch_ptrs,
    **d_Av_array = d_Au_array + batchCount,
    **d_Bu_array = d_Av_array + batchCount,
    **d_Bv_array = d_Bu_array + batchCount,
    **d_C_array  = d_Bv_array + batchCount;

  for(int kTile = 0; kTile < KTiles; kTile++){

    check_ret_error( Xcollect_pointers_2d_5(d_Au_array, d_Au, d_Av_array, d_Av, ld_Aptrs,
                                            d_Bu_array, d_Bu, d_Bv_array, d_Bv, ld_Bptrs,
                                            d_C_array , C, ldC, mb, nb,
                                            MTiles, NTiles, kTile,
                                            transA, transB,
                                            handle->stream) );
    zbeta = (kTile == 0 ? beta : one);

    status = Xgemm_lr_batch( handle,
                              transA, transB,
                              mb, nb, kb,
                              alpha,
                              (T**)d_Au_array, ldAu, 0,
                              (T**)d_Av_array, ldAv, 0, kA,
                              (T**)d_Bu_array, ldBu, 0,
                              (T**)d_Bv_array, ldBv, 0, kB,
                              zbeta,
                              (T**)d_C_array,  ldC,  0,
                              batchCount);
    check_error_ret( status, status);
  }
  return KBLAS_Success;
}

//==============================================================================================
template<typename T>
__global__
void kernel_collect_pointers_2d_7(T **output_Au, const T **input_Au,
                                  T **output_Av, const T **input_Av, int ld_Aptrs,
                                  T **output_Bu, const T **input_Bu,
                                  T **output_Bv, const T **input_Bv, int ld_Bptrs,
                                  T **output_C,  const T  *input_C, const int input_ldC,
                                  int* output_rA, const int* input_rA, int* output_rB, const int* input_rB,
                                  const int mb, const int nb,
                                  bool transA, bool transB,
                                  int kTile)
{
  int ind = blockIdx.x + blockIdx.y * gridDim.x;
  int indA = transA ? kTile + blockIdx.x * ld_Aptrs : blockIdx.x + kTile * ld_Aptrs;
  int indB = transB ? blockIdx.y + kTile * ld_Bptrs : kTile + blockIdx.y * ld_Bptrs;
  output_Au[ind] = (T*)input_Au[indA];
  output_Av[ind] = (T*)input_Av[indA];
  output_Bu[ind] = (T*)input_Bu[indB];
  output_Bv[ind] = (T*)input_Bv[indB];
  output_rA[ind] = (int)input_rA[indA];
  output_rB[ind] = (int)input_rB[indB];
  output_C [ind] = (T*)&input_C[blockIdx.x * mb + blockIdx.y * nb * input_ldC];
}

template<typename T>
inline
int Xcollect_pointers_2d_7( T **output_Au, const T **input_Au,
                            T **output_Av, const T **input_Av, int ld_Aptrs,
                            T **output_Bu, const T **input_Bu,
                            T **output_Bv, const T **input_Bv, int ld_Bptrs,
                            T **output_C,  const T  *input_C, const int input_ldC,
                            int* output_rA, const int* input_rA, int* output_rB, const int* input_rB,
                            const int mb, const int nb,
                            const int mt, const int nt, const int kTile,
                            const char transA, const char transB,
                            cudaStream_t cuda_stream)
{
  dim3 block(1,1);
  dim3 grid(mt, nt);
  kernel_collect_pointers_2d_7<T><<< grid, block, 0, cuda_stream>>>(
                                  output_Au, input_Au,
                                  output_Av, input_Av, ld_Aptrs,
                                  output_Bu, input_Bu,
                                  output_Bv, input_Bv, ld_Bptrs,
                                  output_C,  input_C,  input_ldC,
                                  output_rA, input_rA, output_rB, input_rB,
                                  mb, nb,
                                  transA == KBLAS_Trans, transB == KBLAS_Trans,
                                  kTile);
  check_error_ret( cudaGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

#if 1
//==============================================================================================
template<typename T>
__global__
void kernel_collect_pointers_2d_6(T **output_Au, T **input_Au, T **output_Av, T **input_Av, int ld_Aptrs,
                                  T **output_Bu, T **input_Bu, T **output_Bv, T **input_Bv, int ld_Bptrs,
                                  T **output_Cu, T **input_Cu, T **output_Cv, T **input_Cv, int ld_Cptrs,
                                  bool transA, bool transB,
                                  int kTile)
{
  int ind = blockIdx.x + blockIdx.y * gridDim.x;
  output_Au[ind] = input_Au[transA ? kTile + blockIdx.x * ld_Aptrs : blockIdx.x + kTile * ld_Aptrs];
  output_Av[ind] = input_Av[transA ? kTile + blockIdx.x * ld_Aptrs : blockIdx.x + kTile * ld_Aptrs];
  output_Bu[ind] = input_Bu[transB ? blockIdx.y + kTile * ld_Bptrs : kTile + blockIdx.y * ld_Bptrs];
  output_Bv[ind] = input_Bv[transB ? blockIdx.y + kTile * ld_Bptrs : kTile + blockIdx.y * ld_Bptrs];
  output_Cu[ind] = input_Cu[blockIdx.x + blockIdx.y * ld_Cptrs];
  output_Cv[ind] = input_Cv[blockIdx.x + blockIdx.y * ld_Cptrs];
}
template<typename T>
inline
int Xcollect_pointers_2d_6( T **output_Au, T **input_Au, T **output_Av, T **input_Av, int ld_Aptrs,
                            T **output_Bu, T **input_Bu, T **output_Bv, T **input_Bv, int ld_Bptrs,
                            T **output_Cu, T **input_Cu, T **output_Cv, T **input_Cv, int ld_Cptrs,
                            long mt, long nt, int kTile,
                            char transA, char transB,
                            cudaStream_t cuda_stream)
{
  dim3 block(1,1);
  dim3 grid(mt, nt);
  kernel_collect_pointers_2d_6<T><<< grid, block, 0, cuda_stream>>>(
                                  output_Au, (T**)input_Au, output_Av, (T**)input_Av, ld_Aptrs,
                                  output_Bu, (T**)input_Bu, output_Bv, (T**)input_Bv, ld_Bptrs,
                                  output_Cu, input_Cu, output_Cv, input_Cv, ld_Cptrs,
                                  transA == KBLAS_Trans, transB == KBLAS_Trans,
                                  kTile);
  check_error_ret( cudaGetLastError(), KBLAS_CUDA_Error);
  return KBLAS_Success;
}

//==============================================================================================
// Au_array, Av_array, Bu_array, Bv_array, C_array: array of host pointers to device buffers
template<class T>
int Xgemm_TLR_core( kblasHandle_t handle,
                          char transA, char transB,
                          int MTiles, int NTiles, int KTiles,
                          int mb, int nb, int kb,
                          T alpha,
                          T** d_Au, int ldAu,
                          T** d_Av, int ldAv, int ld_Aptrs, int kA,
                          T** d_Bu, int ldBu,
                          T** d_Bv, int ldBv, int ld_Bptrs, int kB,
                          T beta,
                          T** d_Cu, int ldCu,
                          T** d_Cv, int ldCv, int ld_Cptrs, int& kC,
                          int max_rk, double max_acc)
{
  //TODO input validation
  // if(M % mb || N % nb || K % kb){
  //   printf("please make sure gemm_plr_tied dimensions(%d,%d,%d) are multiple of tile dimensions(%d,%d,%d)\n",
  //     M,N,K,mb,nb,kb);
  //   return KBLAS_Error_WrongInput;
  // }


  // if( (MTiles * NTiles * KTiles) == 1 )
  //   return KBLAS_Error_WrongInput;

  T one = make_one<T>(), zbeta;



  KBlasWorkspaceState ws_needed, ws_child;
  gemm_tlr_wsquery_core<T, true>( MTiles, NTiles, kA, kB, kC, max_rk,
                                  mb, nb,
                                  (kblasWorkspaceState_t)&ws_needed,
                                  (kblasWorkspaceState_t)&ws_child);

  if( !ws_needed.isSufficient( &(handle->work_space.allocated_ws_state) ) )
    return KBLAS_InsufficientWorkspace;

  int status = KBLAS_Success;

  int batchCount = MTiles * NTiles;

  size_t d_batch_ptrs = ws_child.d_ptrs_bytes / sizeof(void*);

  T **d_Au_array = (T**)(handle->work_space.d_ptrs) + d_batch_ptrs,
    **d_Av_array = d_Au_array + batchCount,
    **d_Bu_array = d_Av_array + batchCount,
    **d_Bv_array = d_Bu_array + batchCount,
    **d_Cu_array = d_Bv_array + batchCount,
    **d_Cv_array = d_Cu_array + batchCount;

  for(int kTile = 0; kTile < KTiles; kTile++){
    check_ret_error( Xcollect_pointers_2d_6(d_Au_array, d_Au, d_Av_array, d_Av, ld_Aptrs,
                                            d_Bu_array, d_Bu, d_Bv_array, d_Bv, ld_Bptrs,
                                            d_Cu_array, d_Cu, d_Cv_array, d_Cv, ld_Cptrs,
                                            MTiles, NTiles, kTile,
                                            transA, transB,
                                            handle->stream) );
    zbeta = (kTile == 0 ? beta : one);

    status = Xgemm_lr_batch( handle,
                              transA, transB,
                              mb, nb, kb,
                              alpha,
                              (T**)d_Au_array, ldAu, 0,
                              (T**)d_Av_array, ldAv, 0, kA,
                              (T**)d_Bu_array, ldBu, 0,
                              (T**)d_Bv_array, ldBv, 0, kB,
                              zbeta,
                              (T**)d_Cu_array, ldCu, 0,
                              (T**)d_Cv_array, ldCv, 0, kC,
                              max_rk, max_acc,
                              batchCount);
    check_error_ret( status, status);
  }
  return KBLAS_Success;
}
#endif

#endif //__XGEMM_TLR_CORE__
