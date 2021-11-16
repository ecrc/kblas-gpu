/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/workspace_queries.ch

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef __KBLAS_WORKSPACE_QUERIES_H__
#define __KBLAS_WORKSPACE_QUERIES_H__

#include "kblas_gpu_util.ch"

#define align 32

//==============================================================================================
void kblas_imax_size_2( kblasHandle_t handle,
                        int *m, int *n,
                        int &max_m, int &max_n, int l);
void kblas_imax_size_3( kblasHandle_t handle,
                        int *m, int *n, int *k,
                        int &max_m, int &max_n, int &max_k, int l);

//==============================================================================================
/// Query workspace needed for batch GEMM with offset
void gemm_batch_offset_wsquery_core(int batchCount, bool offseted, kblasWorkspaceState_t ws);

/// Query workspace needed for batch strided GEMM
void gemm_batch_strided_wsquery_core(int batchCount, kblasWorkspaceState_t ws);
void gemm_batch_nonuniform_wsquery_core(kblasWorkspaceState_t ws);

template<bool STRIDED, bool UNIFORM>
inline
void gemm_batch_wsquery_core( int batchCount, bool offseted, kblasWorkspaceState_t ws)
{
  if(STRIDED)
    gemm_batch_strided_wsquery_core(batchCount, ws);
  else
  if(!UNIFORM)
    gemm_batch_nonuniform_wsquery_core(ws);
  else
  if(offseted)
    gemm_batch_offset_wsquery_core( batchCount, offseted, ws);
}
//==============================================================================================
void syrk_batch_wsquery_core(const int m, int batchCount, kblasWorkspaceState_t ws);
void syrk_batch_nonuniform_wsquery_core(kblasWorkspaceState_t ws);

//==============================================================================================
template<bool STRIDED>
inline
void trsm_batch_wsquery_core( int batchCount,
                              char side, int m, int n,
                              kblasWorkspaceState_t wss)
{
  if( ( (side == KBLAS_Right) && (n > 16) ) ||
      ( (side == KBLAS_Left ) && (m > 16) ) ){
    if(STRIDED){
      gemm_batch_strided_wsquery_core(batchCount, wss);
    }else{
      gemm_batch_offset_wsquery_core( batchCount, 1, wss);
    }
  }
}
void trsm_batch_nonuniform_wsquery_core(kblasWorkspaceState_t ws);

//==============================================================================================
template<bool STRIDED>
inline
void trmm_batch_wsquery_core( int batchCount,
                              char side, int m, int n,
                              kblasWorkspaceState_t wss)
{
  //TODO if( side == KBLAS_Right || uplo == KBLAS_Upper || diag == KBLAS_Unit ){
    #ifdef USE_MAGMA
    wss->d_ptrs_bytes = kmax(2 * size_t(batchCount) * sizeof(void*), wss->d_ptrs_bytes);
    #endif
  // }
  if( ( (side == KBLAS_Right) && (n > 16) ) ||
      ( (side == KBLAS_Left ) && (m > 16) ) ){
    if(STRIDED){
      gemm_batch_strided_wsquery_core(batchCount, wss);
    }else{
      gemm_batch_offset_wsquery_core( batchCount, 1, wss);
    }
  }
}

//==============================================================================================
template<bool STRIDED>
inline
void symm_batch_wsquery_core( int batchCount, kblasWorkspaceState_t wss)
{
  if(STRIDED){
    wss->d_ptrs_bytes = kmax(3 * size_t(batchCount) * sizeof(void*), wss->d_ptrs_bytes);
  }
}
void symm_batch_nonuniform_wsquery_core(kblasWorkspaceState_t ws);

//==============================================================================================
template<bool STRIDED>
inline
void potrf_batch_wsquery_core(const int n, int batchCount, kblasWorkspaceState_t wss)
{
  int n1 = CLOSEST_REG_SIZE(n);

  trsm_batch_wsquery_core<STRIDED>( batchCount,
                                    KBLAS_Right, n-n1, n1,
                                    wss);

  syrk_batch_wsquery_core( n-n1, batchCount, wss);
}

//==============================================================================================
template<bool STRIDED>
inline
void lauum_batch_wsquery_core(const int n, int batchCount, kblasWorkspaceState_t wss)
{
  int n1 = CLOSEST_REG_SIZE(n);

  trmm_batch_wsquery_core<STRIDED>( batchCount,
                                    KBLAS_Left, n-n1, n1,
                                    wss);

  syrk_batch_wsquery_core( n1, batchCount, wss);
}

//==============================================================================================
template<bool STRIDED>
inline
void trtri_batch_wsquery_core(const int n, int batchCount, kblasWorkspaceState_t wss)
{
  if(n > 16){
    int n1 = CLOSEST_REG_SIZE(n);

    trsm_batch_wsquery_core<STRIDED>( batchCount,
                                      KBLAS_Left, n-n1, n1,
                                      wss);

    trsm_batch_wsquery_core<STRIDED>( batchCount,
                                      KBLAS_Right, n-n1, n1,
                                      wss);
  }
}

//==============================================================================================
template<bool STRIDED>
inline
void potrs_batch_wsquery_core(const int m, const int n, int batchCount, kblasWorkspaceState_t wss)
{
  int n1 = CLOSEST_REG_SIZE(n);

  trsm_batch_wsquery_core<STRIDED>( batchCount,
                                    KBLAS_Right, m, n1,
                                    wss);
  if(STRIDED){
    gemm_batch_strided_wsquery_core(batchCount, wss);
  }else{
    gemm_batch_offset_wsquery_core( batchCount, 1, wss);
  }
}

//==============================================================================================
template<bool STRIDED>
inline
void potri_batch_wsquery_core(const int n, int batchCount, kblasWorkspaceState_t wss)
{
  trtri_batch_wsquery_core<STRIDED>( n, batchCount, wss);

  lauum_batch_wsquery_core<STRIDED>( n, batchCount, wss);
}

//==============================================================================================
template<bool STRIDED>
inline
void poti_batch_wsquery_core(const int n, int batchCount, kblasWorkspaceState_t wss)
{
  potrf_batch_wsquery_core<STRIDED>( n, batchCount, wss);

  potri_batch_wsquery_core<STRIDED>( n, batchCount, wss);
}

//==============================================================================================
template<bool STRIDED>
inline
void posv_batch_wsquery_core(const int m, const int n, char side, int batchCount, kblasWorkspaceState_t wss)
{
  potrf_batch_wsquery_core<STRIDED>( (side == KBLAS_Right) ? n : m, batchCount, wss);

  potrs_batch_wsquery_core<STRIDED>( m, n, batchCount, wss);
}

//==============================================================================================
//==============================================================================================
//TODO defined elsewhere also
#define SHARED_SVD_DIM_LIMIT 64
#define OSBJ_BS   SHARED_SVD_DIM_LIMIT / 2

template<class T>
void batch_svd_osbj_workspace(int rows, int cols, int num_ops, KBlasWorkspaceState& requested_ws, int top_level)
{
  if(cols <= SHARED_SVD_DIM_LIMIT && rows <= SHARED_SVD_DIM_LIMIT)
    return;

  int block_cols = iDivUp(cols, OSBJ_BS);

  requested_ws.d_data_bytes += size_t(
    4 * OSBJ_BS * OSBJ_BS + // Gram matrix or R
    rows * 2 * OSBJ_BS    + // Temporary block column
    2 * OSBJ_BS           + // Temporary singular values
    2 * OSBJ_BS           + // tau
    1                       // Offdiagonal sum
  ) * sizeof(T) * num_ops;

  requested_ws.d_ptrs_bytes += (block_cols + 7) * sizeof(T*) * num_ops;
}

//==============================================================================================
template<class T>
void batch_tall_svd_workspace(int rows, int cols, int num_ops, KBlasWorkspaceState& requested_ws, int top_level)
{
  requested_ws.d_data_bytes += size_t(
    cols * cols + // R
    rows * cols + // Q
    cols          // tau
  ) * sizeof(T) * num_ops;

  requested_ws.d_ptrs_bytes += 4 * sizeof(T*) * num_ops;
}

//==============================================================================================
template<class T>
void batch_wide_svd_workspace(int rows, int cols, int num_ops, KBlasWorkspaceState& requested_ws, int top_level)
{
  requested_ws.d_data_bytes += size_t(
    rows * rows + // R
    cols * rows + // Q
    rows          // tau
  ) * sizeof(T) * num_ops;

  requested_ws.d_ptrs_bytes += 3 * sizeof(T*) * num_ops;

  // Do we need to do osbj of the rows x rows matrix?
  if(!top_level && rows > SHARED_SVD_DIM_LIMIT)
    batch_svd_osbj_workspace<T>(rows, rows, num_ops, requested_ws, 0);
}

//==============================================================================================
template<class T>
void batch_svd_randomized_workspace(int rows, int cols, int rank, int num_ops, KBlasWorkspaceState& requested_ws, int top_level)
{
  if(rank > cols || rank > rows) return;

  if(rank < cols)
  {
    requested_ws.d_data_bytes += size_t(
      cols * rank + // Omega
      rows * rank + // Y
      rank * cols + // B
      rank * rank + // R
      rank          // tau
    ) * sizeof(T) * num_ops;

    requested_ws.d_ptrs_bytes += 5 * sizeof(T*) * num_ops;
  }

  // Do we need to do osbj of the rank x rank matrix?
  if(!top_level && rank > SHARED_SVD_DIM_LIMIT)
    batch_svd_osbj_workspace<T>(rank, rank, num_ops, requested_ws, 0);
}
//==============================================================================================
template<class T, bool STRIDED, bool UNIFORM>
inline
void svd_full_batch_wsquery_core( int m, int n, int rank, int batchCount, SVD_method variant,
                                  kblasWorkspaceState_t ws_full, kblasWorkspaceState_t ws_child = NULL)
{
  KBlasWorkspaceState ws_local;
  //svd workspace
  if(variant <= SVD_Jacobi_gram){

    if(m > SHARED_SVD_DIM_LIMIT || n > SHARED_SVD_DIM_LIMIT){
      if(variant == SVD_Jacobi && n <= SHARED_SVD_DIM_LIMIT)
        batch_tall_svd_workspace<T>(m, n, batchCount, ws_local, 0);
      else
        batch_svd_osbj_workspace<T>(m, n, batchCount, ws_local, 0);
    }

  }else
  if(variant == SVD_random){

    batch_svd_randomized_workspace<T>(m, n, rank, batchCount, ws_local, 0);

  }else{
    return;
  }

  if(UNIFORM)
    gemm_batch_strided_wsquery_core(batchCount, &ws_local);

  if(ws_child != NULL)
    ws_child->set(&ws_local);

  ws_local.d_data_bytes = kblas_roundup_s(ws_local.d_data_bytes, sizeof(T));
  //A copy
  size_t work_size = kblas_roundup(m, align) * n * size_t(batchCount) * sizeof(T);

  ws_local.d_data_bytes += work_size;

  if(!UNIFORM || !STRIDED)
    ws_local.d_ptrs_bytes += size_t(batchCount) * sizeof(T*);

  ws_full->pad(&ws_local);
}

//==============================================================================================
//==============================================================================================
template<class T>
inline
void gemm_lr_wsquery_core( const int N, int kA, int kB,
                            kblasWorkspaceState_t wss)
{
  int ldW = kblas_roundup(kA, align);

  wss->d_data_bytes = kmax(ldW * size_t(kB + N) * sizeof(T), wss->d_data_bytes);
}

//==============================================================================================
template<class T, bool STRIDED>
inline
void gemm_lr_batch_wsquery_core( const int N, int kA, int kB,
                                  int batchCount,
                                  kblasWorkspaceState_t wss)
{
  int ldW = kblas_roundup(kA, align);

  wss->d_data_bytes = kmax(size_t(batchCount) * ldW * size_t(kB + N) * sizeof(T), wss->d_data_bytes);

  if (STRIDED){
    gemm_batch_strided_wsquery_core(batchCount, wss);
  }else{
    wss->d_ptrs_bytes = kmax(size_t(batchCount) * sizeof(T*) * 2, wss->d_ptrs_bytes);
  }
}

//==============================================================================================
//==============================================================================================
template<class T>
inline
void gemm_lr_wsquery_core( const int M, const int N,
                            int kA, int kB, int kC, int max_rk,
                            kblasWorkspaceState_t wss)
{
  //TODO
  size_t d_data_bytes = 0;
  int CUV_ncols = kC + kA;
  bool clone_CUV = CUV_ncols > max_rk;

  d_data_bytes += kblas_roundup(kA, align) * kB;//AvtBu
  d_data_bytes += CUV_ncols;//kmin(M, CUV_ncols);//qrtauA
  d_data_bytes += CUV_ncols;//kmin(N, CUV_ncols);//qrtauB
  int rA_cols = CUV_ncols;
  d_data_bytes += kblas_roundup(rA_cols, align) * rA_cols;//rA
  if(N < CUV_ncols){
    d_data_bytes += kblas_roundup(rA_cols, align) * rA_cols;//rB
  }
  d_data_bytes += kblas_roundup(CUV_ncols, align) * kmin(CUV_ncols, max_rk) * 2;//Tu & Tv
  d_data_bytes += CUV_ncols;//d_sigma

  d_data_bytes += 2;//acc & rank

  if(clone_CUV){
    d_data_bytes += kblas_roundup(M, align) * CUV_ncols;//_Cu
    d_data_bytes += kblas_roundup(N, align) * CUV_ncols;//_Cv
  }else{
    d_data_bytes += kblas_roundup(kmax(M, N), align) * kmin(max_rk, CUV_ncols);//newUV
  }
  /*
  svd_full_batch_wsquery_core<T>( CUV_ncols, CUV_ncols, max_rk, 1, SVD_random, wss);
  wss->d_data_bytes += d_data_bytes * sizeof(T);
  /*/
  wss->d_data_bytes = kmax(size_t(d_data_bytes) * sizeof(T), wss->d_data_bytes);//*/
}


//==============================================================================================
template<class T, bool STRIDED>
inline
void gemm_lr_batch_wsquery_core( const int M, const int N,
                                  int kA, int kB, int kC, int max_rk,
                                  int batchCount,
                                  kblasWorkspaceState_t ws_full, kblasWorkspaceState_t ws_child = NULL)
{
  KBlasWorkspaceState ws_needed;
  size_t d_data_bytes = 0,
       d_ptrs_bytes = 0;
  int CUV_ncols = kC + kA;
  bool clone_CUV = CUV_ncols > max_rk;
  /*TODO
  svd_full_batch_wsquery_core<T>( CUV_ncols, CUV_ncols, max_rk, 1, SVD_random, wss);
  /*/
  gemm_batch_wsquery_core<STRIDED, true>(batchCount, 1, &ws_needed);
  trmm_batch_wsquery_core<STRIDED>(batchCount,
                                KBLAS_Right, CUV_ncols, CUV_ncols,
                                &ws_needed);
  if(ws_child != NULL)
    ws_child->set(&ws_needed);


  d_data_bytes += kblas_roundup(kA, align) * kB;//AvtBu
  d_ptrs_bytes += batchCount;
  d_data_bytes += CUV_ncols;//kmin(M, CUV_ncols);//qrtauA
  d_ptrs_bytes += batchCount;
  d_data_bytes += CUV_ncols;//kmin(N, CUV_ncols);//qrtauB
  d_ptrs_bytes += batchCount;
  int rA_cols = CUV_ncols;
  d_data_bytes += kblas_roundup(rA_cols, align) * rA_cols;//rA
  d_ptrs_bytes += batchCount;
  if(N < CUV_ncols){
    d_data_bytes += kblas_roundup(rA_cols, align) * rA_cols;//rB
    d_ptrs_bytes += batchCount;
  }
  d_data_bytes += kblas_roundup(CUV_ncols, align) * kmin(CUV_ncols, max_rk) * 2;//Tu & Tv
  d_ptrs_bytes += 2*batchCount;
  d_data_bytes += CUV_ncols;//d_sigma
  d_ptrs_bytes += batchCount;

  if(clone_CUV){
    d_data_bytes += kblas_roundup(M, align) * CUV_ncols;//_Cu
    d_ptrs_bytes += batchCount;
    d_data_bytes += kblas_roundup(N, align) * CUV_ncols;//_Cv
    d_ptrs_bytes += batchCount;
  }else{
    d_data_bytes += kblas_roundup(kmax(M, N), align) * kmin(max_rk, CUV_ncols);//newUV
    d_ptrs_bytes += batchCount;
  }
  d_data_bytes *= size_t(batchCount) * sizeof(T);

  d_data_bytes = kblas_roundup_s((size_t)d_data_bytes, sizeof(double));

  d_data_bytes += size_t(batchCount) * sizeof(double);//acc

  d_data_bytes = kblas_roundup_s((size_t)d_data_bytes, sizeof(int));
  d_data_bytes += size_t(batchCount) * sizeof(int);//rank

  d_ptrs_bytes *= sizeof(T*);

  ws_needed.d_data_bytes += d_data_bytes;
  ws_needed.d_ptrs_bytes += d_ptrs_bytes;

  ws_full->pad(&ws_needed);
}

//==============================================================================================
template<class T, bool STRIDED, bool UNIFORM>
inline
void gemm_tlr_wsquery_core( const int MTiles, const int NTiles, int rA, int rB,
                            const int mb, const int nb,
                            kblasWorkspaceState_t ws_full, kblasWorkspaceState_t ws_child = NULL)
{
  int batchCount = MTiles * NTiles;

  if(STRIDED && batchCount == 1){
    gemm_lr_wsquery_core<T>( nb, rA, rB,
                              ws_full);
  }else{

    KBlasWorkspaceState ws_needed;
    if(UNIFORM){
      gemm_lr_batch_wsquery_core<T, false>( nb, rA, rB, batchCount, &ws_needed);
    }else{
      // gemm_plr_batch_nonuniform_wsquery_core<T>(nb, rA, rB, batchCount, &ws_needed);
    }

    if(ws_child != NULL)
      ws_child->set(&ws_needed);

    ws_needed.d_ptrs_bytes += size_t(batchCount) * sizeof(T*) * 5;

    if(STRIDED)
      ws_needed.h_ptrs_bytes += size_t(batchCount) * sizeof(T*) * 5;

    if(!UNIFORM){
      // ws_needed.d_data_bytes = kblas_roundup(ws_needed.d_data_bytes, sizeof(int));
      // ws_needed.d_data_bytes += batchCount * 2 * sizeof(int);
    }

    ws_full->pad(&ws_needed);
  }
}

//==============================================================================================
template<class T, bool UNIFORM>
inline
void gemm_tlr_wsquery_core( const int MTiles, const int NTiles,
                                  int rA, int rB, int rC, int max_rk,
                                  const int mb, const int nb,
                                  kblasWorkspaceState_t ws_full, kblasWorkspaceState_t ws_child = NULL)
{
  int batchCount = MTiles * NTiles;

  KBlasWorkspaceState ws_needed;
  if(UNIFORM){
    gemm_lr_batch_wsquery_core<T, false>( mb, nb, rA, rB, rC, max_rk, batchCount, &ws_needed);
  }else{
    // TODO
    // gemm_plr_batch_nonuniform_wsquery_core<T>(nb, rA, rB, batchCount, &ws_needed);
  }

  if(ws_child != NULL)
    ws_child->set(&ws_needed);

  ws_needed.d_ptrs_bytes += size_t(batchCount) * sizeof(T*) * 6;

  // if(!UNIFORM){
  //   ws_needed.d_data_bytes = kblas_roundup(ws_needed.d_data_bytes, sizeof(int));
  //   ws_needed.d_data_bytes += batchCount * 2 * sizeof(int);
  // }

  ws_full->pad(&ws_needed);
}


#endif //__KBLAS_WORKSPACE_QUERIES_H__
