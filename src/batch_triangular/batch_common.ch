/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_triangular/batch_common.ch

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 2.0.0
 * @author Ali Charara
 * @date 2017-11-13
 **/

#ifndef __KBLAS_BATCH_COMMON_H__
#define __KBLAS_BATCH_COMMON_H__



//==============================================================================================
/// Query workspace needed for batch GEMM with offset
void gemm_batch_offset_wsquery_core(int batchCount,
                                    int A_row_off, int A_col_off,
                                    int B_row_off, int B_col_off,
                                    int C_row_off, int C_col_off,
                                    kblasWorkspaceState_t ws);

/// Query workspace needed for batch strided GEMM
void gemm_batch_strided_wsquery_core(int batchCount, kblasWorkspaceState_t ws);

//==============================================================================================
void syrk_batch_wsquery_core(const int m, int batchCount, kblasWorkspaceState_t ws);

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
      gemm_batch_offset_wsquery_core( batchCount,
                                      1, 1, 1, 1, 1, 1,
                                      wss);
    }
  }
}

//==============================================================================================
template<bool STRIDED>
inline
void trmm_batch_wsquery_core( int batchCount,
                              char side, int m, int n,
                              kblasWorkspaceState_t wss)
{
  if( ( (side == KBLAS_Right) && (n > 16) ) ||
      ( (side == KBLAS_Left ) && (m > 16) ) ){
    if(STRIDED){
      gemm_batch_strided_wsquery_core(batchCount, wss);
    }else{
      gemm_batch_offset_wsquery_core( batchCount,
                                      1, 1, 1, 1, 1, 1,
                                      wss);
    }
  }
}

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
    gemm_batch_offset_wsquery_core( batchCount,
                                    1, 1, 1, 1, 1, 1,
                                    wss);
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

#endif //__KBLAS_BATCH_COMMON_H__
