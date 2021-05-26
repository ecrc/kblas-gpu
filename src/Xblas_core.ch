/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/Xblas_core.ch

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ali Charara
 * @date 2020-12-10
 **/

#ifndef __XBLAS_CORE__
#define __XBLAS_CORE__


#include "kblas_struct.h"
#include "kblas_prec_def.h"

//==============================================================================================
int kblasXgemm( kblasHandle_t handle,
                char transa, char transb,
                int m, int n, int k,
                const TYPE alpha, const TYPE *A, int lda,
                                  const TYPE *B, int ldb,
                const TYPE beta,        TYPE *C, int ldc);

int kblasXsyrk( kblasHandle_t handle,
                char uplo, char trans,
                int m, int n,
                const TYPE alpha, const TYPE* A, int lda,
                const TYPE beta,        TYPE* B, int ldb);

int kblasXsymm( kblasHandle_t handle,
                char side, char uplo,
                int m, int n,
                const TYPE alpha, const TYPE *A, int lda,
                                  const TYPE *B, int ldb,
                const TYPE beta,        TYPE *C, int ldc);

int kblasXtrsm( kblasHandle_t handle,
                char side, char uplo, char trans, char diag,
                int m, int n,
                const TYPE alpha,
                const TYPE* A, int lda,
                      TYPE* B, int ldb);

int kblasXtrmm( kblasHandle_t handle,
                char side, char uplo, char trans, char diag,
                int m, int n,
                const TYPE alpha,
                const TYPE* A, int lda,
                      TYPE* B, int ldb);

int kblasXscal( kblasHandle_t handle,
                int n,
                const TYPE alpha,
                TYPE *x, int incx);

int kblasXgeam( kblasHandle_t handle,
                char transa, char transb,
                int m, int n,
                const TYPE alpha, const TYPE *A, int lda,
                const TYPE beta,  const TYPE *B, int ldb,
                                        TYPE *C, int ldc);

//==============================================================================================

/// Query workspace needed for GEMM with offset
void kblas_gemm_batch_wsquery(kblasHandle_t handle,
                              int batchCount,
                              int A_row_off, int A_col_off,
                              int B_row_off, int B_col_off,
                              int C_row_off, int C_col_off);

/**
 * @brief Uniform-size batch non-strided GEMM with offset wrapper routine
 */
int kblas_gemm_batch( kblasHandle_t handle,
                      char transA, char transB,
                      const int m, const int n, const int k,
                      const TYPE alpha,
                      const TYPE** A, int A_row_off, int A_col_off, int lda,
                      const TYPE** B, int B_row_off, int B_col_off, int ldb,
                      const TYPE beta,
                            TYPE** C, int C_row_off, int C_col_off, int ldc,
                      int batchCount);

int Xgemm_batch(kblasHandle_t handle,
                char transA, char transB,
                int m, int n, int k,
                TYPE alpha,
                TYPE** A_array, int lda, long strideA,
                TYPE** B_array, int ldb, long strideB,
                TYPE beta,
                TYPE** C_array, int ldc, long strideC,
                int batchCount);

int Xgemm_batch(kblasHandle_t handle,
                char transA, char transB,
                int m, int n, int k,
                TYPE alpha,
                TYPE* A, int lda, long strideA,
                TYPE* B, int ldb, long strideB,
                TYPE beta,
                TYPE* C, int ldc, long strideC,
                int batchCount);

int Xgemm_batch(kblasHandle_t handle,
                char transA, char transB,
                int m, int n, int k,
                TYPE alpha,
                TYPE** A, int A_row_off, int A_col_off, int lda, long strideA,
                TYPE** B, int B_row_off, int B_col_off, int ldb, long strideB,
                TYPE beta,
                TYPE** C, int C_row_off, int C_col_off, int ldc, long strideC,
                int batchCount);

int Xgemm_batch(kblasHandle_t handle,
                char transA, char transB,
                int m, int n, int k,
                TYPE alpha,
                TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                TYPE* B, int B_row_off, int B_col_off, int ldb, long strideB,
                TYPE beta,
                TYPE* C, int C_row_off, int C_col_off, int ldc, long strideC,
                int batchCount);

int Xgemm_batch(kblasHandle_t handle,
                char transA, char transB,
                int* m, int* n, int* k,
                int max_m, int max_n, int max_k,
                const TYPE alpha,
                const TYPE** A, int A_row_off, int A_col_off, int* lda,
                const TYPE** B, int B_row_off, int B_col_off, int* ldb,
                const TYPE beta,
                      TYPE** C, int C_row_off, int C_col_off, int* ldc,
                int batchCount );

//==============================================================================================
int Xsymm_batch(kblasHandle_t handle,
                char side, char uplo,
                int m, int n,
                TYPE alpha, TYPE **dA_array, int ldda, long strideA,
                            TYPE **dB_array, int lddb, long strideB,
                TYPE beta,  TYPE **dC_array, int lddc, long strideC,
                int batchCount);

int Xsymm_batch(kblasHandle_t handle,
                char side, char uplo,
                int m, int n,
                TYPE alpha, TYPE *d_A, int ldda, long strideA,
                            TYPE *d_B, int lddb, long strideB,
                TYPE beta,  TYPE *d_C, int lddc, long strideC,
                int batchCount);

int Xsymm_batch(kblasHandle_t handle,
                char side, char uplo,
                int* m, int* n,
                int max_m, int max_n,
                TYPE alpha, TYPE **dA_array, int* ldda, long strideA,
                            TYPE **dB_array, int* lddb, long strideB,
                TYPE beta,  TYPE **dC_array, int* lddc, long strideC,
                int batchCount);

//==============================================================================================
int Xsyrk_batch(kblasHandle_t handle,
                char uplo, char trans,
                int m, int n,
                TYPE alpha, TYPE** A, int A_row_off, int A_col_off, int lda, long strideA,
                TYPE beta,  TYPE** B, int B_row_off, int B_col_off, int ldb, long strideB,
                int batchCount);
int Xsyrk_batch(kblasHandle_t handle,
                char uplo, char trans,
                int m, int n,
                TYPE alpha, TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                TYPE beta,  TYPE* B, int B_row_off, int B_col_off, int ldb, long strideB,
                int batchCount);
int Xsyrk_batch(kblasHandle_t handle,
                char uplo, char trans,
                int* m, int* n,
                int max_m, int max_n,
                TYPE alpha, TYPE** A, int* lda,
                TYPE beta,  TYPE** B, int* ldb,
                int batchCount);

//==============================================================================================
int Xtrsm_batch(kblasHandle_t handle,
                char side, char uplo, char trans, char diag,
                int m, int n,
                TYPE alpha,
                TYPE** A, int A_row_off, int A_col_off, int lda, long strideA,
                TYPE** B, int B_row_off, int B_col_off, int ldb, long strideB,
                int batchCount);
int Xtrsm_batch(kblasHandle_t handle,
                char side, char uplo, char trans, char diag,
                int m, int n,
                TYPE alpha,
                TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                TYPE* B, int B_row_off, int B_col_off, int ldb, long strideB,
                int batchCount);
int Xtrsm_batch(kblasHandle_t handle,
                char side, char uplo, char trans, char diag,
                int *m, int *n,
                int max_m, int max_n,
                TYPE alpha,
                TYPE** A, int A_row_off, int A_col_off, int* lda, long strideA,
                TYPE** B, int B_row_off, int B_col_off, int* ldb, long strideB,
                int batchCount);

//==============================================================================================
int Xtrmm_batch(kblasHandle_t handle,
                char side, char uplo, char trans, char diag,
                int m, int n,
                TYPE alpha,
                TYPE** A, int A_row_off, int A_col_off, int lda, long strideA,
                TYPE** B, int B_row_off, int B_col_off, int ldb, long strideB,
                int batchCount);

int Xtrmm_batch(kblasHandle_t handle,
                char side, char uplo, char trans, char diag,
                int m, int n,
                TYPE alpha,
                TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                TYPE* B, int B_row_off, int B_col_off, int ldb, long strideB,
                int batchCount);

//==============================================================================================
int Xpotrf_batch(kblasHandle_t handle,
                char uplo,
                const int n,
                TYPE** A, int A_row_off, int A_col_off, int lda, long strideA,
                int batchCount,
                int *info_array);

int Xpotrf_batch(kblasHandle_t handle,
                char uplo,
                const int n,
                TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                int batchCount,
                int *info_array);

int Xpotrf_batch_offset(kblasHandle_t handle,
                        char uplo,
                        const int n,
                        TYPE** A, int A_row_off, int A_col_off, int lda,
                        int batchCount,
                        int *info_array);

int Xpotrf_batch_offset(kblasHandle_t handle,
                        char uplo,
                        const int n,
                        TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                        int batchCount,
                        int *info_array);

//==============================================================================================
int Xpotrs_batch_offset(kblasHandle_t handle,
                        char side, char uplo,
                        const int m, const int n,
                        const TYPE** A, int A_row_off, int A_col_off, int lda,
                              TYPE** B, int B_row_off, int B_col_off, int ldb,
                        int batchCount);

int Xpotrs_batch_offset(kblasHandle_t handle,
                        char side, char uplo,
                        const int m, const int n,
                        const TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                              TYPE* B, int B_row_off, int B_col_off, int ldb, long strideB,
                        int batchCount);

//==============================================================================================
int Xtrtri_batch_offset(kblasHandle_t handle,
                        char uplo, char diag,
                        const int n,
                        TYPE** A, int A_row_off, int A_col_off, int lda,
                        int batchCount,
                        int *info_array);

int Xtrtri_batch_offset(kblasHandle_t handle,
                        char uplo, char diag,
                        const int n,
                        TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                        int batchCount,
                        int *info_array);

//==============================================================================================
int Xlauum_batch_offset(kblasHandle_t handle,
                        char uplo,
                        const int n,
                        TYPE** A, int A_row_off, int A_col_off, int lda,
                        int batchCount,
                        int *info_array);

int Xlauum_batch_offset(kblasHandle_t handle,
                        char uplo,
                        const int n,
                        TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                        int batchCount,
                        int *info_array);

//==============================================================================================
int Xpotri_batch_offset(kblasHandle_t handle,
                        char uplo,
                        const int n,
                        TYPE** A, int A_row_off, int A_col_off, int lda,
                        int batchCount,
                        int *info_array);

int Xpotri_batch_offset(kblasHandle_t handle,
                        char uplo,
                        const int n,
                        TYPE* A, int A_row_off, int A_col_off, int lda, long strideA,
                        int batchCount,
                        int *info_array);

//==============================================================================================
int Xgemm_lr(kblasHandle_t handle,
              char transA, char transB,
              const int M, const int N, const int K,
              const TYPE alpha, const TYPE* Au, int ldAu,
                                const TYPE* Av, int ldAv, int kA,
                                const TYPE* Bu, int ldBu,
                                const TYPE* Bv, int ldBv, int kB,
              TYPE beta,  TYPE* C, int ldC);

int Xgemm_lr_batch(kblasHandle_t handle,
                    char transA, char transB,
                    int M, int N, int K,
                    TYPE alpha, TYPE* Au, int ldAu, long strideAu,
                                TYPE* Av, int ldAv, long strideAv, int kA,
                                TYPE* Bu, int ldBu, long strideBu,
                                TYPE* Bv, int ldBv, long strideBv, int kB,
                    TYPE beta,  TYPE* C,  int ldC,  long strideC,
                    int batchCount);

int Xgemm_lr_batch(kblasHandle_t handle,
                    char transA, char transB,
                    int M, int N, int K,
                    TYPE alpha, TYPE** Au, int ldAu, long strideAu,
                                TYPE** Av, int ldAv, long strideAv, int kA,
                                TYPE** Bu, int ldBu, long strideBu,
                                TYPE** Bv, int ldBv, long strideBv, int kB,
                    TYPE beta,  TYPE** C,  int ldC,  long strideC,
                    int batchCount);

// int Xgemm_plr_batch(kblasHandle_t handle,
//                     char transA, char transB,
//                     int* M, int* N, int* K,
//                     int max_m, int max_n, int max_k,
//                     TYPE alpha, TYPE** Au, int* ldAu,
//                                 TYPE** Av, int* ldAv, int* kA, int max_kA,
//                                 TYPE** Bu, int* ldBu,
//                                 TYPE** Bv, int* ldBv, int* kB, int max_kB,
//                     TYPE beta,  TYPE** C,  int* ldC,
//                     int batchCount);

int Xgemm_lr_batch(kblasHandle_t handle,
                    char transA, char transB,
                    int M, int N, int K,
                    TYPE alpha,
                    TYPE* Au, int ldAu, long strideAu,
                    TYPE* Av, int ldAv, long strideAv, int kA,
                    TYPE* Bu, int ldBu, long strideBu,
                    TYPE* Bv, int ldBv, long strideBv, int kB,
                    TYPE beta,
                    TYPE* Cu, int ldCu, long strideCu,
                    TYPE* Cv, int ldCv, long strideCv, int& kC,
                    int max_rk, double max_acc,
                    int batchCount);

int Xgemm_lr_batch(kblasHandle_t handle,
                    char transA, char transB,
                    int M, int N, int K,
                    TYPE alpha,
                    TYPE** Au, int ldAu, long strideAu,
                    TYPE** Av, int ldAv, long strideAv, int kA,
                    TYPE** Bu, int ldBu, long strideBu,
                    TYPE** Bv, int ldBv, long strideBv, int kB,
                    TYPE beta,
                    TYPE** Cu, int ldCu, long strideCu,
                    TYPE** Cv, int ldCv, long strideCv, int& kC,
                    int max_rk, double max_acc,
                    int batchCount);

int Xgemm_tlr(kblasHandle_t handle,
                    char transA, char transB,
                    int MTiles, int NTiles, int KTiles,
                    int mb, int nb, int kb,
                    TYPE alpha,
                    TYPE** d_Au, int ldAu,
                    TYPE** d_Av, int ldAv, int ld_Aptrs, int kA,
                    TYPE** d_Bu, int ldBu,
                    TYPE** d_Bv, int ldBv, int ld_Bptrs, int kB,
                    TYPE beta,
                    TYPE** d_Cu, int ldCu,
                    TYPE** d_Cv, int ldCv, int ld_Cptrs, int& kC,
                    int max_rk, double max_acc);

//==============================================================================================
int Xacaf_batch(kblasHandle_t handle,
                int m, int n,
                TYPE** A, int lda, long strideA,
                TYPE** U, int ldu, long strideU,
                TYPE** V, int ldv, long strideV,
                TYPE** S, int lds, long strideS,
                double maxacc, int maxrk,
                double* acc, int* rk,
                int batchCount);

int Xacaf_batch(kblasHandle_t handle,
                int* m, int* n,
                int max_m, int max_n,
                TYPE** A, int* lda, long strideA,
                TYPE** U, int* ldu, long strideU,
                TYPE** V, int* ldv, long strideV,
                TYPE** S, int* lds, long strideS,
                double maxacc, int maxrk,
                double* acc, int* rk,
                int batchCount);

int Xacaf_batch(kblasHandle_t handle,
                int m, int n,
                TYPE* A, int lda, long strideA,
                TYPE* U, int ldu, long strideU,
                TYPE* V, int ldv, long strideV,
                TYPE* S, int lds, long strideS,
                double maxacc, int maxrk,
                double* acc, int* rk,
                int batchCount);

int Xacaf_batch(kblasHandle_t handle,
                int* m, int* n,
                int max_m, int max_n,
                TYPE* A, int lda, long strideA,
                TYPE* U, int ldu, long strideU,
                TYPE* V, int ldv, long strideV,
                TYPE* S, int lds, long strideS,
                double maxacc, int maxrk,
                double* acc, int* rk,
                int batchCount);

#endif// __XBLAS_CORE__

