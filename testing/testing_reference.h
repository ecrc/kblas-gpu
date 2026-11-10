/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/testing_reference.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Wajih Halim Boukaram
 * @date 2020-12-10
 **/

#ifndef __TESTING_REFERENCE_H__
#define __TESTING_REFERENCE_H__

#include <cmath>
#include <vector>

/*
 * CPU reference routines for the test checks, one overload per element type, built on the
 * C BLAS and LAPACKE interfaces that every supported CPU BLAS provides. Matrices are column
 * major and the option characters are the KBLAS ones, which are the BLAS and LAPACK letters.
 * testing_helper.h must come first, since it selects the CPU BLAS headers.
 */

inline CBLAS_SIDE ref_side(char side) { return side == KBLAS_Right ? CblasRight : CblasLeft; }
inline CBLAS_UPLO ref_uplo(char uplo) { return uplo == KBLAS_Upper ? CblasUpper : CblasLower; }
inline CBLAS_DIAG ref_diag(char diag) { return diag == KBLAS_Unit ? CblasUnit : CblasNonUnit; }
inline CBLAS_TRANSPOSE ref_trans(char trans)
{
  if(trans == KBLAS_Trans) return CblasTrans;
  if(trans == KBLAS_ConjTrans) return CblasConjTrans;
  return CblasNoTrans;
}

// LAPACKE takes complex arrays as its own complex type, which has the CUDA complex layout
inline lapack_complex_float*        ref_lapack(cuFloatComplex* A)        { return (lapack_complex_float*)A; }
inline const lapack_complex_float*  ref_lapack(const cuFloatComplex* A)  { return (const lapack_complex_float*)A; }
inline lapack_complex_double*       ref_lapack(cuDoubleComplex* A)       { return (lapack_complex_double*)A; }
inline const lapack_complex_double* ref_lapack(const cuDoubleComplex* A) { return (const lapack_complex_double*)A; }

//==============================================================================================
// C = alpha op(A) op(B) + beta C
inline void ref_gemm(char transA, char transB, int m, int n, int k,
                     float alpha, const float* A, int lda, const float* B, int ldb,
                     float beta, float* C, int ldc)
{ cblas_sgemm(CblasColMajor, ref_trans(transA), ref_trans(transB), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); }
inline void ref_gemm(char transA, char transB, int m, int n, int k,
                     double alpha, const double* A, int lda, const double* B, int ldb,
                     double beta, double* C, int ldc)
{ cblas_dgemm(CblasColMajor, ref_trans(transA), ref_trans(transB), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); }
inline void ref_gemm(char transA, char transB, int m, int n, int k,
                     cuFloatComplex alpha, const cuFloatComplex* A, int lda, const cuFloatComplex* B, int ldb,
                     cuFloatComplex beta, cuFloatComplex* C, int ldc)
{ cblas_cgemm(CblasColMajor, ref_trans(transA), ref_trans(transB), m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc); }
inline void ref_gemm(char transA, char transB, int m, int n, int k,
                     cuDoubleComplex alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb,
                     cuDoubleComplex beta, cuDoubleComplex* C, int ldc)
{ cblas_zgemm(CblasColMajor, ref_trans(transA), ref_trans(transB), m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc); }

// C = alpha op(A) op(A)^T + beta C, on the uplo triangle of C
inline void ref_syrk(char uplo, char trans, int n, int k,
                     float alpha, const float* A, int lda, float beta, float* C, int ldc)
{ cblas_ssyrk(CblasColMajor, ref_uplo(uplo), ref_trans(trans), n, k, alpha, A, lda, beta, C, ldc); }
inline void ref_syrk(char uplo, char trans, int n, int k,
                     double alpha, const double* A, int lda, double beta, double* C, int ldc)
{ cblas_dsyrk(CblasColMajor, ref_uplo(uplo), ref_trans(trans), n, k, alpha, A, lda, beta, C, ldc); }
inline void ref_syrk(char uplo, char trans, int n, int k,
                     cuFloatComplex alpha, const cuFloatComplex* A, int lda, cuFloatComplex beta, cuFloatComplex* C, int ldc)
{ cblas_csyrk(CblasColMajor, ref_uplo(uplo), ref_trans(trans), n, k, &alpha, A, lda, &beta, C, ldc); }
inline void ref_syrk(char uplo, char trans, int n, int k,
                     cuDoubleComplex alpha, const cuDoubleComplex* A, int lda, cuDoubleComplex beta, cuDoubleComplex* C, int ldc)
{ cblas_zsyrk(CblasColMajor, ref_uplo(uplo), ref_trans(trans), n, k, &alpha, A, lda, &beta, C, ldc); }

// B = alpha op(A)^-1 B (left side) or alpha B op(A)^-1 (right side), A triangular
inline void ref_trsm(char side, char uplo, char trans, char diag, int m, int n,
                     float alpha, const float* A, int lda, float* B, int ldb)
{ cblas_strsm(CblasColMajor, ref_side(side), ref_uplo(uplo), ref_trans(trans), ref_diag(diag), m, n, alpha, A, lda, B, ldb); }
inline void ref_trsm(char side, char uplo, char trans, char diag, int m, int n,
                     double alpha, const double* A, int lda, double* B, int ldb)
{ cblas_dtrsm(CblasColMajor, ref_side(side), ref_uplo(uplo), ref_trans(trans), ref_diag(diag), m, n, alpha, A, lda, B, ldb); }
inline void ref_trsm(char side, char uplo, char trans, char diag, int m, int n,
                     cuFloatComplex alpha, const cuFloatComplex* A, int lda, cuFloatComplex* B, int ldb)
{ cblas_ctrsm(CblasColMajor, ref_side(side), ref_uplo(uplo), ref_trans(trans), ref_diag(diag), m, n, &alpha, A, lda, B, ldb); }
inline void ref_trsm(char side, char uplo, char trans, char diag, int m, int n,
                     cuDoubleComplex alpha, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb)
{ cblas_ztrsm(CblasColMajor, ref_side(side), ref_uplo(uplo), ref_trans(trans), ref_diag(diag), m, n, &alpha, A, lda, B, ldb); }

// B = alpha op(A) B (left side) or alpha B op(A) (right side), A triangular
inline void ref_trmm(char side, char uplo, char trans, char diag, int m, int n,
                     float alpha, const float* A, int lda, float* B, int ldb)
{ cblas_strmm(CblasColMajor, ref_side(side), ref_uplo(uplo), ref_trans(trans), ref_diag(diag), m, n, alpha, A, lda, B, ldb); }
inline void ref_trmm(char side, char uplo, char trans, char diag, int m, int n,
                     double alpha, const double* A, int lda, double* B, int ldb)
{ cblas_dtrmm(CblasColMajor, ref_side(side), ref_uplo(uplo), ref_trans(trans), ref_diag(diag), m, n, alpha, A, lda, B, ldb); }
inline void ref_trmm(char side, char uplo, char trans, char diag, int m, int n,
                     cuFloatComplex alpha, const cuFloatComplex* A, int lda, cuFloatComplex* B, int ldb)
{ cblas_ctrmm(CblasColMajor, ref_side(side), ref_uplo(uplo), ref_trans(trans), ref_diag(diag), m, n, &alpha, A, lda, B, ldb); }
inline void ref_trmm(char side, char uplo, char trans, char diag, int m, int n,
                     cuDoubleComplex alpha, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb)
{ cblas_ztrmm(CblasColMajor, ref_side(side), ref_uplo(uplo), ref_trans(trans), ref_diag(diag), m, n, &alpha, A, lda, B, ldb); }

// y = alpha x + y on contiguous vectors
inline void ref_axpy(int n, float alpha, const float* x, float* y)                            { cblas_saxpy(n, alpha, x, 1, y, 1); }
inline void ref_axpy(int n, double alpha, const double* x, double* y)                         { cblas_daxpy(n, alpha, x, 1, y, 1); }
inline void ref_axpy(int n, cuFloatComplex alpha, const cuFloatComplex* x, cuFloatComplex* y)    { cblas_caxpy(n, &alpha, x, 1, y, 1); }
inline void ref_axpy(int n, cuDoubleComplex alpha, const cuDoubleComplex* x, cuDoubleComplex* y) { cblas_zaxpy(n, &alpha, x, 1, y, 1); }

//==============================================================================================
// Cholesky factorization of the uplo triangle of A; returns the LAPACK info
inline int ref_potrf(char uplo, int n, float* A, int lda)           { return LAPACKE_spotrf(LAPACK_COL_MAJOR, uplo, n, A, lda); }
inline int ref_potrf(char uplo, int n, double* A, int lda)          { return LAPACKE_dpotrf(LAPACK_COL_MAJOR, uplo, n, A, lda); }
inline int ref_potrf(char uplo, int n, cuFloatComplex* A, int lda)  { return LAPACKE_cpotrf(LAPACK_COL_MAJOR, uplo, n, ref_lapack(A), lda); }
inline int ref_potrf(char uplo, int n, cuDoubleComplex* A, int lda) { return LAPACKE_zpotrf(LAPACK_COL_MAJOR, uplo, n, ref_lapack(A), lda); }

// Inverse of A from its Cholesky factor in the uplo triangle; returns the LAPACK info
inline int ref_potri(char uplo, int n, float* A, int lda)           { return LAPACKE_spotri(LAPACK_COL_MAJOR, uplo, n, A, lda); }
inline int ref_potri(char uplo, int n, double* A, int lda)          { return LAPACKE_dpotri(LAPACK_COL_MAJOR, uplo, n, A, lda); }
inline int ref_potri(char uplo, int n, cuFloatComplex* A, int lda)  { return LAPACKE_cpotri(LAPACK_COL_MAJOR, uplo, n, ref_lapack(A), lda); }
inline int ref_potri(char uplo, int n, cuDoubleComplex* A, int lda) { return LAPACKE_zpotri(LAPACK_COL_MAJOR, uplo, n, ref_lapack(A), lda); }

// Product of the uplo triangle of A with its conjugate transpose; returns the LAPACK info
inline int ref_lauum(char uplo, int n, float* A, int lda)           { return LAPACKE_slauum(LAPACK_COL_MAJOR, uplo, n, A, lda); }
inline int ref_lauum(char uplo, int n, double* A, int lda)          { return LAPACKE_dlauum(LAPACK_COL_MAJOR, uplo, n, A, lda); }
inline int ref_lauum(char uplo, int n, cuFloatComplex* A, int lda)  { return LAPACKE_clauum(LAPACK_COL_MAJOR, uplo, n, ref_lapack(A), lda); }
inline int ref_lauum(char uplo, int n, cuDoubleComplex* A, int lda) { return LAPACKE_zlauum(LAPACK_COL_MAJOR, uplo, n, ref_lapack(A), lda); }

// Inverse of the triangular matrix in the uplo triangle of A; returns the LAPACK info
inline int ref_trtri(char uplo, char diag, int n, float* A, int lda)           { return LAPACKE_strtri(LAPACK_COL_MAJOR, uplo, diag, n, A, lda); }
inline int ref_trtri(char uplo, char diag, int n, double* A, int lda)          { return LAPACKE_dtrtri(LAPACK_COL_MAJOR, uplo, diag, n, A, lda); }
inline int ref_trtri(char uplo, char diag, int n, cuFloatComplex* A, int lda)  { return LAPACKE_ctrtri(LAPACK_COL_MAJOR, uplo, diag, n, ref_lapack(A), lda); }
inline int ref_trtri(char uplo, char diag, int n, cuDoubleComplex* A, int lda) { return LAPACKE_ztrtri(LAPACK_COL_MAJOR, uplo, diag, n, ref_lapack(A), lda); }

// Matrix norm of a general m x n matrix ('M' max entry, 'F' Frobenius, '1', 'I'); NaN when A holds a NaN
inline double ref_lange(char norm, int m, int n, const float* A, int lda)
{ std::vector<float> work(m); return LAPACKE_slange_work(LAPACK_COL_MAJOR, norm, m, n, A, lda, work.data()); }
inline double ref_lange(char norm, int m, int n, const double* A, int lda)
{ std::vector<double> work(m); return LAPACKE_dlange_work(LAPACK_COL_MAJOR, norm, m, n, A, lda, work.data()); }
inline double ref_lange(char norm, int m, int n, const cuFloatComplex* A, int lda)
{ std::vector<float> work(m); return LAPACKE_clange_work(LAPACK_COL_MAJOR, norm, m, n, ref_lapack(A), lda, work.data()); }
inline double ref_lange(char norm, int m, int n, const cuDoubleComplex* A, int lda)
{ std::vector<double> work(m); return LAPACKE_zlange_work(LAPACK_COL_MAJOR, norm, m, n, ref_lapack(A), lda, work.data()); }

// Matrix norm of the symmetric matrix stored in the uplo triangle of A; NaN when that triangle holds a NaN
inline double ref_lansy(char norm, char uplo, int n, const float* A, int lda)
{ std::vector<float> work(n); return LAPACKE_slansy_work(LAPACK_COL_MAJOR, norm, uplo, n, A, lda, work.data()); }
inline double ref_lansy(char norm, char uplo, int n, const double* A, int lda)
{ std::vector<double> work(n); return LAPACKE_dlansy_work(LAPACK_COL_MAJOR, norm, uplo, n, A, lda, work.data()); }
inline double ref_lansy(char norm, char uplo, int n, const cuFloatComplex* A, int lda)
{ std::vector<float> work(n); return LAPACKE_clansy_work(LAPACK_COL_MAJOR, norm, uplo, n, ref_lapack(A), lda, work.data()); }
inline double ref_lansy(char norm, char uplo, int n, const cuDoubleComplex* A, int lda)
{ std::vector<double> work(n); return LAPACKE_zlansy_work(LAPACK_COL_MAJOR, norm, uplo, n, ref_lapack(A), lda, work.data()); }

// Larger of two errors, NaN when either is NaN, so a failed result never reads as a small error
inline double ref_max_error(double a, double b) { return (std::isnan(a) || std::isnan(b)) ? NAN : fmax(a, b); }

#endif //__TESTING_REFERENCE_H__
