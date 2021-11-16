/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file include/kblas_tlr.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

#ifndef _KBLAS_TLR_H_
#define _KBLAS_TLR_H_


//###############################################################
// GEMM TLR
//###############################################################

/** @addtogroup CPP_API
*  @{
*/
#ifdef __cplusplus
    /**
     * @name TODO
     */
    //@{
    //------------------------------------------------------------------------------
    /**
     * @brief TODO
     *
     * @param[in] handle  KBLAS handle
     * @param[in] transA  specifies the form of op( A ) to be used in
     *                    the matrix multiplication as follows:
     *                    = KBLAS_NoTrans:   op( A ) = A.
     *                    = KBLAS_Trans:     op( A ) = A**T.
     * @param[in] transB  specifies the form of op( B ) to be used in
     *                    the matrix multiplication as follows:
     *                    = KBLAS_NoTrans:   op( B ) = B.
     *                    = KBLAS_Trans:     op( B ) = B**T.
     * @param[in] m       specifies the number of rows of the matrices
     *                    op( A ) and of the matrices C. Must be at least zero.
     * @param[in] n       specifies the number of columns of the matrices
     *                    op( B ) and of the matrices C. Must be at least zero.
     * @param[in] k             specifies the number of columns of each matrix
     *                          op( A ) and the number of rows of each matrix op( B ). Must
     *                          be at least  zero.
     * @param[in] alpha         Specifies the scalar alpha.
     * @param[in] A_array       Host pointer to array of device pointers to device buffers.
     *                          Each buffer contians a matrix of dimension (lda, ka), where ka
     *                          is k when transA is KBLAS_NoTrans, m otherwise. Leading (m, ka)
     *                          of each buffer must contain the data of A's.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] B_array       Host pointer to array of device pointers to device buffers.
     *                          Each buffer contians a matrix of dimension (ldb, kb), where kb
     *                          is n when transB is KBLAS_NoTrans, k otherwise. Leading (k, kb)
     *                          of each buffer must contain the data of B's.
     * @param[in] ldb           Leading dimension of each matrix of B.
     * @param[in] beta          Specifies the scalar beta.
     * @param[in, out] C_array  Host pointer to array of device pointers to device buffers.
     *                          Each buffer contians a matrix of dimension (ldc, n). Leading (m, n)
     *                          of each buffer must contain the data of C's.
     * @param[in] ldc           Leading dimension of each matrix of C.
     * @param[in] batchCount    Number of matrices to be batch processed
     *
     * Workspace needed: none
     *
     */
    int kblas_gemm_lr(kblasHandle_t handle,
                      char transA, char transB,
                      const int M, const int N, const int K,
                      const float alpha,
                      const float* Au, int ldAu, const float* Av, int ldAv, int kA,
                      const float* Bu, int ldBu, const float* Bv, int ldBv, int kB,
                      const float beta,
                            float* C, int ldC);

    /**
     * @brief TODO
     */
    int kblas_gemm_lr(kblasHandle_t handle,
                      char transA, char transB,
                      const int M, const int N, const int K,
                      const double alpha,
                      const double* Au, int ldAu, const double* Av, int ldAv, int kA,
                      const double* Bu, int ldBu, const double* Bv, int ldBv, int kB,
                      const double beta,
                            double* C, int ldC);

    /**
     * @brief TODO
     */
    int kblas_gemm_lr(kblasHandle_t handle,
                      char transA, char transB,
                      const int M, const int N, const int K,
                      const hipFloatComplex alpha,
                      const hipFloatComplex* Au, int ldAu, const hipFloatComplex* Av, int ldAv, int kA,
                      const hipFloatComplex* Bu, int ldBu, const hipFloatComplex* Bv, int ldBv, int kB,
                      const hipFloatComplex beta,
                            hipFloatComplex* C, int ldC);

    /**
     * @brief TODO
     */
    int kblas_gemm_lr(kblasHandle_t handle,
                      char transA, char transB,
                      const int M, const int N, const int K,
                      const hipDoubleComplex alpha,
                      const hipDoubleComplex* Au, int ldAu, const hipDoubleComplex* Av, int ldAv, int kA,
                      const hipDoubleComplex* Bu, int ldBu, const hipDoubleComplex* Bv, int ldBv, int kB,
                      const hipDoubleComplex beta,
                            hipDoubleComplex* C, int ldC);

    //@}

    /**
     * @name TODO
     */
    //@{

    /**
     * @brief TODO
     */
    int kblas_gemm_lr( kblasHandle_t handle,
                        char transA, char transB,
                        const int M, const int N, const int K,
                        const float alpha,
                        const float* Au, int ldAu, const float* Av, int ldAv, int kA,
                        const float* Bu, int ldBu, const float* Bv, int ldBv, int kB,
                        const float beta,
                              float* Cu, int ldCu,       float* Cv, int ldCv, int& kC,
                        int max_rk, double max_acc);
    /**
     * @brief TODO
     */
    int kblas_gemm_lr( kblasHandle_t handle,
                        char transA, char transB,
                        const int M, const int N, const int K,
                        const double alpha,
                        const double* Au, int ldAu, const double* Av, int ldAv, int kA,
                        const double* Bu, int ldBu, const double* Bv, int ldBv, int kB,
                        const double beta,
                              double* Cu, int ldCu,       double* Cv, int ldCv, int& kC,
                        int max_rk, double max_acc);
    //@}

    /**
     * @name TODO
     */
    //@{

    /**
     * @brief TODO
     */
    int kblas_gemm_lr_batch( kblasHandle_t handle,
                              char transA, char transB,
                              const int M, const int N, const int K,
                              const float alpha,
                              const float** Au_array, int ldAu,
                              const float** Av_array, int ldAv, int kA,
                              const float** Bu_array, int ldBu,
                              const float** Bv_array, int ldBv, int kB,
                              const float beta,
                                    float** C_array, int ldC,
                              int batchCount);
    /**
     * @brief TODO
     */
    int kblas_gemm_lr_batch( kblasHandle_t handle,
                              char transA, char transB,
                              const int M, const int N, const int K,
                              const double alpha,
                              const double** Au_array, int ldAu,
                              const double** Av_array, int ldAv, int kA,
                              const double** Bu_array, int ldBu,
                              const double** Bv_array, int ldBv, int kB,
                              const double beta,
                                    double** C_array, int ldC,
                              int batchCount);
    /**
     * @brief TODO
     */
    int kblas_gemm_lr_batch( kblasHandle_t handle,
                              char transA, char transB,
                              const int M, const int N, const int K,
                              const hipFloatComplex alpha,
                              const hipFloatComplex** Au_array, int ldAu,
                              const hipFloatComplex** Av_array, int ldAv, int kA,
                              const hipFloatComplex** Bu_array, int ldBu,
                              const hipFloatComplex** Bv_array, int ldBv, int kB,
                              const hipFloatComplex beta,
                                    hipFloatComplex** C_array, int ldC,
                              int batchCount);
    /**
     * @brief TODO
     */
    int kblas_gemm_lr_batch( kblasHandle_t handle,
                              char transA, char transB,
                              const int M, const int N, const int K,
                              const hipDoubleComplex alpha,
                              const hipDoubleComplex** Au_array, int ldAu,
                              const hipDoubleComplex** Av_array, int ldAv, int kA,
                              const hipDoubleComplex** Bu_array, int ldBu,
                              const hipDoubleComplex** Bv_array, int ldBv, int kB,
                              const hipDoubleComplex beta,
                                    hipDoubleComplex** C_array, int ldC,
                              int batchCount);
    //@}

    /**
     * @name TODO
     */
    //@{

    /**
     * @brief TODO
     */
    int kblas_gemm_lr_batch( kblasHandle_t handle,
                              char transA, char transB,
                              const int M, const int N, const int K,
                              const float alpha,
                              const float* Au, int ldAu, long strideAu,
                              const float* Av, int ldAv, long strideAv, int kA,
                              const float* Bu, int ldBu, long strideBu,
                              const float* Bv, int ldBv, long strideBv, int kB,
                              const float beta,
                                    float* C, int ldC, long strideC,
                              int batchCount);
    /**
     * @brief TODO
     */
    int kblas_gemm_lr_batch( kblasHandle_t handle,
                              char transA, char transB,
                              const int M, const int N, const int K,
                              const double alpha,
                              const double* Au, int ldAu, long strideAu,
                              const double* Av, int ldAv, long strideAv, int kA,
                              const double* Bu, int ldBu, long strideBu,
                              const double* Bv, int ldBv, long strideBv, int kB,
                              const double beta,
                                    double* C, int ldC, long strideC,
                              int batchCount);
    /**
     * @brief TODO
     */
    int kblas_gemm_lr_batch( kblasHandle_t handle,
                              char transA, char transB,
                              const int M, const int N, const int K,
                              const hipFloatComplex alpha,
                              const hipFloatComplex* Au, int ldAu, long strideAu,
                              const hipFloatComplex* Av, int ldAv, long strideAv, int kA,
                              const hipFloatComplex* Bu, int ldBu, long strideBu,
                              const hipFloatComplex* Bv, int ldBv, long strideBv, int kB,
                              const hipFloatComplex beta,
                                    hipFloatComplex* C, int ldC, long strideC,
                              int batchCount);
    /**
     * @brief TODO
     */
    int kblas_gemm_lr_batch( kblasHandle_t handle,
                              char transA, char transB,
                              const int M, const int N, const int K,
                              const hipDoubleComplex alpha,
                              const hipDoubleComplex* Au, int ldAu, long strideAu,
                              const hipDoubleComplex* Av, int ldAv, long strideAv, int kA,
                              const hipDoubleComplex* Bu, int ldBu, long strideBu,
                              const hipDoubleComplex* Bv, int ldBv, long strideBv, int kB,
                              const hipDoubleComplex beta,
                                    hipDoubleComplex* C, int ldC, long strideC,
                              int batchCount);
    //@}

    /**
     * @name TODO
     */
    //@{

    /**
     * @brief TODO
     */
    int kblas_gemm_lr_batch( kblasHandle_t handle,
                              char transA, char transB,
                              const int M, const int N, const int K,
                              const float alpha,
                              const float* Au, int ldAu, long strideAu,
                              const float* Av, int ldAv, long strideAv, int kA,
                              const float* Bu, int ldBu, long strideBu,
                              const float* Bv, int ldBv, long strideBv, int kB,
                              const float beta,
                                    float* Cu, int ldCu, long strideCu,
                                    float* Cv, int ldCv, long strideCv, int& kC,
                              int max_rk, double max_acc,
                              int batchCount);
    /**
     * @brief TODO
     */
    int kblas_gemm_lr_batch( kblasHandle_t handle,
                              char transA, char transB,
                              const int M, const int N, const int K,
                              const double alpha,
                              const double* Au, int ldAu, long strideAu,
                              const double* Av, int ldAv, long strideAv, int kA,
                              const double* Bu, int ldBu, long strideBu,
                              const double* Bv, int ldBv, long strideBv, int kB,
                              const double beta,
                                    double* Cu, int ldCu, long strideCu,
                                    double* Cv, int ldCv, long strideCv, int& kC,
                              int max_rk, double max_acc,
                              int batchCount);
    /**
     * @brief TODO
     */
    int kblas_gemm_lr_batch( kblasHandle_t handle,
                              char transA, char transB,
                              const int M, const int N, const int K,
                              const float alpha,
                              const float** Au, int ldAu, long strideAu,
                              const float** Av, int ldAv, long strideAv, int kA,
                              const float** Bu, int ldBu, long strideBu,
                              const float** Bv, int ldBv, long strideBv, int kB,
                              const float beta,
                                    float** Cu, int ldCu, long strideCu,
                                    float** Cv, int ldCv, long strideCv, int& kC,
                              int max_rk, double max_acc,
                              int batchCount);
    /**
     * @brief TODO
     */
    int kblas_gemm_lr_batch( kblasHandle_t handle,
                              char transA, char transB,
                              const int M, const int N, const int K,
                              const double alpha,
                              const double** Au, int ldAu, long strideAu,
                              const double** Av, int ldAv, long strideAv, int kA,
                              const double** Bu, int ldBu, long strideBu,
                              const double** Bv, int ldBv, long strideBv, int kB,
                              const double beta,
                                    double** Cu, int ldCu, long strideCu,
                                    double** Cv, int ldCv, long strideCv, int& kC,
                              int max_rk, double max_acc,
                              int batchCount);
    //@}

    /**
     * @name TODO
     */
    //@{

    /**
     * @brief TODO
     */
    int kblas_gemm_tlr( kblasHandle_t handle,
                              char transA, char transB,
                              const int MTiles, const int NTiles, const int KTiles,
                              const int mb, const int nb, const int kb,
                              const float alpha,
                              const float** Au_array, int ldAu,
                              const float** Av_array, int ldAv, int kA,
                              const float** Bu_array, int ldBu,
                              const float** Bv_array, int ldBv, int kB,
                              const float beta,
                                    float* C, int ldC);
    /**
     * @brief TODO
     */
    int kblas_gemm_tlr( kblasHandle_t handle,
                              char transA, char transB,
                              const int MTiles, const int NTiles, const int KTiles,
                              const int mb, const int nb, const int kb,
                              const double alpha,
                              const double** Au_array, int ldAu,
                              const double** Av_array, int ldAv, int kA,
                              const double** Bu_array, int ldBu,
                              const double** Bv_array, int ldBv, int kB,
                              const double beta,
                                    double* C, int ldC);
    /**
     * @brief TODO
     */
    int kblas_gemm_tlr( kblasHandle_t handle,
                              char transA, char transB,
                              const int MTiles, const int NTiles, const int KTiles,
                              const int mb, const int nb, const int kb,
                              const hipFloatComplex alpha,
                              const hipFloatComplex** Au_array, int ldAu,
                              const hipFloatComplex** Av_array, int ldAv, int kA,
                              const hipFloatComplex** Bu_array, int ldBu,
                              const hipFloatComplex** Bv_array, int ldBv, int kB,
                              const hipFloatComplex beta,
                                    hipFloatComplex* C, int ldC);
    /**
     * @brief TODO
     */
    int kblas_gemm_tlr( kblasHandle_t handle,
                              char transA, char transB,
                              const int MTiles, const int NTiles, const int KTiles,
                              const int mb, const int nb, const int kb,
                              const hipDoubleComplex alpha,
                              const hipDoubleComplex** Au_array, int ldAu,
                              const hipDoubleComplex** Av_array, int ldAv, int kA,
                              const hipDoubleComplex** Bu_array, int ldBu,
                              const hipDoubleComplex** Bv_array, int ldBv, int kB,
                              const hipDoubleComplex beta,
                                    hipDoubleComplex* C, int ldC);
    //@}



    /**
     * @name TODO
     */
    //@{

    /**
     * @brief TODO
     */
    int kblas_gemm_tlr( kblasHandle_t handle,
                              char transA, char transB,
                              const int MTiles, const int NTiles, const int KTiles,
                              const int mb, const int nb, const int kb,
                              const float alpha,
                              const float** d_Au, int ldAu,
                              const float** d_Av, int ldAv, int ld_Aptrs, int kA,
                              const float** d_Bu, int ldBu,
                              const float** d_Bv, int ldBv, int ld_Bptrs, int kB,
                              const float beta,
                                    float* C, int ldC);
    /**
     * @brief TODO
     */
    int kblas_gemm_tlr( kblasHandle_t handle,
                              char transA, char transB,
                              const int MTiles, const int NTiles, const int KTiles,
                              const int mb, const int nb, const int kb,
                              const double alpha,
                              const double** d_Au, int ldAu,
                              const double** d_Av, int ldAv, int ld_Aptrs, int kA,
                              const double** d_Bu, int ldBu,
                              const double** d_Bv, int ldBv, int ld_Bptrs, int kB,
                              const double beta,
                                    double* C, int ldC);
    //@}

    /**
     * @name TODO
     */
    //@{

    /**
     * @brief TODO
     */
    int kblas_gemm_tlr( kblasHandle_t handle,
                              char transA, char transB,
                              int MTiles, int NTiles, int KTiles,
                              int mb, int nb, int kb,
                              float alpha,
                              float** d_Au, int ldAu,
                              float** d_Av, int ldAv, int ld_Aptrs, int kA,
                              float** d_Bu, int ldBu,
                              float** d_Bv, int ldBv, int ld_Bptrs, int kB,
                              float beta,
                              float** d_Cu, int ldCu,
                              float** d_Cv, int ldCv, int ld_Cptrs, int& kC,
                              int max_rk, double max_acc);
    /**
     * @brief TODO
     */
    int kblas_gemm_tlr( kblasHandle_t handle,
                              char transA, char transB,
                              int MTiles, int NTiles, int KTiles,
                              int mb, int nb, int kb,
                              double alpha,
                              double** d_Au, int ldAu,
                              double** d_Av, int ldAv, int ld_Aptrs, int kA,
                              double** d_Bu, int ldBu,
                              double** d_Bv, int ldBv, int ld_Bptrs, int kB,
                              double beta,
                              double** d_Cu, int ldCu,
                              double** d_Cv, int ldCv, int ld_Cptrs, int& kC,
                              int max_rk, double max_acc);
    //@}

    /**
     * @name TODO
     */
    //@{

    /**
     * @brief TODO
     */
    int kblas_gemm_tlr( kblasHandle_t handle,
                              char transA, char transB,
                              const int MTiles, const int NTiles, const int KTiles,
                              const int* mb_array, const int* nb_array, const int* kb_array,
                              const int mb, const int nb, const int kb,
                              const float alpha,
                              const float** d_Au, int* ldAu,
                              const float** d_Av, int* ldAv, int ld_Aptrs, int* rA, int max_rA,
                              const float** d_Bu, int* ldBu,
                              const float** d_Bv, int* ldBv, int ld_Bptrs, int* rB, int max_rB,
                              const float beta,
                                    float* C, int ldC, int* ldC_array);
    /**
     * @brief TODO
     */
    int kblas_gemm_tlr( kblasHandle_t handle,
                              char transA, char transB,
                              const int MTiles, const int NTiles, const int KTiles,
                              const int* mb_array, const int* nb_array, const int* kb_array,
                              const int mb, const int nb, const int kb,
                              const double alpha,
                              const double** d_Au, int* ldAu,
                              const double** d_Av, int* ldAv, int ld_Aptrs, int* rA, int max_rA,
                              const double** d_Bu, int* ldBu,
                              const double** d_Bv, int* ldBv, int ld_Bptrs, int* rB, int max_rB,
                              const double beta,
                                    double* C, int ldC, int* ldC_array);
    //@}

#endif
/** @} */

/** @addtogroup C_API
*  @{
*/
#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @ingroup WSQUERY
     * @brief Workspace query for partial Low Rank GEMM routines.
     *
     * @param[in,out] handle KBLAS handle. On return, stores the needed workspace size in corresponding data field.
     * @param[in]     N Number of columns for output matrix C.
     * @param[in]     kA rank of input matrix A.
     * @param[in]     kB rank of input matrix B.
     */
    void kblasSgemm_lr_lld_wsquery(kblasHandle_t handle, const int N, int kA, int kB);
    void kblasDgemm_lr_lld_wsquery(kblasHandle_t handle, const int N, int kA, int kB);
    void kblasCgemm_lr_lld_wsquery(kblasHandle_t handle, const int N, int kA, int kB);
    void kblasZgemm_lr_lld_wsquery(kblasHandle_t handle, const int N, int kA, int kB);


    void kblasSgemm_lr_lll_wsquery(kblasHandle_t handle,
                                const int M, const int N,
                                int kA, int kB, int kC, int max_rk);
    void kblasDgemm_lr_lll_wsquery(kblasHandle_t handle,
                                const int M, const int N,
                                int kA, int kB, int kC, int max_rk);
    void kblasCgemm_lr_lll_wsquery(kblasHandle_t handle,
                                const int M, const int N,
                                int kA, int kB, int kC, int max_rk);
    void kblasZgemm_lr_lll_wsquery(kblasHandle_t handle,
                                const int M, const int N,
                                int kA, int kB, int kC, int max_rk);

    void kblasSgemm_lr_lll_batch_strided_wsquery(kblasHandle_t handle,
                                              const int M, const int N,
                                              int kA, int kB, int kC, int max_rk,
                                              int batchCount);
    void kblasDgemm_lr_lll_batch_strided_wsquery(kblasHandle_t handle,
                                              const int M, const int N,
                                              int kA, int kB, int kC, int max_rk,
                                              int batchCount);
    void kblasSgemm_lr_lll_batch_wsquery(kblasHandle_t handle,
                                      const int M, const int N,
                                      int kA, int kB, int kC, int max_rk,
                                      int batchCount);
    void kblasDgemm_lr_lll_batch_wsquery(kblasHandle_t handle,
                                      const int M, const int N,
                                      int kA, int kB, int kC, int max_rk,
                                      int batchCount);

    /**
     * @name TODO
     *
     * @{
     */
    /**
     * @brief TODO
     *
     * @param[in] handle  KBLAS handle
     * @param[in] transA  Specifies the form of op( A ) to be used in
     *                    the matrix multiplication as follows:
     *                    - KBLAS_NoTrans:   op( A ) = A.
     *                    - KBLAS_Trans:     op( A ) = A**T.
     * @param[in] transB  Specifies the form of op( B ) to be used in
     *                    the matrix multiplication as follows:
     *                    - KBLAS_NoTrans:   op( B ) = B.
     *                    - KBLAS_Trans:     op( B ) = B**T.
     * @param[in] m       Specifies the number of rows of the matrices
     *                    op( A ) and of the matrices C. Must be at least zero.
     * @param[in] n       Specifies the number of columns of the matrices
     *                    op( B ) and of the matrices C. Must be at least zero.
     * @param[in] k             Specifies the number of columns of each matrix
     *                          op( A ) and the number of rows of each matrix op( B ). Must
     *                          be at least  zero.
     * @param[in] alpha         Specifies the scalar alpha.
     * @param[in] A_array       Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (lda, ka), where ka
     *                          is k when transA is KBLAS_NoTrans, m otherwise. Leading (m, ka)
     *                          of each buffer must contain the data of A's.
     * @param[in] lda           Leading dimension of each matrix of A.
     * @param[in] B_array       Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (ldb, kb), where kb
     *                          is n when transB is KBLAS_NoTrans, k otherwise. Leading (k, kb)
     *                          of each buffer must contain the data of B's.
     * @param[in] ldb           Leading dimension of each matrix of B.
     * @param[in] beta          Specifies the scalar beta.
     * @param[in, out] C_array  Host pointer to array of device pointers to device buffers.
     *                          Each buffer contains a matrix of dimension (ldc, n). Leading (m, n)
     *                          of each buffer must contain the data of C's.
     * @param[in] ldc           Leading dimension of each matrix of C.
     * @param[in] batchCount    Number of matrices to be batch processed
     *
     * Workspace needed: none
     *
     */
    int kblasSgemm_lr_lld( kblasHandle_t handle,
                        char transA, char transB,
                        const int M, const int N, const int K,
                        const float alpha,
                        const float* Au, int ldAu, const float* Av, int ldAv, int kA,
                        const float* Bu, int ldBu, const float* Bv, int ldBv, int kB,
                        const float beta,
                              float* C, int ldC);
    /**
     * @brief TODO
     */
    int kblasDgemm_lr_lld( kblasHandle_t handle,
                        char transA, char transB,
                        const int M, const int N, const int K,
                        const double alpha,
                        const double* Au, int ldAu, const double* Av, int ldAv, int kA,
                        const double* Bu, int ldBu, const double* Bv, int ldBv, int kB,
                        const double beta,
                              double* C, int ldC);
    /**
     * @brief TODO
     */
    int kblasCgemm_lr_lld( kblasHandle_t handle,
                        char transA, char transB,
                        const int M, const int N, const int K,
                        const hipFloatComplex alpha,
                        const hipFloatComplex* Au, int ldAu, const hipFloatComplex* Av, int ldAv, int kA,
                        const hipFloatComplex* Bu, int ldBu, const hipFloatComplex* Bv, int ldBv, int kB,
                        const hipFloatComplex beta,
                              hipFloatComplex* C, int ldC);
    /**
     * @brief TODO
     */
    int kblasZgemm_lr_lld( kblasHandle_t handle,
                        char transA, char transB,
                        const int M, const int N, const int K,
                        const hipDoubleComplex alpha,
                        const hipDoubleComplex* Au, int ldAu, const hipDoubleComplex* Av, int ldAv, int kA,
                        const hipDoubleComplex* Bu, int ldBu, const hipDoubleComplex* Bv, int ldBv, int kB,
                        const hipDoubleComplex beta,
                              hipDoubleComplex* C, int ldC);
    //@}

    void kblasSgemm_lr_lld_batch_wsquery(kblasHandle_t handle, const int N, int kA, int kB, int batchCount);
    void kblasDgemm_lr_lld_batch_wsquery(kblasHandle_t handle, const int N, int kA, int kB, int batchCount);
    void kblasCgemm_lr_lld_batch_wsquery(kblasHandle_t handle, const int N, int kA, int kB, int batchCount);
    void kblasZgemm_lr_lld_batch_wsquery(kblasHandle_t handle, const int N, int kA, int kB, int batchCount);
    /**
     * @name TODO
     */
    //@{
    /**
     * @brief TODO
     */
    int kblasSgemm_lr_lld_batch( kblasHandle_t handle,
                              char transA, char transB,
                              const int M, const int N, const int K,
                              const float alpha,
                              const float** Au_array, int ldAu,
                              const float** Av_array, int ldAv, int kA,
                              const float** Bu_array, int ldBu,
                              const float** Bv_array, int ldBv, int kB,
                              const float beta,
                                    float** C_array, int ldC,
                              int batchCount);
    /**
     * @brief TODO
     */
    int kblasDgemm_lr_lld_batch( kblasHandle_t handle,
                              char transA, char transB,
                              const int M, const int N, const int K,
                              const double alpha,
                              const double** Au_array, int ldAu,
                              const double** Av_array, int ldAv, int kA,
                              const double** Bu_array, int ldBu,
                              const double** Bv_array, int ldBv, int kB,
                              const double beta,
                                    double** C_array, int ldC,
                              int batchCount);
    /**
     * @brief TODO
     */
    int kblasCgemm_lr_lld_batch( kblasHandle_t handle,
                              char transA, char transB,
                              const int M, const int N, const int K,
                              const hipFloatComplex alpha,
                              const hipFloatComplex** Au_array, int ldAu,
                              const hipFloatComplex** Av_array, int ldAv, int kA,
                              const hipFloatComplex** Bu_array, int ldBu,
                              const hipFloatComplex** Bv_array, int ldBv, int kB,
                              const hipFloatComplex beta,
                                    hipFloatComplex** C_array, int ldC,
                              int batchCount);
    /**
     * @brief TODO
     */
    int kblasZgemm_lr_lld_batch( kblasHandle_t handle,
                              char transA, char transB,
                              const int M, const int N, const int K,
                              const hipDoubleComplex alpha,
                              const hipDoubleComplex** Au_array, int ldAu,
                              const hipDoubleComplex** Av_array, int ldAv, int kA,
                              const hipDoubleComplex** Bu_array, int ldBu,
                              const hipDoubleComplex** Bv_array, int ldBv, int kB,
                              const hipDoubleComplex beta,
                                    hipDoubleComplex** C_array, int ldC,
                              int batchCount);
    //@}

    void kblasSgemm_lr_lld_batch_strided_wsquery(kblasHandle_t handle, const int N, int kA, int kB, int batchCount);
    void kblasDgemm_lr_lld_batch_strided_wsquery(kblasHandle_t handle, const int N, int kA, int kB, int batchCount);
    void kblasCgemm_lr_lld_batch_strided_wsquery(kblasHandle_t handle, const int N, int kA, int kB, int batchCount);
    void kblasZgemm_lr_lld_batch_strided_wsquery(kblasHandle_t handle, const int N, int kA, int kB, int batchCount);
    /**
     * @name TODO
     */
    //@{
    /**
     * @brief TODO
     */
    int kblasSgemm_lr_lld_batch_strided( kblasHandle_t handle,
                                      char transA, char transB,
                                      const int M, const int N, const int K,
                                      const float alpha,
                                      const float* Au, int ldAu, long strideAu,
                                      const float* Av, int ldAv, long strideAv, int kA,
                                      const float* Bu, int ldBu, long strideBu,
                                      const float* Bv, int ldBv, long strideBv, int kB,
                                      const float beta,
                                            float* C, int ldC, long strideC,
                                      int batchCount);
    /**
     * @brief TODO
     */
    int kblasDgemm_lr_lld_batch_strided( kblasHandle_t handle,
                                      char transA, char transB,
                                      const int M, const int N, const int K,
                                      const double alpha,
                                      const double* Au, int ldAu, long strideAu,
                                      const double* Av, int ldAv, long strideAv, int kA,
                                      const double* Bu, int ldBu, long strideBu,
                                      const double* Bv, int ldBv, long strideBv, int kB,
                                      const double beta,
                                            double* C, int ldC, long strideC,
                                      int batchCount);
    /**
     * @brief TODO
     */
    int kblasCgemm_lr_lld_batch_strided( kblasHandle_t handle,
                                      char transA, char transB,
                                      const int M, const int N, const int K,
                                      const hipFloatComplex alpha,
                                      const hipFloatComplex* Au, int ldAu, long strideAu,
                                      const hipFloatComplex* Av, int ldAv, long strideAv, int kA,
                                      const hipFloatComplex* Bu, int ldBu, long strideBu,
                                      const hipFloatComplex* Bv, int ldBv, long strideBv, int kB,
                                      const hipFloatComplex beta,
                                            hipFloatComplex* C, int ldC, long strideC,
                                      int batchCount);
    /**
     * @brief TODO
     */
    int kblasZgemm_lr_lld_batch_strided( kblasHandle_t handle,
                                      char transA, char transB,
                                      const int M, const int N, const int K,
                                      const hipDoubleComplex alpha,
                                      const hipDoubleComplex* Au, int ldAu, long strideAu,
                                      const hipDoubleComplex* Av, int ldAv, long strideAv, int kA,
                                      const hipDoubleComplex* Bu, int ldBu, long strideBu,
                                      const hipDoubleComplex* Bv, int ldBv, long strideBv, int kB,
                                      const hipDoubleComplex beta,
                                            hipDoubleComplex* C, int ldC, long strideC,
                                      int batchCount);
    //@}

    void kblasSgemm_tlr_lld_wsquery(kblasHandle_t handle,
                                      const int MTiles, const int NTiles, int kA, int kB,
                                      const int mb, const int nb);
    void kblasDgemm_tlr_lld_wsquery(kblasHandle_t handle,
                                      const int MTiles, const int NTiles, int kA, int kB,
                                      const int mb, const int nb);
    void kblasCgemm_tlr_lld_wsquery(kblasHandle_t handle,
                                      const int MTiles, const int NTiles, int kA, int kB,
                                      const int mb, const int nb);
    void kblasZgemm_tlr_lld_wsquery(kblasHandle_t handle,
                                      const int MTiles, const int NTiles, int kA, int kB,
                                      const int mb, const int nb);
    void kblasSgemm_plr_dev_tiled_wsquery(kblasHandle_t handle,
                                          const int MTiles, const int NTiles, int kA, int kB,
                                          const int mb, const int nb);
    void kblasDgemm_plr_dev_tiled_wsquery(kblasHandle_t handle,
                                          const int MTiles, const int NTiles, int kA, int kB,
                                          const int mb, const int nb);
    void kblasSgemm_tlr_lll_wsquery(kblasHandle_t handle,
                                      const int MTiles, const int NTiles,
                                      int kA, int kB, int kC, int max_rk,
                                      const int mb, const int nb);
    void kblasDgemm_tlr_lll_wsquery(kblasHandle_t handle,
                                      const int MTiles, const int NTiles,
                                      int kA, int kB, int kC, int max_rk,
                                      const int mb, const int nb);

    /**
     * @name TODO
     */
    //@{
    /**
     * @brief TODO
     */
    int kblasSgemm_tlr_lld( kblasHandle_t handle,
                              char transA, char transB,
                              const int MTiles, const int NTiles, const int KTiles,
                              const int mb, const int nb, const int kb,
                              const float alpha,
                              const float** Au_array, int ldAu,
                              const float** Av_array, int ldAv, int kA,
                              const float** Bu_array, int ldBu,
                              const float** Bv_array, int ldBv, int kB,
                              const float beta,
                                    float* C, int ldC);
    /**
     * @brief TODO
     */
    int kblasDgemm_tlr_lld( kblasHandle_t handle,
                              char transA, char transB,
                              const int MTiles, const int NTiles, const int KTiles,
                              const int mb, const int nb, const int kb,
                              const double alpha,
                              const double** Au_array, int ldAu,
                              const double** Av_array, int ldAv, int kA,
                              const double** Bu_array, int ldBu,
                              const double** Bv_array, int ldBv, int kB,
                              const double beta,
                                    double* C, int ldC);
    /**
     * @brief TODO
     */
    int kblasCgemm_tlr_lld( kblasHandle_t handle,
                              char transA, char transB,
                              const int MTiles, const int NTiles, const int KTiles,
                              const int mb, const int nb, const int kb,
                              const hipFloatComplex alpha,
                              const hipFloatComplex** Au_array, int ldAu,
                              const hipFloatComplex** Av_array, int ldAv, int kA,
                              const hipFloatComplex** Bu_array, int ldBu,
                              const hipFloatComplex** Bv_array, int ldBv, int kB,
                              const hipFloatComplex beta,
                                    hipFloatComplex* C, int ldC);
    /**
     * @brief TODO
     */
    int kblasZgemm_tlr_lld( kblasHandle_t handle,
                              char transA, char transB,
                              const int MTiles, const int NTiles, const int KTiles,
                              const int mb, const int nb, const int kb,
                              const hipDoubleComplex alpha,
                              const hipDoubleComplex** Au_array, int ldAu,
                              const hipDoubleComplex** Av_array, int ldAv, int kA,
                              const hipDoubleComplex** Bu_array, int ldBu,
                              const hipDoubleComplex** Bv_array, int ldBv, int kB,
                              const hipDoubleComplex beta,
                                    hipDoubleComplex* C, int ldC);
    //@}
#ifdef __cplusplus
}
#endif
/** @} */


#endif // _KBLAS_TLR_H_
