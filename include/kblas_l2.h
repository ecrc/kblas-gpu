/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file include/kblas_l2.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ahmad Abdelfattah
 * @date 2020-12-10
 **/

#ifndef _KBLAS_L2_H_
#define _KBLAS_L2_H_

#ifdef __cplusplus
extern "C" {
#endif

// scal
int kblas_cscal(int n, cuFloatComplex alpha, cuFloatComplex *x, int incx);
int kblas_dscal(int n, double alpha, double *x, int incx);
int kblas_sscal(int n, float alpha, float *x, int incx);
int kblas_zscal(int n, cuDoubleComplex alpha, cuDoubleComplex *x, int incx);

int kblas_cscal_async(int n, cuFloatComplex alpha, cuFloatComplex *x, int incx, cudaStream_t stream);
int kblas_dscal_async(int n, double alpha, double *x, int incx, cudaStream_t stream);
int kblas_sscal_async(int n, float alpha, float *x, int incx, cudaStream_t stream);
int kblas_zscal_async(int n, cuDoubleComplex alpha, cuDoubleComplex *x, int incx, cudaStream_t stream);

// kblas SYMV/HEMV
int kblas_ssymv( char uplo, int m, float alpha, float *dA, int lda, float *dX, int incx, float  beta, float *dY, int incy);
int kblas_dsymv( char uplo, int m, double alpha, double *dA, int lda, double *dX, int incx, double  beta, double *dY, int incy);
int kblas_chemv( char uplo, int m, cuFloatComplex alpha, cuFloatComplex *dA, int lda, cuFloatComplex *dX, int incx, cuFloatComplex  beta, cuFloatComplex *dY, int incy);
int kblas_zhemv( char uplo, int m, cuDoubleComplex alpha, cuDoubleComplex *dA, int lda, cuDoubleComplex *dX, int incx, cuDoubleComplex  beta, cuDoubleComplex *dY, int incy);

int kblas_ssymv_async( char uplo, int m, float alpha, float *dA, int lda, float *dX, int incx, float  beta, float *dY, int incy, cudaStream_t stream);
int kblas_dsymv_async( char uplo, int m, double alpha, double *dA, int lda, double *dX, int incx, double  beta, double *dY, int incy, cudaStream_t stream);
int kblas_chemv_async( char uplo, int m, cuFloatComplex alpha, cuFloatComplex *dA, int lda, cuFloatComplex *dX, int incx, cuFloatComplex  beta, cuFloatComplex *dY, int incy, cudaStream_t stream);
int kblas_zhemv_async( char uplo, int m, cuDoubleComplex alpha, cuDoubleComplex *dA, int lda, cuDoubleComplex *dX, int incx, cuDoubleComplex  beta, cuDoubleComplex *dY, int incy, cudaStream_t stream);

// kblas GEMV
int kblas_sgemv( char trans, int rows, int cols, float alpha, float *dA, int lda, float *dX, int incx, float  beta, float *dY, int incy);
int kblas_dgemv( char trans, int rows, int cols, double alpha, double *dA, int lda, double *dX, int incx, double  beta, double *dY, int incy);
int kblas_cgemv( char trans, int rows, int cols, cuFloatComplex alpha, cuFloatComplex *dA, int lda, cuFloatComplex *dX, int incx, cuFloatComplex  beta, cuFloatComplex *dY, int incy);
int kblas_zgemv( char trans, int rows, int cols, cuDoubleComplex alpha, cuDoubleComplex *dA, int lda, cuDoubleComplex *dX, int incx, cuDoubleComplex  beta, cuDoubleComplex *dY, int incy);

int kblas_sgemv_async( char trans, int rows, int cols, float alpha, float *dA, int lda, float *dX, int incx, float  beta, float *dY, int incy, cudaStream_t stream);
int kblas_dgemv_async( char trans, int rows, int cols, double alpha, double *dA, int lda, double *dX, int incx, double  beta, double *dY, int incy, cudaStream_t stream);
int kblas_cgemv_async( char trans, int rows, int cols, cuFloatComplex alpha, cuFloatComplex *dA, int lda, cuFloatComplex *dX, int incx, cuFloatComplex  beta, cuFloatComplex *dY, int incy, cudaStream_t stream);
int kblas_zgemv_async( char trans, int rows, int cols, cuDoubleComplex alpha, cuDoubleComplex *dA, int lda, cuDoubleComplex *dX, int incx, cuDoubleComplex  beta, cuDoubleComplex *dY, int incy, cudaStream_t stream);

// kblas GEMV2
int kblas_sgemv2(char trans, int rows, int cols, float alpha, float *dA, int lda, float *dX, int incx, float  beta, float *dY, int incy);
int kblas_dgemv2(char trans, int rows, int cols, double alpha, double *dA, int lda, double *dX, int incx, double  beta, double *dY, int incy);
int kblas_cgemv2(char trans, int rows, int cols, cuFloatComplex alpha, cuFloatComplex *dA, int lda, cuFloatComplex *dX, int incx, cuFloatComplex  beta, cuFloatComplex *dY, int incy);
int kblas_zgemv2(char trans, int rows, int cols, cuDoubleComplex alpha, cuDoubleComplex *dA, int lda, cuDoubleComplex *dX, int incx, cuDoubleComplex  beta, cuDoubleComplex *dY, int incy);

int kblas_sgemv2_async(	char trans, int rows, int cols, float alpha, float *dA, int lda, float *dX, int incx, float  beta, float *dY, int incy, cudaStream_t stream);
int kblas_dgemv2_async(	char trans, int rows, int cols, double alpha, double *dA, int lda, double *dX, int incx, double  beta, double *dY, int incy, cudaStream_t stream);
int kblas_cgemv2_async(	char trans, int rows, int cols, cuFloatComplex alpha, cuFloatComplex *dA, int lda, cuFloatComplex *dX, int incx, cuFloatComplex  beta, cuFloatComplex *dY, int incy, cudaStream_t stream);
int kblas_zgemv2_async(	char trans, int rows, int cols, cuDoubleComplex alpha, cuDoubleComplex *dA, int lda, cuDoubleComplex *dX, int incx, cuDoubleComplex  beta, cuDoubleComplex *dY, int incy, cudaStream_t stream);

// GEMV offset
int kblas_sgemv_offset( char trans, int rows, int cols, float alpha, float *dA, int lda, float *dX, int incx, float beta, float *dY, int incy, int offset_r, int offset_c);
int kblas_dgemv_offset( char trans, int rows, int cols, double alpha, double *dA, int lda, double *dX, int incx, double beta, double *dY, int incy, int offset_r, int offset_c);
int kblas_cgemv_offset( char trans, int rows, int cols, cuFloatComplex alpha, cuFloatComplex *dA, int lda, cuFloatComplex *dX, int incx, cuFloatComplex beta, cuFloatComplex *dY, int incy, int offset_r, int offset_c);
int kblas_zgemv_offset( char trans, int rows, int cols, cuDoubleComplex alpha, cuDoubleComplex *dA, int lda, cuDoubleComplex *dX, int incx, cuDoubleComplex beta, cuDoubleComplex *dY, int incy, int offset_r, int offset_c);

int kblas_sgemv_offset_async( char trans, int rows, int cols, float alpha, float *dA, int lda, float *dX, int incx, float  beta, float *dY, int incy, int offset_r, int offset_c, cudaStream_t stream);
int kblas_dgemv_offset_async( char trans, int rows, int cols, double alpha, double *dA, int lda, double *dX, int incx, double  beta, double *dY, int incy, int offset_r, int offset_c, cudaStream_t stream);
int kblas_cgemv_offset_async( char trans, int rows, int cols, cuFloatComplex alpha, cuFloatComplex *dA, int lda, cuFloatComplex *dX, int incx, cuFloatComplex  beta, cuFloatComplex *dY, int incy, int offset_r, int offset_c, cudaStream_t stream);
int kblas_zgemv_offset_async( char trans, int rows, int cols, cuDoubleComplex alpha, cuDoubleComplex *dA, int lda, cuDoubleComplex *dX, int incx, cuDoubleComplex  beta, cuDoubleComplex *dY, int incy, int offset_r, int offset_c, cudaStream_t stream);

int kblas_sgemv2_offset( char trans, int rows, int cols, float alpha, float *dA, int lda, float *dX, int incx, float beta, float *dY, int incy, int offset_r, int offset_c);
int kblas_dgemv2_offset( char trans, int rows, int cols, double alpha, double *dA, int lda, double *dX, int incx, double beta, double *dY, int incy, int offset_r, int offset_c);
int kblas_cgemv2_offset( char trans, int rows, int cols, cuFloatComplex alpha, cuFloatComplex *dA, int lda, cuFloatComplex *dX, int incx, cuFloatComplex beta, cuFloatComplex *dY, int incy, int offset_r, int offset_c);
int kblas_zgemv2_offset( char trans, int rows, int cols, cuDoubleComplex alpha, cuDoubleComplex *dA, int lda, cuDoubleComplex *dX, int incx, cuDoubleComplex beta, cuDoubleComplex *dY, int incy, int offset_r, int offset_c);

int kblas_sgemv2_offset_async( char trans, int rows, int cols, float alpha, float *dA, int lda, float *dX, int incx, float  beta, float *dY, int incy, int offset_r, int offset_c, cudaStream_t stream);
int kblas_dgemv2_offset_async( char trans, int rows, int cols, double alpha, double *dA, int lda, double *dX, int incx, double  beta, double *dY, int incy, int offset_r, int offset_c, cudaStream_t stream);
int kblas_cgemv2_offset_async( char trans, int rows, int cols, cuFloatComplex alpha, cuFloatComplex *dA, int lda, cuFloatComplex *dX, int incx, cuFloatComplex  beta, cuFloatComplex *dY, int incy, int offset_r, int offset_c, cudaStream_t stream);
int kblas_zgemv2_offset_async( char trans, int rows, int cols, cuDoubleComplex alpha, cuDoubleComplex *dA, int lda, cuDoubleComplex *dX, int incx, cuDoubleComplex  beta, cuDoubleComplex *dY, int incy, int offset_r, int offset_c, cudaStream_t stream);

// SYHEMV offset
int kblas_ssymv_offset( char uplo, int m, float alpha, float *dA, int lda, float *dX, int incx, float beta, float *dY, int incy, int offset);
int kblas_dsymv_offset( char uplo, int m, double alpha, double *dA, int lda, double *dX, int incx, double beta, double *dY, int incy, int offset);
int kblas_chemv_offset( char uplo, int m, cuFloatComplex alpha, cuFloatComplex *dA, int lda, cuFloatComplex *dX, int incx, cuFloatComplex beta, cuFloatComplex *dY, int incy, int offset);
int kblas_zhemv_offset( char uplo, int m, cuDoubleComplex alpha, cuDoubleComplex *dA, int lda, cuDoubleComplex *dX, int incx, cuDoubleComplex beta, cuDoubleComplex *dY, int incy, int offset);

int kblas_ssymv_offset_async( char uplo, int m, float alpha, float *dA, int lda, float *dX, int incx, float  beta, float *dY, int incy, int offset, cudaStream_t stream);
int kblas_dsymv_offset_async( char uplo, int m, double alpha, double *dA, int lda, double *dX, int incx, double  beta, double *dY, int incy, int offset, cudaStream_t stream);
int kblas_chemv_offset_async( char uplo, int m, cuFloatComplex alpha, cuFloatComplex *dA, int lda, cuFloatComplex *dX, int incx, cuFloatComplex  beta, cuFloatComplex *dY, int incy, int offset, cudaStream_t stream);

int kblas_zhemv_offset_async( char uplo, int m, cuDoubleComplex alpha, cuDoubleComplex *dA, int lda, cuDoubleComplex *dX, int incx, cuDoubleComplex  beta, cuDoubleComplex *dY, int incy, int offset, cudaStream_t stream);

// multi gpu symv
//sync
int kblas_ssymv_mgpu( char uplo, int m,
                      float alpha, float **dA, int lda,
                      float **dX, int incx,
                      float  beta, float **dY, int incy,
                      int ngpus,
                      int offset);

int kblas_dsymv_mgpu( char uplo, int m,
                      double alpha, double **dA, int lda,
                      double **dX, int incx,
                      double  beta, double **dY, int incy,
                      int ngpus,
                      int offset);

int kblas_chemv_mgpu( char uplo, int m,
                      cuFloatComplex alpha, cuFloatComplex **dA, int lda,
                      cuFloatComplex **dX, int incx,
                      cuFloatComplex  beta, cuFloatComplex **dY, int incy,
                      int ngpus,
                      int offset);

int kblas_zhemv_mgpu( char uplo, int m,
                      cuDoubleComplex alpha, cuDoubleComplex **dA, int lda,
                      cuDoubleComplex **dX, int incx,
                      cuDoubleComplex  beta, cuDoubleComplex **dY, int incy,
                      int ngpus,
                      int offset);
// async
int kblas_ssymv_mgpu_async( char uplo, int m,
                            float alpha, float **dA, int lda,
                            float **dX, int incx,
                            float  beta, float **dY, int incy,
                            int ngpus,
                            int offset,
                            cudaStream_t stream[MAX_NGPUS][MAX_STREAMS]);

int kblas_dsymv_mgpu_async( char uplo, int m,
                            double alpha, double **dA, int lda,
                            double **dX, int incx,
                            double  beta, double **dY, int incy,
                            int ngpus,
                            int offset,
                            cudaStream_t stream[MAX_NGPUS][MAX_STREAMS]);

int kblas_chemv_mgpu_async( char uplo, int m,
                            cuFloatComplex alpha, cuFloatComplex **dA, int lda,
                            cuFloatComplex **dX, int incx,
                            cuFloatComplex  beta, cuFloatComplex **dY, int incy,
                            int ngpus,
                            int offset,
                            cudaStream_t stream[MAX_NGPUS][MAX_STREAMS]);

int kblas_zhemv_mgpu_async( char uplo, int m,
                            cuDoubleComplex alpha, cuDoubleComplex **dA, int lda,
                            cuDoubleComplex **dX, int incx,
                            cuDoubleComplex  beta, cuDoubleComplex **dY, int incy,
                            int ngpus,
                            int offset,
                            cudaStream_t stream[MAX_NGPUS][MAX_STREAMS]);


// multi gpu gemv
// sync
int kblas_sgemv_mgpu( char trans, int rows, int cols,
                      float alpha, float **dA, int lda,
                      float **dX, int incx,
                      float  beta, float **dY, int incy,
                      int ngpus,
                      int offset_r, int offset_c);

int kblas_dgemv_mgpu( char trans, int rows, int cols,
                      double alpha, double **dA, int lda,
                      double **dX, int incx,
                      double  beta, double **dY, int incy,
                      int ngpus,
                      int offset_r, int offset_c);

int kblas_cgemv_mgpu( char trans, int rows, int cols,
                      cuFloatComplex alpha, cuFloatComplex **dA, int lda,
                      cuFloatComplex **dX, int incx,
                      cuFloatComplex  beta, cuFloatComplex **dY, int incy,
                      int ngpus,
                      int offset_r, int offset_c);

int kblas_zgemv_mgpu( char trans, int rows, int cols,
                      cuDoubleComplex alpha, cuDoubleComplex **dA, int lda,
                      cuDoubleComplex **dX, int incx,
                      cuDoubleComplex  beta, cuDoubleComplex **dY, int incy,
                      int ngpus,
                      int offset_r, int offset_c);

// async
int kblas_sgemv_mgpu_async( char trans, int rows, int cols,
                            float alpha, float **dA, int lda,
                            float **dX, int incx,
                            float  beta, float **dY, int incy,
                            int ngpus,
                            int offset_r, int offset_c,
                            cudaStream_t stream[MAX_NGPUS][MAX_STREAMS]);

int kblas_dgemv_mgpu_async( char trans, int rows, int cols,
                            double alpha, double **dA, int lda,
                            double **dX, int incx,
                            double  beta, double **dY, int incy,
                            int ngpus,
                            int offset_r, int offset_c,
                            cudaStream_t stream[MAX_NGPUS][MAX_STREAMS]);

int kblas_cgemv_mgpu_async( char trans, int rows, int cols,
                            cuFloatComplex alpha, cuFloatComplex **dA, int lda,
                            cuFloatComplex **dX, int incx,
                            cuFloatComplex  beta, cuFloatComplex **dY, int incy,
                            int ngpus,
                            int offset_r, int offset_c,
                            cudaStream_t stream[MAX_NGPUS][MAX_STREAMS]);

int kblas_zgemv_mgpu_async( char trans, int rows, int cols,
                            cuDoubleComplex alpha, cuDoubleComplex **dA, int lda,
                            cuDoubleComplex **dX, int incx,
                            cuDoubleComplex  beta, cuDoubleComplex **dY, int incy,
                            int ngpus,
                            int offset_r, int offset_c,
                            cudaStream_t stream[MAX_NGPUS][MAX_STREAMS]);

// gemm_mgpu (out of core)
void kblas_sgemm_mgpu(char transa, char transb, long m, long n, long k,
                      float alpha, const float* A, long lda,
                      const float* B, long ldb,
                      float beta, float* C, long ldc,
                      long ngpus, long* gpu_id,
                      long *tile);
void kblas_dgemm_mgpu(char transa, char transb, long m, long n, long k,
                      double alpha, const double* A, long lda,
                      const double* B, long ldb,
                      double beta, double* C, long ldc,
                      long ngpus, long* gpu_id,
                      long *tile);
void kblas_cgemm_mgpu(char transa, char transb, long m, long n, long k,
                      cuFloatComplex alpha, const cuFloatComplex* A, long lda,
                      const cuFloatComplex* B, long ldb,
                      cuFloatComplex beta, cuFloatComplex* C, long ldc,
                      long ngpus, long* gpu_id,
                      long *tile);
void kblas_zgemm_mgpu(char transa, char transb, long m, long n, long k,
                      cuDoubleComplex alpha, const cuDoubleComplex* A, long lda,
                      const cuDoubleComplex* B, long ldb,
                      cuDoubleComplex beta, cuDoubleComplex* C, long ldc,
                      long ngpus, long* gpu_id,
                      long *tile);

// auxiliary mgpu control functions
void kblas_smalloc_mgpu_1D(	int rows, int cols, float** dA, int ngpus, int ldb, int block_size);
void kblas_dmalloc_mgpu_1D(	int rows, int cols, double** dA, int ngpus, int ldb, int block_size);
void kblas_cmalloc_mgpu_1D(	int rows, int cols, cuFloatComplex** dA, int ngpus, int ldb, int block_size);
void kblas_zmalloc_mgpu_1D(	int rows, int cols, cuDoubleComplex** dA, int ngpus, int ldb, int block_size);

void kblas_ssetmatrix_mgpu_1D(int rows, int cols, float* A, int LDA, float** dA, int LDB, int ngpus, int block_size);
void kblas_dsetmatrix_mgpu_1D(int rows, int cols, double* A, int LDA, double** dA, int LDB, int ngpus, int block_size);
void kblas_csetmatrix_mgpu_1D(int rows, int cols, cuFloatComplex* A, int LDA, cuFloatComplex** dA, int LDB, int ngpus, int block_size);
void kblas_zsetmatrix_mgpu_1D(int rows, int cols, cuDoubleComplex* A, int LDA, cuDoubleComplex** dA, int LDB, int ngpus, int block_size);

void kblas_ssetvector_mgpu_1D(int n, float* Y, float** dY, int ngpus, int block_size);
void kblas_dsetvector_mgpu_1D(int n, double* Y, double** dY, int ngpus, int block_size);
void kblas_csetvector_mgpu_1D(int n, cuFloatComplex* Y, cuFloatComplex** dY, int ngpus, int block_size);
void kblas_zsetvector_mgpu_1D(int n, cuDoubleComplex* Y, cuDoubleComplex** dY, int ngpus, int block_size);

int get_ssymv_mgpu_bs(char uplo);
int get_dsymv_mgpu_bs(char uplo);
int get_chemv_mgpu_bs(char uplo);
int get_zhemv_mgpu_bs(char uplo);

int get_sgemv_mgpu_bs(char trans);
int get_dgemv_mgpu_bs(char trans);
int get_cgemv_mgpu_bs(char trans);
int get_zgemv_mgpu_bs(char trans);

#ifdef __cplusplus
}
#endif

#endif // _KBLAS_L2_H_
