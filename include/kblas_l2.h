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
 * @version 3.0.0
 * @author Ahmad Abdelfattah
 * @date 2018-11-14
 **/

#ifndef _KBLAS_L2_H_
#define _KBLAS_L2_H_

#ifdef __cplusplus
extern "C" {
#endif

// scal
int kblas_dscal(int n, double alpha, double *x, int incx);
int kblas_sscal(int n, float alpha, float *x, int incx);

int kblas_dscal_async(int n, double alpha, double *x, int incx, hipStream_t stream);
int kblas_sscal_async(int n, float alpha, float *x, int incx, hipStream_t stream);

// kblas SYMV/HEMV
int kblas_ssymv( char uplo, int m, float alpha, float *dA, int lda, float *dX, int incx, float  beta, float *dY, int incy);
int kblas_dsymv( char uplo, int m, double alpha, double *dA, int lda, double *dX, int incx, double  beta, double *dY, int incy);


int kblas_ssymv_async( char uplo, int m, float alpha, float *dA, int lda, float *dX, int incx, float  beta, float *dY, int incy, hipStream_t stream);
int kblas_dsymv_async( char uplo, int m, double alpha, double *dA, int lda, double *dX, int incx, double  beta, double *dY, int incy, hipStream_t stream);

// kblas GEMV
int kblas_sgemv( char trans, int rows, int cols, float alpha, float *dA, int lda, float *dX, int incx, float  beta, float *dY, int incy);
int kblas_dgemv( char trans, int rows, int cols, double alpha, double *dA, int lda, double *dX, int incx, double  beta, double *dY, int incy);

int kblas_sgemv_async( char trans, int rows, int cols, float alpha, float *dA, int lda, float *dX, int incx, float  beta, float *dY, int incy, hipStream_t stream);
int kblas_dgemv_async( char trans, int rows, int cols, double alpha, double *dA, int lda, double *dX, int incx, double  beta, double *dY, int incy, hipStream_t stream);

// kblas GEMV2
int kblas_sgemv2(char trans, int rows, int cols, float alpha, float *dA, int lda, float *dX, int incx, float  beta, float *dY, int incy);
int kblas_dgemv2(char trans, int rows, int cols, double alpha, double *dA, int lda, double *dX, int incx, double  beta, double *dY, int incy);

int kblas_sgemv2_async(	char trans, int rows, int cols, float alpha, float *dA, int lda, float *dX, int incx, float  beta, float *dY, int incy, hipStream_t stream);
int kblas_dgemv2_async(	char trans, int rows, int cols, double alpha, double *dA, int lda, double *dX, int incx, double  beta, double *dY, int incy, hipStream_t stream);

// GEMV offset
int kblas_sgemv_offset( char trans, int rows, int cols, float alpha, float *dA, int lda, float *dX, int incx, float beta, float *dY, int incy, int offset_r, int offset_c);
int kblas_dgemv_offset( char trans, int rows, int cols, double alpha, double *dA, int lda, double *dX, int incx, double beta, double *dY, int incy, int offset_r, int offset_c);

int kblas_sgemv_offset_async( char trans, int rows, int cols, float alpha, float *dA, int lda, float *dX, int incx, float  beta, float *dY, int incy, int offset_r, int offset_c, hipStream_t stream);
int kblas_dgemv_offset_async( char trans, int rows, int cols, double alpha, double *dA, int lda, double *dX, int incx, double  beta, double *dY, int incy, int offset_r, int offset_c, hipStream_t stream);

int kblas_sgemv2_offset( char trans, int rows, int cols, float alpha, float *dA, int lda, float *dX, int incx, float beta, float *dY, int incy, int offset_r, int offset_c);
int kblas_dgemv2_offset( char trans, int rows, int cols, double alpha, double *dA, int lda, double *dX, int incx, double beta, double *dY, int incy, int offset_r, int offset_c);

int kblas_sgemv2_offset_async( char trans, int rows, int cols, float alpha, float *dA, int lda, float *dX, int incx, float  beta, float *dY, int incy, int offset_r, int offset_c, hipStream_t stream);
int kblas_dgemv2_offset_async( char trans, int rows, int cols, double alpha, double *dA, int lda, double *dX, int incx, double  beta, double *dY, int incy, int offset_r, int offset_c, hipStream_t stream);

// SYHEMV offset
int kblas_ssymv_offset( char uplo, int m, float alpha, float *dA, int lda, float *dX, int incx, float beta, float *dY, int incy, int offset);
int kblas_dsymv_offset( char uplo, int m, double alpha, double *dA, int lda, double *dX, int incx, double beta, double *dY, int incy, int offset);

int kblas_ssymv_offset_async( char uplo, int m, float alpha, float *dA, int lda, float *dX, int incx, float  beta, float *dY, int incy, int offset, hipStream_t stream);
int kblas_dsymv_offset_async( char uplo, int m, double alpha, double *dA, int lda, double *dX, int incx, double  beta, double *dY, int incy, int offset, hipStream_t stream);


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
                      hipFloatComplex alpha, hipFloatComplex **dA, int lda,
                      hipFloatComplex **dX, int incx,
                      hipFloatComplex  beta, hipFloatComplex **dY, int incy,
                      int ngpus,
                      int offset);

int kblas_zhemv_mgpu( char uplo, int m,
                      hipDoubleComplex alpha, hipDoubleComplex **dA, int lda,
                      hipDoubleComplex **dX, int incx,
                      hipDoubleComplex  beta, hipDoubleComplex **dY, int incy,
                      int ngpus,
                      int offset);
// async
int kblas_ssymv_mgpu_async( char uplo, int m,
                            float alpha, float **dA, int lda,
                            float **dX, int incx,
                            float  beta, float **dY, int incy,
                            int ngpus,
                            int offset,
                            hipStream_t stream[MAX_NGPUS][MAX_STREAMS]);

int kblas_dsymv_mgpu_async( char uplo, int m,
                            double alpha, double **dA, int lda,
                            double **dX, int incx,
                            double  beta, double **dY, int incy,
                            int ngpus,
                            int offset,
                            hipStream_t stream[MAX_NGPUS][MAX_STREAMS]);

int kblas_chemv_mgpu_async( char uplo, int m,
                            hipFloatComplex alpha, hipFloatComplex **dA, int lda,
                            hipFloatComplex **dX, int incx,
                            hipFloatComplex  beta, hipFloatComplex **dY, int incy,
                            int ngpus,
                            int offset,
                            hipStream_t stream[MAX_NGPUS][MAX_STREAMS]);

int kblas_zhemv_mgpu_async( char uplo, int m,
                            hipDoubleComplex alpha, hipDoubleComplex **dA, int lda,
                            hipDoubleComplex **dX, int incx,
                            hipDoubleComplex  beta, hipDoubleComplex **dY, int incy,
                            int ngpus,
                            int offset,
                            hipStream_t stream[MAX_NGPUS][MAX_STREAMS]);


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
                      hipFloatComplex alpha, hipFloatComplex **dA, int lda,
                      hipFloatComplex **dX, int incx,
                      hipFloatComplex  beta, hipFloatComplex **dY, int incy,
                      int ngpus,
                      int offset_r, int offset_c);

int kblas_zgemv_mgpu( char trans, int rows, int cols,
                      hipDoubleComplex alpha, hipDoubleComplex **dA, int lda,
                      hipDoubleComplex **dX, int incx,
                      hipDoubleComplex  beta, hipDoubleComplex **dY, int incy,
                      int ngpus,
                      int offset_r, int offset_c);

// async
int kblas_sgemv_mgpu_async( char trans, int rows, int cols,
                            float alpha, float **dA, int lda,
                            float **dX, int incx,
                            float  beta, float **dY, int incy,
                            int ngpus,
                            int offset_r, int offset_c,
                            hipStream_t stream[MAX_NGPUS][MAX_STREAMS]);

int kblas_dgemv_mgpu_async( char trans, int rows, int cols,
                            double alpha, double **dA, int lda,
                            double **dX, int incx,
                            double  beta, double **dY, int incy,
                            int ngpus,
                            int offset_r, int offset_c,
                            hipStream_t stream[MAX_NGPUS][MAX_STREAMS]);

int kblas_cgemv_mgpu_async( char trans, int rows, int cols,
                            hipFloatComplex alpha, hipFloatComplex **dA, int lda,
                            hipFloatComplex **dX, int incx,
                            hipFloatComplex  beta, hipFloatComplex **dY, int incy,
                            int ngpus,
                            int offset_r, int offset_c,
                            hipStream_t stream[MAX_NGPUS][MAX_STREAMS]);

int kblas_zgemv_mgpu_async( char trans, int rows, int cols,
                            hipDoubleComplex alpha, hipDoubleComplex **dA, int lda,
                            hipDoubleComplex **dX, int incx,
                            hipDoubleComplex  beta, hipDoubleComplex **dY, int incy,
                            int ngpus,
                            int offset_r, int offset_c,
                            hipStream_t stream[MAX_NGPUS][MAX_STREAMS]);

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
                      hipFloatComplex alpha, const hipFloatComplex* A, long lda,
                      const hipFloatComplex* B, long ldb,
                      hipFloatComplex beta, hipFloatComplex* C, long ldc,
                      long ngpus, long* gpu_id,
                      long *tile);
void kblas_zgemm_mgpu(char transa, char transb, long m, long n, long k,
                      hipDoubleComplex alpha, const hipDoubleComplex* A, long lda,
                      const hipDoubleComplex* B, long ldb,
                      hipDoubleComplex beta, hipDoubleComplex* C, long ldc,
                      long ngpus, long* gpu_id,
                      long *tile);

// auxiliary mgpu control functions
void kblas_smalloc_mgpu_1D(	int rows, int cols, float** dA, int ngpus, int ldb, int block_size);
void kblas_dmalloc_mgpu_1D(	int rows, int cols, double** dA, int ngpus, int ldb, int block_size);
void kblas_cmalloc_mgpu_1D(	int rows, int cols, hipFloatComplex** dA, int ngpus, int ldb, int block_size);
void kblas_zmalloc_mgpu_1D(	int rows, int cols, hipDoubleComplex** dA, int ngpus, int ldb, int block_size);

void kblas_ssetmatrix_mgpu_1D(int rows, int cols, float* A, int LDA, float** dA, int LDB, int ngpus, int block_size);
void kblas_dsetmatrix_mgpu_1D(int rows, int cols, double* A, int LDA, double** dA, int LDB, int ngpus, int block_size);
void kblas_csetmatrix_mgpu_1D(int rows, int cols, hipFloatComplex* A, int LDA, hipFloatComplex** dA, int LDB, int ngpus, int block_size);
void kblas_zsetmatrix_mgpu_1D(int rows, int cols, hipDoubleComplex* A, int LDA, hipDoubleComplex** dA, int LDB, int ngpus, int block_size);

void kblas_ssetvector_mgpu_1D(int n, float* Y, float** dY, int ngpus, int block_size);
void kblas_dsetvector_mgpu_1D(int n, double* Y, double** dY, int ngpus, int block_size);
void kblas_csetvector_mgpu_1D(int n, hipFloatComplex* Y, hipFloatComplex** dY, int ngpus, int block_size);
void kblas_zsetvector_mgpu_1D(int n, hipDoubleComplex* Y, hipDoubleComplex** dY, int ngpus, int block_size);

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
