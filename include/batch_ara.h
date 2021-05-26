/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/

/**
 * @file include/batch_ara.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Wajih Halim Boukaram
 * @date 2020-12-10
 **/
 
#ifndef __BATCH_ARA_H__
#define __BATCH_ARA_H__

#ifdef __cplusplus
extern "C" {
#endif

// Workspace query routines
void kblasSara_batch_wsquery(kblasHandle_t handle, int bs, int num_ops);
void kblasDara_batch_wsquery(kblasHandle_t handle, int bs, int num_ops);
void kblasSara_trsm_batch_wsquery(kblasHandle_t handle, int num_ops);
void kblasDara_trsm_batch_wsquery(kblasHandle_t handle, int num_ops);

typedef int (*dara_sampler)(void* data, double** B_batch, int* ldb_batch, int* samples_batch, double** A_batch, int* lda_batch, int max_samples, int num_ops, int transpose);
typedef int (*sara_sampler)(void* data, float** B_batch, int* ldb_batch, int* samples_batch, float** A_batch, int* lda_batch, int max_samples, int num_ops, int transpose);

// Main ARA routines
int kblas_sara_batch_fn(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, sara_sampler sampler, void* sampler_data, 
	float** A_batch, int* lda_batch, float** B_batch, int* ldb_batch, int* ranks_batch, 
	float tol, int max_rows, int max_cols, int max_rank, int bs, int r, kblasRandState_t rand_state, 
	int relative, int num_ops
);

int kblas_dara_batch_fn(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, dara_sampler sampler, void* sampler_data, 
	double** A_batch, int* lda_batch, double** B_batch, int* ldb_batch, int* ranks_batch, 
	double tol, int max_rows, int max_cols, int max_rank, int bs, int r, kblasRandState_t rand_state, 
	int relative, int num_ops
);

int kblas_sara_batch(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, float** M_batch, int* ldm_batch, 
	float** A_batch, int* lda_batch, float** B_batch, int* ldb_batch, int* ranks_batch, 
	float tol, int max_rows, int max_cols, int max_rank, int bs, int r, kblasRandState_t rand_state, 
	int relative, int num_ops
);

int kblas_dara_batch(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, double** M_batch, int* ldm_batch, 
	double** A_batch, int* lda_batch, double** B_batch, int* ldb_batch, int* ranks_batch, 
	double tol, int max_rows, int max_cols, int max_rank, int bs, int r, kblasRandState_t rand_state, 
	int relative, int num_ops
);

// Utility routines for ara-based algorithms
int kblas_sara_svec_count_batch(
	double* diag_R, int bs, int* op_samples, int* ranks_batch, double* max_diag, float** Y_batch, 
	int* ldy_batch, float tol, int r, int* small_vectors, int relative, int num_ops, cudaStream_t stream
);

int kblas_dara_svec_count_batch(
	double* diag_R, int bs, int* op_samples, int* ranks_batch, double* max_diag, double** Y_batch, 
	int* ldy_batch, double tol, int r, int* small_vectors, int relative, int num_ops, cudaStream_t stream
);

int kblas_sara_trsm_batch(
	kblasHandle_t handle, float** B_batch, int* ldb_batch, float** A_batch, int* lda_batch, 
	int* rows_batch, int* cols_batch, int num_ops, int max_rows, int bs
);

int kblas_dara_trsm_batch(
	kblasHandle_t handle, double** B_batch, int* ldb_batch, double** A_batch, int* lda_batch, 
	int* rows_batch, int* cols_batch, int num_ops, int max_rows, int bs
);

int kblas_sara_fused_potrf_batch( 
	int* op_samples, double** A_batch, int* lda_batch, float** R_batch, int* ldr_batch, 
	double* diag_R, int bs, int* block_ranks, int num_ops, cudaStream_t stream
);

int kblas_dara_fused_potrf_batch( 
	int* op_samples, double** A_batch, int* lda_batch, double** R_batch, int* ldr_batch, 
	double* diag_R, int bs, int* block_ranks, int num_ops, cudaStream_t stream
);

int kblas_sara_mp_syrk_batch(
	kblasHandle_t handle, int* m, int* n, int max_m, int max_n, 
	const float** A, int* lda, double** B, int* ldb, int batchCount 
);

int kblas_dara_mp_syrk_batch(
	kblasHandle_t handle, int* m, int* n, int max_m, int max_n, 
	const double** A, int* lda, double** B, int* ldb, int batchCount 
);

int kblas_ara_batch_set_samples(
	int* op_samples, int* small_vectors, 
	int samples, int r, int num_ops, cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

// Workspace routine wrappers
template<class T> inline void kblas_ara_batch_wsquery(kblasHandle_t handle, int bs, int num_ops);
template<> 
inline void kblas_ara_batch_wsquery<float>(kblasHandle_t handle, int bs, int num_ops) 
{ kblasSara_batch_wsquery(handle, bs, num_ops); }

template<> 
inline void kblas_ara_batch_wsquery<double>(kblasHandle_t handle, int bs, int num_ops) 
{ kblasDara_batch_wsquery(handle, bs, num_ops); }

template<class T> inline void kblas_ara_trsm_batch_wsquery(kblasHandle_t handle, int num_ops);
template<> 
inline void kblas_ara_trsm_batch_wsquery<float>(kblasHandle_t handle, int num_ops) 
{ kblasSara_trsm_batch_wsquery(handle, num_ops); }

template<> 
inline void kblas_ara_trsm_batch_wsquery<double>(kblasHandle_t handle, int num_ops) 
{ kblasDara_trsm_batch_wsquery(handle, num_ops); }

// Main ARA routine wrappers
inline int kblas_ara_batch_fn(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, sara_sampler sampler, void* sampler_data, 
	float** A_batch, int* lda_batch, float** B_batch, int* ldb_batch, int* ranks_batch, 
	float tol, int max_rows, int max_cols, int max_rank, int bs, int r, kblasRandState_t rand_state, 
	int relative, int num_ops
)
{
	return kblas_sara_batch_fn(
		handle, rows_batch, cols_batch, sampler, sampler_data, A_batch, lda_batch, B_batch, ldb_batch, 
		ranks_batch, tol, max_rows, max_cols, max_rank, bs, r, rand_state, relative, num_ops
	);
}

inline int kblas_ara_batch_fn(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, dara_sampler sampler, void* sampler_data, 
	double** A_batch, int* lda_batch, double** B_batch, int* ldb_batch, int* ranks_batch, 
	double tol, int max_rows, int max_cols, int max_rank, int bs, int r, kblasRandState_t rand_state, 
	int relative, int num_ops
)
{
	return kblas_dara_batch_fn(
		handle, rows_batch, cols_batch, sampler, sampler_data, A_batch, lda_batch, B_batch, ldb_batch, 
		ranks_batch, tol, max_rows, max_cols, max_rank, bs, r, rand_state, relative, num_ops
	);
}

inline int kblas_ara_batch(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, float** M_batch, int* ldm_batch, 
	float** A_batch, int* lda_batch, float** B_batch, int* ldb_batch, int* ranks_batch, 
	float tol, int max_rows, int max_cols, int max_rank, int bs, int r, kblasRandState_t rand_state, 
	int relative, int num_ops
)
{
	return kblas_sara_batch(
		handle, rows_batch, cols_batch, M_batch, ldm_batch, A_batch, lda_batch, B_batch, ldb_batch, 
		ranks_batch, tol, max_rows, max_cols, max_rank, bs, r, rand_state, relative, num_ops
	);
}

inline int kblas_ara_batch(
	kblasHandle_t handle, int* rows_batch, int* cols_batch, double** M_batch, int* ldm_batch, 
	double** A_batch, int* lda_batch, double** B_batch, int* ldb_batch, int* ranks_batch, 
	double tol, int max_rows, int max_cols, int max_rank, int bs, int r, kblasRandState_t rand_state, 
	int relative, int num_ops
)
{
	return kblas_dara_batch(
		handle, rows_batch, cols_batch, M_batch, ldm_batch, A_batch, lda_batch, B_batch, ldb_batch, 
		ranks_batch, tol, max_rows, max_cols, max_rank, bs, r, rand_state, relative, num_ops
	);
}

// utility routine wrappers
inline int kblas_ara_svec_count_batch(
	double* diag_R, int bs, int* op_samples, int* ranks_batch, double* max_diag, float** Y_batch, 
	int* ldy_batch, float tol, int r, int* small_vectors, int relative, int num_ops, cudaStream_t stream
)
{
	return kblas_sara_svec_count_batch(
		diag_R, bs, op_samples, ranks_batch, max_diag, 
		Y_batch, ldy_batch, tol, r, small_vectors, relative, 
		num_ops, stream
	);
}

inline int kblas_ara_svec_count_batch(
	double* diag_R, int bs, int* op_samples, int* ranks_batch, double* max_diag, double** Y_batch, 
	int* ldy_batch, double tol, int r, int* small_vectors, int relative, int num_ops, cudaStream_t stream
)
{
	return kblas_dara_svec_count_batch(
		diag_R, bs, op_samples, ranks_batch, max_diag, 
		Y_batch, ldy_batch, tol, r, small_vectors, relative, 
		num_ops, stream
	);
}

inline int kblas_ara_trsm_batch(
	kblasHandle_t handle, float** B_batch, int* ldb_batch, float** A_batch, int* lda_batch, 
	int* rows_batch, int* cols_batch, int num_ops, int max_rows, int bs
)
{
	return kblas_sara_trsm_batch(
		handle, B_batch, ldb_batch, A_batch, lda_batch, 
		rows_batch, cols_batch, num_ops, max_rows, bs
	);
}

inline int kblas_ara_trsm_batch(
	kblasHandle_t handle, double** B_batch, int* ldb_batch, double** A_batch, int* lda_batch, 
	int* rows_batch, int* cols_batch, int num_ops, int max_rows, int bs
)
{
	return kblas_dara_trsm_batch(
		handle, B_batch, ldb_batch, A_batch, lda_batch, 
		rows_batch, cols_batch, num_ops, max_rows, bs
	);
}

inline int kblas_ara_fused_potrf_batch( 
	int* op_samples, double** A_batch, int* lda_batch, float** R_batch, int* ldr_batch, 
	double* diag_R, int bs, int* block_ranks, int num_ops, cudaStream_t stream
)
{
	return kblas_sara_fused_potrf_batch( 
		op_samples, A_batch, lda_batch, R_batch, ldr_batch, 
		diag_R, bs, block_ranks, num_ops, stream
	);
}

inline int kblas_ara_fused_potrf_batch( 
	int* op_samples, double** A_batch, int* lda_batch, double** R_batch, int* ldr_batch, 
	double* diag_R, int bs, int* block_ranks, int num_ops, cudaStream_t stream
)
{
	return kblas_dara_fused_potrf_batch( 
		op_samples, A_batch, lda_batch, R_batch, ldr_batch, 
		diag_R, bs, block_ranks, num_ops, stream
	);
}

inline int kblas_ara_mp_syrk_batch(
	kblasHandle_t handle, int* m, int* n, int max_m, int max_n, 
	const float** A, int* lda, double** B, int* ldb, int batchCount 
)
{
	return kblas_sara_mp_syrk_batch(handle, m, n, max_m, max_n, A, lda, B, ldb, batchCount);
}

inline int kblas_ara_mp_syrk_batch(
	kblasHandle_t handle, int* m, int* n, int max_m, int max_n, 
	const double** A, int* lda, double** B, int* ldb, int batchCount 
)
{
	return kblas_dara_mp_syrk_batch(handle, m, n, max_m, max_n, A, lda, B, ldb, batchCount);
}

#endif

#endif
