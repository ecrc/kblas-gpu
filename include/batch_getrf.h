#ifndef __KBLAS_BATCH_GETRF_H__
#define __KBLAS_BATCH_GETRF_H__

#ifdef __cplusplus
extern "C" {
#endif

// Workspace interface 
size_t kblasSgetrf_vbatch_workspace(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount);
size_t kblasDgetrf_vbatch_workspace(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount);
size_t kblasCgetrf_vbatch_workspace(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount);
size_t kblasZgetrf_vbatch_workspace(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount);
size_t kblasSgetrf_vbatch_workspace_host(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount);
size_t kblasDgetrf_vbatch_workspace_host(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount);
size_t kblasCgetrf_vbatch_workspace_host(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount);
size_t kblasZgetrf_vbatch_workspace_host(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount);

// Batched LU interface 
int kblasSgetrf_vbatch(
	kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int max_m, int max_n, 
	float **A_ptrs, int *lda_batch, int **ipiv_ptrs, void* workspace, size_t ws_bytes,
	int *info_batch, int batchCount
);
int kblasDgetrf_vbatch(
	kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int max_m, int max_n, 
	double **A_ptrs, int *lda_batch, int **ipiv_ptrs, void* workspace, size_t ws_bytes,
	int *info_batch, int batchCount
);
int kblasCgetrf_vbatch(
	kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int max_m, int max_n, 
	cuFloatComplex **A_ptrs, int *lda_batch, int **ipiv_ptrs, void* workspace, size_t ws_bytes,
	int *info_batch, int batchCount
);
int kblasZgetrf_vbatch(
	kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int max_m, int max_n, 
	cuDoubleComplex **A_ptrs, int *lda_batch, int **ipiv_ptrs, void* workspace, size_t ws_bytes,
	int *info_batch, int batchCount
);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

template<class T> inline size_t kblas_getrf_vbatch_workspace(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount);
template<class T> inline size_t kblas_getrf_vbatch_workspace_host(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount);

template<> inline size_t kblas_getrf_vbatch_workspace<float>(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount)
{ return kblasSgetrf_vbatch_workspace(kblas_handle, m_batch, n_batch, batchCount); }
template<> inline size_t kblas_getrf_vbatch_workspace<double>(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount)
{ return kblasDgetrf_vbatch_workspace(kblas_handle, m_batch, n_batch, batchCount); }
template<> inline size_t kblas_getrf_vbatch_workspace<cuFloatComplex>(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount)
{ return kblasCgetrf_vbatch_workspace(kblas_handle, m_batch, n_batch, batchCount); }
template<> inline size_t kblas_getrf_vbatch_workspace<cuDoubleComplex>(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount)
{ return kblasZgetrf_vbatch_workspace(kblas_handle, m_batch, n_batch, batchCount); }

template<> inline size_t kblas_getrf_vbatch_workspace_host<float>(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount)
{ return kblasSgetrf_vbatch_workspace_host(kblas_handle, m_batch, n_batch, batchCount); }
template<> inline size_t kblas_getrf_vbatch_workspace_host<double>(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount)
{ return kblasDgetrf_vbatch_workspace_host(kblas_handle, m_batch, n_batch, batchCount); }
template<> inline size_t kblas_getrf_vbatch_workspace_host<cuFloatComplex>(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount)
{ return kblasCgetrf_vbatch_workspace_host(kblas_handle, m_batch, n_batch, batchCount); }
template<> inline size_t kblas_getrf_vbatch_workspace_host<cuDoubleComplex>(kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int batchCount)
{ return kblasZgetrf_vbatch_workspace_host(kblas_handle, m_batch, n_batch, batchCount); }

inline int kblas_getrf_vbatch(
	kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int max_m, int max_n, 
	float **A_ptrs, int *lda_batch, int **ipiv_ptrs, void* workspace, size_t ws_bytes,
	int *info_batch, int batchCount
)
{ return kblasSgetrf_vbatch(kblas_handle, m_batch, n_batch, max_m, max_n, A_ptrs, lda_batch, ipiv_ptrs, workspace, ws_bytes, info_batch, batchCount); }

inline int kblas_getrf_vbatch(
	kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int max_m, int max_n, 
	double **A_ptrs, int *lda_batch, int **ipiv_ptrs, void* workspace, size_t ws_bytes,
	int *info_batch, int batchCount
)
{ return kblasDgetrf_vbatch(kblas_handle, m_batch, n_batch, max_m, max_n, A_ptrs, lda_batch, ipiv_ptrs, workspace, ws_bytes, info_batch, batchCount); }

inline int kblas_getrf_vbatch(
	kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int max_m, int max_n, 
	cuFloatComplex **A_ptrs, int *lda_batch, int **ipiv_ptrs, void* workspace, size_t ws_bytes,
	int *info_batch, int batchCount
)
{ return kblasCgetrf_vbatch(kblas_handle, m_batch, n_batch, max_m, max_n, A_ptrs, lda_batch, ipiv_ptrs, workspace, ws_bytes, info_batch, batchCount); }

inline int kblas_getrf_vbatch(
	kblasHandle_t kblas_handle, int *m_batch, int* n_batch, int max_m, int max_n, 
	cuDoubleComplex **A_ptrs, int *lda_batch, int **ipiv_ptrs, void* workspace, size_t ws_bytes,
	int *info_batch, int batchCount
)
{ return kblasZgetrf_vbatch(kblas_handle, m_batch, n_batch, max_m, max_n, A_ptrs, lda_batch, ipiv_ptrs, workspace, ws_bytes, info_batch, batchCount); }

#endif 

#endif 
