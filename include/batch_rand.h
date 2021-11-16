#ifndef __BATCH_RAND_H__
#define __BATCH_RAND_H__

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Random state initialization
int kblasInitRandState(kblasHandle_t handle, kblasRandState_t* state, int num_states, unsigned int seed);
int kblasDestroyRandState(kblasRandState_t state);

//------------------------------------------------------------------------------
// Uniform array of pointers interface
int kblasDrand_batch(kblasHandle_t handle, int m, int n, double** A_ptrs, int lda, kblasRandState_t state, int num_ops);
int kblasSrand_batch(kblasHandle_t handle, int m, int n, float** A_ptrs, int lda, kblasRandState_t state, int num_ops);

//------------------------------------------------------------------------------
// Uniform strided interface
int kblasDrand_batch_strided(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, kblasRandState_t state, int num_ops);
int kblasSrand_batch_strided(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, kblasRandState_t state, int num_ops);

//------------------------------------------------------------------------------
// Non-uniform array of pointers interface
int kblasDrand_vbatch(kblasHandle_t handle, int* m, int* n, double** A_ptrs, int* lda, int max_m, kblasRandState_t state, int num_ops);
int kblasSrand_vbatch(kblasHandle_t handle, int* m, int* n, float** A_ptrs, int* lda, int max_m, kblasRandState_t state, int num_ops);

//------------------------------------------------------------------------------
// Non-uniform strided interface
int kblasDrand_vbatch_strided(kblasHandle_t handle, int* m, int* n, double* A_strided, int* lda, int stride_a, int max_m, kblasRandState_t state, int num_ops);
int kblasSrand_vbatch_strided(kblasHandle_t handle, int* m, int* n, float* A_strided, int* lda, int stride_a, int max_m, kblasRandState_t state, int num_ops);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
inline int kblas_rand_batch(kblasHandle_t handle, int m, int n, double** A_ptrs, int lda, kblasRandState_t state, int num_ops) 
{ return kblasDrand_batch(handle, m, n, A_ptrs, lda, state, num_ops); }
inline int kblas_rand_batch(kblasHandle_t handle, int m, int n, float** A_ptrs, int lda, kblasRandState_t state, int num_ops)
{ return kblasSrand_batch(handle, m, n, A_ptrs, lda, state, num_ops); }

inline int kblas_rand_batch(kblasHandle_t handle, int m, int n, double* A_strided, int lda, int stride_a, kblasRandState_t state, int num_ops)
{ return kblasDrand_batch_strided(handle, m, n, A_strided, lda, stride_a, state, num_ops); }
inline int kblas_rand_batch(kblasHandle_t handle, int m, int n, float* A_strided, int lda, int stride_a, kblasRandState_t state, int num_ops)
{ return kblasSrand_batch_strided(handle, m, n, A_strided, lda, stride_a, state, num_ops); }

inline int kblas_rand_batch(kblasHandle_t handle, int* m, int* n, double** A_ptrs, int* lda, int max_m, kblasRandState_t state, int num_ops)
{ return kblasDrand_vbatch(handle, m, n, A_ptrs, lda, max_m, state, num_ops); }
inline int kblas_rand_batch(kblasHandle_t handle, int* m, int* n, float** A_ptrs, int* lda, int max_m, kblasRandState_t state, int num_ops)
{ return kblasSrand_vbatch(handle, m, n, A_ptrs, lda, max_m, state, num_ops); }

inline int kblas_rand_batch(kblasHandle_t handle, int* m, int* n, double* A_strided, int* lda, int stride_a, int max_m, kblasRandState_t state, int num_ops)
{ return kblasDrand_vbatch_strided(handle, m, n, A_strided, lda, stride_a, max_m, state, num_ops); }
inline int kblas_rand_batch(kblasHandle_t handle, int* m, int* n, float* A_strided, int* lda, int stride_a, int max_m, kblasRandState_t state, int num_ops)
{ return kblasSrand_vbatch_strided(handle, m, n, A_strided, lda, stride_a, max_m, state, num_ops); }

#endif

#endif
