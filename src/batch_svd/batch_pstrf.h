#ifndef __BATCH_PSTRF_H__
#define __BATCH_PSTRF_H__

void kblas_pstrf_batched(float** M_ptrs, int ldm, int** piv, int* r, int dim, int num_ops);
void kblas_pstrf_batched(double** M_ptrs, int ldm, int** piv, int* r, int dim, int num_ops);

#endif
