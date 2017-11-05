#ifndef __BATCH_PERMUTE_H__
#define __BATCH_PERMUTE_H__

void kblas_permute_columns_batched(int rows, int cols, float** M_ptrs, int ldm, int** perm, int* rank, int num_ops);
void kblas_permute_columns_batched(int rows, int cols, double** M_ptrs, int ldm, int** perm, int* rank, int num_ops);

#endif
