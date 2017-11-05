#include "common.h"
#include "batch_tsqr_chol.h"
#include "batch_pstrf.h"
#include "batch_permute.h"
#include "debug_routines.h"
#include "batch_mm_wrappers.h"

#include <mkl.h>
void batch_pstrf_cpu(Real** R_ptrs, int ldr, int** piv_ptrs, int* ranks, int dim, int num_ops)
{
	thrust::device_ptr<Real*> dev_R_ptrs(R_ptrs);
	thrust::host_vector<Real*> host_R_ptrs(dev_R_ptrs, dev_R_ptrs + num_ops);
	thrust::host_vector<int> host_ranks(num_ops);
	
	thrust::device_ptr<int*> dev_piv_ptrs(piv_ptrs);
	thrust::host_vector<int*> host_piv_ptrs(dev_piv_ptrs, dev_piv_ptrs + num_ops);
	
	//#pragma omp parallel for
	for(int op_id = 0; op_id < num_ops; op_id++)
	{
		thrust::device_ptr<Real> dev_R_op(host_R_ptrs[op_id]);
		thrust::host_vector<Real> host_R_op(dev_R_op, dev_R_op + ldr * dim);

		Real* R = vec_ptr(host_R_op);
		int rank = dim;
		thrust::host_vector<int> piv(dim);
		
		//printDenseMatrix(R, ldr, dim, dim, 5, "A");
		// Now try and get the cholesky factorization of R, overwritten in R
		#ifdef DOUBLE_PRECISION
		int info = LAPACKE_dpstrf(LAPACK_COL_MAJOR, 'U', dim, R, dim, &piv[0], &rank, -1);
		#else
		int info = LAPACKE_spstrf(LAPACK_COL_MAJOR, 'U', dim, R, dim, &piv[0], &rank, -1);
		#endif
		if(info != 0) 
			printf("Operation %d failed at cholesky factorization with error %d - rank was %d\n", op_id, info, rank);
		
		host_ranks[op_id] = rank;
		for(int i = 0; i < dim; i++) piv[i]--;
		
		thrust::device_ptr<int> dev_pivots(host_piv_ptrs[op_id]);
		thrust::copy(piv.begin(), piv.end(), dev_pivots);
		thrust::copy(host_R_op.begin(), host_R_op.end(), dev_R_op);
	}
	thrust::device_ptr<int> dev_ranks(ranks);
	thrust::copy(host_ranks.begin(), host_ranks.end(), dev_ranks);
}

void batch_tsqr_chol(float** M_ptrs, int ldm, float** R_ptrs, int ldr, int** piv, int* rank, int rows, int cols, int num_ops, GPUBlasHandle& handle)
{
	// Form the gram matrices
	//batch_gemm(1, 0, cols, cols, rows, 1, M_ptrs, ldm, M_ptrs, ldm, 0, R_ptrs, ldr, num_ops, handle);
	batch_syrk(1, 1, cols, rows, 1, M_ptrs, ldm, 0, R_ptrs, ldr, num_ops, handle);
	
	// Do the cholesky of the gram matrix 
	kblas_pstrf_batched(R_ptrs, ldr, piv, rank, cols, num_ops);
	
	// Apply the pivots
	kblas_permute_columns_batched(rows, cols, M_ptrs, ldm, piv, rank, num_ops);
	
	// Finally get the Q factor as A*R^-1 by doing a trsm 
	thrust::device_vector<int> v_rows(num_ops + 1, rows), v_ldr(num_ops + 1, ldr), v_ldm(num_ops + 1, ldm);
	magmablas_strsm_vbatched(MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit, vec_ptr(v_rows), rank, 1, R_ptrs, vec_ptr(v_ldr), M_ptrs, vec_ptr(v_ldm), num_ops, handle.magma_queue);
	
	gpuErrchk( cudaGetLastError() );
}

void batch_tsqr_chol(double** M_ptrs, int ldm, double** R_ptrs, int ldr, int** piv, int* rank, int rows, int cols, int num_ops, GPUBlasHandle& handle)
{
	// Form the gram matrices
	//handle.timer.start();
	//Real phase_time[4] = {0};
	//printDenseMatrixGPU(M_ptrs, 0, ldm, rows, cols, 14, "M");
	
	//batch_gemm(1, 0, cols, cols, rows, 1, M_ptrs, ldm, M_ptrs, ldm, 0, R_ptrs, ldr, num_ops, handle);
	batch_syrk(1, 1, cols, rows, 1, M_ptrs, ldm, 0, R_ptrs, ldr, num_ops, handle);
	//phase_time[0] = handle.timer.stop();
	//printDenseMatrixGPU(R_ptrs, 0, ldr, cols, cols, 14, "G_gpu");
	// Do the cholesky of the gram matrix 
	//handle.timer.start();
	kblas_pstrf_batched(R_ptrs, ldr, piv, rank, cols, num_ops);
	//batch_pstrf_cpu(R_ptrs, ldr, piv, rank, cols, num_ops);
	//phase_time[1] = handle.timer.stop();
	/*thrust::host_vector<int> host_piv = copyGPUArray(piv, 0, cols);
	thrust::host_vector<int> host_ranks = copyGPUArray(rank, num_ops);
	printf("ranks=["); for(int i = 0; i < num_ops; i++) printf("%d ", host_ranks[i]); printf("];\n");
	printf("piv=["); for(int i = 0; i < cols; i++) printf("%d ", host_piv[i]); printf("];\n");
	printDenseMatrixGPU(R_ptrs, 0, ldr, cols, cols, 14, "R");
	*/
	// Apply the pivots
	//handle.timer.start();
	kblas_permute_columns_batched(rows, cols, M_ptrs, ldm, piv, rank, num_ops);
	//phase_time[2] = handle.timer.stop();

	// Finally get the Q factor as A*R^-1 by doing a trsm 
	//handle.timer.start();
	thrust::device_vector<int> v_rows(num_ops + 1, rows), v_ldr(num_ops + 1, ldr), v_ldm(num_ops + 1, ldm);
	magmablas_dtrsm_vbatched(MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit, vec_ptr(v_rows), rank, 1, R_ptrs, vec_ptr(v_ldr), M_ptrs, vec_ptr(v_ldm), num_ops, handle.magma_queue);

	//phase_time[3] = handle.timer.stop();
	
	//printf("Syrk time = %fs - Chol time = %fs - Pivot time = %fs - TRSM time = %fs\n", phase_time[0], phase_time[1], phase_time[2], phase_time[3]);
	//printDenseMatrixGPU(M_ptrs, 0, ldm, rows, cols, 14, "Q");
	
	gpuErrchk( cudaGetLastError() );
}
