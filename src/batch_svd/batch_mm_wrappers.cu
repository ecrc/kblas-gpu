#include <batch_mm_wrappers.h>

void batch_gemm(int trans_a, int trans_b, int m, int n, int k, float alpha, float** dA_array, int lda, float** dB_array, int ldb, float beta, float** dC_array, int ldc, int batchCount, GPUBlasHandle& handle)
{
	if(m == 0 || n == 0 || k == 0 || batchCount == 0) 
		return;
	
	#ifdef HLIB_PROFILING_ENABLED
	handle.timer.start(handle.stream);
	#endif
	
	if(handle.use_magma)
	{
		magma_trans_t magma_ta = (trans_a ? MagmaTrans : MagmaNoTrans);
		magma_trans_t magma_tb = (trans_b ? MagmaTrans : MagmaNoTrans);
		
		int batch_increment = 65535;
		int batch_start = 0;
		
		while(batch_start != batchCount)
		{
			int batch_size = std::min(batch_increment, batchCount - batch_start);

			magmablas_sgemm_batched ( 
				magma_ta, magma_tb, m, n, k, 
				alpha, (const float**) dA_array, lda, 
				(const float**)dB_array, ldb, beta,
				dC_array, ldc, batch_size, 
				handle.magma_queue //, handle.cublas_handle
			);		
			
			dA_array += batch_size;
			dB_array += batch_size;
			dC_array += batch_size;
			
			batch_start += batch_size;
			gpuErrchk( cudaGetLastError() );
		}
	}
	else
	{
		cublasOperation_t cublas_ta = (trans_a ? CUBLAS_OP_T : CUBLAS_OP_N);
		cublasOperation_t cublas_tb = (trans_b ? CUBLAS_OP_T : CUBLAS_OP_N);
		
        gpuCublasErrchk(
            cublasSgemmBatched(
                    handle.cublas_handle, cublas_ta, cublas_tb, m, n, k, 
                    &alpha, (const float**)dA_array, lda, 
					(const float**)dB_array, ldb, &beta, 
					dC_array, ldc, batchCount
            )
        );
	}
	#ifdef HLIB_PROFILING_ENABLED
	double time_elapsed = handle.timer.stop(handle.stream);
	double gemm_gflops = (double)(2 * m * n * k * 1e-9) * batchCount;
	PerformanceCounter::addOpCount(PerformanceCounter::GEMM, gemm_gflops);
    PerformanceCounter::addOpTime(PerformanceCounter::GEMM, time_elapsed);
	#endif
}

void batch_gemm(int trans_a, int trans_b, int m, int n, int k, double alpha, double** dA_array, int lda, double** dB_array, int ldb, double beta, double** dC_array, int ldc, int batchCount, GPUBlasHandle& handle)
{
	if(m == 0 || n == 0 || k == 0 || batchCount == 0) 
		return;
	
	#ifdef HLIB_PROFILING_ENABLED
	handle.timer.start(handle.stream);
	#endif
	
	if(handle.use_magma)
	{
		magma_trans_t magma_ta = (trans_a ? MagmaTrans : MagmaNoTrans);
		magma_trans_t magma_tb = (trans_b ? MagmaTrans : MagmaNoTrans);
		
		int batch_increment = 65535;
		int batch_start = 0;
		
		while(batch_start != batchCount)
		{
			int batch_size = std::min(batch_increment, batchCount - batch_start);

			magmablas_dgemm_batched ( 
				magma_ta, magma_tb, m, n, k, 
				alpha, (const double**) dA_array, lda, 
				(const double**)dB_array, ldb, beta,
				dC_array, ldc, batch_size, 
				handle.magma_queue //, handle.cublas_handle
			);		
			
			dA_array += batch_size;
			dB_array += batch_size;
			dC_array += batch_size;
			
			batch_start += batch_size;
			gpuErrchk( cudaGetLastError() );
		}
	}
	else
	{
		cublasOperation_t cublas_ta = (trans_a ? CUBLAS_OP_T : CUBLAS_OP_N);
		cublasOperation_t cublas_tb = (trans_b ? CUBLAS_OP_T : CUBLAS_OP_N);
		
        gpuCublasErrchk(
            cublasDgemmBatched(
                    handle.cublas_handle, cublas_ta, cublas_tb, m, n, k, 
                    &alpha, (const double**)dA_array, lda, 
					(const double**)dB_array, ldb, &beta, 
					dC_array, ldc, batchCount
            )
        );
	}
	#ifdef HLIB_PROFILING_ENABLED
	double time_elapsed = handle.timer.stop(handle.stream);
	double gemm_gflops = (double)(2 * m * n * k * 1e-9) * batchCount;
	PerformanceCounter::addOpCount(PerformanceCounter::GEMM, gemm_gflops);
    PerformanceCounter::addOpTime(PerformanceCounter::GEMM, time_elapsed);
	#endif
}

void batch_syrk(int uppper, int trans, int n, int k, double alpha, double** dA_array, int lda, double beta, double** dC_array, int ldc, int batchCount, GPUBlasHandle& handle)
{
	if(n == 0 || k == 0 || batchCount == 0) 
		return;
	
	#ifdef HLIB_PROFILING_ENABLED
	handle.timer.start(handle.stream);
	#endif
	
	if(!handle.use_magma) printf("Batched SYRK needs magma\n");
	
	magma_trans_t magma_trans = (trans  ? MagmaTrans : MagmaNoTrans);
	magma_uplo_t  magma_uplo  = (uppper ? MagmaUpper : MagmaLower);
	
	int batch_increment = 65535;
	int batch_start = 0;
	
	while(batch_start != batchCount)
	{
		int batch_size = std::min(batch_increment, batchCount - batch_start);

		magmablas_dsyrk_batched( 
			magma_uplo, magma_trans, n, k, 
			alpha, (const double**) dA_array, lda, 
			beta, dC_array, ldc, 
			batch_size, handle.magma_queue 
		);		
		
		dA_array += batch_size;
		dC_array += batch_size;
		
		batch_start += batch_size;
		gpuErrchk( cudaGetLastError() );
	}

	#ifdef HLIB_PROFILING_ENABLED
	double time_elapsed = handle.timer.stop(handle.stream);
	double gemm_gflops = (double)(n * n * k * 1e-9) * batchCount;
	PerformanceCounter::addOpCount(PerformanceCounter::GEMM, gemm_gflops);
    PerformanceCounter::addOpTime(PerformanceCounter::GEMM, time_elapsed);
	#endif
}

void batch_syrk(int uppper, int trans, int n, int k, float alpha, float** dA_array, int lda, float beta, float** dC_array, int ldc, int batchCount, GPUBlasHandle& handle)
{
	if(n == 0 || k == 0 || batchCount == 0) 
		return;
	
	#ifdef HLIB_PROFILING_ENABLED
	handle.timer.start(handle.stream);
	#endif
	
	if(!handle.use_magma) printf("Batched SYRK needs magma\n");
	
	magma_trans_t magma_trans = (trans  ? MagmaTrans : MagmaNoTrans);
	magma_uplo_t  magma_uplo  = (uppper ? MagmaUpper : MagmaLower);
	
	int batch_increment = 65535;
	int batch_start = 0;
	
	while(batch_start != batchCount)
	{
		int batch_size = std::min(batch_increment, batchCount - batch_start);

		magmablas_ssyrk_batched( 
			magma_uplo, magma_trans, n, k, 
			alpha, (const float**) dA_array, lda, 
			beta, dC_array, ldc, 
			batch_size, handle.magma_queue 
		);		
		
		dA_array += batch_size;
		dC_array += batch_size;
		
		batch_start += batch_size;
		gpuErrchk( cudaGetLastError() );
	}

	#ifdef HLIB_PROFILING_ENABLED
	double time_elapsed = handle.timer.stop(handle.stream);
	double gemm_gflops = (double)(n * n * k * 1e-9) * batchCount;
	PerformanceCounter::addOpCount(PerformanceCounter::GEMM, gemm_gflops);
    PerformanceCounter::addOpTime(PerformanceCounter::GEMM, time_elapsed);
	#endif
}
