#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/transform.h>

#include <cuda.h>
#include <cusolverDn.h>
#include <magma.h>
#include <mkl.h>

#include <mini_blas_gpu.h>
#include <batch_svd.h>
#include <timer.h>
#include "util.h"
#include <gpu_util.h>

#define  vec_ptr(v)  thrust::raw_pointer_cast((v.data()))

#ifdef DOUBLE_PRECISION
typedef double Real;
#else
typedef float Real;
#endif

#define __USE_RAND_SVD__

int streams = 24;

extern void dgesvd( char* jobu, char* jobvt, int* m, int* n, double* a,
                int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
                double* work, int* lwork, int* info );

void batch_cpu_svd(Real* A, int stride_A, Real* U, int stride_U, Real* S, int stride_S, Real* V, int stride_V, int rows, int cols, int num_ops)
{
	#pragma omp parallel for
	for(int i = 0; i < num_ops; i++)
	{
		Real* a_op = A + i * stride_A;
		//Real* u_op = U + i * stride_U;
		Real* s_op = S + i * stride_S;
		Real* v_op = V + i * stride_V;

        Real stat[6] = {1, 0, 0, 0, 0, 0};
		#ifdef DOUBLE_PRECISION
        int info = LAPACKE_dgesvj(CblasColMajor, 'G', 'U', 'N', rows, cols, a_op, rows, s_op, cols, v_op, cols, &stat[0]);
		#else
		int info = LAPACKE_sgesvj(CblasColMajor, 'G', 'U', 'N', rows, cols, a_op, rows, s_op, cols, v_op, cols, &stat[0]);
		#endif
        if( info > 0 )
		{
			printf( "The algorithm computing SVD failed to converge for operation %d.\n", i);
			exit( 1 );
        }
	}
}

void batch_cpu_svd(Real* A, Real* U, Real* S, Real* V, int rows, int cols, int num_ops)
{
	batch_cpu_svd(A, rows * cols, U, rows * rows, S, cols, V, cols * cols, rows, cols, num_ops);
}

double batch_qr_cpu(Real* m, Real* tau, int rows, int cols, int num_ops)
{
	Timer timer;
	timer.start();
	#pragma omp parallel for
    for(int i = 0; i < num_ops; i++)
    {
        Real* m_op = m + i * rows * cols;
        Real* tau_op = tau + i * cols;
		#ifdef DOUBLE_PRECISION
		LAPACKE_dgeqrf(LAPACK_COL_MAJOR, rows, cols, m_op, rows, tau_op);
		#else
		LAPACKE_sgeqrf(LAPACK_COL_MAJOR, rows, cols, m_op, rows, tau_op);
		#endif
    }
	timer.stop();
	return timer.getElapsedTimeInSec();
}

double batch_unpackQ_cpu(Real* m, Real* tau, int rows, int cols, int num_ops)
{
	Timer timer;
	timer.start();
	#pragma omp parallel for
    for(int i = 0; i < num_ops; i++)
    {
        Real* m_op = m + i * rows * cols;
        Real* tau_op = tau + i * cols;
		#ifdef DOUBLE_PRECISION
        LAPACKE_dorgqr(LAPACK_COL_MAJOR, rows, cols, cols, m_op, rows, tau_op);
		#else
		LAPACKE_sorgqr(LAPACK_COL_MAJOR, rows, cols, cols, m_op, rows, tau_op);
		#endif
    }
	timer.stop();
	return timer.getElapsedTimeInSec();
}

double batch_qr_copyR_cpu(Real* A, Real* R, int rows, int cols, int num_ops)
{
	Timer timer;
	timer.start();
	#pragma omp parallel for
    for(int i = 0; i < num_ops; i++)
    {
        Real* A_op = A + i * rows * cols;
        Real* R_op = R + i * cols * cols;
        for(int j = 0; j < cols; j++)
            for(int k = j; k < cols; k++)
                R_op[j + k * cols] = A_op[j + k * rows];
    }
	timer.stop();
	return timer.getElapsedTimeInSec();
}

template <int transa, int transb>
double batch_gemm_cpu(Real* c, int ldc, Real* a, int lda, Real* b, int ldb, int m, int n, int p, Real alpha, Real beta, int num_ops)
{
	Timer timer;
	timer.start();
	#pragma omp parallel for
    for(int i = 0; i < num_ops; i++)
    {
        Real* c_op = c + i * m * n;
        Real* a_op = a + i * m * p;
        Real* b_op = b + i * p * n;

        CBLAS_TRANSPOSE trans_a = (transa ? CblasTrans : CblasNoTrans);
        CBLAS_TRANSPOSE trans_b = (transb ? CblasTrans : CblasNoTrans);
		#ifdef DOUBLE_PRECISION
        cblas_dgemm(CblasColMajor, trans_a, trans_b, m, n, p, alpha, a_op, lda, b_op, ldb, beta, c_op, ldc);
		#else
		cblas_sgemm(CblasColMajor, trans_a, trans_b, m, n, p, alpha, a_op, lda, b_op, ldb, beta, c_op, ldc);
		#endif
    }
    timer.stop();
	return timer.getElapsedTimeInSec();
}

struct prg
{
    Real a, b;

    __host__ __device__
    prg(Real _a=0, Real _b=1) : a(_a), b(_b) {};

    __host__ __device__
        Real operator()(const unsigned int n) const
        {
            thrust::default_random_engine rng;
            thrust::random::normal_distribution<Real> dist(a, b);
            rng.discard(n);

            return dist(rng);
        }
};

void generateRandomData(Real* random_data, int rows, int cols, int num_ops)
{
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);

    thrust::transform(thrust::system::omp::par,
            index_sequence_begin,
            index_sequence_begin + rows * cols * num_ops,
            random_data,
            prg());
}

void batch_svd_randomized_cpu(Real* M, Real* S, int rows, int cols, int rank, int num_ops)
{
    // generate the sampling matrices
    thrust::host_vector<Real> omega(num_ops * cols * rank);
    generateRandomData(vec_ptr(omega), cols, rank, num_ops);

    // Memory for temporary matrix data
    thrust::host_vector<Real>
        Y(num_ops * rows * rank), B(num_ops * rank * cols),
        R(num_ops * rank * rank), tau(num_ops * rank),
        U(num_ops * rank * rank), V(num_ops * rank * rank);

    Real alpha = 1, beta = 0;

    // First form the sampled matrix Y = A * omega
	batch_gemm_cpu<0, 0>(
        vec_ptr(Y), rows,
        M, rows,
        vec_ptr(omega), cols,
        rows, rank, cols,
        alpha, beta, num_ops
    );

    // Overwrite Y with Q of its QR decomposition
    batch_qr_cpu(vec_ptr(Y), vec_ptr(tau), rows, rank, num_ops);
    batch_unpackQ_cpu(vec_ptr(Y), vec_ptr(tau), rows, rank, num_ops);

	// Form B = A' * Q_Y
    batch_gemm_cpu<1, 0>(
		vec_ptr(B), cols,
		M, rows,
		vec_ptr(Y), rows,
		cols, rank, rows,
        alpha, beta, num_ops
	);

    // Do the QR of B - we only need the Q of B if we want the right singular vectors
    batch_qr_cpu(vec_ptr(B), vec_ptr(tau), cols, rank, num_ops);
    batch_qr_copyR_cpu(vec_ptr(B), vec_ptr(R), cols, rank, num_ops);

    // Now do the SVD of R
	batch_cpu_svd(vec_ptr(R), rank * rank, vec_ptr(U), rank * rank, S, cols, vec_ptr(V), rank * rank, rank, rank, num_ops);

    // Finally, we overwrite the matrix with left singular values of the
    // truncated SVD as the product U_A = Q_Y * V_B
	batch_gemm_cpu<0, 0>(
        M, rows,
		vec_ptr(Y), rows,
		vec_ptr(V), rank,
        rows, rank, rank,
		alpha, beta, num_ops
	);
}

void batch_cudaSVD(Real* A, Real* U, Real* S, Real* V, int rows, int cols, int num_ops, int streams)
{
	thrust::device_vector<int> dev_info(streams);
	int* info_ptr = vec_ptr(dev_info);

	// Allocate the streams
	std::vector<cudaStream_t> cuda_streams(streams);
	std::vector<cusolverDnHandle_t> solver_handles(streams);

	for(int i = 0 ; i < streams; i++)
	{
		gpuErrchk(cudaStreamCreate(&cuda_streams[i]));
		cusolveSafeCall(cusolverDnCreate(&solver_handles[i]));
		cusolveSafeCall(cusolverDnSetStream(solver_handles[i], cuda_streams[i]));
	}

	int work_size = 0;
	#ifdef DOUBLE_PRECISION
	cusolveSafeCall(cusolverDnDgesvd_bufferSize(solver_handles[0], rows, cols, &work_size));
	#else
	cusolveSafeCall(cusolverDnSgesvd_bufferSize(solver_handles[0], rows, cols, &work_size));
	#endif
	// Allocate workspace for all operations
	thrust::device_vector<Real> workspace(streams * work_size);
	Real* work_ptr = vec_ptr(workspace);

	int ops_done = 0;
	while(ops_done < num_ops)
	{
		int ops_to_do = std::min(streams, num_ops - ops_done);

        #pragma omp parallel for num_threads(streams)
		for(int i = 0; i < ops_to_do; i++)
		{
			int op_index = ops_done + i;
			Real* matrix = A + rows * cols * op_index;
			Real* U_m = U + rows * rows * op_index;
			Real* S_m = S + cols * op_index;
			Real* V_m = V + cols * cols * op_index;

			Real* work_m = work_ptr + work_size * i;
			int* dev_info_m = info_ptr + i;
			#ifdef DOUBLE_PRECISION
			cusolveSafeCall(
				cusolverDnDgesvd(
					solver_handles[i], 'A', 'N', rows, cols,
					matrix, rows, S_m, U_m, rows,
					V_m, cols, work_m, work_size, NULL,
					dev_info_m
				)
			);
			#else
			cusolveSafeCall(
				cusolverDnSgesvd(
					solver_handles[i], 'A', 'N', rows, cols,
					matrix, rows, S_m, U_m, rows,
					V_m, cols, work_m, work_size, NULL,
					dev_info_m
				)
			);
			#endif

		}
		ops_done += streams;
	}

	for(int i = 0 ; i < streams; i++)
	{
		gpuErrchk(cudaStreamDestroy(cuda_streams[i]));
		cusolveSafeCall(cusolverDnDestroy(solver_handles[i]));
	}
    cudaDeviceSynchronize();
}

void testSValues(Real* S1, Real* S2, int stride_s, int rank, int num_ops)
{
	Real avg_diff = 0;
	for(int i = 0; i < num_ops; i++)
	{
		Real diff = 0, norm2 = 0;
		Real* s1_op = S1 + i * stride_s;
		Real* s2_op = S2 + i * stride_s;
		for(int j = 0; j < rank; j++)
		{
			diff += (s1_op[j] - s2_op[j]) * (s1_op[j] - s2_op[j]);
			norm2 += s2_op[j] * s2_op[j];
		}
		avg_diff += sqrt(diff / norm2);
	}
	printf("S Computation error: %e\n", avg_diff / num_ops);
}

void testDim(int rows, int cols, int num_ops, size_t rand_seed, Real& svd_gops, Real& svd_time, Real& gemm_gops, Real& gemm_time, Real& qr_time, Real& qr_gops, Real& total_time, Real& cusolver_time, Real& cpu_time)
{
    kblas_gpu_timer timer;
	timer.init();
    Timer timer_cpu;

	thrust::host_vector<Real> host_A(rows * cols * num_ops);
	thrust::host_vector<Real> host_U(rows * rows * num_ops), host_S(cols * num_ops), host_V(cols * cols * num_ops);
	thrust::host_vector<Real> exact_S(cols * num_ops, 0);
	thrust::host_vector<Real> workspace(3 * rows * num_ops);

	// Generate a bunch of random seeds from the master seed
	thrust::host_vector<int> random_seeds(num_ops);
	thrust::default_random_engine seed_rng; seed_rng.seed(rand_seed);
	thrust::uniform_int_distribution <int> seed_dist(0, 10000);
	for(int i = 0; i < num_ops; i++)
		random_seeds[i] = seed_dist(seed_rng);

	#pragma omp parallel for
	for(int op_index = 0; op_index < num_ops; op_index++)
	{
		thrust::default_random_engine rng; rng.seed(random_seeds[op_index]);
		thrust::uniform_int_distribution<int> dist(0, 4095);
		int seeds[4] = {dist(rng), dist(rng), dist(rng), dist(rng)};
		if(seeds[3] % 2 != 1) seeds[3]++;

		int info;
		Real* matrix = vec_ptr(host_A) + rows * cols * op_index;
		Real* svals  = vec_ptr(exact_S) + cols * op_index;
		Real* work   = vec_ptr(workspace) + 3 * rows * op_index;

		int mode = 5;
		Real cond = 1e7, norm_2 = 1;

		mode = 0;
		Real decay = 0.4;
		for(int i = 0; i < cols; i++) svals[i] = exp(-decay*i);
		#ifdef DOUBLE_PRECISION
		info = LAPACKE_dlatms(LAPACK_COL_MAJOR, rows, cols, 'N', &seeds[0], 'N', svals, mode, cond, norm_2, rows, cols, 'N', matrix, rows);
		#else
		info = LAPACKE_slatms(LAPACK_COL_MAJOR, rows, cols, 'N', &seeds[0], 'N', svals, mode, cond, norm_2, rows, cols, 'N', matrix, rows);
		#endif
		if(info != 0) printf("Error generating random matrix: %d\n", info);
		thrust::sort(svals, svals + cols, thrust::greater<Real>());
	}

	thrust::device_vector<Real> dev_A = host_A;
	thrust::device_vector<Real> dev_U(rows * rows * num_ops), dev_S(cols * num_ops), dev_V(cols * cols * num_ops);

    int col_limit = 64;
    #ifdef __USE_RAND_SVD__
    int rank = (cols > col_limit ? col_limit : cols);
    #else
    int rank = cols;
    #endif
    rank = (rank > rows ? rows : rank);

    ///////////////////////////
    // CPU
    ///////////////////////////
	thrust::host_vector<Real> temp_A = host_A;
	timer_cpu.start();
	temp_A = dev_A;
    if(cols > col_limit)
    {
        #ifdef __USE_RAND_SVD__
        batch_svd_randomized_cpu(vec_ptr(temp_A), vec_ptr(host_S), rows, cols, col_limit, num_ops);
        #else
        batch_cpu_svd(vec_ptr(temp_A), vec_ptr(host_U), vec_ptr(host_S), vec_ptr(host_V), rows, cols, num_ops);
        #endif
    }
    else
        batch_cpu_svd(vec_ptr(temp_A), vec_ptr(host_U), vec_ptr(host_S), vec_ptr(host_V), rows, cols, num_ops);
    dev_A = host_U; dev_S = host_S;
	timer_cpu.stop();
    cpu_time = timer_cpu.getElapsedTimeInSec();

    testSValues(vec_ptr(host_S), vec_ptr(exact_S), cols, rank, num_ops);

    ///////////////////////////
    // CUBLAS
    ///////////////////////////
    dev_A = host_A;
	dev_S = thrust::device_vector<Real>(cols * num_ops, 0);
	timer.start();
	batch_cudaSVD(vec_ptr(dev_A), vec_ptr(dev_U), vec_ptr(dev_S), vec_ptr(dev_V), rows, cols, num_ops, streams);
	cusolver_time = timer.stop();
	thrust::host_vector<Real> host_U_cublas = dev_U, host_S_cublas = dev_S, host_V_cublas = dev_V;

    testSValues(vec_ptr(host_S_cublas), vec_ptr(exact_S), cols, rank, num_ops);

    ///////////////////////////
    // KBLAS
    ///////////////////////////
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	GPUBlasHandle handle(1, stream);
	handle.setWorkspace(200 * 1024 * 1024);

	dev_A = host_A;
	dev_S = thrust::device_vector<Real>(cols * num_ops, 0);
	timer.start();
	if(cols > col_limit)
    {
        #ifdef __USE_RAND_SVD__
        kblas_rsvd_batched(rows, cols, col_limit, vec_ptr(dev_A), rows, rows * cols, vec_ptr(dev_S), cols, num_ops, handle);
        #else
		kblas_gesvj_qr_batched(rows, cols, vec_ptr(dev_A), rows, rows * cols, vec_ptr(dev_S), cols, num_ops, handle);
		#endif
    }
    else
        kblas_gesvj_batched(rows, cols, vec_ptr(dev_A), rows, rows * cols, vec_ptr(dev_S), cols, num_ops, handle);

	total_time = timer.stop();

	svd_time  = PerformanceCounter::getOpTime(PerformanceCounter::SVD);
    svd_gops  = PerformanceCounter::getOpCount(PerformanceCounter::SVD);

    gemm_time = PerformanceCounter::getOpTime(PerformanceCounter::GEMM);
    gemm_gops = PerformanceCounter::getOpCount(PerformanceCounter::GEMM);

	qr_time = PerformanceCounter::getOpTime(PerformanceCounter::QR);
    qr_gops = PerformanceCounter::getOpCount(PerformanceCounter::QR);

	PerformanceCounter::clearCounters();

	thrust::host_vector<Real> host_S_kblas = dev_S, host_U_kblas = dev_A;
	testSValues(vec_ptr(host_S_kblas), vec_ptr(exact_S), cols, rank, num_ops);
}

void avg_and_stdev(Real* values, int num_vals, Real& avg, Real& std_dev)
{
	assert(num_vals > 0);
	if(num_vals == 1) {avg = values[0]; std_dev = 0; return; }

	avg = 0;
    // Skip the first run
	for(int i = 1; i < num_vals; i++) avg += values[i];
	avg /= (num_vals - 1);
	std_dev = 0;
	for(int i = 1; i < num_vals; i++)
		std_dev += (values[i] - avg) * (values[i] - avg);
	std_dev = sqrt(std_dev / (num_vals - 1));
}

int main(int argc, char** argv)
{
	if(argc != 5)
	{
		printf("Usage: %s rows cols num_ops streams\n", argv[0]);
		exit(0);
	}
	magma_init();
	#ifdef DOUBLE_PRECISION
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	#endif

	int rows = std::strtol( argv[1], NULL, 0 );
	int cols = std::strtol( argv[2], NULL, 0 );
	int num_ops = std::strtol( argv[3], NULL, 0 );
	streams = std::strtol( argv[4], NULL, 0 );

	const int RUNS = 1;

	Real svd_gops[RUNS], svd_time[RUNS], gemm_gops[RUNS], gemm_time[RUNS], qr_gops[RUNS], qr_time[RUNS], total_time[RUNS], cusolver_time[RUNS];
	Real svd_gflops[RUNS], gemm_gflops[RUNS], qr_gflops[RUNS], total_gflops[RUNS];
	Real cpu_time[RUNS];

	for(int i = 0; i < RUNS; i++)
	{
		testDim(rows, cols, num_ops, i, svd_gops[i], svd_time[i], gemm_gops[i], gemm_time[i], qr_time[i], qr_gops[i], total_time[i], cusolver_time[i], cpu_time[i]);
		svd_gflops[i] = (svd_time[i] > 0 ? svd_gops[i] / svd_time[i] : 0);
		gemm_gflops[i] = (gemm_time[i] > 0 ? gemm_gops[i] / gemm_time[i] : 0);
		qr_gflops[i] = (qr_time[i] > 0 ? qr_gops[i] / qr_time[i] : 0);
		total_gflops[i] = (qr_gops[i] + svd_gops[i] + gemm_gops[i]) / total_time[i];
	}

	Real avg_time, std_dev_time, avg_gflops, std_dev_gflops;
	avg_and_stdev(&total_time[0], RUNS, avg_time, std_dev_time);
	avg_and_stdev(&total_gflops[0], RUNS, avg_gflops, std_dev_gflops);

	Real avg_svd_time, avg_svd_gflops, dummy;
	avg_and_stdev(&svd_time[0], RUNS, avg_svd_time, dummy);
	avg_and_stdev(&svd_gflops[0], RUNS, avg_svd_gflops, dummy);

	Real avg_gemm_time, avg_gemm_gflops;
	avg_and_stdev(&gemm_time[0], RUNS, avg_gemm_time, dummy);
	avg_and_stdev(&gemm_gflops[0], RUNS, avg_gemm_gflops, dummy);

	Real avg_qr_time, avg_qr_gflops;
	avg_and_stdev(&qr_time[0], RUNS, avg_qr_time, dummy);
	avg_and_stdev(&qr_gflops[0], RUNS, avg_qr_gflops, dummy);

	Real avg_cusolver_time, std_dev_cusolver_time;
	avg_and_stdev(&cusolver_time[0], RUNS, avg_cusolver_time, std_dev_cusolver_time);

	Real avg_other = avg_time - avg_svd_time - avg_gemm_time - avg_qr_time;

    Real avg_cpu_time, std_dev_cpu_time;
    avg_and_stdev(&cpu_time[0], RUNS, avg_cpu_time, std_dev_cpu_time);

    printf("CPU Time: %.6f ,std_dev: %.6f\n", avg_cpu_time, std_dev_cpu_time);

	if(rows > 64)
	{
		printf("GEMM performance:\t %.6f (%.6f %% of total) at %.6f GFLOPs\n", avg_gemm_time, avg_gemm_time / avg_time * 100, avg_gemm_gflops);
		printf("QR performance:\t\t %.6f (%.6f %% of total) at %.6f GFLOPs\n", avg_qr_time, avg_qr_time / avg_time * 100, avg_qr_gflops);
		printf("SVD performance:\t %.6f (%.6f %% of total) at %.6f GFLOPs\n", avg_svd_time, avg_svd_time / avg_time * 100, avg_svd_gflops);
		printf("Other computations:\t %.6f (%.6f %% of total)\n", avg_other, avg_other / avg_time * 100);
	}
    printf("Batch SVD total time:\t %.6f at %.6f GFLOPs ,std_dev: %.6f %.6f (%.6f %%)\n", avg_time, avg_gflops, std_dev_time, std_dev_gflops, std_dev_gflops / avg_gflops * 100);
	printf("Streamed SVD:\t %.6f ,std_dev: %.6f\n", avg_cusolver_time, std_dev_cusolver_time);

    return 0;
}
