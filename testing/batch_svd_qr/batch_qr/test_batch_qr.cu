#include <stdio.h>
#include <string.h>
#include <cstdlib>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

#include <mini_blas_gpu.h>
#include <batch_qr.h>
#include <magma.h>
#include <mkl.h>

#include <thrust_wrappers.h>
#include <timer.h>
// #include <timer_gpu.h>

#define  vec_ptr(v)  thrust::raw_pointer_cast((v.data()))

#ifdef DOUBLE_PRECISION
typedef double Real;
#else
typedef float Real;
#endif

void printDenseMatrix(Real *matrix, int m, int n, int digits, const char* name)
{
    char format[10];
    sprintf(format, "%%.%df ", digits);
    printf("%s = [\n", name);
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++)
            printf(format, matrix[i + j * m]);
        printf(";\n");
    }
    printf("];\n");
}

double batch_qr_magma(Real* m, Real* tau, int rows, int cols, int num_ops, magma_queue_t& queue)
{
    Real** A_array, **Tau_array;
    magma_int_t* info;

    gpuErrchk( cudaMalloc((void**)&A_array, num_ops * sizeof(Real*)) );
    gpuErrchk( cudaMalloc((void**)&Tau_array, num_ops * sizeof(Real*)) );
    gpuErrchk( cudaMalloc((void**)&info, num_ops * sizeof(magma_int_t)) );

    generateArrayOfPointers(m, A_array, rows * cols, num_ops);
    generateArrayOfPointers(tau, Tau_array, cols, num_ops);
    gpuErrchk( cudaGetLastError() );

	kblas_gpu_timer timer;
	timer.init();
	timer.start();
    #ifdef DOUBLE_PRECISION
	magma_dgeqrf_batched(rows, cols, A_array, rows, Tau_array, info, num_ops, queue);
    #else
	magma_sgeqrf_batched(rows, cols, A_array, rows, Tau_array, info, num_ops, queue);
    #endif
    double total_time = timer.stop();

	gpuErrchk( cudaGetLastError() );
    gpuErrchk( cudaFree(A_array) );
    gpuErrchk( cudaFree(Tau_array) );
    gpuErrchk( cudaFree(info) );

    return total_time;
}

double batch_qr_cublas(Real* m, Real* tau, int rows, int cols, int num_ops)
{
    cublasHandle_t handle;
    gpuCublasErrchk( cublasCreate(&handle) );
    int info;

    Real** A_array, **Tau_array;
    gpuErrchk( cudaMalloc((void**)&A_array, num_ops * sizeof(Real*)) );
    gpuErrchk( cudaMalloc((void**)&Tau_array, num_ops * sizeof(Real*)) );

    generateArrayOfPointers(m, A_array, rows * cols, num_ops);
    generateArrayOfPointers(tau, Tau_array, cols, num_ops);

	kblas_gpu_timer timer;
	timer.init();
	timer.start();
    #ifdef DOUBLE_PRECISION
    gpuCublasErrchk(
            cublasDgeqrfBatched(
                        handle, rows, cols, A_array, rows,
                        Tau_array, &info, num_ops
        )
    );
    #else
    gpuCublasErrchk(
            cublasSgeqrfBatched(
                        handle, rows, cols, A_array, rows,
                        Tau_array, &info, num_ops
        )
    );
    #endif
	double total_time = timer.stop();

    gpuErrchk( cudaGetLastError() );
    gpuErrchk( cudaFree(A_array) );
    gpuErrchk( cudaFree(Tau_array) );

	return total_time;
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

Real compare_results_R(Real* m1, Real* m2, int rows, int cols, int num_ops)
{
    Real err = 0;
    for(int op = 0; op < num_ops; op++)
    {
        Real* m1_op = m1 + op * rows * cols;
        Real* m2_op = m2 + op * rows * cols;
        Real err_op = 0, norm_f_op = 0;
        for(int i = 0; i < cols; i++)
        {
            for(int j = i; j < cols; j++)
            {
                Real diff_entry = abs(abs(m1_op[i + j * rows]) - abs(m2_op[i + j * rows]));
                err_op += diff_entry * diff_entry;
                norm_f_op += m1_op[i + j * rows] * m1_op[i + j * rows];
            }
        }
        err += sqrt(err_op / norm_f_op);
    }
    return  err / num_ops;
}

void avg_and_stdev(Real* values, int num_vals, Real& avg, Real& std_dev)
{
	if(num_vals == 0) return;

	avg = 0;
	for(int i = 0; i < num_vals; i++) avg += values[i];
	avg /= num_vals;
	std_dev = 0;
	for(int i = 0; i < num_vals; i++)
		std_dev += (values[i] - avg) * (values[i] - avg);
	std_dev = sqrt(std_dev / num_vals);
}

int main(int argc, char** argv)
{
    if(argc != 4)
    {
        printf("Usage: %s rows columns num_ops\n", argv[0]);
        return -1;
    }

    int rows = std::strtol( argv[1], NULL, 0 );
    int columns = std::strtol( argv[2], NULL, 0 );
    int num_ops = std::strtol( argv[3], NULL, 0 );

    #ifdef DOUBLE_PRECISION
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    #endif

    magma_init();
    magma_queue_t queue;
    magma_queue_create(&queue);

    thrust::host_vector<Real> m(num_ops * rows * columns), m_original, q;
    thrust::host_vector<Real> tau(num_ops * columns);
    for(int i = 0; i < m.size(); i++)
        m[i] = (Real)rand() / RAND_MAX;
    m_original = m;

    thrust::device_vector<Real> d_m;
    thrust::device_vector<Real> d_tau(num_ops * columns, 0);
    thrust::host_vector<Real> gpu_results;

    const int RUNS = 10;
    Real cpu_time[RUNS], kblas_time[RUNS], magma_time[RUNS], cublas_time[RUNS];
    Real hh_ops = (Real)(2.0 * rows * columns * columns - (2.0 / 3.0) * columns * columns * columns) * num_ops * 1e-9;

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	GPUBlasHandle handle(1, stream);

	Real kblas_err = 0, magma_err = 0, cublas_err = 0;

	for(int i = 0; i < RUNS; i++)
    {
        m = m_original;
        cpu_time[i] = batch_qr_cpu(vec_ptr(m), vec_ptr(tau), rows, columns, num_ops);

        d_m = m_original;
        handle.tic();
		kblas_geqrf_batched(rows, columns, vec_ptr(d_m), rows, rows * columns, vec_ptr(d_tau), columns, num_ops, handle);
		//kblas_tsqrf_batched(rows, columns, vec_ptr(d_m), rows, rows * columns, vec_ptr(d_tau), columns, num_ops, handle);
        kblas_time[i] = handle.toc();
        gpu_results = d_m; kblas_err += compare_results_R(vec_ptr(gpu_results), vec_ptr(m), rows, columns, num_ops);

        d_m = m_original;
        cublas_time[i] = batch_qr_cublas(vec_ptr(d_m), vec_ptr(d_tau), rows, columns, num_ops);
        gpu_results = d_m; cublas_err += compare_results_R(vec_ptr(gpu_results), vec_ptr(m), rows, columns, num_ops);

		d_m = m_original;
        magma_time[i] = batch_qr_magma(vec_ptr(d_m), vec_ptr(d_tau), rows, columns, num_ops, queue);
        gpu_results = d_m; magma_err += compare_results_R(vec_ptr(gpu_results), vec_ptr(m), rows, columns, num_ops);
	}

    Real avg_cpu_time, sdev_cpu_time;
    Real avg_kblas_time, sdev_kblas_time;
    Real avg_magma_time, sdev_magma_time;
    Real avg_cublas_time, sdev_cublas_time;
    avg_and_stdev(cpu_time, RUNS, avg_cpu_time, sdev_cpu_time);
    avg_and_stdev(kblas_time, RUNS, avg_kblas_time, sdev_kblas_time);
    avg_and_stdev(magma_time, RUNS, avg_magma_time, sdev_magma_time);
    avg_and_stdev(cublas_time, RUNS, avg_cublas_time, sdev_cublas_time);

    printf("MKL:\t completed in %f s at %f GFLOPS\n", avg_cpu_time, hh_ops / avg_cpu_time);
    printf("KBLAS:\t completed in %f s at %f GFLOPS  (%e error)\n", avg_kblas_time, hh_ops / avg_kblas_time, kblas_err / RUNS);
    printf("CUBLAS:\t completed in %f s at %f GFLOPS  (%e error)\n", avg_cublas_time, hh_ops / avg_cublas_time, cublas_err / RUNS);
    printf("MAGMA:\t completed in %f s at %f GFLOPS  (%e error)\n", avg_magma_time, hh_ops / avg_magma_time, magma_err / RUNS);

    return 0;
}