/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/testing_helper.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Wajih Halim Boukaram
 * @author Ali Charara
 * @date 2018-11-14
 **/

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/random.h>

#include <sys/time.h>
#include <stdarg.h>

#include "testing_helper.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Generating array of pointers from a strided array
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
struct UnaryAoAAssign : public thrust::unary_function<int, T*>
{
  T* original_array;
  int stride;
  UnaryAoAAssign(T* original_array, int stride) { this->original_array = original_array; this->stride = stride; }
  __host__ __device__
  T* operator()(const unsigned int& thread_id) const { return original_array + thread_id * stride; }
};

template<class T>
void generateArrayOfPointersT(T* original_array, T** array_of_arrays, int stride, int num_arrays, cudaStream_t stream)
{
  thrust::device_ptr<T*> dev_data(array_of_arrays);

  thrust::transform(
    thrust::cuda::par.on(stream),
    thrust::counting_iterator<int>(0),
    thrust::counting_iterator<int>(num_arrays),
    dev_data,
    UnaryAoAAssign<T>(original_array, stride)
    );

  check_error( cudaGetLastError() );
}

extern "C" void generateDArrayOfPointers(double* original_array, double** array_of_arrays, int stride, int num_arrays, cudaStream_t stream)
{ generateArrayOfPointersT<double>(original_array, array_of_arrays, stride, num_arrays, stream); }

extern "C" void generateSArrayOfPointers(float* original_array, float** array_of_arrays, int stride, int num_arrays, cudaStream_t stream)
{ generateArrayOfPointersT<float>(original_array, array_of_arrays, stride, num_arrays, stream); }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Timer helpers
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct GPU_Timer
{
	cudaEvent_t start_event, stop_event;
	float elapsed_time;
	cudaStream_t stream;

  GPU_Timer(cudaStream_t stream = 0)
  {
    init(stream);
  }

  void init(cudaStream_t stream = 0)
  {
    //TODO: this critical region is dead locking
    // #pragma omp critical (create_timer)
    {
      check_error( cudaEventCreate(&start_event) );
      check_error( cudaEventCreate(&stop_event ) );
      elapsed_time = 0;
      this->stream = stream;
    }
  }

  void destroy()
  {
    //TODO: this critical region is dead locking
    // #pragma omp critical (delete_timer)
    {
      check_error( cudaEventDestroy(start_event));
      check_error( cudaEventDestroy(stop_event ));
    }
  }

  void start()
  {
    check_error( cudaEventRecord(start_event, stream) );
  }

  void recordEnd()
  {
    check_error( cudaEventRecord(stop_event, stream) );
  }

  float stop()
  {
    check_error( cudaEventSynchronize(stop_event) );

    float time_since_last_start;
    check_error( cudaEventElapsedTime(&time_since_last_start, start_event, stop_event) );
    elapsed_time = (time_since_last_start * 0.001);

    return elapsed_time;
  }

  float elapsedSec()
  {
    return elapsed_time;
  }
};

extern "C" double gettime(void)
{
	struct timeval tp;
	gettimeofday( &tp, NULL );
	return tp.tv_sec + 1e-6 * tp.tv_usec;
}

extern "C" GPU_Timer* newGPU_Timer(cudaStream_t stream)
{
	GPU_Timer* timer = new GPU_Timer(stream);
	// timer->init(stream);
	return timer;
}

extern "C" void deleteGPU_Timer(GPU_Timer* timer)
{
	timer->destroy();
	delete timer;
}

extern "C" void gpuTimerTic(GPU_Timer* timer)
{
	timer->start();
}

extern "C" void gpuTimerRecordEnd(GPU_Timer* timer)
{
	timer->recordEnd();
}

extern "C" double gpuTimerToc(GPU_Timer* timer)
{
	return timer->stop();
}

extern "C" void avg_and_stdev(double* values, int num_vals, double* avg, double* std_dev, int warmup)
{
	if(num_vals == 0) return;

	int start = 0;
	if(warmup == 1 && num_vals != 1)
		start = 1;

	*avg = 0;
	for(int i = start; i < num_vals; i++)
		*avg += values[i];
	*avg /= num_vals;

	*std_dev = 0;
	for(int i = 0; i < num_vals; i++)
		*std_dev += (values[i] - *avg) * (values[i] - *avg);
	*std_dev = sqrt(*std_dev / num_vals);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Error helpers
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void gpuAssert(cudaError_t code, const char *file, int line)
{
  if(code != cudaSuccess)
  {
    printf("gpuAssert: %s(%d) %s %d\n", cudaGetErrorString(code), (int)code, file, line);
    exit(-1);
  }
}

const char* cublasGetErrorString(cublasStatus_t error)
{
  switch(error)
  {
    case CUBLAS_STATUS_SUCCESS:
    return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED:
    return "not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:
    return "out of memory";
    case CUBLAS_STATUS_INVALID_VALUE:
    return "invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH:
    return "architecture mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR:
    return "memory mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED:
    return "execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR:
    return "internal error";
    default:
    return "unknown error code";
  }
}

const char *cusolverGetErrorString(cusolverStatus_t error)
{
  switch (error)
  {
    case CUSOLVER_STATUS_SUCCESS:
    return "CUSOLVER_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
    return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
    return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
    return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
    return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
    return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
    return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    default:
    return "unknown error code";
  }
}

extern "C" void gpuCusolverAssert(cusolverStatus_t code, const char *file, int line)
{
	if(code != CUSOLVER_STATUS_SUCCESS)
	{
    printf("gpuCusolverAssert: %s %s %d\n", cusolverGetErrorString(code), file, line);
    exit(-1);
  }
}

extern "C" void gpuCublasAssert(cublasStatus_t code, const char *file, int line)
{
	if(code != CUBLAS_STATUS_SUCCESS)
	{
    printf("gpuCublasAssert: %s %s %d\n", cublasGetErrorString(code), file, line);
    exit(-1);
  }
}

extern "C" void gpuKblasAssert(int code, const char *file, int line)
{
	if(code != 1)  // TODO replace by KBlas_Success
	{
    printf("gpuKblasAssert: %s %s %d\n", kblasGetErrorString(code), file, line);
    exit(-1);
  }
}

////////////////////////////////////////////////////////////
// Data generation
////////////////////////////////////////////////////////////
template<class Real>
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

template<class T>
void generate_random(T* random_data, long num_elements, int num_ops)
{
  thrust::counting_iterator<long> index_sequence_begin(0);

  thrust::transform(thrust::system::omp::par,
    index_sequence_begin,
    index_sequence_begin + num_elements * num_ops,
    random_data,
    prg<T>());
}

extern "C" void generateDrandom(double* random_data, long num_elements, int num_ops)
{
	generate_random<double>(random_data, num_elements, num_ops);
}

extern "C" void generateSrandom(float* random_data, long num_elements, int num_ops)
{
  generate_random<float>(random_data, num_elements, num_ops);
}

extern "C" void srand_matrix(long rows, long cols, float* A, long LDA)
{
	//generate_random<float>(A, cols * LDA, 1);
	long i;
  long size_a = cols * LDA;
  for(i = 0; i < size_a; i++) A[i] = ( (float)rand() ) / (float)RAND_MAX;
}

extern "C" void drand_matrix(long rows, long cols, double* A, long LDA)
{
    //generate_random<double>(A, cols * LDA, 1);
	long i;
  long size_a = cols * LDA;
  for(i = 0; i < size_a; i++) A[i] = ( (double)rand() ) / (double)RAND_MAX;
}

extern "C" void crand_matrix(long rows, long cols, cuFloatComplex* A, long LDA)
{
    // fill in the entire matrix with random values
  long i;
  long size_a = cols * LDA;
  for(i = 0; i < size_a; i++)
  {
   A[i].x = ( (float)rand() ) / (float)RAND_MAX;
   A[i].y = ( (float)rand() ) / (float)RAND_MAX;
 }
}

extern "C" void zrand_matrix(long rows, long cols, cuDoubleComplex* A, long LDA)
{
	// fill in the entire matrix with random values
	long i;
	long size_a = cols * LDA;
  for(i = 0; i < size_a; i++)
  {
   A[i].x = ( (double)rand() ) / (double)RAND_MAX;
   A[i].y = ( (double)rand() ) / (double)RAND_MAX;
 }
}

template<class T>
void matrix_make_hpd(int N, T* A, int lda)
{
	for(int i = 0; i < N; i++)
	{
		A[i + i * lda] += N;
		for(int j = 0; j < i; j++)
			A[j + i*lda] = A[i + j*lda];
	}
}

template<class T>
void matrix_make_hpd_complex(int N, T* A, int lda)
{
	for(int i = 0; i < N; i++)
	{
		A[i + i *  lda].x += N;
		A[i + i *  lda].y += N;
		for(int j = 0; j < i; j++)
			A[j + i*lda] = A[i + j*lda];
	}
}

extern "C" void smatrix_make_hpd(int N, float* A, int lda)
{ matrix_make_hpd<float>(N, A, lda); }

extern "C" void dmatrix_make_hpd(int N, double* A, int lda)
{ matrix_make_hpd<double>(N, A, lda); }

extern "C" void cmatrix_make_hpd(int N, cuFloatComplex* A, int lda)
{ matrix_make_hpd_complex<cuFloatComplex>(N, A, lda); }

extern "C" void zmatrix_make_hpd(int N, cuDoubleComplex* A, int lda)
{ matrix_make_hpd_complex<cuDoubleComplex>(N, A, lda); }

lapack_int LAPACKE_latms(int matrix_layout, lapack_int m, lapack_int n, char dist, lapack_int *iseed, char sym, float *d, lapack_int mode, float cond, float dmax, lapack_int kl, lapack_int ku, char pack, float *a, lapack_int lda)
{ return LAPACKE_slatms(matrix_layout, m, n, dist, iseed, sym, d, mode, cond, dmax, kl, ku, pack, a, lda); }

lapack_int LAPACKE_latms(int matrix_layout, lapack_int m, lapack_int n, char dist, lapack_int *iseed, char sym, double *d, lapack_int mode, double cond, double dmax, lapack_int kl, lapack_int ku, char pack, double *a, lapack_int lda)
{ return LAPACKE_dlatms(matrix_layout, m, n, dist, iseed, sym, d, mode, cond, dmax, kl, ku, pack, a, lda); }

template<class Real>
void generate_randomMatrices(
	Real* M_strided, int stride_M, Real* svals_strided, int stride_S, int rows, int cols,
	Real cond, Real exp_decay, int seed, int num_ops, int threads
  )
{
	// Generate a bunch of random seeds from the master seed
	thrust::host_vector<int> random_seeds(num_ops);
	thrust::default_random_engine seed_rng; seed_rng.seed(seed);
	thrust::uniform_int_distribution <int> seed_dist(0, 10000);
	for(int i = 0; i < num_ops; i++)
		random_seeds[i] = seed_dist(seed_rng);

	#pragma omp parallel for num_threads(threads)
	for(int op_index = 0; op_index < num_ops; op_index++)
	{
		thrust::default_random_engine rng; rng.seed(random_seeds[op_index]);
		thrust::uniform_int_distribution<int> dist(0, 4095);
		int seeds[4] = {dist(rng), dist(rng), dist(rng), dist(rng)};
		if(seeds[3] % 2 != 1) seeds[3]++;

		int info;
		Real* matrix = M_strided + stride_M * op_index;
		Real* svals  = svals_strided + stride_S * op_index;

		// Sanitize input
		for(int j = 0; j < cols; j++)
      svals[j] = 0;

    for(int i = 0; i < rows; i++)
     for(int j = 0; j < cols; j++)
      matrix[i + j * rows] = 0;

    int mode = 5;
    if(cond == 0)
    {
     mode = 0;
     for(int i = 0; i < cols; i++)
      svals[i] = exp(-exp_decay*i);
  }
  info = LAPACKE_latms(LAPACK_COL_MAJOR, rows, cols, 'N', &seeds[0], 'N', svals, mode, cond, 1, rows, cols, 'N', matrix, rows);
  if(info != 0) printf("Error generating random matrix: %d\n", info);
  thrust::sort(svals, svals + cols, thrust::greater<Real>());
}
}

extern "C" void generateDrandomMatrices(
	double* M_strided, int stride_M, double* svals_strided, int stride_S, int rows, int cols,
	double cond, double exp_decay, int seed, int num_ops, int threads
  )
{
	generate_randomMatrices<double>(
		M_strided, stride_M, svals_strided, stride_S, rows, cols,
		cond, exp_decay, seed, num_ops, threads
   );
}

extern "C" void generateSrandomMatrices(
	float* M_strided, int stride_M, float* svals_strided, int stride_S, int rows, int cols,
	float cond, float exp_decay, int seed, int num_ops, int threads
  )
{
	generate_randomMatrices<float>(
		M_strided, stride_M, svals_strided, stride_S, rows, cols,
		cond, exp_decay, seed, num_ops, threads
   );
}


////////////////////////////////////////////////////////////
// Result checking
////////////////////////////////////////////////////////////

// Vectors
// template<class T, class T_complex>
// T get_magnitude(T_complex a){ return sqrt(a.x * a.x + a.y * a.y); }
// template<class T>
// T get_magnitude(T ax, T ay){ return sqrt(ax * ax + ay * ay); }

template<class T>
T get_max_error(T* ref, T *res, int n, int inc)
{
	int i;
	T max_err = -1.0;
	T err = -1.0;
	inc = abs(inc);
	for(i = 0; i < n; i++)
	{
		err = fabs(res[i * inc] - ref[i * inc]);
		if(ref[i * inc] != 0.0)err /= fabs(ref[i * inc]);
		if(err > max_err)max_err = err;
		//printf("[%2d]   %-.2f   %-.2f   %-.2e \n", i, ref[i], res[i], err);
	}
	return max_err;
}

template<class T, class T_complex>
T get_max_error_complex(T_complex* ref, T_complex *res, int n, int inc)
{
	int i;
	T max_err = get_magnitude<T>(0, 0);
	T err;
	inc = abs(inc);
	for(i = 0; i < n; i++)
	{
		err = get_magnitude<T>(res[i * inc].x - ref[i * inc].x, res[i * inc].y - ref[i * inc].y);
		if(get_magnitude<T, T_complex>(ref[i * inc]) > 0)
			err /= get_magnitude<T, T_complex>(ref[i * inc]);
		if(err > max_err) max_err = err;
	}
	return max_err;
}

extern "C" float sget_max_error(float* ref, float *res, int n, int inc)
{ return get_max_error<float>(ref, res, n, inc); }

extern "C" double dget_max_error(double* ref, double *res, int n, int inc)
{ return get_max_error<double>(ref, res, n, inc); }

extern "C" float cget_max_error(cuFloatComplex* ref, cuFloatComplex *res, int n, int inc)
{ return get_max_error_complex<float, cuFloatComplex>(ref, res, n, inc); }

double zget_max_error(cuDoubleComplex* ref, cuDoubleComplex *res, int n, int inc)
{ return get_max_error_complex<double, cuDoubleComplex>(ref, res, n, inc); }

// Matrices
template<class T>
T get_max_error_matrix(T* ref, T *res, long m, long n, long lda)
{
	long i, j;
	T max_err = -1.0;
	T err = -1.0;
	for(i = 0; i < m; i++)
	{
		for(j = 0; j < n; j++)
		{
			T ref_ = ref[j * lda + i];
			T res_ = res[j * lda + i];
			err = fabs(res_ - ref_);
			if(ref_ != 0.0) err /= fabs(ref_);
			if(err > max_err) max_err = err;
			//printf("\n[%2d]   %-.2f   %-.2f   %-.2e \n", i, ref_, res_, err);
		}
	}
	return max_err;
}

template<class T, class T_complex>
T get_max_error_matrix_complex(T_complex* ref, T_complex *res, long m, long n, long lda)
{
	long i, j;
	T max_err = -1.0;
	T err = -1.0;
	for(i = 0; i < m; i++)
	{
		for(j = 0; j < n; j++)
		{
			err = get_magnitude<T>(res[j * lda + i].x - ref[j * lda + i].x, res[j * lda + i].y - ref[j * lda + i].y);
			if(get_magnitude<T, T_complex>(ref[j * lda + i]) > 0) err /= get_magnitude<T, T_complex>(ref[j * lda + i]);
			if(err > max_err) max_err = err;
			//printf("\n[%2d]   %-.2f   %-.2f   %-.2e \n", i, ref_, res_, err);
		}
	}
	return max_err;
}

template<class T>
T get_max_error_matrix_uplo(T* ref, T *res, long m, long n, long lda, char uplo)
{
  long i, j;
  T max_err = -1.0;
  T err = -1.0;
  for(i = 0; i < m; i++){
    for(j = (uplo == KBLAS_Upper ? i : 0); j < (uplo == KBLAS_Lower ? i+1 : n); j++)
    {
      T ref_ = ref[j * lda + i];
      T res_ = res[j * lda + i];
      err = fabs(res_ - ref_);
      if(ref_ != 0.0)err /= fabs(ref_);
      if(err > max_err)max_err = err;
      //printf("\n[%2d]   %-.2f   %-.2f   %-.2e \n", i, ref_, res_, err);
    }
  }
  return max_err;
}

extern "C" float sget_max_error_matrix(float* ref, float *res, long m, long n, long lda)
{ return get_max_error_matrix<float>(ref, res, m, n, lda); }

extern "C" double dget_max_error_matrix(double* ref, double *res, long m, long n, long lda)
{ return get_max_error_matrix<double>(ref, res, m, n, lda); }

extern "C" float cget_max_error_matrix(cuFloatComplex* ref, cuFloatComplex *res, long m, long n, long lda)
{ return get_max_error_matrix_complex<float, cuFloatComplex>(ref, res, m, n, lda); }

extern "C" double zget_max_error_matrix(cuDoubleComplex* ref, cuDoubleComplex *res, long m, long n, long lda)
{ return get_max_error_matrix_complex<double, cuDoubleComplex>(ref, res, m, n, lda); }

extern "C" double dget_max_error_matrix_uplo(double* ref, double *res, long m, long n, long lda, char uplo)
{ return get_max_error_matrix_uplo<double>(ref, res, m, n, lda, uplo); }

////////////////////////////////////////////////////////////
// Command line parser
////////////////////////////////////////////////////////////
static const char *usage =
"  --range start:stop:step\n"
"                   Adds test cases with range for sizes m,n. Can be repeated.\n"
"  -N m[:n]         Adds one test case with sizes m,n. Can be repeated.\n"
"                   If only m given then n=m.\n"
"  -m m             Sets m for all tests, overriding -N and --range.\n"
"  -n n             Sets n for all tests, overriding -N and --range.\n"
"\n"
"  -c  --[no]check  Whether to check results against CUBLAS, default is on.\n"
"  --dev x          GPU device to use, default 0.\n"
"\n"
"  --niter x        Number of iterations to repeat each test, default 1.\n"
"  -L -U            uplo   = Lower*, Upper.\n"
"  -[NTC][NTC]      transA = NoTrans*, Trans, or ConjTrans (first letter) and\n"
"                   transB = NoTrans*, Trans, or ConjTrans (second letter).\n"
"  -[TC]            transA = Trans or ConjTrans. Default is NoTrans. Doesn't change transB.\n"
"  -S[LR]           side   = Left*, Right.\n"
"  -D[NU]           diag   = NonUnit*, Unit.\n"
"                   * default values\n"
"\n"
"examples: \n"
"to test trmm with matrix A[512,512], B[2000,512] do\n"
"   test_dtrmm -N 2000:512 \n"
"to test trmm for range of sizes starting at 1024, stoping at 4096, steping at 1024, sizes will be for both A and B, with A upper traingular and transposed, do\n"
"   test_dtrmm --range 1024:4096:1024 -U -T";


#define USAGE printf("usage: -N m[:n] --range m-start:m-end:m-step -m INT -n INT -L|U -SL|SR -DN|DU -[NTC][NTC] -c --niter INT --dev devID\n\n"); \
printf("%s\n", usage);

#define USING printf("side %c, uplo %c, trans %c, diag %c, db %d\n", opts.side, opts.uplo, opts.transA, opts.diag, opts.db);

void kblas_assert( int condition, const char* msg, ... )
{
	if ( ! condition ) {
		printf( "Assert failed: " );
		va_list va;
		va_start( va, msg );
		vprintf( msg, va );
		printf( "\n" );
		exit(1);
	}
}

extern "C" int parse_opts(int argc, char** argv, kblas_opts *opts)
{
    // negative flag indicating -m, -n, -k not given
  int m = -1;
  int n = -1;
  int k = -1;
  int m_start = 0, m_stop = 0, m_step = 0;
  int n_start = 0, n_stop = 0, n_step = 0;
  int k_start = 0, k_stop = 0, k_step = 0;
  char m_op = '+', n_op = '+', k_op = '+';

    // fill in default values
  for(int d = 0; d < MAX_NGPUS; d++)
    opts->devices[d] = d;

  opts->nstream    = 1;
  opts->ngpu       = 1;
  opts->niter      = 1;
  opts->nruns      = 4;
  opts->nb         = 64;  // ??
  opts->db         = 512;  // ??
  opts->tolerance  = 0.;
  opts->check      = 0;
  opts->verbose    = 0;
  opts->custom     = 0;
  opts->warmup     = 0;
  opts->time       = 0;
  opts->lapack     = 0;
  opts->cuda       = 0;
  opts->nonUniform = 0;
  opts->magma      = 0;
  opts->svd        = 0;
  opts->batchCount = 4;
  opts->strided    = 0;
  opts->btest      = 1;
  opts->rtest      = 1;
  //opts->bd[0]      = -1;
  opts->omp_numthreads = 20;
  opts->LR = 't';

  opts->uplo      = 'L';      // potrf, etc.
  opts->transA    = 'N';    // gemm, etc.
  opts->transB    = 'N';    // gemm
  opts->side      = 'L';       // trsm, etc.
  opts->diag      = 'N';    // trsm, etc.
  opts->version    = 1;

  if(argc < 2){
    USAGE
    exit(0);
  }

  int ndevices;
  cudaGetDeviceCount( &ndevices );
  int info;
  int ntest = 0;
  for( int i = 1; i < argc; ++i ) {
    // ----- matrix size
    // each -N fills in next entry of msize, nsize, ksize and increments ntest
    if ( strcmp("-N", argv[i]) == 0 && i+1 < argc ) {
      kblas_assert( ntest < MAX_NTEST, "error: -N %s, max number of tests exceeded, ntest=%d.\n", argv[i], ntest );
      i++;
      int m2, n2, k2, q2;
      info = sscanf( argv[i], "%d:%d:%d:%d", &m2, &n2, &k2, &q2 );
      if ( info == 4 && m2 >= 0 && n2 >= 0 && k2 >= 0 && q2 >= 0 ) {
        opts->msize[ ntest ] = m2;
        opts->nsize[ ntest ] = n2;
        opts->ksize[ ntest ] = k2;
        opts->rsize[ ntest ] = q2;
      }
      else
      if ( info == 3 && m2 >= 0 && n2 >= 0 && k2 >= 0 ) {
        opts->msize[ ntest ] = m2;
        opts->nsize[ ntest ] = n2;
        opts->ksize[ ntest ] = k2;
          opts->rsize[ ntest ] = k2;  // implicitly
      }
      else
      if ( info == 2 && m2 >= 0 && n2 >= 0 ) {
        opts->msize[ ntest ] = m2;
        opts->nsize[ ntest ] = n2;
        opts->ksize[ ntest ] = n2;  // implicitly
        opts->rsize[ ntest ] = n2;  // implicitly
      }
      else
      if ( info == 1 && m2 >= 0 ) {
        opts->msize[ ntest ] = m2;
        opts->nsize[ ntest ] = m2;  // implicitly
        opts->ksize[ ntest ] = m2;  // implicitly
        opts->rsize[ ntest ] = m2;  // implicitly
      }
      else {
        fprintf( stderr, "error: -N %s is invalid; ensure m >= 0, n >= 0, k >= 0, info=%d, m2=%d, n2=%d, k2=%d, q2=%d.\n",
          argv[i],info,m2,n2,k2,q2 );
        exit(1);
      }
      ntest++;
    }
    // --range start:stop:step fills in msize[ntest:], nsize[ntest:], ksize[ntest:]
    // with given range and updates ntest
    else if ( strcmp("--range", argv[i]) == 0 && i+1 < argc ) {
      i++;
      int start, stop, step;
      char op;
      info = sscanf( argv[i], "%d:%d%c%d", &start, &stop, &op, &step );
      if ( info == 4 && start >= 0 && stop >= 0 && step != 0 && (op == '+' || op == '*' || op == ':')) {
        for( int n = start; (step > 0 ? n <= stop : n >= stop); ) {
          if ( ntest >= MAX_NTEST ) {
            printf( "warning: --range %s, max number of tests reached, ntest=%d.\n",
             argv[i], ntest );
            break;
          }
          opts->msize[ ntest ] = n;
          opts->nsize[ ntest ] = n;
          opts->ksize[ ntest ] = n;
          opts->rsize[ ntest ] = n;
          ntest++;
          if(op == '*') n *= step; else n += step;
        }
      }
      else {
        fprintf( stderr, "error: --range %s is invalid; ensure start >= 0, stop >= 0, step != 0 && op in (+,*,:).\n",
          argv[i] );
        exit(1);
      }
    }
    else if ( strcmp("--nrange", argv[i]) == 0 && i+1 < argc ) {
      i++;
      int start, stop, step;
      char op;
      info = sscanf( argv[i], "%d:%d%c%d", &start, &stop, &op, &step );
      if ( info == 4 && start >= 0 && stop >= 0 && step != 0 && (op == '+' || op == '*' || op == ':')) {
        n_start = start;
        n_stop = stop;
        n_step = step;
        n_op = (op == ':'? '+' : op);
      }
      else {
        fprintf( stderr, "error: --nrange %s is invalid; ensure start >= 0, stop >= 0, step != 0 && op in (+,*,:).\n",
          argv[i] );
        exit(1);
      }
    }
    else if ( strcmp("--mrange", argv[i]) == 0 && i+1 < argc ) {
      i++;
      int start, stop, step;
      char op;
      info = sscanf( argv[i], "%d:%d%c%d", &start, &stop, &op, &step );
      if ( info == 4 && start >= 0 && stop >= 0 && step != 0 && (op == '+' || op == '*' || op == ':')) {
        m_start = start;
        m_stop = stop;
        m_step = step;
        m_op = (op == ':'? '+' : op);
      }
      else {
        fprintf( stderr, "error: --mrange %s is invalid; ensure start >= 0, stop >= 0, step != 0 && op in (+,*,:).\n",
          argv[i] );
        exit(1);
      }
    }
    else if ( strcmp("--krange", argv[i]) == 0 && i+1 < argc ) {
      i++;
      int start, stop, step;
      char op;
      info = sscanf( argv[i], "%d:%d%c%d", &start, &stop, &op, &step );
      if ( info == 4 && start >= 0 && stop >= 0 && step != 0 && (op == '+' || op == '*' || op == ':')) {
        k_start = start;
        k_stop = stop;
        k_step = step;
        k_op = (op == ':'? '+' : op);
      }
      else {
        fprintf( stderr, "error: --krange %s is invalid; ensure start >= 0, stop >= 0, step != 0 && op in (+,*,:).\n",
          argv[i] );
        exit(1);
      }
    }
    // save m, n, k if -m, -n, -k is given; applied after loop
    else if ( strcmp("-m", argv[i]) == 0 && i+1 < argc ) {
      m = atoi( argv[++i] );
      kblas_assert( m >= 0, "error: -m %s is invalid; ensure m >= 0.\n", argv[i] );
    }
    else if ( strcmp("-n", argv[i]) == 0 && i+1 < argc ) {
      n = atoi( argv[++i] );
      kblas_assert( n >= 0, "error: -n %s is invalid; ensure n >= 0.\n", argv[i] );
    }
    else if ( strcmp("-k", argv[i]) == 0 && i+1 < argc ) {
      k = atoi( argv[++i] );
      kblas_assert( k >= 0, "error: -k %s is invalid; ensure k >= 0.\n", argv[i] );
    }

    // ----- scalar arguments
    else if ( strcmp("--dev", argv[i]) == 0 && i+1 < argc ) {
      int n;
      info = sscanf( argv[++i], "%d", &n );
      if ( info == 1) {
        char inp[512];
        char * pch;
        int ngpus = 0;
        strcpy(inp, argv[i]);
        pch = strtok (inp,",");
        do{
          info = sscanf( pch, "%d", &n );
          if ( ngpus >= MAX_NGPUS ) {
            printf( "warning: selected number exceeds KBLAS max number of GPUs, ngpus=%d.\n", ngpus);
            break;
          }
          if ( ngpus >= ndevices ) {
            printf( "warning: max number of available devices reached, ngpus=%d.\n", ngpus);
            break;
          }
          if ( n >= ndevices || n < 0) {
            printf( "error: device %d is invalid; ensure dev in [0,%d].\n", n, ndevices-1 );
            break;
          }
          opts->devices[ ngpus++ ] = n;
          pch = strtok (NULL,",");
        }while(pch != NULL);
        opts->ngpu = ngpus;
      }
      else {
        fprintf( stderr, "error: --dev %s is invalid; ensure you have comma seperated list of integers.\n",
         argv[i] );
        exit(1);
      }
      kblas_assert( opts->ngpu > 0 && opts->ngpu <= ndevices,
       "error: --dev %s is invalid; ensure dev in [0,%d].\n", argv[i], ndevices-1 );
    }
    else if ( strcmp("--ngpu",    argv[i]) == 0 && i+1 < argc ) {
      opts->ngpu = atoi( argv[++i] );
      kblas_assert( opts->ngpu <= MAX_NGPUS ,
        "error: --ngpu %s exceeds MAX_NGPUS, %d.\n", argv[i], MAX_NGPUS  );
      kblas_assert( opts->ngpu <= ndevices,
       "error: --ngpu %s exceeds number of CUDA devices, %d.\n", argv[i], ndevices );
      kblas_assert( opts->ngpu > 0,
       "error: --ngpu %s is invalid; ensure ngpu > 0.\n", argv[i] );
    }
    else if ( strcmp("--nstream", argv[i]) == 0 && i+1 < argc ) {
      opts->nstream = atoi( argv[++i] );
      kblas_assert( opts->nstream > 0,
       "error: --nstream %s is invalid; ensure nstream > 0.\n", argv[i] );
    }
    else if ( strcmp("--niter",   argv[i]) == 0 && i+1 < argc ) {
      opts->niter = atoi( argv[++i] );
      kblas_assert( opts->niter > 0,
       "error: --niter %s is invalid; ensure niter > 0.\n", argv[i] );
    }
    else if ( strcmp("--nruns",   argv[i]) == 0 && i+1 < argc ) {
      opts->nruns = atoi( argv[++i] );
      kblas_assert( opts->nruns > 0,
        "error: --nruns %s is invalid; ensure nruns > 0.\n", argv[i] );
    }
    else if ( strcmp("--ver",   argv[i]) == 0 && i+1 < argc ) {
      opts->version = atoi( argv[++i] );
      // kblas_assert( opts->nruns > 0,
      //   "error: --nruns %s is invalid; ensure nruns > 0.\n", argv[i] );
    }
    else if ( (strcmp("--batchCount",   argv[i]) == 0 || strcmp("--batch",   argv[i]) == 0) && i+1 < argc ) {
      i++;
      int start, stop, step;
      char op;
      info = sscanf( argv[i], "%d:%d%c%d", &start, &stop, &op, &step );
      if( info == 1 ){
        opts->batch[0] = opts->batchCount = start;
      }else
      if ( info == 4 && start >= 0 && stop >= 0 && step != 0 && (op == '+' || op == '*' || op == ':')) {
        opts->btest = 0;
        for( int b = start; (step > 0 ? b <= stop : b >= stop); ) {
          opts->batch[ opts->btest++ ] = b;
          if(op == '*') b *= step; else b += step;
        }
      }
      else {
        fprintf( stderr, "error: --range %s is invalid; ensure start >= 0, stop >= 0, step != 0 && op in (+,*,:).\n",
          argv[i] );
        exit(1);
      }
      //opts->batchCount = atoi( argv[++i] );
      //kblas_assert( opts->batchCount > 0, "error: --batchCount %s is invalid; ensure batchCount > 0.\n", argv[i] );
    }
    else if ( (strcmp("--rank",   argv[i]) == 0) && i+1 < argc ) {
      i++;
      int start, stop, step;
      char op, sep;
      info = sscanf( argv[i], "%d%c%d%c%d", &start, &sep, &stop, &op, &step );
      if( info == 1 ){
        opts->rank[0] = opts->rank[1] = start;
      }else
      if( info == 3 ){
        opts->rank[0] = start;
        opts->rank[1] = stop;
      }else
      if ( info == 5 && start >= 0 && stop >= 0 && step != 0 && (op == '+' || op == '*' || op == ':')) {
        opts->rtest = 0;
        for( int b = start; (step > 0 ? b <= stop : b >= stop); ) {
          opts->rank[ opts->rtest++ ] = b;
          if(op == '*') b *= step; else b += step;
        }
      }
      else {
        fprintf( stderr, "error: --range %s is invalid; ensure start >= 0, stop >= 0, step != 0 && op in (+,*,:).\n",
          argv[i] );
        exit(1);
      }
      //opts->batchCount = atoi( argv[++i] );
      //kblas_assert( opts->batchCount > 0, "error: --batchCount %s is invalid; ensure batchCount > 0.\n", argv[i] );
    }
    else if ( strcmp("--cuda",  argv[i]) == 0  ) { opts->cuda = 1;  }
    else if ( strcmp("--magma",  argv[i]) == 0  ) { opts->magma = 1;  }
    else if ( strcmp("--strided",  argv[i]) == 0 || strcmp("-s",  argv[i]) == 0 ) { opts->strided = 1;  }
    else if ( (strcmp("--tolerance", argv[i]) == 0 || strcmp("--tol", argv[i]) == 0) && i+1 < argc ) {
      int tol = atoi( argv[++i] );
      if (tol != 0)
        opts->tolerance = pow(10, tol);
      else
        opts->tolerance = 0;
    }
    else if ( strcmp("--svd",      argv[i]) == 0 && i+1 < argc ) {
      opts->svd = atoi( argv[++i] );
      kblas_assert( opts->svd >= SVD_Jacobi && opts->svd <= SVD_aca,
       "error: --svd %s is invalid; ensure 0 <= svd <= 2.\n", argv[i] );
    }
    else if ( strcmp("--nb",      argv[i]) == 0 && i+1 < argc ) {
      opts->nb = atoi( argv[++i] );
      kblas_assert( opts->nb > 0,
       "error: --nb %s is invalid; ensure nb > 0.\n", argv[i] );
    }
    else if ( strcmp("--db",      argv[i]) == 0 && i+1 < argc ) {
      opts->db = atoi( argv[++i] );
      kblas_assert( opts->db > 0,
       "error: --db %s is invalid; ensure db > 0.\n", argv[i] );
    }
    else if ( (strcmp("--LR",      argv[i]) == 0  || strcmp("--lr",      argv[i]) == 0 )&& i+1 < argc ) {
      opts->LR = char(argv[++i][0]);
      kblas_assert( opts->LR == 't' || opts->LR == 'b',
       "error: --LR %s is invalid; ensure LT = t|l.\n", argv[i] );
    }
    #ifdef KBLAS_ENABLE_BACKDOORS
    else if ( strcmp("--bd",      argv[i]) == 0 && i+1 < argc ) {
      //opts->bd = atoi( argv[++i] );
      //kblas_assert( opts->bd[0] >= 0,"error: --bd %s is invalid; ensure bd's >= 0.\n", argv[i] );
      int n;
      info = sscanf( argv[++i], "%d", &n );
      if ( info == 1) {
        char inp[512];
        char * pch;
        int nbd = 0;
        strcpy(inp, argv[i]);
        pch = strtok (inp,",");
        do{
          info = sscanf( pch, "%d", &n );
          if ( nbd >= 16 ) {
            printf( "warning: selected number exceeds KBLAS max number of back doors\n");
            break;
          }
          kblas_back_door[ nbd++ ] = n;
          pch = strtok (NULL,",");
        }while(pch != NULL);
      }
      else {
        fprintf( stderr, "error: --bd %s is invalid; ensure you have comma seperated list of integers.\n",
         argv[i] );
        exit(1);
      }
    }
    #endif
    else if ( strcmp("--omp_threads", argv[i]) == 0 && i+1 < argc ) {
      opts->omp_numthreads = atoi( argv[++i] );
      kblas_assert( opts->omp_numthreads >= 1,
        "error: --omp_numthreads %s is invalid; ensure omp_numthreads >= 1.\n", argv[i] );
    }
    // ----- boolean arguments
    // check results
    else if ( strcmp("--var",      argv[i]) == 0 ) { opts->nonUniform  = 1; }
    else if ( strcmp("-c",         argv[i]) == 0 ) { opts->check  = 1; }
    else if ( strcmp("-t",         argv[i]) == 0 ) { opts->time  = 1; }
    else if ( strcmp("-l",         argv[i]) == 0 ) { opts->lapack  = 1; }
    else if ( strcmp("-v",  argv[i]) == 0 ) { opts->verbose= 1;  }
    else if ( strcmp("-cu",         argv[i]) == 0 ) { opts->custom  = 1; }
    else if ( strcmp("-w",  argv[i]) == 0 ) { opts->warmup = 1;  }

    // ----- lapack flag arguments
    else if ( strcmp("-L",  argv[i]) == 0 ) { opts->uplo = KBLAS_Lower; }
    else if ( strcmp("-U",  argv[i]) == 0 ) { opts->uplo = KBLAS_Upper; }
    else if ( strcmp("-NN", argv[i]) == 0 ) { opts->transA = KBLAS_NoTrans;   opts->transB = KBLAS_NoTrans;   }
    else if ( strcmp("-NT", argv[i]) == 0 ) { opts->transA = KBLAS_NoTrans;   opts->transB = KBLAS_Trans;     }
    else if ( strcmp("-TN", argv[i]) == 0 ) { opts->transA = KBLAS_Trans;     opts->transB = KBLAS_NoTrans;   }
    else if ( strcmp("-TT", argv[i]) == 0 ) { opts->transA = KBLAS_Trans;     opts->transB = KBLAS_Trans;     }
    else if ( strcmp("-T",  argv[i]) == 0 ) { opts->transA = KBLAS_Trans;     }

    else if ( strcmp("-SL", argv[i]) == 0 ) { opts->side  = KBLAS_Left;  }
    else if ( strcmp("-SR", argv[i]) == 0 ) { opts->side  = KBLAS_Right; }

    else if ( strcmp("-DN", argv[i]) == 0 ) { opts->diag  = KBLAS_NonUnit; }
    else if ( strcmp("-DU", argv[i]) == 0 ) { opts->diag  = KBLAS_Unit;    }

    // ----- usage
    else if ( strcmp("-h",     argv[i]) == 0 || strcmp("--help", argv[i]) == 0 ) {
      USAGE
      exit(0);
    }
    else {
      fprintf( stderr, "error: unrecognized option %s\n", argv[i] );
      exit(1);
    }
  }

  // fill in msize[:], nsize[:], ksize[:] if -m, -n, -k were given
  if(m_step != 0 && n_step != 0 && k_step != 0){
    for( int m = m_start; (m_step > 0 ? m <= m_stop : m >= m_stop); ) {
      for( int n = n_start; (n_step > 0 ? n <= n_stop : n >= n_stop); ) {
        for( int k = k_start; (k_step > 0 ? k <= k_stop : k >= k_stop); ) {
          if ( ntest >= MAX_NTEST ) {
            printf( "warning: --m/n_range, max number of tests reached, ntest=%d.\n",
              ntest );
            break;
          }
          opts->msize[ ntest ] = m;
          opts->nsize[ ntest ] = n;
          opts->ksize[ ntest ] = k;
          if(k_op == '*') k *= k_step;else k += k_step ;
          ntest++;
        }
        if(n_op == '*') n *= n_step;else n += n_step ;
      }
      if(m_op == '*') m *= m_step;else m += m_step ;
    }
  }else
  if(m_step != 0 && n_step != 0){
    for( int m = m_start; (m_step > 0 ? m <= m_stop : m >= m_stop); ) {
      for( int n = n_start; (n_step > 0 ? n <= n_stop : n >= n_stop); ) {
        if ( ntest >= MAX_NTEST ) {
          printf( "warning: --m/n_range, max number of tests reached, ntest=%d.\n",
            ntest );
          break;
        }
        opts->msize[ ntest ] = m;
        opts->nsize[ ntest ] = n;
        if(n_op == '*') n *= n_step;else n += n_step ;
        if(k >= 0)
          opts->ksize[ntest] = k;
        else
          opts->ksize[ ntest ] = n;
        ntest++;
      }
      if(m_op == '*') m *= m_step;else m += m_step ;
    }
  }else
  if(k_step != 0 && m >= 0 && n >= 0){
    for( int k = k_start; (k_step > 0 ? k <= k_stop : k >= k_stop); ) {
      if ( ntest >= MAX_NTEST ) {
        printf( "warning: --m/n_range, max number of tests reached, ntest=%d.\n",
          ntest );
        break;
      }
      opts->msize[ ntest ] = m;
      opts->nsize[ ntest ] = n;
      opts->ksize[ ntest ] = k;
      if(k_op == '*') k *= k_step; else k += k_step;
      ntest++;
    }
  }else
  if(n_step != 0 && m >= 0){
    for( int n = n_start; (n_step > 0 ? n <= n_stop : n >= n_stop); ) {
      if ( ntest >= MAX_NTEST ) {
        printf( "warning: --m/n_range, max number of tests reached, ntest=%d.\n",
          ntest );
        break;
      }
      opts->msize[ ntest ] = m;
      opts->nsize[ ntest ] = n;
      if(n_op == '*') n *= n_step;else n += n_step ;
      if(k >= 0)
        opts->ksize[ntest] = k;
      ntest++;
    }
  }else
  if(m_step != 0 && n >= 0){
    for( int m = m_start; (m_step > 0 ? m <= m_stop : m >= m_stop); ) {
      if ( ntest >= MAX_NTEST ) {
        printf( "warning: --m/n_range, max number of tests reached, ntest=%d.\n",
          ntest );
        break;
      }
      opts->msize[ ntest ] = m;
      if(m_op == '*') m *= m_step;else m += m_step ;
      opts->nsize[ ntest ] = n;
      if(k >= 0)
        opts->ksize[ntest] = k;
      ntest++;
    }
  }else{
    if ( m >= 0 ) {
      for( int j = 0; j < MAX_NTEST; ++j ) {
        opts->msize[j] = m;
      }
    }
    if ( n >= 0 ) {
      for( int j = 0; j < MAX_NTEST; ++j ) {
        opts->nsize[j] = n;
      }
    }
    if ( k >= 0 ) {
      for( int j = 0; j < MAX_NTEST; ++j ) {
        opts->ksize[j] = k;
      }
    }
    if ( m > 0 && n > 0) {
      ntest = 1;
    }
  }
  // if size not specified
  if ( ntest == 0 ) {
    fprintf(stderr, "please specify matrix size\n\n");
    //USAGE
    exit(0);
  }
  kblas_assert( ntest <= MAX_NTEST, "error: tests exceeded max allowed tests!\n" );
  opts->ntest = ntest;


  // set device
  cudaError_t ed = cudaSetDevice( opts->devices[0] );
  if(ed != cudaSuccess){
    printf("Error setting device : %s \n", cudaGetErrorString(ed) ); exit(-1);
  }

  return 1;
}// end parse_opts
