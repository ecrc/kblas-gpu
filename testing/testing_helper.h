#ifndef __TESTING_HELPER_H__
#define __TESTING_HELPER_H__

#include <cublas_v2.h>
#include <kblas.h>

#ifdef __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////
// Error checking
////////////////////////////////////////////////////////////

#define check_error(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line);

#define check_cublas_error(ans) { gpuCublasAssert((ans), __FILE__, __LINE__); }
void gpuCublasAssert(cublasStatus_t code, const char *file, int line);

////////////////////////////////////////////////////////////
// Timers
////////////////////////////////////////////////////////////
double gettime(void);

struct GPU_Timer;
GPU_Timer* newGPU_Timer();
void deleteGPU_Timer(GPU_Timer* timer);
void gpuTimerTic(GPU_Timer* timer, cudaStream_t stream);
double gpuTimerToc(GPU_Timer* timer, cudaStream_t stream);

////////////////////////////////////////////////////////////
// Generate array of pointers from a strided array 
////////////////////////////////////////////////////////////
void generateDArrayOfPointers(double* original_array, double** array_of_arrays, int stride, int num_arrays, cudaStream_t stream);
void generateSArrayOfPointers(float* original_array, float** array_of_arrays, int stride, int num_arrays, cudaStream_t stream);

////////////////////////////////////////////////////////////
// Command line parser
////////////////////////////////////////////////////////////
#define MAX_NTEST 1000

typedef struct kblas_opts
{
	// matrix size
	int ntest;
	int msize[ MAX_NTEST ];
	int nsize[ MAX_NTEST ];
	int ksize[ MAX_NTEST ];
	int rsize[ MAX_NTEST ];

	// scalars
	int devices[MAX_NGPUS];
	int nstream;
	int ngpu;
	int niter;
	int nruns;
	double      tolerance;
	int check;
	int verbose;
	int nb;
	int db;
	int custom;
	int warmup;
	int time;
	int lapack;
	//int bd[KBLAS_BACKDOORS];
	int batchCount;
	int strided;
	int btest, batch[MAX_NTEST];
	int rtest, rank[MAX_NTEST];
	int omp_numthreads;

	// lapack flags
	char uplo;
	char transA;
	char transB;
	char side;
	char diag;
} kblas_opts;

int parse_opts(int argc, char** argv, kblas_opts *opts);

#ifdef __cplusplus
}
#endif

#endif // __TESTING_HELPER_H__