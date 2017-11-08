#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

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
	
    void init(cudaStream_t stream = 0)
    {
        #pragma omp critical (create_timer)
        {
            check_error( cudaEventCreate(&start_event) );
            check_error( cudaEventCreate(&stop_event ) );
            elapsed_time = 0;
			this->stream = stream;
        }
    }

    void destroy()
    {
        #pragma omp critical (delete_timer)
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
	GPU_Timer* timer = new GPU_Timer();
	timer->init(stream);
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Error helpers
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void gpuAssert(cudaError_t code, const char *file, int line)
{
    if(code != cudaSuccess) 
	{
        printf("GPUassert: %s(%d) %s %d\n", cudaGetErrorString(code), (int)code, file, line);
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

extern "C" void gpuCublasAssert(cublasStatus_t code, const char *file, int line)
{
	if(code != CUBLAS_STATUS_SUCCESS) 
	{
        printf("GPUassert: %s %s %d\n", cublasGetErrorString(code), file, line);
		exit(-1);
	}
}

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
    opts->tolerance  = 30.;
    opts->check      = 0;
    opts->verbose    = 0;
    opts->custom     = 0;
    opts->warmup     = 0;
    opts->time       = 0;
    opts->lapack     = 0;
    opts->batchCount = 4;
    opts->strided    = 0;
    opts->btest      = 1;
    opts->rtest      = 1;
    //opts->bd[0]      = -1;
    opts->omp_numthreads = 20;

    opts->uplo      = 'L';      // potrf, etc.
    opts->transA    = 'N';    // gemm, etc.
    opts->transB    = 'N';    // gemm
    opts->side      = 'L';       // trsm, etc.
    opts->diag      = 'N';    // trsm, etc.

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
        kblas_assert( ntest < MAX_NTEST, "error: -N %s, max number of tests exceeded, ntest=%d.\n",
                     argv[i], ntest );
        i++;
        int m2, n2, k2, q2;
        info = sscanf( argv[i], "%d:%d:%d:%d", &m2, &n2, &k2, &q2 );
        if ( info == 4 && m2 >= 0 && n2 >= 0 && k2 >= 0 && q2 >= 0 ) {
          opts->msize[ ntest ] = m2;
          opts->nsize[ ntest ] = n2;
          opts->ksize[ ntest ] = k2;
          opts->rsize[ ntest ] = q2;
        }
        else if ( info == 3 && m2 >= 0 && n2 >= 0 && k2 >= 0 ) {
          opts->msize[ ntest ] = m2;
          opts->nsize[ ntest ] = n2;
          opts->ksize[ ntest ] = k2;
          opts->rsize[ ntest ] = k2;  // implicitly
        }
        else if ( info == 2 && m2 >= 0 && n2 >= 0 ) {
          opts->msize[ ntest ] = m2;
          opts->nsize[ ntest ] = n2;
          opts->ksize[ ntest ] = n2;  // implicitly
          opts->rsize[ ntest ] = n2;  // implicitly
        }
        else if ( info == 1 && m2 >= 0 ) {
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
        char op;
        info = sscanf( argv[i], "%d:%d%c%d", &start, &stop, &op, &step );
        if( info == 1 ){
          opts->rank[0] = start;
        }else
        if ( info == 4 && start >= 0 && stop >= 0 && step != 0 && (op == '+' || op == '*' || op == ':')) {
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
      else if ( strcmp("--strided",  argv[i]) == 0 || strcmp("-s",  argv[i]) == 0 ) { opts->strided = 1;  }
      else if ( strcmp("--tolerance", argv[i]) == 0 && i+1 < argc ) {
        opts->tolerance = atof( argv[++i] );
        kblas_assert( opts->tolerance >= 0 && opts->tolerance <= 1000,
                     "error: --tolerance %s is invalid; ensure tolerance in [0,1000].\n", argv[i] );
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
    if(ed != cudaSuccess){printf("Error setting device : %s \n", cudaGetErrorString(ed) ); exit(-1);}

    return 1;
}// end parse_opts