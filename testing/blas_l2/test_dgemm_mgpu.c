/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file testing/blas_l2/test_dgemm_mgpu.c

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ahmad Abdelfattah
 * @date 2018-11-14
 **/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <hip/hip_runtime_api.h>
#include <hipblas.h>
#include "kblas.h"
#include "testing_utils.h"

#define FMULS_GEMM(m_, n_, k_) ((m_) * (n_) * (k_))
#define FADDS_GEMM(m_, n_, k_) ((m_) * (n_) * (k_))

#define FLOPS_ZGEMM(m_, n_, k_) (6. * FMULS_GEMM((double)(m_), (double)(n_), (double)(k_)) + 2.0 * FADDS_GEMM((double)(m_), (double)(n_), (double)(k_)) )
#define FLOPS_CGEMM(m_, n_, k_) (6. * FMULS_GEMM((double)(m_), (double)(n_), (double)(k_)) + 2.0 * FADDS_GEMM((double)(m_), (double)(n_), (double)(k_)) )
#define FLOPS_DGEMM(m_, n_, k_) (     FMULS_GEMM((double)(m_), (double)(n_), (double)(k_)) +       FADDS_GEMM((double)(m_), (double)(n_), (double)(k_)) )
#define FLOPS_SGEMM(m_, n_, k_) (     FMULS_GEMM((double)(m_), (double)(n_), (double)(k_)) +       FADDS_GEMM((double)(m_), (double)(n_), (double)(k_)) )

#define MAX_NGPUS	(16)

int main(int argc, char** argv)
{
  if(argc < 9)
  {
    printf("USAGE: %s <ngpus-start> <ngpus-end> <transa> <transb> <start-dim> <stop-dim> <step-dim> <tile-size>\n", argv[0]);
    printf("==> <ngpus-start> <ngpus-end>: Run the test on <ngpus-start> up to <ngpus-end>, adding 1 GPU each time\n");
    printf("==> <transa>: 'n' or 'N' (no-traspose) or 't' or 'T' (transpose) or 'c' or 'C' (conjugate) \n");
    printf("==> <transb>: 'n' or 'N' (no-traspose) or 't' or 'T' (transpose) or 'c' or 'C' (conjugate) \n");
    printf("==> <start-dim> <stop-dim> <step-dim>: test for dimensions in the range start-dim : stop-dim with step step-dim \n");
    printf("==> <tile-size>: must divide every dimension to be tested\n");
    exit(-1);
  }

  long ngpus1 = atoi(argv[1]);
  long ngpus2 = atoi(argv[2]);
  char transa = *argv[3];
  char transb = *argv[4];
  long istart = atoi(argv[5]);
  long istop = atoi(argv[6]);
  long istep = atoi(argv[7]);
  long tile_size = atoi(argv[8]);

  int gpus_avail;
  hipGetDeviceCount(&gpus_avail);
  long *gpu_id = (long*)malloc(ngpus2 * sizeof(long));

  if(argc > 9){
    int n;
    int info = sscanf( argv[9], "%d", &n );
    if ( info == 1) {
      char inp[512];
      char * pch;
      int ngpus = 0;
      strcpy(inp, argv[9]);
      pch = strtok (inp,",");
      do{
        info = sscanf( pch, "%d", &n );
        if ( ngpus >= MAX_NGPUS ) {
          printf( "warning: selected number exceeds KBLAS max number of GPUs, ngpus=%d.\n", ngpus);
          break;
        }
        if ( ngpus >= gpus_avail ) {
          printf( "warning: max number of available devices reached, ngpus=%d.\n", ngpus);
          break;
        }
        if ( n >= gpus_avail || n < 0) {
          printf( "error: device %d is invalid; ensure dev in [0,%d].\n", n, gpus_avail-1 );
          break;
        }
        gpu_id[ ngpus++ ] = n;
        pch = strtok (NULL,",");
      }while(pch != NULL);
      if(ngpus2 > ngpus) ngpus2 = ngpus;
      if(ngpus1 > ngpus) ngpus1 = ngpus;
    }
  }else{
    // init ids - for now assume one node
    long k;
    for(k = 0; k < ngpus2; k++)
            gpu_id[k] = (long)k;
  }

  if(ngpus1 > gpus_avail)
  {printf("Error: Can't run on %ld gpus, only %d gpus are available \n", ngpus1, gpus_avail); exit(-1);}

  if(ngpus2 > gpus_avail)
  {printf("Error: Can't run on %ld gpus, only %d gpus are available \n", ngpus2, gpus_avail); exit(-1);}

  if(ngpus1 > ngpus2)
  {printf("Error: ngpus-end is larger than ngpu-start \n"); exit(1);}

  long M = istop;
  long N = M;
  long K = M;
  long LDA = M;
  long LDB = K;
  long LDC = M;

  hipblasOperation_t transa_, transb_;

  if(transa == 'N' || transa == 'n')
    transa_ = HIPBLAS_OP_N;
  else
  if (transa == 'T' || transa == 't')
    transa_ = HIPBLAS_OP_T;
  else
  if (transa == 'C' || transa == 'c')
    transa_ = HIPBLAS_OP_C;
  else
    {printf("wrong parameter transa = %c\n", transa); exit(1);}

  if(transb == 'N' || transb == 'n')
    transb_ = HIPBLAS_OP_N;
  else
  if (transb == 'T' || transb == 't')
    transb_ = HIPBLAS_OP_T;
  else
  if (transb == 'C' || transb == 'c')
    transb_ = HIPBLAS_OP_C;
  else
    {printf("wrong parameter transb = %c\n", transb); exit(1);}

  //double alpha = 2.3, beta = -1.0;
  double alpha = 1.0, beta = 0.0;

  hipError_t err;
  hipEvent_t start, stop;

  hipSetDevice(gpu_id[0]);
  hipEventCreate(&start);
  hipEventCreate(&stop);

  // point to host memory
  double* A = NULL;
  double* B = NULL;
  double* C = NULL;
  double* csingle = NULL;
  double* cmgpu = NULL;

  printf("transa = %c - transb = %c .. \n", transa, transb);

  long size_a = M * LDA;
  long size_b = K * LDB;
  long size_c = M * LDC;

  // cpu alloc / init
  {
    // alloc memory cpu
    printf("Allocating matrices on cpu .. \n");

    //A = (double*)malloc(size_a*sizeof(double));
    //B = (double*)malloc(size_b*sizeof(double));
    C = (double*)malloc(size_c*sizeof(double));
    csingle = (double*)malloc(size_c*sizeof(double));
    //cmgpu = (double*)malloc(size_c*sizeof(double));

    hipHostMalloc((void**)&A, size_a*sizeof(double));			// better for mgpu version
    hipHostMalloc((void**)&B, size_b*sizeof(double));			// better for mgpu version
    //hipHostMalloc((void**)&C, size_c*sizeof(double));			// better for mgpu version
    //hipHostMalloc((void**)&csingle, size_c*sizeof(double));	// better for mgpu version
    hipHostMalloc((void**)&cmgpu, size_c*sizeof(double));		// better for mgpu version


    printf("Initializing on cpu .. \n");
    // Initialize matrix and vector on cpu
    long i, j, m;
    drand_matrix(M, K, A, LDA);
    drand_matrix(K, N, B, LDB);
    drand_matrix(M, N, C, LDC);
  }

  long alloc = 1, alloc_mgpu = 1;
  long ngpus;
  for(ngpus = ngpus1; ngpus <= ngpus2; ngpus++)
  {
    printf("-------------------------------- Testing DGEMM -------------------------------\n");
    printf("  Matrix         1 GPU        Tile Size      KBLAS %-2ld GPUs          Max.    \n", ngpus);
    printf(" Dimension     (Gflop/s)         Used     (Gflop/s)     Time(s)      Error    \n");
    printf("-----------   -------------   ---------   ----------   ----------   ----------\n");

    long dim;
    for(dim = istart; dim <= istop; dim += istep)
    {
      float elapsedTime;
      float dimf = (float)dim;
      float flops = FLOPS_DGEMM(dimf, dimf, dimf) / 1e6;
      float single_gpu_perf;
      float mgpu_perf;
      double error;
      long m = dim;
      long n = dim;
      long k = dim;
      long lda = ((k+31)/32)*32;
      long ldb = ((n+31)/32)*32;
      long ldc = ((n+31)/32)*32;

      long sizea = m * lda;
      long sizeb = k * ldb;
      long sizec = m * ldc;

      printf("%-11ld   ", m);
      //printf("check point 1\n");
      // single gpu test
      {
        double* dA_single = NULL;
        double* dB_single = NULL;
        double* dC_single = NULL;

        alloc = 1;
        // alloc A, B, C on gpus
        // single gpu test
        hipError_t e1, e2, e3;
        hipSetDevice(gpu_id[0]);
        e1 = hipMalloc((void**)&dA_single, sizea*sizeof(double));
        e2 = hipMalloc((void**)&dB_single, sizeb*sizeof(double));
        e3 = hipMalloc((void**)&dC_single, sizec*sizeof(double));

        //printf("check point 1.1\n");

        if((e1 != hipSuccess) || (e2 != hipSuccess) || (e3 != hipSuccess) )
        {
          if(dA_single)hipFree(dA_single);
          if(dB_single)hipFree(dB_single);
          if(dC_single)hipFree(dC_single);
          alloc = 0;
          printf("-----           ");
        }
        //printf("check point 1.2\n");
        if(alloc == 1)
        {
          // offload A, y to gpus
          hipSetDevice(gpu_id[0]);
          //cublasSetKernelStream(0);
          float time = 0.0, copy_time = 0.0;

          hipEventRecord(start, 0);
          hipblasSetMatrix((long)m, (long)k, sizeof(double), A, (long)LDA, dA_single, (long)lda);
          hipblasSetMatrix((long)k, (long)n, sizeof(double), B, (long)LDB, dB_single, (long)ldb);
          hipEventRecord(stop, 0);
          hipEventSynchronize(stop);
          elapsedTime = 0.0;
          hipEventElapsedTime(&elapsedTime, start, stop);
          copy_time += elapsedTime;
          long r;
          for(r = 0; r < (NRUNS+1); r++)
          {
            hipSetDevice(gpu_id[0]);
            hipEventRecord(start, 0);
            hipblasSetMatrix((long)m, (long)n, sizeof(double), C, (long)LDC, dC_single, (long)ldc);
            //printf("check point 1.3\n");
            // call cublas dgemm
            hipblasDgemm(transa, transb,
                            (long)m, (long)n, (long)k,
                            alpha, dA_single, (long)lda,
                            dB_single, (long)ldb,
                            beta, dC_single, (long)ldc);
            //printf("check point 1.4\n");
            hipEventRecord(stop, 0);
            hipEventSynchronize(stop);
            elapsedTime = 0.0;
            hipEventElapsedTime(&elapsedTime, start, stop);
            if(r > 0)time += elapsedTime;		// skip first iteration
          }

          hipEventRecord(start, 0);
          hipblasGetMatrix((long)m, (long)n, sizeof(double), dC_single, (long)ldc, csingle, (long)LDC);
          hipEventRecord(stop, 0);
          hipEventSynchronize(stop);
          elapsedTime = 0.0;
          hipEventElapsedTime(&elapsedTime, start, stop);
          copy_time += elapsedTime;

          elapsedTime = time/NRUNS + copy_time;
          single_gpu_perf = flops / elapsedTime;

          printf("%-13.2f   ", single_gpu_perf);
          //printf("check point 1.5\n");
          if(dA_single)hipFree(dA_single);
          if(dB_single)hipFree(dB_single);
          if(dC_single)hipFree(dC_single);
          //printf("check point 1.6\n");
        }
      } // end of 1 gpu test
      //printf("check point 2\n");
      // mgpu test
      {
        float time  = 0.0;
        long tile;
        long r;
        for(r = 0; r < (NRUNS+1); r++)
        {
          // make a copy of C
          memcpy(cmgpu, C, size_c * sizeof(double));

          hipSetDevice(gpu_id[0]);
          hipEventRecord(start, 0);
          // dgemm_mgpu
          tile = tile_size;
          kblas_dgemm_mgpu( transa, transb,
                                  (long)m, (long)n, (long)k,
                                  alpha, A, (long)LDA,
                                  B, (long)LDB,
                                  beta, cmgpu, (long)LDC,
                                  ngpus, gpu_id,
                                  (long*)&tile);
          hipSetDevice(gpu_id[0]);
          hipEventRecord(stop, 0);
          hipEventSynchronize(stop);

          elapsedTime = 0.0;
          hipEventElapsedTime(&elapsedTime, start, stop);
          if(r > 0)time += elapsedTime;		// skip first iteration
        }
        elapsedTime = time / NRUNS;

        printf("%-9ld   ", tile);

        // flops/s including alloc time
        mgpu_perf = flops / elapsedTime;
        printf("%-10.2f   ", mgpu_perf);

        // print time
        printf("%-10.2f   ", elapsedTime * 1e-3);
      } // end of mgpu test
      //printf("check point 3\n");
      // testing correctness
      {
        if(alloc == 1 && alloc_mgpu == 1)
        {
          // testing error -- specify ref. vector and result vector
          double* cref = csingle;
          double* cres = cmgpu;
          //error = get_max_error(cref, cres, m , n, LDC);
          error = dget_max_error_matrix(cref, cres, m, n, LDC);
          printf("%-10e;\n", error);
        }
        else
          printf ("N/A       \n");
      }
      //printf("check point 4\n");
      if(alloc == 0 && alloc_mgpu == 0)break;

    } // end of for loop over dim
    printf("\n\n");
  } // end of for loop over ngpus
  // finalize
  hipSetDevice(gpu_id[0]);
  hipEventDestroy(start);
  hipEventDestroy(stop);

  //if(A)free(A);
  //if(B)free(B);
  if(C)free(C);
  if(csingle)free(csingle);
  //if(cmgpu)free(cmgpu);

  if(A)hipHostFree(A);
  if(B)hipHostFree(B);
  //if(C)hipHostFree(C);
  //if(csingle)hipHostFree(csingle);
  if(cmgpu)hipHostFree(cmgpu);
  if(gpu_id)free(gpu_id);

  return 0;
}

