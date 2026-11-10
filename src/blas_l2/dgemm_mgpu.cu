/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/blas_l2/dgemm_mgpu.cu

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ahmad Abdelfattah
 * @date 2020-12-10
 **/

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <stdio.h>
#include "gemm_aux.cuh"
#include "kblas_operators.h"


#define DGEMM_MAX_TILE	(4096)

extern "C"
void kblas_dgemm_mgpu(char transa, char transb, long m, long n, long k,
                      double alpha, const double* A, long lda,
                      const double* B, long ldb,
                      double beta, double* C, long ldc,
                      long ngpus, long* gpu_id,
                      long *tile)
{

  cublasStatus_t se;
  int current_gpu;
  cudaGetDevice(&current_gpu);

  long tile_size = (*tile);
  if(tile_size == -1)
  {
    tile_size = recommend_tile(m, n, k, ngpus, DGEMM_MAX_TILE);
    (*tile) = tile_size;
  }
  // set to 1 to print info
  long pflag = 0;

  // compute #waves of full stripes
  long stripes = (m + tile_size-1)/tile_size; //(m / tile_size) + (m%tile_size != 0);
  long full_waves = stripes / ngpus;
  long remaining_stripes = stripes % ngpus;

  // compute the memory space required per gpu
  // first, wrap up k to be multiple of tile_size
  long k__ = ( (k + tile_size-1)/tile_size ) * tile_size;
  long width = tile_size;
  long height = k__; 			// height of a h-stripe of A or v-stripe of B
  height += 2 * tile_size;		// 2 extra tiles for multiplication
  height += 2 * tile_size;		// 2 output tiles
  height = ( (height+31)/32 ) * 32;	// for coalesced memory access
  long mem_space = height * width;

  // gpu pointers/worspace
  double* gpu_ws[MAX_NGPUS];
  double* a[MAX_NGPUS];
  double* b[MAX_NGPUS][2];
  double* c[MAX_NGPUS][2];
  double* a_[MAX_NGPUS];
  double* b_[MAX_NGPUS][2];
  double* c_[MAX_NGPUS][2];

  // streams
  cudaStream_t	stream[MAX_NGPUS][4];

  // events
  long nevents = (max(n, k)+tile_size-1) / tile_size;
  cudaEvent_t _ain_[MAX_NGPUS][MAX_EVENTS];
  cudaEvent_t _bin_[MAX_NGPUS][MAX_EVENTS];

  cudaEvent_t _afree_[MAX_NGPUS][MAX_EVENTS];
  cudaEvent_t _bfree_[MAX_NGPUS][MAX_EVENTS];

  cudaEvent_t _cin_[MAX_NGPUS][MAX_EVENTS];
  cudaEvent_t _cout_[MAX_NGPUS][MAX_EVENTS];
  cudaEvent_t _compute_[MAX_NGPUS][MAX_EVENTS];

  // allocate gpu memory
  {
    if(pflag)printf("memory allocation\n");
    cudaError_t e;
    for(long i = 0; i <  ngpus; i++)
    {
      cudaSetDevice(gpu_id[i]);
      e = cudaMalloc((void**)&gpu_ws[i], mem_space * sizeof(double));
      if(e != cudaSuccess)
      {
        printf("ERROR: failed to allocate memory on gpu %ld \n", i);
        for(long j = 0; j <= i; j++) { if(gpu_ws[i]) cudaFree(gpu_ws[i]); }
        exit(1);
      }
    }
  }

  // aux host pointers
  // aux pointers
  double *A_[MAX_NGPUS], *B_[MAX_NGPUS], *C_[MAX_NGPUS];

  // Adjust pointers
  {
    if(pflag)printf("adjust pointers\n");
    // host
    for(long i = 0; i < ngpus; i++)
    {
      A_[i] = (double*)A;
      B_[i] = (double*)B;
      C_[i] = (double*)C;
    }
    // device
    for(long i = 0; i < ngpus; i++)
    {
      a[i] = gpu_ws[i];
      b[i][0] = a[i]     + tile_size * k__;
      b[i][1] = b[i][0]  + tile_size * tile_size;
      c[i][0] = b[i][1]  + tile_size * tile_size;
      c[i][1] = c[i][0]  + tile_size * tile_size;
    }
  }

  // create streams and events
  {
    if(pflag)printf("stream create\n");
    for(long i = 0; i < ngpus; i++)
    {
      cudaSetDevice(gpu_id[i]);

      cudaStreamCreate(&stream[i][0]);	// compute
      cudaStreamCreate(&stream[i][1]);	// copy a in and c out
      cudaStreamCreate(&stream[i][2]);	// copy b in
      cudaStreamCreate(&stream[i][3]);	// copy c in


      for(long j = 0; j < nevents; j++)
      {
        cudaEventCreate(&_ain_[i][j], cudaEventDisableTiming);
        cudaEventCreate(&_bin_[i][j], cudaEventDisableTiming);
        cudaEventCreate(&_afree_[i][j], cudaEventDisableTiming);
        cudaEventCreate(&_bfree_[i][j], cudaEventDisableTiming);
        cudaEventCreate(&_compute_[i][j], cudaEventDisableTiming);
        cudaEventCreate(&_cin_[i][j], cudaEventDisableTiming);
        cudaEventCreate(&_cout_[i][j], cudaEventDisableTiming);
}
    }
  }

  // set stream for the gemm calls
  for(long id = 0; id < ngpus; id++)
  {
    if(pflag)printf("set kernel stream\n");
    cudaSetDevice(gpu_id[id]);
    cublasSetKernelStream(stream[id][0]);
  }


  // compute stepping in A and B
  long step_a, step_b;
  if(transa == 'n' || transa == 'N'){
    step_a = tile_size * lda;
  }else{
    step_a = tile_size;
  }
  if(transb == 'n' || transb == 'N'){
    step_b = tile_size;
  }else{
    step_b = tile_size * ldb;
  }

  // selector to switch between 2 gpu buffers
  long bselect[MAX_NGPUS] = {0};
  long cselect[MAX_NGPUS] = {0};

  // variables that store the actual tile sizes from A, B, and C for every GPU
  long ra[MAX_NGPUS] = {0};
  long ca[MAX_NGPUS] = {0};
  long rb[MAX_NGPUS] = {0};
  long cb[MAX_NGPUS] = {0};
  long rc[MAX_NGPUS] = {0};
  long cc[MAX_NGPUS] = {0};

  //main loop
  {
    if(pflag)printf("main loop\n");
    long total_iterations = full_waves + (remaining_stripes!=0);
    long ngpus_active;
    long n_ = (n + tile_size-1) / tile_size;
    long k_ = (k + tile_size-1) / tile_size;

    // i - loop over full waves (m)
    for(long i = 0; i < total_iterations; i++)
    {
      ngpus_active = ngpus;
      if(i == total_iterations-1){
        if(remaining_stripes != 0)
          ngpus_active = remaining_stripes;
      }

      // advance A_
      if(pflag)printf("i = %ld, advance A_\n", i);
      if(transa == 'n' || transa == 'N'){
        for(long id = 0; id < ngpus_active; id++) {
          A_[id] = (double*)A + (i *ngpus + id) * tile_size;
        }
      }else{
        for(long id = 0; id < ngpus_active; id++) {
          A_[id] = (double*)A + (i * ngpus + id) * tile_size * lda;
        }
      }

      // compute #rows of current tiles in A and C
      for(long id = 0; id < ngpus_active; id++){
        rc[id] = min(m - (i*ngpus+id)*tile_size , tile_size);
      }
      if(transa == 'n' || transa == 'N'){
        for(long id = 0; id < ngpus_active; id++)
          ra[id] = min(m - (i*ngpus+id)*tile_size , tile_size);
      }else{
        for(long id = 0; id < ngpus_active; id++){
          ca[id] = min(m - (i*ngpus+id)*tile_size , tile_size);
        }
      }

      // j - loop over (n) -
      for(long j = 0; j < n_ ; j++)
      {
        if(pflag)printf("\t j = %ld, advance B_ and C_\n", j);

        // compute #cols in current tiles in B and C
        for(long id = 0; id < ngpus_active; id++){
          cc[id] = min(n - j*tile_size , tile_size);
        }
        if(transb == 'n' || transb == 'N'){
          for(long id = 0; id < ngpus_active; id++){
            cb[id] = min(n - j*tile_size , tile_size);
          }
        }else{
          for(long id = 0; id < ngpus_active; id++){
            rb[id] = min(n - j*tile_size , tile_size);
          }
        }

        // Advance B_
        if(transb == 'n' || transb == 'N'){
          for(long id = 0; id < ngpus_active; id++) {
            B_[id] = (double*)B + j * tile_size * ldb;
          }
        }else{
          for(long id = 0; id < ngpus_active; id++) {
            B_[id] = (double*)B + j * tile_size;
          }
        }

        // Advance C_
        for(long id = 0; id < ngpus_active; id++)
        {
            //C_[id] = (double*)C + ( (i *ngpus + id) * tile_size ) + ( j * tile_size * ldc);
            C_[id] = (double*)C;
            //if(transa == 'n' || transa == 'N')
            C_[id] += (i *ngpus + id) * tile_size;
            //else C_[id] += (i * ngpus + id) * tile_size * ldc;

            //if(transb == 'n' || transb == 'N')
            C_[id] += j * tile_size * ldc;
            //else C_[id] += j * tile_size;
        }

        // copy device pointers
        for(long id = 0; id < ngpus_active; id++)
        {
            a_[id] = a[id];
            b_[id][0] = b[id][0];
            b_[id][1] = b[id][1];
            c_[id][0] = c[id][0];
            c_[id][1] = c[id][1];
        }

        // if starting to compute new row of tiles in C
        // copy the first tile of C in the row into devices
        if(j == 0)
        {
          for(long id = 0; id < ngpus_active; id++)
          {
            cudaSetDevice(gpu_id[id]);
            cudaStreamWaitEvent(stream[id][3], _cout_[id][cselect[id]], 0);
            se = cublasSetMatrixAsync(rc[id], cc[id], sizeof(double), C_[id], ldc, c[id][cselect[id]], tile_size, stream[id][3]);
            process_error(se, "copy cin new row of tiles");
            cudaEventRecord(_cin_[id][cselect[id]], stream[id][3]);
          }
        }

        if(pflag)printf("\t j = %ld, copy a, b tile in\n", j);
        // prepare a first input offload
        for(long id = 0; id < ngpus_active; id ++)
        {
          // as if p = 0 (first iteration in the inner-most loop)
          if(transa == 'n' || transa == 'N')
            ca[id] = min(k - 0*tile_size , tile_size);
          else
            ra[id] = min(k - 0*tile_size , tile_size);

          if(transb == 'n' || transb == 'N')
            rb[id] = min(k - 0*tile_size , tile_size);
          else
            cb[id] = min(k - 0*tile_size , tile_size);


          cudaSetDevice(gpu_id[id]);
          if(j == 0)
          {
            cudaStreamWaitEvent(stream[id][1], _afree_[id][0], 0);
            se = cublasSetMatrixAsync(ra[id], ca[id], sizeof(double), A_[id], lda, a_[id], tile_size, stream[id][1]);
            char ss[100];
            sprintf(ss, " i =%ld, j = %ld copy ain new row of tiles: [%ld]x[%ld]", i, j, ra[id], ca[id]);
            process_error(se, ss);
            cudaEventRecord(_ain_[id][0], stream[id][1]);
          }
          cudaStreamWaitEvent(stream[id][2], _bfree_[id][bselect[id]], 0);
          se = cublasSetMatrixAsync(rb[id], cb[id], sizeof(double), B_[id], ldb, b_[id][bselect[id]], tile_size, stream[id][2]);
          process_error(se, "copy bin new row of tiles");
          cudaEventRecord(_bin_[id][bselect[id]], stream[id][2]);
        }

        // init b selector
        //for(long id = 0; id < ngpus; id++) bselect[id] = 0;

        // p - loop over k
        long p = 0;
        for(p = 0;  p < k_; p++)
        {
          double beta_;
          if(p == 0)beta_ = beta; else beta_ = 1;

          for(long id = 0; id < ngpus_active; id++)
          {
            cudaSetDevice(gpu_id[id]);

            if(pflag)printf("\t\t p = %ld, wait for communication\n", p);

            if(transa == 'n' || transa == 'N')
              ca[id] = min(k - p*tile_size , tile_size);
            else
              ra[id] = min(k - p*tile_size , tile_size);

            if(transb == 'n' || transb == 'N')
              rb[id] = min(k - p*tile_size , tile_size);
            else
              cb[id] = min(k - p*tile_size , tile_size);


            // wait for communication
            //if(p == 0)cudaStreamSynchronize(stream[id][3]);
            //if(j == 0)cudaStreamSynchronize(stream[id][1]);
            //cudaStreamSynchronize(stream[id][2]);
            if(p == 0) cudaStreamWaitEvent(stream[id][0], _cin_[id][cselect[id]], 0);
            if(j == 0) cudaStreamWaitEvent(stream[id][0], _ain_[id][p], 0);
            cudaStreamWaitEvent(stream[id][0], _bin_[id][bselect[id]], 0);

            if(pflag)printf("\t\t p = %ld, gpu = %ld, invoke dgemm\n", p, id);
            if(pflag)printf("\t\t ------------------------------\n");
            if(pflag)printf("\t\t cselect[%ld] = %ld \n", id, cselect[id]);
            long msmall = rc[id];
            long nsmall = cc[id];
            long ksmall;
            if(transa == 'n' || transa == 'N') ksmall = ca[id];
            else ksmall = ra[id];

            //{
            //	printf("\n");
            //	printf("gpu %ld: [%ld][%ld] x [%ld][%ld] = [%ld][%ld]\n", id, msmall, ksmall, ksmall, nsmall, msmall, nsmall);
            //	//print a
            //	printf("A\n--------\n");
            //	myprint_matrix(transa, msmall, ksmall, a_[id], tile_size);
            //	//print b
            //	printf("B\n--------\n");
            //	myprint_matrix(transb, ksmall, nsmall, b_[id][bselect[id]], tile_size);
            //}
            // invoke dgemm
            cublasDgemm(transa, transb,
                        msmall, nsmall, ksmall,
                        alpha, a_[id], tile_size,
                        b_[id][bselect[id]], tile_size,
                        beta_, c_[id][cselect[id]], tile_size);
            cudaEventRecord(_bfree_[id][bselect[id]], stream[id][0]);
            if(j == n_-1) cudaEventRecord(_afree_[id][p], stream[id][0]);
            if(p == k_-1) cudaEventRecord(_compute_[id][j], stream[id][0]);

            // prepare next input
            bselect[id] = 1 - bselect[id];
            a_[id] += tile_size * tile_size;
            if(p != k_-1)
            {
              if(pflag)printf("\t\t p = %ld, prepare next input\n", p);
              if(j == 0)
              {
                A_[id] += step_a;

                if(transa == 'n' || transa == 'N')ca[id] = min(k - (p+1)*tile_size, tile_size);
                else ra[id] = min(k - (p+1)*tile_size, tile_size);

                cudaStreamWaitEvent(stream[id][1], _afree_[id][p+1], 0);
                se = cublasSetMatrixAsync(ra[id], ca[id], sizeof(double), A_[id], lda, a_[id], tile_size, stream[id][1]);
                process_error(se, "prefetch ain");
                cudaEventRecord(_ain_[id][p+1], stream[id][1]);

                if(transa == 'n' || transa == 'N')ca[id] = min(k - (p)*tile_size, tile_size);
                else ra[id] = min(k - (p)*tile_size, tile_size);

              }
              B_[id] += step_b;

              if(transb == 'n' || transb == 'N') rb[id] = min(k - (p+1)*tile_size, tile_size);
              else cb[id] = min(k - (p+1)*tile_size, tile_size);

              cudaStreamWaitEvent(stream[id][2], _bfree_[id][bselect[id]], 0);
              se = cublasSetMatrixAsync(rb[id], cb[id], sizeof(double), B_[id], ldb, b_[id][bselect[id]], tile_size, stream[id][2]);
              process_error(se, "prefetch bin");
              cudaEventRecord(_bin_[id][bselect[id]], stream[id][2]);


              if(transb == 'n' || transb == 'N') rb[id] = min(k - (p)*tile_size, tile_size);
              else cb[id] = min(k - (p)*tile_size, tile_size);
            }
            if( p == 0)
            {
              if(j != n_-1)
              {
                // early copy of the next tile of C
                double* Ctmp = C_[id] + tile_size * ldc;
                cselect[id] = 1 - cselect[id];
                // rc[id] is the same, but we need to compute cc
                cc[id] = min(n - (j+1)*tile_size, tile_size);
                if(pflag)printf("\t\t cselect[%ld] = %ld \n", id, cselect[id]);
                cudaStreamWaitEvent(stream[id][3], _cout_[id][cselect[id]], 0);
                se = cublasSetMatrixAsync(rc[id], cc[id], sizeof(double), Ctmp, ldc, c_[id][cselect[id]], tile_size, stream[id][3]);
                char ss[100];
                sprintf(ss, "gpu[%ld]: prefetch cin [%ld]x[%ld]", id, rc[id], cc[id]);

                process_error(se, ss);
                cudaEventRecord(_cin_[id][cselect[id]], stream[id][3]);
                cselect[id] = 1 - cselect[id];
                // restore cc
                cc[id] = min(n - j*tile_size, tile_size);
              }
            }
            if(pflag)printf("\n");
          }
        }// p - loop over k

        // copy c into cpu
        for(long id = 0; id < ngpus_active; id++)
        {
          if(pflag)printf("i = %ld, j = %ld, gpu = %ld, copy c output\n", i, j, id);
          cudaSetDevice(gpu_id[id]);
          //cudaStreamSynchronize(stream[id][0]);
          cudaStreamWaitEvent(stream[id][3], _compute_[id][j], 0);
          se = cublasGetMatrixAsync(rc[id], cc[id], sizeof(double), c_[id][cselect[id]], tile_size, C_[id], ldc, stream[id][3]);
          process_error(se, "read output c");
          cudaEventRecord(_cout_[id][cselect[id]], stream[id][3]);
          cselect[id] = 1 - cselect[id];
        }
      }// j - loop over (n)
    } // i - loop over full waves (n)
  }// main compute part

  // global sync point
  {
  if(pflag)printf("global sync\n");
    for(long id = 0; id < ngpus; id++)
    {
      cudaSetDevice(gpu_id[id]);
      cudaDeviceSynchronize();
    }
  }

  // switch cublas streams to the default one
  for(long id = 0; id < ngpus; id ++)
  {
    cudaSetDevice(gpu_id[id]);
    cublasSetKernelStream(0);
  }

  // destroy streams
  {
    if(pflag)printf("destroy stream\n");
    for(long i = 0; i < ngpus; i++)
    {
      cudaSetDevice(gpu_id[i]);
      cudaStreamDestroy(stream[i][0]);
      cudaStreamDestroy(stream[i][1]);
      cudaStreamDestroy(stream[i][2]);
      cudaStreamDestroy(stream[i][3]);
    }
  }

  // destroy events
  {
    for(long i = 0; i < ngpus; i++)
    {
      for(long j = 0; j < nevents; j++)
      {
        cudaEventDestroy(_ain_[i][j]);
        cudaEventDestroy(_bin_[i][j]);
        cudaEventDestroy(_afree_[i][j]);
        cudaEventDestroy(_bfree_[i][j]);
        cudaEventDestroy(_compute_[i][j]);
        cudaEventDestroy(_cin_[i][j]);
        cudaEventDestroy(_cout_[i][j]);
      }
    }
  }

  // free resources
  {
    if(pflag)printf("free resources\n");
    for(long i = 0; i < ngpus; i++)
      if(gpu_ws[i]) cudaFree(gpu_ws[i]);
  }

  // retrieve current gpu
  cudaSetDevice(current_gpu);
}
