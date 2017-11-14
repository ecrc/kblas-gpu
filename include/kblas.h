/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file include/kblas.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 2.0.0
 * @author Ali Charara
 * @author Wajih Halim Boukaram
 * @author Ahmad Abdelfattah
 * @date 2017-11-13
 **/

#ifndef _KBLAS_H_
#define _KBLAS_H_

/** @defgroup C_API KBLAS C style API routines
 */

/** @defgroup CPP_API KBLAS C++ style API routines
 */

//TODO is this include necessary here?
#include "defs.h"

struct KBlasHandle;
typedef struct KBlasWorkspace *kblasWorkspace_t;
typedef struct KBlasHandle *kblasHandle_t;

int kblasCreate(kblasHandle_t *handle);
int kblasDestroy(kblasHandle_t *handle);

int kblasAllocateWorkspace(kblasHandle_t handle);
int kblasFreeeWorkspace(kblasHandle_t handle);

void kblasTimerTic(kblasHandle_t handle);
void kblasTimerRecordEnd(kblasHandle_t handle);
double kblasTimerToc(kblasHandle_t handle);

cudaStream_t kblasGetStream(kblasHandle_t handle);
void kblasSetStream(kblasHandle_t handle, cudaStream_t stream);

cublasHandle_t kblasGetCublasHandle(kblasHandle_t handle);

const char* kblasGetErrorString(int error);

//============================================================================
//BLAS2 routines
//============================================================================
#include "kblas_l2.h"


//============================================================================
//BLAS3 routines
//============================================================================
#include "kblas_l3.h"

//============================================================================
//BATCH routines
//============================================================================
#include "kblas_batch.h"
#include "batch_qr.h"
#include "batch_svd.h"

#endif // _KBLAS_H_
