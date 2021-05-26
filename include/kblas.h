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
 * @version 4.0.0
 * @author Ali Charara
 * @author Wajih Halim Boukaram
 * @author Ahmad Abdelfattah
 * @date 2020-12-10
 **/

#ifndef _KBLAS_H_
#define _KBLAS_H_

/** @defgroup C_API KBLAS C style API routines
 */

/** @defgroup CPP_API KBLAS C++ style API routines
 */

/** @defgroup WSQUERY KBLAS workspace helper routines routines
 */

#include "kblas_defs.h"

// Extern to make it work with C
#ifdef __cplusplus
extern "C"
{
#endif

struct KBlasHandle;
typedef struct KBlasWorkspace *kblasWorkspace_t;
typedef struct KBlasHandle *kblasHandle_t;

struct KBlasRandState;
typedef struct KBlasRandState* kblasRandState_t;

/** @addtogroup C_API
*  @{
*/

/**
 * @name KBLAS helper routines
 * @{
 */

/**
 * @brief Create a KBLAS handle.
 */
int kblasCreate(kblasHandle_t *handle);

/**
 * @brief Destroy a KBLAS handle.
 */
int kblasDestroy(kblasHandle_t *handle);
void kblasTimerTic(kblasHandle_t handle);
void kblasTimerRecordEnd(kblasHandle_t handle);
double kblasTimerToc(kblasHandle_t handle);

/**
 * @brief Create CUDA stream to be used internally.
 */
int kblasCreateStreams(kblasHandle_t handle, int nStreams);
/**
 * @brief Retrieve the CUDA stream used in the KBLAS handle structure.
 */
cudaStream_t kblasGetStream(kblasHandle_t handle);

/**
 * @brief Set the CUDA stream used in the KBLAS handle structure.
 */
void kblasSetStream(kblasHandle_t handle, cudaStream_t stream);

/**
 * @brief Retrieve the cuBLAS handle used in the KBLAS handle structure.
 */
cublasHandle_t kblasGetCublasHandle(kblasHandle_t handle);

/**
 * @brief Enable MAGMA support.
 */
int kblasEnableMagma(kblasHandle_t handle);

/**
 * @brief Set MAGMA queue in KBLAS handle.
 */
// int kblasSetMagma(kblasHandle_t handle, magma_queue_t queue);
const char* kblasGetErrorString(int error);

/** @} */
/** @} */


/**
 * @ingroup WSQUERY
 * @brief Allocate workspace buffers hosted in the handle structure.
 */
int kblasAllocateWorkspace(kblasHandle_t handle);

/**
 * @ingroup WSQUERY
 * @brief Free workspace buffers hosted in the handle structure.
 */
int kblasFreeWorkspace(kblasHandle_t handle);

// End of extern "C"
#ifdef __cplusplus
}
#endif

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
#include "batch_rand.h"
#include "batch_geqp.h"
#include "batch_ara.h"
#include "batch_pstrf.h"
#include "batch_block_copy.h"
#include "batch_transpose.h"

//============================================================================
//TLR routines
//============================================================================
#include "kblas_tlr.h"

#endif // _KBLAS_H_
