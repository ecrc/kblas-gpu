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
 * @version 3.0.0
 * @author Ali Charara
 * @author Wajih Halim Boukaram
 * @author Ahmad Abdelfattah
 * @date 2018-11-14
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
hipStream_t kblasGetStream(kblasHandle_t handle);

/**
 * @brief Set the CUDA stream used in the KBLAS handle structure.
 */
void kblasSetStream(kblasHandle_t handle, hipStream_t stream);

/**
 * @brief Retrieve the cuBLAS handle used in the KBLAS handle structure.
 */
hipblasHandle_t kblasGetCublasHandle(kblasHandle_t handle);

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




#include "kblas_l3.h"


#endif // _KBLAS_H_
