/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file include/kblas_defs.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Ali Charara
 * @author Ahmad Abdelfattah
 * @date 2018-11-14
 **/


#ifndef	_KBLAS_DEFS_
#define _KBLAS_DEFS_

#define KBLAS_Lower 'L'
#define KBLAS_Upper 'U'
#define KBLAS_Left 'L'
#define KBLAS_Right 'R'
#define KBLAS_Trans 'T'
#define KBLAS_NoTrans 'N'
#define KBLAS_Unit 'U'
#define KBLAS_NonUnit 'N'
#define KBLAS_Symm 'S'
#define KBLAS_NonSymm 'N'

// ----------------------------------------
#define KBLAS_Success 1
#define KBLAS_UnknownError 0
#define KBLAS_NotSupported -1
#define KBLAS_NotImplemented -2
#define KBLAS_cuBLAS_Error -3
#define KBLAS_WrongConfig -4
#define KBLAS_CUDA_Error -5
#define KBLAS_InsufficientWorkspace -6
#define KBLAS_Error_Allocation -7
#define KBLAS_Error_Deallocation -8
#define KBLAS_Error_NotInitialized -9
#define KBLAS_Error_WrongInput -10
#define KBLAS_MAGMA_Error -11
#define KBLAS_SVD_NoConvergence -12


// ----------------------------------------
#define MAX_NGPUS	(16)
#define MAX_STREAMS	(1)
#define KBLAS_NSTREAMS	10

const int gpu_lid[MAX_NGPUS] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
const int gpu_gid[MAX_NGPUS] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

#endif	// _KBLAS_DEFS_
