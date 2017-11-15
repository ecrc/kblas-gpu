/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file include/defs.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 2.0.0
 * @author Ali Charara
 * @author Ahmad Abdelfattah
 * @date 2017-11-13
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

#define MAX_NGPUS	(16)
#define MAX_STREAMS	(1)

const int gpu_lid[MAX_NGPUS] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
const int gpu_gid[MAX_NGPUS] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

#endif	// _KBLAS_DEFS_
