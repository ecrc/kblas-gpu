/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/kblas_gpu_timer.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 3.0.0
 * @author Wajih Halim Boukaram
 * @date 2018-11-14
 **/

#ifndef __KBLAS_GPU_TIMER__
#define __KBLAS_GPU_TIMER__

#include "kblas_error.h"

struct kblas_gpu_timer
{
	hipEvent_t start_event, stop_event;
	float elapsed_time;

    void init()
    {
        // #pragma omp critical (create_timer)
        {
            check_error( hipEventCreate(&start_event) );
            check_error( hipEventCreate(&stop_event ) );
            elapsed_time = 0;
        }
    }

    void destroy()
    {
        // #pragma omp critical (delete_timer)
        {
            check_error( hipEventDestroy(start_event));
            check_error( hipEventDestroy(stop_event ));
        }
    }

    void start(hipStream_t stream = 0)
    {
        check_error( hipEventRecord(start_event, stream) );
    }

	void recordEnd(hipStream_t stream = 0)
	{
		check_error( hipEventRecord(stop_event, stream) );
	}

    float stop(hipStream_t stream = 0)
    {
        check_error( hipEventSynchronize(stop_event) );

        float time_since_last_start;
        check_error( hipEventElapsedTime(&time_since_last_start, start_event, stop_event) );
        elapsed_time = (time_since_last_start * 0.001);

        return elapsed_time;
    }

    float elapsedSec()
    {
        return elapsed_time;
    }
};

#endif //__KBLAS_GPU_TIMER__
