/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/batch_svd/qr_kernels.cuh

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 2.0.0
 * @author Wajih Halim Boukaram
 * @date 2017-11-13
 **/

#ifndef __QR_KERNELS_H__
#define __QR_KERNELS_H__

#define OPS_PER_BLOCK  	2
#define QR_MAX_SMEM		32768
#define ROWS_PER_BLOCK 	64
#define MULT_REDUCTION_SMEM(warps, width) ((warps) * WARP_SIZE * (width) + (warps) * (width))

template<class T> struct QR_Config{};
template<> struct QR_Config<float>  { static const int HH_CB = 16; static const int HH_CB_PANEL = HH_CB / 2; static const int HH_MAX_ROWS = QR_MAX_SMEM / (HH_CB * sizeof(float)); };
template<> struct QR_Config<double> { static const int HH_CB =  8; static const int HH_CB_PANEL = HH_CB / 2; static const int HH_MAX_ROWS = QR_MAX_SMEM / (HH_CB * sizeof(double)); };
//template<> struct QR_Config<double> { static const int HH_CB =  4; static const int HH_CB_PANEL = HH_CB / 2; static const int HH_MAX_ROWS = QR_MAX_SMEM / (HH_CB * sizeof(double)); };

template<class T, int columns, int i, int BLOCK_SIZE, int PANEL_WIDTH>
inline __device__ void multreduction_gemv(T matrix_row[columns], T v, int threadId, int warp_tid, int warp_id, volatile T* smem)
{
    int thread_smem_offset = threadId * PANEL_WIDTH + (threadId * PANEL_WIDTH) / WARP_SIZE;
    volatile T* smem_w = smem + threadId + warp_id;
    const int warps = BLOCK_SIZE / WARP_SIZE;

    // Fill in shared memory
    #pragma unroll
    for(int j = 0; j < PANEL_WIDTH; j++)
        smem[j + thread_smem_offset] = v * matrix_row[i + j];

    if(warps != 1) __syncthreads();

    // Now do the partial reduction starting with the serial part
    #pragma unroll
    for(int j = 1; j < PANEL_WIDTH; j++)
        smem_w[0] += smem_w[warps * (WARP_SIZE + 1) * j];

	// Pad with zeros if the block size is not a power of two
	if(warps != 1 && warps != 2 && warps != 4 && warps != 8 && warps != 16 && warps != 32)
		smem_w[warps * (WARP_SIZE + 1)] = 0;

    if(warps != 1) __syncthreads();

    // Now the tree partial reduction
	if(BLOCK_SIZE >= 544) { if(threadId < 512) smem_w[0] += smem_w[16 * (WARP_SIZE + 1)]; __syncthreads(); }
	if(BLOCK_SIZE >= 288) { if(threadId < 256) smem_w[0] += smem_w[ 8 * (WARP_SIZE + 1)]; __syncthreads(); }
	if(BLOCK_SIZE >= 160) { if(threadId < 128) smem_w[0] += smem_w[ 4 * (WARP_SIZE + 1)]; __syncthreads(); }
    if(BLOCK_SIZE >= 96 ) { if(threadId < 64 ) smem_w[0] += smem_w[ 2 * (WARP_SIZE + 1)]; __syncthreads(); }
    if(threadId < 32)
    {
        if(BLOCK_SIZE >= 64) 					  smem_w[0] += smem_w[WARP_SIZE + 1];
        if(BLOCK_SIZE >= 32 && PANEL_WIDTH <= 16) smem_w[0] += smem_w[16];
        if(BLOCK_SIZE >= 16 && PANEL_WIDTH <= 8 ) smem_w[0] += smem_w[8 ];
        if(BLOCK_SIZE >=  8 && PANEL_WIDTH <= 4 ) smem_w[0] += smem_w[4 ];
        if(BLOCK_SIZE >=  4 && PANEL_WIDTH <= 2 ) smem_w[0] += smem_w[2 ];
        if(BLOCK_SIZE >=  2 && PANEL_WIDTH <= 1 ) smem_w[0] += smem_w[1 ];
    }

    // Synch to make sure all threads have access to the reductions
    if(warps != 1) __syncthreads();
}

// Apply a househoulder reflector H(k) to A(k:end, k:end)
// We will do some redundant work for the rows, to avoid branching
template<class T, int columns, int k, int BLOCK_SIZE, int PANEL_WIDTH>
struct qr_apply_househoulder_unroll
{
    static inline __device__
    void call(T matrix_row[columns], T v, T tau, int threadId, int warp_tid, int warp_id, volatile T* smem)
    {
		if(tau == 0) return;

        T tv = tau * v;
        const int limit = k + (columns - k) / PANEL_WIDTH * PANEL_WIDTH;
        const int remaining = columns - limit;

        if(k + PANEL_WIDTH <= columns)
        {
            // First compute w in shared memory as
            // w = A' * v
            multreduction_gemv<T, columns, k, BLOCK_SIZE, PANEL_WIDTH>(matrix_row, v, threadId, warp_tid, warp_id, smem);

            // Now shared memory has the vector w in the first PANEL_WIDTH entries
            // so we can apply the rank 1 update: A = A - tau * v * w'
            #pragma unroll
            for(int j = 0; j < PANEL_WIDTH; j++)
                matrix_row[k + j] -= tv * smem[j];

            if(BLOCK_SIZE != WARP_SIZE) __syncthreads();
        }

        if(remaining < 2)
        {
            #pragma unroll
            for(int i = limit; i < columns; i++)
            {
                T w = blockAllReduceSum<T, BLOCK_SIZE>(v * matrix_row[i], warp_tid, warp_id, smem);
                matrix_row[i] -= tv * w;
            }
        }
        else
            qr_apply_househoulder_unroll<T, columns, limit, BLOCK_SIZE, PANEL_WIDTH / 2>::call(matrix_row, v, tau, threadId, warp_tid, warp_id, smem);
    }
};

// Apply a househoulder reflector H(k) to A(k:end, k:end)
// and to A1(k, k:en)
template<class T, int columns, int k, int BLOCK_SIZE, int PANEL_WIDTH>
struct dtsqrt_apply_househoulder_unroll
{
    static inline __device__
    void call(T matrix_row[columns], T v, T tau, int threadId, int warp_tid, int warp_id, volatile T* smem, volatile T* A1, int A1_row)
    {
        T tv = tau * v;
        const int limit = k + (columns - k) / PANEL_WIDTH * PANEL_WIDTH;
        const int remaining = columns - limit;
        const int HH_CB = QR_Config<T>::HH_CB;

        if(k + PANEL_WIDTH <= columns)
        {
            // First compute w in shared memory as
            // w = A2(:, k:end)' * v
            multreduction_gemv<T, columns, k, BLOCK_SIZE, PANEL_WIDTH>(matrix_row, v, threadId, warp_tid, warp_id, smem);

            // Now shared memory has the vector w in the first PANEL_WIDTH entries
            // so we can compute w += A1(k, k:end) and then update
            // A1(k, k:end) -= tau * w';
            // A2(:, k:end) -= tau * v * w';
            if(threadId < PANEL_WIDTH)
            {
                smem[threadId] += A1[A1_row + (k + threadId) * HH_CB];
                A1[A1_row + (k + threadId) * HH_CB] -= tau * smem[threadId];
            }
            if(BLOCK_SIZE != WARP_SIZE) __syncthreads();

            #pragma unroll
            for(int j = 0; j < PANEL_WIDTH; j++)
                matrix_row[k + j] -= tv * smem[j];

            if(BLOCK_SIZE != WARP_SIZE) __syncthreads();
        }

        if(remaining < 2)
        {
            #pragma unroll
            for(int i = limit; i < columns; i++)
            {
                // The block reduction has a sync point so no need to sync here
                T a1_entry = A1[A1_row + i * HH_CB];
                T w = blockAllReduceSum<T, BLOCK_SIZE>(v * matrix_row[i], warp_tid, warp_id, smem);
                w += a1_entry;
                if(threadId == 0) A1[A1_row + i * HH_CB] -= tau * w;
                matrix_row[i] -= tv * w;
            }
        }
        else
            dtsqrt_apply_househoulder_unroll<T, columns, limit, BLOCK_SIZE, PANEL_WIDTH / 2>::call(matrix_row, v, tau, threadId, warp_tid, warp_id, smem, A1, A1_row);
    }
};

template<class T, int k, int columns, int BLOCK_SIZE>
struct qr_panel_unroll
{
    static inline __device__
    void call(T matrix_row[columns], T* tau, int threadId, int warp_tid, int warp_id, volatile T* smem, volatile T* pivot)
    {
		const int HH_CB_PANEL = QR_Config<T>::HH_CB_PANEL;

        qr_panel_unroll<T, k - 1, columns, BLOCK_SIZE>::call(matrix_row, tau, threadId, warp_tid, warp_id, smem, pivot);

        // Save the pivot in shared memory at this point since we do a sync during
        // the reduction for computing the norm of the vector
        T temp_pivot = matrix_row[k];
        if(threadId == k) *pivot = temp_pivot;

        ///////////////////////////////////////////////////
        // Get the househoulder vector for this column
        ///////////////////////////////////////////////////
        T val = (threadId >= k ? matrix_row[k] * matrix_row[k] : 0);
        T beta = sqrt(blockAllReduceSum<T, BLOCK_SIZE>(val, warp_tid, warp_id, smem));

		if(beta != 0)
		{
			// Get the pivot by reading from another thread's registers
			T alpha = *pivot;
			if(alpha >= 0) beta *= -1;

			T tau_k = (beta - alpha) / beta;
			if(threadId == 0) tau[k] = tau_k;

			// Doing some unnecessary work here by masking, but it should make
			// the GPU version work better by avoiding conditionals on threads
			// that are going to be idle anyway in the update phase
			T scal = (T)1.0 / (alpha - beta);
			T v = (threadId > k ? scal * matrix_row[k] : 0);
			if(threadId == k) { v = 1; matrix_row[k] = beta; }

			///////////////////////////////////////////////////
			// Now do the update
			///////////////////////////////////////////////////
			qr_apply_househoulder_unroll<T, columns, k + 1, BLOCK_SIZE, HH_CB_PANEL>::call(matrix_row, v, tau_k, threadId, warp_tid, warp_id, smem);

			///////////////////////////////////////////////////
			// Finally, store the reflector in the lower part of A
			///////////////////////////////////////////////////
			if(threadId >= k + 1) { matrix_row[k] = v; }
		}
		else
		{
			if(threadId == 0)
				tau[k] = 0;
		}
    }
};

template<class T, int k, int columns, int BLOCK_SIZE>
struct dtsqrt_panel_unroll
{
    static inline __device__
    void call(T matrix_row[columns], T* tau, int threadId, int warp_tid, int warp_id, volatile T* smem, volatile T* A1)
    {
        dtsqrt_panel_unroll<T, k - 1, columns, BLOCK_SIZE>::call(matrix_row, tau, threadId, warp_tid, warp_id, smem, A1);

		const int HH_CB = QR_Config<T>::HH_CB;
		const int HH_CB_PANEL = QR_Config<T>::HH_CB_PANEL;

        ///////////////////////////////////////////////////
        // Get the househoulder vector for this column
        ///////////////////////////////////////////////////
        T alpha = A1[k + k * HH_CB];
        T beta = sqrt(blockAllReduceSum<T, BLOCK_SIZE>(matrix_row[k] * matrix_row[k], warp_tid, warp_id, smem) + alpha * alpha);
		if(beta != 0)
		{
			if(alpha >= 0) beta *= -1;

			T tau_k = (beta - alpha) / beta;
			if(threadId == 0)
			{
				tau[k] = tau_k;
				A1[k + k * HH_CB] = beta;
			}

			T scal = (T)1.0 / (alpha - beta);
			T v = scal * matrix_row[k];

			///////////////////////////////////////////////////
			// Now do the update
			///////////////////////////////////////////////////
			dtsqrt_apply_househoulder_unroll<T, columns, k + 1, BLOCK_SIZE, HH_CB_PANEL>::call(matrix_row, v, tau_k, threadId, warp_tid, warp_id, smem, A1, k);

			///////////////////////////////////////////////////
			// Finally, store the reflector in the lower part of A
			///////////////////////////////////////////////////
			matrix_row[k] = v;
		}
		else
		{
			if(threadId == 0)
				tau[k] = 0;
		}
    }
};

template<class T, int k, int columns, int BLOCK_SIZE>
struct qrunpack_loop_unroll
{
    static inline __device__
    void call(T matrix_row[columns], T* tau, int threadId, int warp_tid, int warp_id, volatile T* smem)
    {
        T v = (threadId < k ? 0 : matrix_row[k]);
        if(threadId == k) v = 1;

		const int HH_CB_PANEL = QR_Config<T>::HH_CB_PANEL;

        qr_apply_househoulder_unroll<T, columns, k + 1, BLOCK_SIZE, HH_CB_PANEL>::call(matrix_row, v, tau[k], threadId, warp_tid, warp_id, smem);

        T tmp = (threadId == k ? tau[k] : 1);
        matrix_row[k] = (threadId <= k ? 1 - tmp : -tau[k] * matrix_row[k]);

        qrunpack_loop_unroll<T, k - 1, columns, BLOCK_SIZE>::call(matrix_row, tau, threadId, warp_tid, warp_id, smem);
    }
};

template<class T, int columns, int BLOCK_SIZE>
inline __device__
void qr_apply_househoulder_panel(T matrix_row[columns], T v, T tau, int threadId, volatile T* smem)
{
    int warp_tid = threadId % WARP_SIZE;
    int warp_id = threadId / WARP_SIZE;

    qr_apply_househoulder_unroll<T, columns, 0, BLOCK_SIZE, columns>::call(matrix_row, v, tau, threadId, warp_tid, warp_id, smem);
}

template<class T, int columns, int BLOCK_SIZE>
inline __device__
void qr_householder_panel(T matrix_row[columns], T* tau, int threadId, volatile T* smem, volatile T* pivot)
{
    int warp_tid = threadId % WARP_SIZE;
    int warp_id = threadId / WARP_SIZE;

    qr_panel_unroll<T, columns - 1, columns, BLOCK_SIZE>::call(matrix_row, tau, threadId, warp_tid, warp_id, smem, pivot);
}

// DTSQRT of a panel stored in registers (matrix_row) with the triangular portion stored in A1
template<class T, int columns, int BLOCK_SIZE>
inline __device__
void dtsqrt_panel(T matrix_row[columns], T* tau, int threadId, volatile T* smem, volatile T* A1)
{
    int warp_tid = threadId % WARP_SIZE;
    int warp_id = threadId / WARP_SIZE;

    dtsqrt_panel_unroll<T, columns - 1, columns, BLOCK_SIZE>::call(matrix_row, tau, threadId, warp_tid, warp_id, smem, A1);
}

// DTSQRT of a panel stored in registers (matrix_row) with the triangular portion stored in A1
template<class T, int columns, int BLOCK_SIZE>
inline __device__
void dtsqrt_apply_panel(T A2_matrix_row[columns], T v, T tau_k, int threadId, volatile T* smem, volatile T* A1, int A1_row)
{
    int warp_tid = threadId % WARP_SIZE;
    int warp_id = threadId / WARP_SIZE;

    dtsqrt_apply_househoulder_unroll<T, columns, 0, BLOCK_SIZE, columns>::call(A2_matrix_row, v, tau_k, threadId, warp_tid, warp_id, smem, A1, A1_row);
}

template<class T, int columns, int BLOCK_SIZE>
inline __device__
void qr_unpackQ_panel(T matrix_row[columns], T* tau, int threadId, volatile T* smem)
{
    // First handle the last column of A outside the loop
    // since it doesn't involve a househoulder update
    matrix_row[columns - 1] = (threadId < columns ? 0 : -tau[columns - 1] * matrix_row[columns - 1]);
    if(threadId == columns - 1)
        matrix_row[columns - 1] = (T)1.0 - tau[columns - 1];

    int warp_tid = threadId % WARP_SIZE;
    int warp_id = threadId / WARP_SIZE;

    // Now go through the columns of A starting from the end and going to the
    // first column, applying the househoulder vectors stored in the column to
    // the submatrix of A that is to the right of the column
    qrunpack_loop_unroll<T, columns - 2, columns, BLOCK_SIZE>::call(matrix_row, tau, threadId, warp_tid, warp_id, smem);
}

// Specializations to end loop recursions
template<class T, int columns, int k, int BLOCK_SIZE>
struct qr_apply_househoulder_unroll<T, columns, k, BLOCK_SIZE, 0> {
    static inline __device__
    void call(T matrix_row[columns], T v, T tau, int threadId, int warp_tid, int warp_id, volatile T* smem) {}
};

template<class T, int columns, int k, int BLOCK_SIZE>
struct dtsqrt_apply_househoulder_unroll<T, columns, k, BLOCK_SIZE, 0> {
    static inline __device__
    void call(T matrix_row[columns], T v, T tau, int threadId, int warp_tid, int warp_id, volatile T* smem, volatile T* A1, int A1_row){}
};

template<class T, int columns, int BLOCK_SIZE>
struct qr_panel_unroll<T, -1, columns, BLOCK_SIZE> {
    static inline __device__
    void call(T matrix_row[columns], T* tau, int threadId, int warp_tid, int warp_id, volatile T* smem, volatile T* pivot) {}
};

template<class T, int columns, int BLOCK_SIZE>
struct dtsqrt_panel_unroll<T, -1, columns, BLOCK_SIZE> {
    static inline __device__
    void call(T matrix_row[columns], T* tau, int threadId, int warp_tid, int warp_id, volatile T* smem, volatile T* A1) {}
};

template<class T, int columns, int BLOCK_SIZE>
struct qrunpack_loop_unroll <T, -1, columns, BLOCK_SIZE> {
    static inline __device__
    void call(T matrix_row[columns], T* tau, int threadId, int warp_tid, int warp_id, volatile T* smem) {}
};

#endif
