
__forceinline__ __device__ 
void gen_househoulder(Real& v, Real x1, int thread_x, int k, Real& alpha, Real& beta)
{
    Real norm_v = sqrt(warpAllReduceSum(v * v));
    alpha = (x1 > 0 ? -1 : 1) * norm_v;
    beta  = (alpha == 0 ? 0 : (Real)1.0 / (abs(alpha) * (abs(alpha) + abs(x1))));
    if(thread_x == k) v -= alpha;
}
    
__global__ 
void blockBidiagKernel(Real* __restrict__ M, Real* __restrict__ D, Real* __restrict__ E, int rows, int cols, int num_ops)
{
    extern __shared__ char sdata[];

    int op_id = blockIdx.x;
    if(op_id >= num_ops) return;
    
    int thread_x = threadIdx.x;
    
    Real* shared_matrix = (Real*)sdata;

    Real* m_op = M + rows * cols * op_id;
    Real* d_op = D + cols * op_id;
    Real* e_op = E + cols * op_id;
    
    int srows = rows + 1;
    // Load the data in from gmem into padded smem
    for(int i = 0; i < cols; i++)
        shared_matrix[thread_x + i * srows] = m_op[thread_x + i * rows];
    
    for(int k = 0; k < cols - 1; k++)
    {
        // Column reflector
        Real v = (thread_x < k ? 0 : shared_matrix[thread_x + k * srows]);
        Real x1 = shared_matrix[k + k * srows];
        Real alpha, beta;
        gen_househoulder(v, x1, thread_x, k, alpha, beta);
        if(thread_x >= k) shared_matrix[thread_x + k * srows] = v;
        
        d_op[k] = alpha;
        
        // Apply column reflector
        for(int i = k + 1; i < cols; i++)
        {
            Real w = beta * shared_matrix[thread_x + i * srows] * v;
            w = warpAllReduceSum(w);
            shared_matrix[thread_x + i * srows] -= v * w;
        }

        if(k < cols - 2)
        {
            // Row reflector
            v = (thread_x <= k ? 0 : shared_matrix[k + thread_x * srows]);
            x1 = shared_matrix[k + (k + 1) * srows];
            gen_househoulder(v, x1, thread_x, k+1, alpha, beta);
            if(thread_x > k) shared_matrix[k + thread_x * srows] = v;
            
            e_op[k] = alpha;
            
            // Apply row reflector
            for(int i = k + 1; i < cols; i++)
            {
                Real w = beta * shared_matrix[i + thread_x * srows] * v;
                w = warpAllReduceSum(w);
                shared_matrix[i + thread_x * srows] -= v * w;
            }
        }
        else if(k == cols - 2)
            e_op[k] = shared_matrix[k + (k+1)*srows];
    }
    d_op[cols - 1] = shared_matrix[cols - 1 + (cols - 1) * srows];
    e_op[cols - 1] = 0;
    
    // Store reflectors in gmem 
    for(int i = 0; i < cols; i++)
        m_op[thread_x + i * rows] = shared_matrix[thread_x + i * srows];
}

 
__global__ 
void blockBidiagKernel2(Real* __restrict__ M, Real* __restrict__ D, Real* __restrict__ E, int rows, int cols, int num_ops)
{
    extern __shared__ char sdata[];

    int op_id = blockIdx.x;
    if(op_id >= num_ops) return;
    
    int thread_x = threadIdx.x;
    int srows = rows + 1;
	
    Real* shared_matrix = (Real*)sdata;
	Real* v = (Real*)&shared_matrix[srows * cols];
	Real* w = (Real*)&v[rows];
	
    Real* m_op = M + rows * cols * op_id;
    Real* d_op = D + cols * op_id;
    Real* e_op = E + cols * op_id;
    
    
    // Load the data in from gmem into padded smem
    for(int i = 0; i < cols; i++)
        shared_matrix[thread_x + i * srows] = m_op[thread_x + i * rows];
    
    for(int k = 0; k < cols - 1; k++)
    {
        // Column reflector
        v[thread_x] = (thread_x < k ? 0 : shared_matrix[thread_x + k * srows]);
        Real x1 = shared_matrix[k + k * srows];
        Real alpha, beta;
        gen_househoulder(v[thread_x], x1, thread_x, k, alpha, beta);
        if(thread_x >= k) shared_matrix[thread_x + k * srows] = v[thread_x];
        
        d_op[k] = alpha;
        
        // Apply column reflector
		w[thread_x] = 0;
		for(int i = k; i < rows; i++)
			w[thread_x] += shared_matrix[i + thread_x * srows] * v[i];

		for(int i = k + 1; i < cols; i++)			
			shared_matrix[thread_x + i * srows] -= beta * v[thread_x] * w[i];

        if(k < cols - 2)
        {
            // Row reflector
            v[thread_x] = (thread_x <= k ? 0 : shared_matrix[k + thread_x * srows]);
            x1 = shared_matrix[k + (k + 1) * srows];
            gen_househoulder(v[thread_x], x1, thread_x, k+1, alpha, beta);
            if(thread_x > k) shared_matrix[k + thread_x * srows] = v[thread_x];
            
            e_op[k] = alpha;
            
            // Apply row reflector
			w[thread_x] = 0;
			for(int i = k + 1; i < cols; i++)
				w[thread_x] += shared_matrix[thread_x + i * srows] * v[i];
			
            for(int i = k + 1; i < cols; i++)
                shared_matrix[thread_x + i * srows] -= beta * v[i] * w[thread_x];
		}
        else if(k == cols - 2)
            e_op[k] = shared_matrix[k + (k+1)*srows];
    }
    d_op[cols - 1] = shared_matrix[cols - 1 + (cols - 1) * srows];
    e_op[cols - 1] = 0;
    
    // Store reflectors in gmem 
    for(int i = 0; i < cols; i++)
        m_op[thread_x + i * rows] = shared_matrix[thread_x + i * srows];
}

void batch_bidiag(Real* M, Real* D, Real* E, int rows, int cols, int num_ops)
{
    int block_size = WARP_SIZE * iDivUp(rows, WARP_SIZE);
	
	dim3 dimBlock(block_size, 1);
	dim3 dimGrid(num_ops, 1);
	
	size_t smem_needed = ((rows + 1) * cols + cols + rows) * sizeof(Real);
    
    blockBidiagKernel2<<< dimGrid, dimBlock, smem_needed >>>(M, D, E, rows, cols, num_ops);
}


void batch_bidiag_cpu(Real* M, Real* D, Real* E, Real* tauQ, Real* tauP, int rows, int cols, int num_ops)
{
	#pragma omp parallel for
	for(int i = 0; i < num_ops; i++)
	{
		Real* m_op = M + i * rows * cols;
		Real* d_op = D + i * cols;
		Real* e_op = E + i * cols;
		Real* tq_op = tauQ + i * cols;
		Real* tp_op = tauP + i * cols;
        
		#ifdef DOUBLE_PRECISION
		int info = LAPACKE_dgebrd(LAPACK_COL_MAJOR, rows, cols, m_op, rows, d_op, e_op, tq_op, tp_op);
		#else
		int info = LAPACKE_sgebrd(LAPACK_COL_MAJOR, rows, cols, m_op, rows, d_op, e_op, tq_op, tp_op);	
		#endif
        if(info != 0) printf("Error %d computing bidiag of op %d\n", info, i);
	}
}
