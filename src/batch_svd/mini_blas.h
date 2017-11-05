#ifndef __MINI_BLAS_H__
#define __MINI_BLAS_H__

#include <string.h>
#include "debug_routines.h"

#define HLIB_USE_MKL

#ifdef HLIB_USE_MKL
#include <mkl.h>
#endif

// Does the dot product of two columns of a matrix
inline Real dot_product(Real* matrix, int col_a, int col_b, int lda, int m, int n)
{
    Real dp = 0;
    for(int i = 0; i < m; i++)
        dp += matrix[i + col_a * lda] * matrix[i + col_b * lda];
    return dp;
}

inline Real dot_product(Real* a, Real* b, int m)
{
    Real dp = 0;
    for(int i = 0; i < m; i++)
        dp += a[i] * b[i];
    return dp;
}

inline Real norm_vec(Real* v, int m)
{
    Real dp = 0;
    for(int i = 0; i < m; i++)
        dp += v[i] * v[i];
    return sqrt(dp);
}

inline Real norm_vec2(Real* v, int m)
{
    Real dp = 0;
    for(int i = 0; i < m; i++)
        dp += v[i] * v[i];
    return dp;
}

inline void transpose(Real* trans, Real* matrix, int ld, int m, int n)
{
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            trans[j + i * n] = matrix[i + j * ld];
}

inline int svd_jacobi_core(Real* matrix, Real* S, Real* V, int lda, int m, int n, int* dot_products, int* total_rotations)
{
    int count = 1, sweep = 0;
    int sweep_max = 5 * n;
    if(sweep_max < 12) sweep_max = 12;
    
    Real rtol = REAL_EPSILON * 10 * m;
    memset(V, 0, sizeof(Real) * n * n);
    // Set V to the identity matrix and save the norms of the columns to use as error thresholds
    for(int i = 0; i < n; i++)
    {
        S[i] = REAL_EPSILON * sqrt(dot_product(matrix, i, i, lda, m, n));
        V[i + i * n] = 1;
    }
    //printDenseMatrix(matrix, m, n, 12, "A1");
    int dps = 0, rots = 0;
    
    while(count > 0 && sweep < sweep_max)
    {
        // Keep track of the number of column combinations that aren't orthogonal
        count = n * (n - 1) / 2;
        for(int j = 0; j < n - 1; j++)
        {
            for(int k = j + 1; k < n; k++)
            {
                // compute [a p; p b]=(i,j) submatrix of A'*A
                Real a = dot_product(matrix, j, j, lda, m, n);
                Real b = dot_product(matrix, k, k, lda, m, n);
                Real p = dot_product(matrix, j, k, lda, m, n);
				
				//printf("%d %d -- %e %e %e\n", j, k, a, b, p);
                
                Real a_norm = sqrt(a), b_norm = sqrt(b);
                
                // Make sure the singular values are sorted before outputting them
                int sorted = (a >= b);
                dps++;
                
                // If this pair of columns is orthogonal skip it
                if(sorted && (fabs(p) <= rtol * a_norm * b_norm || a_norm < S[j] || b_norm < S[k]))
                {
                    count--;
                    continue;
                }
                rots++;
                
                // Calculate the rotation matrix that will diagonalize the submatrix of A'*A
                Real c, s;
                Real q = a - b;
                Real v = hypot(2 * p, q);
                
                if(v == 0 || !sorted)
                {
                    c = 0;
                    s = 1;
                }
                else
                {
                    c = sqrt((v + q) / (2 * v));
                    s = p / (v * c);
                }
                
                // Apply the rotations to the matrix and its V factor
                for(int i = 0; i < m; i++)
                {
                    Real m_ij = matrix[i + j * lda];
                    Real m_ik = matrix[i + k * lda];
                    
                    matrix[i + j * lda] = c * m_ij + s * m_ik;
                    matrix[i + k * lda] = -s * m_ij + c * m_ik;
                }
                
                for(int i = 0; i < n; i++)
                {
                    Real v_ij = V[i + j * n];
                    Real v_ik = V[i + k * n];
                    
                    V[i + j * n] = c * v_ij + s * v_ik;
                    V[i + k * n] = -s * v_ij + c * v_ik;
                }
                
                // Update the error estimates
                c = fabs(c); s = fabs(s);
                q = S[j];
                S[j] = c * S[j] + s * S[k];
                S[k] = s * q    + c * S[k];
                /*c *= c; s *= s;
                q = S[j] * S[j]; v = S[k] * S[k];
                S[j] = sqrt(c * q + s * v);
                S[k] = sqrt(s * q + c * v);*/
            }
        }
        sweep++;
    }
    
	int ret_val = 0;
    if(sweep == sweep_max)
	{
		ret_val = -1;
        fprintf(stderr, "\033[1;31m Warning: svd max iterations reached\033[0m\n");
	}

    for(int i = 0; i < n; i++)
    {
        S[i] = sqrt(dot_product(matrix, i, i, lda, m, n));
        if(S[i] != 0.0)
            for(int j = 0; j < m; j++)
                matrix[j + i * lda] /= S[i];
    }
    if(total_rotations) *total_rotations = rots;
    if(dot_products) *dot_products = dps;
	
	return ret_val;
    /*printDenseMatrix(matrix, m, n, 12, "U1");
    printDenseMatrix(S, n, 1, 12, "S1");
    printDenseMatrix(V, n, n, 12, "V1");*/
}

inline void svd_osbj_pair(Real* matrix, Real* S, int m, int n, Real tolerance)
{
    int count = 1, sweep = 0;
    int sweep_max = std::max(5 * n, 12);
	for(int i = 0; i < n; i++)
        S[i] = sqrt(dot_product(matrix, i, i, m, m, n));
	
    while(count > 0 && sweep < sweep_max)
    {
        // Keep track of the number of column combinations that aren't orthogonal
        count = n * (n - 1) / 2;
        for(int j = 0; j < n - 1; j++)
        {
            for(int k = j + 1; k < n; k++)
            {
                // compute [a p; p b]=(i,j) submatrix of A'*A
                Real a = dot_product(matrix, j, j, m, m, n);
                Real b = dot_product(matrix, k, k, m, m, n);
                Real p = dot_product(matrix, j, k, m, m, n);

                Real a_norm = sqrt(a), b_norm = sqrt(b);
                
                // Make sure the singular values are sorted before outputting them
                int sorted = (a >= b);
                
                // If this pair of columns is orthogonal skip it
                if(sorted && (fabs(p) <= tolerance * a_norm * b_norm || a_norm < REAL_EPSILON * S[j] || b_norm < REAL_EPSILON * S[k]))
                {
                    count--;
                    continue;
                }
                
                // Calculate the rotation matrix that will diagonalize the submatrix of A'*A
                Real c, s;
                Real q = a - b;
                Real v = hypot(2 * p, q);
                
                if(v == 0 || !sorted)
                {
                    c = 0;
                    s = 1;
                }
                else
                {
                    c = sqrt((v + q) / (2 * v));
                    s = p / (v * c);
                }
                
                // Apply the rotations to the matrix and its V factor
                for(int i = 0; i < m; i++)
                {
                    Real m_ij = matrix[i + j * m];
                    Real m_ik = matrix[i + k * m];
                    
                    matrix[i + j * m] = c * m_ij + s * m_ik;
                    matrix[i + k * m] = -s * m_ij + c * m_ik;
                }

                c *= c; s *= s;
                q = S[j] * S[j]; v = S[k] * S[k];
                S[j] = sqrt(c * q + s * v);
                S[k] = sqrt(s * q + c * v);
            }
        }
        sweep++;
    }

    if(sweep == sweep_max)
        fprintf(stderr, "\033[1;31m Warning: inner svd max iterations reached\033[0m\n");
}

inline Real getOrthogMeasure(Real* matrix, int m, int nb)
{
	// First compute c = A_j * e
	// where e = ones(nb, 1)
	// Have to implicitily normalize A_j 
	std::vector<Real> c(m, 0), norm_Ai(nb, 0);
	for(int j = 0; j < nb; j++)
		for(int i = 0; i < m; i++)
			norm_Ai[j] += matrix[i + (j + nb) * m] * matrix[i + (j + nb) * m];
	for(int j = 0; j < nb; j++)
		norm_Ai[j] = sqrt(norm_Ai[j]);
	
	for(int i = 0; i < m; i++)
		for(int j = 0; j < nb; j++)
			c[i] += matrix[i + (j + nb) * m] / norm_Ai[j];
		
	// Now we get w = norm(A_i^t * c) / norm(e)
	// where norm(e) = sqrt(nb)
	// Have to implicitily normalize A_i 
	Real w_norm = 0;
	for(int j = 0 ; j < nb; j++)
	{
		Real w_j = 0, norm_ai = 0;
		for(int i = 0; i < m; i++)
		{
			norm_ai += matrix[i + j * m] * matrix[i + j * m];
			w_j += matrix[i + j * m] * c[i];
		}
		w_norm += w_j * w_j / norm_ai;
	}
	return sqrt(w_norm / nb);
}

inline void svd_osbj(Real* matrix, Real* S, int lda, int m, int n, int nb)
{
	int num_blocks = (n + nb - 1) / nb;
	int sweep_max = std::max(5 * num_blocks, 12);
	Real outer_tol = m * 1e-12;
	Real inner_tol = m * REAL_EPSILON;
	// Real prev_max_w = 1;
	int converged = 0, sweeps = 0;
	
	std::vector<Real> block_matrix(m * 2 * nb);
	
	while(converged == 0 && sweeps < sweep_max)
	{
		Real max_w = 0;
		
		for(int block_i = 0; block_i < num_blocks - 1; block_i++)
		{
			// Copy block_i into a buffer
			int i1 = block_i * nb;
			for(int i = 0; i < m; i++)
				for(int j = 0; j < nb; j++)
					block_matrix[i + j * m] = matrix[i + (j + i1) * lda];
				
			for(int block_j = block_i + 1; block_j < num_blocks; block_j++)
			{
				int j1 = block_j * nb, j2 = std::min(j1 + nb, n);
				int ncols = nb + j2 - j1;
				
				// Copy block_j into a buffer
				for(int i = 0; i < m; i++)
					for(int j = 0; j < nb; j++)
						block_matrix[i + (nb + j) * m] = (j + j1 < n ? matrix[i + (j + j1) * lda] : 0);
					
				// Get the orthogonalization measure w of the two blocks
				Real w_ij = getOrthogMeasure(&block_matrix[0], m, nb);
				max_w = std::max(max_w, w_ij);
				// Perform the SVD of the copied matrix
				svd_osbj_pair(&block_matrix[0], S, m, ncols, inner_tol);
				
				// Copy the block_j out of the buffer back into memory
				for(int i = 0; i < m; i++)
					for(int j = 0; j < nb; j++)
						if(j1 + j < n) 
							matrix[i + (j + j1) * lda] = block_matrix[i + (nb + j) * m];
				
			}
			
			// Copy block_i out of the buffer back into memory
			for(int i = 0; i < m; i++)
				for(int j = 0; j < nb; j++)
					matrix[i + (j + i1) * lda] = block_matrix[i + j * m];
		}
		sweeps++;
		//if(fabs(max_w - prev_max_w) <= max_w * REAL_EPSILON) 
		//	converged = 1;
		if(max_w < outer_tol)
			converged = 1;
		//prev_max_w = max_w;
	}
	//printf("%d sweeps\n", sweeps);
    if(sweeps == sweep_max)
        fprintf(stderr, "\033[1;31m Warning: outer svd max iterations reached\033[0m\n");
	
    for(int i = 0; i < n; i++)
    {
        S[i] = sqrt(dot_product(matrix, i, i, lda, m, n));
        if(S[i] != 0.0)
            for(int j = 0; j < m; j++)
                matrix[j + i * lda] /= S[i];
    }
}

inline void svd_jacobi(Real* matrix, Real* S, Real* V, int lda, int m, int n, int* dot_products = NULL, int* total_rotations = NULL)
{
	if(m == 0 || n == 0) 
        return;
#ifndef HLIB_USE_MKL
    if(m >= n)
        svd_jacobi_core(matrix, S, V, lda, m, n, dot_products, total_rotations);
    else
    {
        Real* matrix_transpose = (Real*)malloc(sizeof(Real) * m * n);
        assert(matrix_transpose);
        transpose(matrix_transpose, matrix, lda, m, n);
        
        int ret_val = svd_jacobi_core(matrix_transpose, S, V, n, n, m, dot_products, total_rotations);
		/*
		if(ret_val < 0)
		{
			printDenseMatrix(matrix, m, n, 12, "A");
			printDenseMatrix(matrix_transpose, n, m, 12, "U");
			printDenseMatrix(S, n, 1, 12, "S");
			printDenseMatrix(V, n, m, 12, "V");
		}
        */
        // copy V over to the original matrix and set the extra columns to zero
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < m; j++)
                matrix[i + j * m] = V[i + j * m];
            for(int j = m; j < n; j++)
            {
                matrix[i + j * m] = 0;
                // also zero out the extra singular values
                S[j] = 0;
            }
        }
        // Copy U (stored in the transpose matrix) to V and set the extra columns to 0
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < m; j++)
                V[i + j * n] = matrix_transpose[i + j * n];
            for(int j = m; j < n; j++)
                V[i + j * n] = 0;
        }
        /*printDenseMatrix(matrix, m, n, 12, "U");
        printDenseMatrix(S, n, 1, 12, "S");
        printDenseMatrix(V, n, n, 12, "V");*/
            
        free(matrix_transpose);
    }
#else
	std::vector<Real> superb(n);
	#ifdef DOUBLE_PRECISION
	LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'O', 'A', m, n, matrix, lda, S, NULL, m, V, n, &superb[0]);
	#else
	LAPACKE_sgesvd(LAPACK_COL_MAJOR, 'O', 'A', m, n, matrix, lda, S, NULL, m, V, n, &superb[0]);
	#endif
#endif
}

// QR using classical Gram-Schmidt orthogonalization with iterative
// reorthogonalization - the reorth loop usually is repeated only 
// once if reorthogonalization is needed
inline void icgs_qr(Real* A, int lda, int m, int n, Real* R)
{
    const Real kappa = 2;
    
    Real* p = (Real*)malloc(sizeof(Real) * m);
    assert(p);
    for(int k = 0; k < n; k++)
    {
        Real t = norm_vec(A + k * lda, m);
        int reorth = 1;
        //int count = 0;
        while(reorth == 1)
        {
            //count++;
            for(int i = 0; i < m; i++)
                p[i] = A[i + k * lda];
            
            for(int i = 0; i < k; i++)
            {
                Real s = dot_product(A + i * lda, p, m);
                R[i + k * n] += s;
                for(int j = 0; j < m; j++)
                    A[j + k * lda] -= s * A[j + i * lda];
            }
            Real tt = norm_vec(A + k * lda, m);
            if(tt < kappa * REAL_EPSILON * t || tt > t / kappa)
            {
                reorth = 0;
                if(tt < kappa * REAL_EPSILON * t)
                    tt = 0;
            }
            t = tt;
        }
        //printf("Count: %d\n", count);
        R[k + k * n] = t;
        t = (t > kappa * REAL_EPSILON ? 1.0 / t : 0); 
        for(int i = 0; i < m; i++)
            A[i + k * lda] *= t;
    }
    free(p);
}

// Modified Gram-Schmidt orthogonalizations
inline void qr_mgs(Real* A, int lda, int m, int n, Real* R)
{
    for(int k = 0; k < n; k++)
    {
        Real t = norm_vec(A + k * lda, m);
        int reorth = 1;
        Real tt = 0;
        //int count = 0;
        while(reorth == 1)
        {
            //count++;
            for(int i = 0; i < k; i++)
            {
                Real s = dot_product(A, i, k, lda, m, n);
                R[i + k * n] += s;
                for(int j = 0; j < m; j++)
                    A[j + k * lda] -= s * A[j + i * lda];
            }
            tt = norm_vec(A + k * lda, m);
            if(tt > 10 * REAL_EPSILON * t && tt < t / 10)
                t = tt;
            else
            {
                reorth = 0;
                if(tt < 10 * REAL_EPSILON * t)
                    tt = 0;
            }
        }
        //printf("Count: %d\n", count);
        R[k + k * n] = tt;
        tt = (tt * REAL_EPSILON != 0 ? 1.0 / tt : 0); 
        for(int i = 0; i < m; i++)
            A[i + k * lda] *= tt;
    }
}

// Apply a househoulder reflector H(k) to the left of A(k:end, k:end) 
// We will do some redundant work for the rows, to avoid branching
inline void qr_apply_househoulder(Real* A, int lda, int m, int n, Real* v, Real* w, Real tau, int k)
{
    // First compute w
    // w = A' * v
    // This part is going to be difficult to parallelize since the number of columns
    // is usually less than the warp size - going to ignore it for now
    for(int i = k; i < n; i++)
    {
        w[i] = 0;
        for(int j = 0; j < m; j++)
            w[i] += A[j + i * lda] * v[j];
    }
    
    // Now we update A as:
    // A = A - tau * v * w'
    // i should start from k, but we're doing things using masks
    // to avoid branching on the GPU
    for(int i = 0; i < m; i++)
        for(int j = k; j < n; j++)
            A[i + j * lda] -= tau * v[i] * w[j];
}

// Calculate the qr decomposition of a mxn matrix A
// The upper triangular part of A is overwritten by R and
// the lower part contains the househoulder reflectors
// v and w are temporary workspace of size m and n respectively 
// that should be allocated by the caller
// tau is a nx1 vector of the househoulder vector coefficients
inline void qr_householder(Real* A, int lda, int m, int n, Real* v, Real* w, Real* tau)
{
	if(m <= 0 || n <= 0) return;
#if 1//ndef HLIB_USE_MKL
    int k_it = (m - 1 < n ? m - 1 : n);
    
    for(int k = 0; k < k_it; k++)
    {
        ///////////////////////////////////////////////////
        // Get the househoulder vector for this column
        ///////////////////////////////////////////////////
        Real alpha = A[k + k * lda];
        Real beta = norm_vec(A + k + k * lda, m - k);
        if(alpha >= 0) beta *= -1;
        
        tau[k] = (beta == 0 ? 0 : (beta - alpha) / beta);
        
        // Doing some unnecessary work here by masking, but it should make 
        // the GPU version work better by avoiding conditionals on threads
        // that are going to be idle anyway in the update phase
        Real scal = (beta == 0 ? 0 : (Real)1.0 / (alpha - beta));
        for(int i = 0; i < m; i++)
            v[i] = (i < k ? 0 : scal * A[i + k * lda]);
        v[k] = 1;
        
        ///////////////////////////////////////////////////
        // Now do the update
        ///////////////////////////////////////////////////
        qr_apply_househoulder(A, lda, m, n, v, w, tau[k], k);

        ///////////////////////////////////////////////////
        // Finally, store the reflector in the lower part of A
        ///////////////////////////////////////////////////   
        for(int i = k + 1; i < m; i++)
            A[i + k * lda] = v[i];
    }
#else
    #ifdef DOUBLE_PRECISION
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, A, lda, tau);
    #else
    LAPACKE_sgeqrf(LAPACK_COL_MAJOR, m, n, A, lda, tau);
    #endif
#endif
}

// Calculate the Q orthogonal factor of a mxn matrix A
// A is overwritten by Q
// The lower part of A contains the househoulder reflectors
// v and w are temporary workspace of size m and n respectively 
// that should be allocated by the caller
// tau is a nx1 vector of the househoulder vector coefficients
inline void qr_unpackQ(Real* A, int lda, int m, int n, Real* v, Real* w, Real* tau)
{
	if(m <= 0 || n <= 0) return;
	
    int k_it = (n < m ? n : m);
    // First handle the last column of A outside the loop
    // since it doesn't involve a househoulder update
    for(int i = 0; i < m; i++)
        A[i + (k_it - 1) * lda] = (i < k_it ? 0 : A[i + (k_it - 1) * lda] * -tau[k_it - 1]);
    A[(k_it - 1) + (k_it - 1) * lda] = (Real)1.0 - tau[k_it - 1];
    
    // Now go through the columns of A starting from the end and going to the 
    // first column, applying the househoulder vectors stored in the column to
    // the submatrix of A that is to the right of the column
    for(int k = k_it - 2; k >=0; k--)
    {
        for(int i = 0; i < m; i++)
            v[i] = (i < k ? 0 : A[i + k * lda]);
        v[k] = 1;
        
        qr_apply_househoulder(A, lda, m, n, v, w, tau[k], k + 1);
        
        for(int i = 0; i < m; i++)
            A[i + k * lda] = (i < k ? 0 : A[i + k * lda] * -tau[k]);
        A[k + k * lda] = (Real)1.0 - tau[k];
    }
}

inline void qr_copyR(Real* A, int lda, Real* R, int ldr, int m, int n)
{
    for(int i = 0; i < n; i++)
		for(int j = 0; j < n; j++)
			R[i + j * ldr] = (i > j ? 0 : A[i + j * lda]);
}

// C = alpha * C + beta * A * B
template <int transa, int transb>
inline void gemm(Real* c, int ldc, Real* a, int lda, Real* b, int ldb, int m, int n, int p, Real alpha, Real beta)
{
    if(m == 0 || n == 0 || p == 0) return;
    if(alpha == 0 && beta == 0)    return;
    
#ifndef HLIB_USE_MKL
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            Real c_ij = 0;
            for(int k = 0; k < p; k++)
            {
                int index_a = (transa == 0 ? i + k * lda : k + i * lda);
                int index_b = (transb == 0 ? k + j * ldb : j + k * ldb);
                c_ij += a[index_a] * b[index_b];
            }
            int index_c = i + j * ldc;
            c[index_c] = alpha * c[index_c] + beta * c_ij;
        }
    }
#else
    CBLAS_TRANSPOSE trans_a = (transa ? CblasTrans : CblasNoTrans);
    CBLAS_TRANSPOSE trans_b = (transb ? CblasTrans : CblasNoTrans);
    #ifdef DOUBLE_PRECISION
    cblas_dgemm(CblasColMajor, trans_a, trans_b, m, n, p, beta, a, lda, b, ldb, alpha, c, ldc);
    #else
    cblas_sgemm(CblasColMajor, trans_a, trans_b, m, n, p, beta, a, lda, b, ldb, alpha, c, ldc);
    #endif
#endif
}

template <int transa, int transb>
inline void gemm(Real* c, Real* a, Real* b, int m, int n, int p, Real alpha, Real beta)
{
    int ldc = m, lda = (transa == 0 ? m : p), ldb = (transb == 0 ? p : n);
    gemm<transa, transb>(c, ldc, a, lda, b, ldb, m, n, p, alpha, beta);
}

template <int transa, int transb>
inline void gemm(Real* c, int ldc, Real* a, Real* b, int m, int n, int p, Real alpha, Real beta)
{
    int lda = (transa == 0 ? m : p), ldb = (transb == 0 ? p : n);
    gemm<transa, transb>(c, ldc, a, lda, b, ldb, m, n, p, alpha, beta);
}

// y = beta * y + alpha * A * x
template<int transa>
inline void gemv(Real* a, int lda, int m, int n, Real* x, Real* y, Real alpha, Real beta)
{
    if(!transa)
    {
        for(int i = 0; i < m; i++)
        {
            Real y_i = 0;
            for(int j = 0; j < n; j++)
                y_i += a[i + j * lda] * x[j];
            y[i] = beta * y[i] + alpha * y_i;
        }
    }
    else
    {
        for(int i = 0; i < n; i++)
        {
            Real y_i = 0;
            for(int j = 0; j < m; j++)
                y_i += a[j + i * lda] * x[j];
            y[i] = beta * y[i] + alpha * y_i;
        }
    }
}

inline void bsr_gemv(int block_rows, int block_size, Real* a, int* ia, int* ja, Real* x, Real* y)
{
    for(int i = 0; i < block_rows; i++)
    {
        Real* y_ptr = y + i * block_size;
        for(int block = ia[i]; block < ia[i+1]; block++)
        {
            int j = ja[block - HLIB_BSR_INDEX_BASE] - HLIB_BSR_INDEX_BASE;
            Real* a_ptr = a + (block - HLIB_BSR_INDEX_BASE) * block_size * block_size;
            Real* x_ptr = x + j * block_size;
            for(int block_i = 0; block_i < block_size; block_i++)
            {
                //y_ptr[block_i] = 0;
                for(int block_j = 0; block_j < block_size; block_j++)
                    y_ptr[block_i] += a_ptr[block_i + block_j * block_size] * x_ptr[block_j];
            }
        }
    }
}

inline void scale_rows(Real* matrix, Real* s, int m, int n)
{
    for(int i = 0; i < m; i++)
    {
        Real scale = s[i];
        for(int j = 0; j < n; j++)
            matrix[i + j * m] *= scale;
    }
}

inline void scale_cols(Real* matrix, Real* s, int m, int n)
{
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            matrix[i + j * m] *= s[j];
}

// A += B
inline void add(Real* a, Real* b, int m, int n)
{
    for(int i = 0; i < m * n; i++)
        a[i] += b[i];
}

#endif
