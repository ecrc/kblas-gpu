#ifndef __KBLAS_GETRF_FUNCTORS_H__
#define __KBLAS_GETRF_FUNCTORS_H__

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/transform_scan.h>
#include "vbatch_thrust_util.cuh"

template <class T>
struct GetrfPanelSet_Functor
{
    T **A_ptrs, **panel_ptrs;
	int *m_batch, *n_batch, *lda_batch;
	int *panel_m_batch, *panel_n_batch;
	int bs,	j;
	
    GetrfPanelSet_Functor(
		T **A_ptrs, int *lda_batch, int *m_batch, int *n_batch,
		T **panel_ptrs, int *panel_m_batch, int *panel_n_batch,
		int bs, int j)
    {
        this->A_ptrs = A_ptrs;
		this->lda_batch = lda_batch;
        this->m_batch = m_batch;
		this->n_batch = n_batch;
		this->panel_ptrs = panel_ptrs;
		this->panel_m_batch = panel_m_batch;
		this->panel_n_batch = panel_n_batch;
		this->bs = bs;
		this->j = j;
    }
	
    inline __host__ __device__ void operator()(size_t i)
    {
		panel_m_batch[i] = (m_batch[i] - j < bs ? m_batch[i] - j : bs);
		panel_n_batch[i] = n_batch[i] - j;
		
		if(panel_m_batch[i] <= 0 || panel_n_batch[i] <= 0)
		{
			panel_m_batch[i] = panel_n_batch[i] = 0;
			panel_ptrs[i] = NULL;
		}
		else
			panel_ptrs[i] = A_ptrs[i] + j + j * lda_batch[i];
	    }
};

template <class T>
struct GetrfTrsmSet_Functor
{
    T **A_ptrs, **B_ptrs;
	int *m_batch, *n_batch, *lda_batch;
	int *panel_m_batch, *panel_n_batch;
	int bs,	j;
	
    GetrfTrsmSet_Functor(
		T **A_ptrs, int *lda_batch, int *m_batch, int *n_batch,
		T **B_ptrs, int *panel_m_batch, int *panel_n_batch,
		int bs, int j)
    {
        this->A_ptrs = A_ptrs;
		this->lda_batch = lda_batch;
        this->m_batch = m_batch;
		this->n_batch = n_batch;
		this->B_ptrs = B_ptrs;
		this->panel_m_batch = panel_m_batch;
		this->panel_n_batch = panel_n_batch;
		this->bs = bs;
		this->j = j;
    }
	
    inline __host__ __device__ void operator()(size_t i)
    {
		panel_m_batch[i] = m_batch[i] - j - bs;
		panel_n_batch[i] = (n_batch[i] - j < bs ? n_batch[i] - j : bs);
	
		if(panel_m_batch[i] <= 0 || panel_n_batch[i] <= 0) 
		{
			panel_m_batch[i] = panel_n_batch[i] = 0;
			B_ptrs[i] = NULL;
		}
		else
			B_ptrs[i] = A_ptrs[i] + j + bs + j * lda_batch[i];
	}
};

template <class T>
struct GetrfUpdateSet_Functor
{
    T **A_ptrs, **U_ptrs, **B_ptrs;
	int *m_batch, *n_batch, *lda_batch;
	int *update_m_batch, *update_n_batch, *update_k_batch;
	int bs,	j;
	
    GetrfUpdateSet_Functor(
		T **A_ptrs, int *lda_batch, int *m_batch, int *n_batch,
		T** U_ptrs, T **B_ptrs, int *update_m_batch, int *update_n_batch,
		int *update_k_batch, int bs, int j)
    {
        this->A_ptrs = A_ptrs;
		this->lda_batch = lda_batch;
        this->m_batch = m_batch;
		this->n_batch = n_batch;
		this->U_ptrs = U_ptrs;
		this->B_ptrs = B_ptrs;
		this->update_m_batch = update_m_batch;
		this->update_n_batch = update_n_batch;
		this->update_k_batch = update_k_batch;
		this->bs = bs;
		this->j = j;
    }
	
    inline __host__ __device__ void operator()(size_t i)
    {
		update_m_batch[i] = m_batch[i] - j - bs;
		update_n_batch[i] = n_batch[i] - j - bs;
		update_k_batch[i] = bs;
		if(update_m_batch[i] <= 0 || update_n_batch[i] <= 0) 
		{
			update_m_batch[i] = update_n_batch[i] = update_k_batch[i] = 0;
			U_ptrs[i] = B_ptrs[i] = NULL;
		}
		else
		{
			U_ptrs[i] = A_ptrs[i] + j + (j + bs) * lda_batch[i];
			B_ptrs[i] = A_ptrs[i] + j + bs + (j + bs) * lda_batch[i];
		}
	}
};

#endif 
