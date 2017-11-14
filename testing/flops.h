#ifndef _TESTING_FLOPS_
#define _TESTING_FLOPS_

#include <typeinfo>

#define is_complex(t) ( typeid(t) == typeid(cuFloatComplex) || typeid(t) == typeid(cuComplex) || typeid(t) == typeid(cuDoubleComplex) )
//==============================================================================================
#define FMULS_GEMM(m_, n_, k_) ((m_) * (n_) * (k_))
#define FADDS_GEMM(m_, n_, k_) ((m_) * (n_) * (k_))

template<class T>
double FLOPS_GEMM(int m, int n, int k){
  return (is_complex(T) ? 6. : 1.) * FMULS_GEMM((double)(m), (double)(n), (double)(k))
       + (is_complex(T) ? 2. : 1.) * FADDS_GEMM((double)(m), (double)(n), (double)(k));
}

//==============================================================================================
template<class T>
double FLOPS_GEMM_PLR(int m, int n, int k, int ra, int rb){
  return FLOPS_GEMM<T>(ra, rb, k) + FLOPS_GEMM<T>(ra, n, rb) + FLOPS_GEMM<T>(m, n, ra);
}

//==============================================================================================
#define FMULS_TRMM_2(m_, n_) (0.5 * (n_) * (m_) * ((m_)+1))
#define FADDS_TRMM_2(m_, n_) (0.5 * (n_) * (m_) * ((m_)-1))
#define FMULS_TRMM(side_, m_, n_) ( ( (side_) == KBLAS_Left ) ? FMULS_TRMM_2((m_), (n_)) : FMULS_TRMM_2((n_), (m_)) )
#define FADDS_TRMM(side_, m_, n_) ( ( (side_) == KBLAS_Left ) ? FADDS_TRMM_2((m_), (n_)) : FADDS_TRMM_2((n_), (m_)) )


template<class T>
double FLOPS_TRMM(char side, int m, int n){
  return (is_complex(T) ? 6. : 1.) * FMULS_TRMM(side, (double)(m), (double)(n))
       + (is_complex(T) ? 2. : 1.) * FADDS_TRMM(side, (double)(m), (double)(n));
}

//==============================================================================================
#define FMULS_TRSM_2(m_, n_) (0.5 * (n_) * (m_) * ((m_)+1))
#define FADDS_TRSM_2(m_, n_) (0.5 * (n_) * (m_) * ((m_)-1))
#define FMULS_TRSM(side_, m_, n_) ( ( (side_) == KBLAS_Left ) ? FMULS_TRSM_2((m_), (n_)) : FMULS_TRSM_2((n_), (m_)) )
#define FADDS_TRSM(side_, m_, n_) ( ( (side_) == KBLAS_Left ) ? FADDS_TRSM_2((m_), (n_)) : FADDS_TRSM_2((n_), (m_)) )

template<class T>
double FLOPS_TRSM(char side, int m, int n){
  return (is_complex(T) ? 6. : 1.) * FMULS_TRSM(side, (double)(m), (double)(n))
       + (is_complex(T) ? 2. : 1.) * FADDS_TRSM(side, (double)(m), (double)(n));
}

//==============================================================================================
#define FMULS_POTRF(n_) ((n_) * (((1. / 6.) * (n_) + 0.5) * (n_) + (1. / 3.)))
#define FADDS_POTRF(n_) ((n_) * (((1. / 6.) * (n_)      ) * (n_) - (1. / 6.)))

template<class T>
double FLOPS_POTRF(int n){
  return (is_complex(T) ? 6. : 1.) * FMULS_POTRF((double)(n))
       + (is_complex(T) ? 2. : 1.) * FADDS_POTRF((double)(n));
}

//==============================================================================================
template<class T>
double FLOPS_POTRS(char side, int m, int n){
  return 2. * FLOPS_TRSM<T>(side, m, n);
}
//==============================================================================================
#define FMULS_TRTRI(n_) ((n_) * ((n_) * ( 1./6. * (n_) + 0.5 ) + 1./3.))
#define FADDS_TRTRI(n_) ((n_) * ((n_) * ( 1./6. * (n_) - 0.5 ) + 1./3.))

template<class T>
double FLOPS_TRTRI(int n){
  return (is_complex(T) ? 6. : 1.) * FMULS_TRTRI((double)(n))
       + (is_complex(T) ? 2. : 1.) * FADDS_TRTRI((double)(n));
}

//==============================================================================================

#define FLOPS_LAUUM FLOPS_POTRF

//==============================================================================================
#define FMULS_SYMM(side_, m_, n_) ( ( (side_) == KBLAS_Left ) ? FMULS_GEMM((m_), (m_), (n_)) : FMULS_GEMM((m_), (n_), (n_)) )
#define FADDS_SYMM(side_, m_, n_) ( ( (side_) == KBLAS_Left ) ? FADDS_GEMM((m_), (m_), (n_)) : FADDS_GEMM((m_), (n_), (n_)) )

template<class T>
double FLOPS_SYMM(char side_, int m_, int n_){
  return (is_complex(T) ? 6. : 1.) * FMULS_SYMM(side_, (double)(m_), (double)(n_))
       + (is_complex(T) ? 2. : 1.) * FADDS_SYMM(side_, (double)(m_), (double)(n_));
}

//==============================================================================================
#define FMULS_SYRK(k_, n_) (0.5 * (k_) * (n_) * ((n_)+1))
#define FADDS_SYRK(k_, n_) (0.5 * (k_) * (n_) * ((n_)+1))

template<class T>
double FLOPS_SYRK(int k_, int n_){
  return (is_complex(T) ? 6. : 1.) * FMULS_SYRK((double)(k_), (double)(n_))
       + (is_complex(T) ? 2. : 1.) * FADDS_SYRK((double)(k_), (double)(n_));
}

//==============================================================================================
template<class T>
double FLOPS_SYRK_PLR(int n, int k, int ra, bool simple){
  return (!simple ? (FLOPS_SYRK<T>(ra, k) + FLOPS_SYMM<T>(KBLAS_Right, n, ra) + FLOPS_GEMM<T>(n, n, ra)) : FLOPS_SYRK<T>(n, ra));
}

//==============================================================================================
//TODO assuming SchurA, handle SchurD, nonsymmetric should use LU flop count
template<class T>
double FLOPS_SCHUR(T t, char symm, int n, int p){
  if(symm == KBLAS_NonSymm)
    return FLOPS_POTRF(t, p) + 2. * FLOPS_TRSM(t, KBLAS_Right, n-p, p) + FLOPS_GEMM<T>(n-p, n-p, p);
  else
    return FLOPS_POTRF(t, p) + FLOPS_TRSM(t, KBLAS_Right, n-p, p) + FLOPS_SYRK<T>(p, n-p);
}
/*#define FLOPS_SCHUR_NonSYMM return FLOPS_POTRF(t, p) + 2. * FLOPS_TRSM(t, KBLAS_Right, n-p, p);//TODO + FLOPS_GEMM<T>(n-p, n-p, p);
#define FLOPS_SCHUR_SYMM return FLOPS_POTRF(t, p) + FLOPS_TRSM(t, KBLAS_Right, n-p, p) + FLOPS_SYRK(t, p, n-p);
*/

#endif //_TESTING_FLOPS_
