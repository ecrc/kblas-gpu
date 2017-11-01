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
/*template<class T>
double FLOPS_GEMM(T p, int m, int n, int k){
  return (is_complex(p) ? 6. : 1.) * FMULS_GEMM((double)(m), (double)(n), (double)(k))
       + (is_complex(p) ? 2. : 1.) * FADDS_GEMM((double)(m), (double)(n), (double)(k));
}
/*double FLOPS_GEMM(float p, int m, int n, int k){
  return FMULS_GEMM((double)(m), (double)(n), (double)(k)) + FADDS_GEMM((double)(m), (double)(n), (double)(k));
}
double FLOPS_GEMM(double p, int m, int n, int k){
  return FMULS_GEMM((double)(m), (double)(n), (double)(k)) + FADDS_GEMM((double)(m), (double)(n), (double)(k));
}
#ifdef SUPPORT_GPU
double FLOPS_GEMM(cuFloatComplex p, int m, int n, int k){
  return 6. * FMULS_GEMM((double)(m), (double)(n), (double)(k)) + 2.0 * FADDS_GEMM((double)(m), (double)(n), (double)(k));
}
double FLOPS_GEMM(cuDoubleComplex p, int m, int n, int k){
  return 6. * FMULS_GEMM((double)(m), (double)(n), (double)(k)) + 2.0 * FADDS_GEMM((double)(m), (double)(n), (double)(k));
}
#endif*/

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
/*double FLOPS_TRMM(float p, char side, int m, int n){
  return FMULS_TRMM(side, (double)(m), (double)(n)) + FADDS_TRMM(side, (double)(m), (double)(n));
}
double FLOPS_TRMM(double p, char side, int m, int n){
  return FMULS_TRMM(side, (double)(m), (double)(n)) + FADDS_TRMM(side, (double)(m), (double)(n));
}
#ifdef SUPPORT_GPU
double FLOPS_TRMM(cuFloatComplex p, char side, int m, int n){
  return 6. * FMULS_TRMM(side, (double)(m), (double)(n)) + 2. * FADDS_TRMM(side, (double)(m), (double)(n));
}
double FLOPS_TRMM(cuDoubleComplex p, char side, int m, int n){
  return 6. * FMULS_TRMM(side, (double)(m), (double)(n)) + 2. * FADDS_TRMM(side, (double)(m), (double)(n));
}
#endif*/

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
/*double FLOPS_TRSM(float p, char side, int m, int n){
  return FMULS_TRSM(side, (double)(m), (double)(n)) + FADDS_TRSM(side, (double)(m), (double)(n));
}
double FLOPS_TRSM(double p, char side, int m, int n){
  return FMULS_TRSM(side, (double)(m), (double)(n)) + FADDS_TRSM(side, (double)(m), (double)(n));
}
#ifdef SUPPORT_GPU
double FLOPS_TRSM(cuFloatComplex p, char side, int m, int n){
  return 6. * FMULS_TRSM(side, (double)(m), (double)(n)) + 2. * FADDS_TRSM(side, (double)(m), (double)(n));
}
double FLOPS_TRSM(cuDoubleComplex p, char side, int m, int n){
  return 6. * FMULS_TRSM(side, (double)(m), (double)(n)) + 2. * FADDS_TRSM(side, (double)(m), (double)(n));
}
#endif*/


//==============================================================================================
#define FMULS_POTRF(n_) ((n_) * (((1. / 6.) * (n_) + 0.5) * (n_) + (1. / 3.)))
#define FADDS_POTRF(n_) ((n_) * (((1. / 6.) * (n_)      ) * (n_) - (1. / 6.)))

template<class T>
double FLOPS_POTRF(int n){
  return (is_complex(T) ? 6. : 1.) * FMULS_POTRF((double)(n))
       + (is_complex(T) ? 2. : 1.) * FADDS_POTRF((double)(n));
}
/*double FLOPS_POTRF(float p, int n){
  return FMULS_POTRF((double)(n)) + FADDS_POTRF((double)(n));
}
double FLOPS_POTRF(double p, int n){
  return FMULS_POTRF((double)(n)) + FADDS_POTRF((double)(n));
}
#ifdef SUPPORT_GPU
double FLOPS_POTRF(cuFloatComplex p, int n){
  return 6. * FMULS_POTRF((double)(n)) + 2. * FADDS_POTRF((double)(n));
}
double FLOPS_POTRF(cuDoubleComplex p, int n){
  return 6. * FMULS_POTRF((double)(n)) + 2. * FADDS_POTRF((double)(n));
}
#endif*/

//==============================================================================================
#define FMULS_TRTRI(n_) ((n_) * ((n_) * ( 1./6. * (n_) + 0.5 ) + 1./3.))
#define FADDS_TRTRI(n_) ((n_) * ((n_) * ( 1./6. * (n_) - 0.5 ) + 1./3.))

template<class T>
double FLOPS_TRTRI(int n){
  return (is_complex(T) ? 6. : 1.) * FMULS_TRTRI((double)(n))
       + (is_complex(T) ? 2. : 1.) * FADDS_TRTRI((double)(n));
}
/*double FLOPS_TRTRI(float p, int n){
  return FMULS_TRTRI((double)(n)) + FADDS_TRTRI((double)(n));
}
double FLOPS_TRTRI(double p, int n){
  return FMULS_TRTRI((double)(n)) + FADDS_TRTRI((double)(n));
}
#ifdef SUPPORT_GPU
double FLOPS_TRTRI(cuFloatComplex p, int n){
  return 6. * FMULS_TRTRI((double)(n)) + 2. * FADDS_TRTRI((double)(n));
}
double FLOPS_TRTRI(cuDoubleComplex p, int n){
  return 6. * FMULS_TRTRI((double)(n)) + 2. * FADDS_TRTRI((double)(n));
}
#endif*/

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

/*double FLOPS_SYRK(float t, int k_, int n_){
  return (     FMULS_SYRK((double)(k_), (double)(n_)) +       FADDS_SYRK((double)(k_), (double)(n_)) );
}
double FLOPS_SYRK(double t, int k_, int n_){
  return (     FMULS_SYRK((double)(k_), (double)(n_)) +       FADDS_SYRK((double)(k_), (double)(n_)) );
}
double FLOPS_SYRK(cuFloatComplex t, int k_, int n_){
  return (6. * FMULS_SYRK((double)(k_), (double)(n_)) + 2.0 * FADDS_SYRK((double)(k_), (double)(n_)) );
}
double FLOPS_SYRK(cuDoubleComplex t, int k_, int n_){
  return (6. * FMULS_SYRK((double)(k_), (double)(n_)) + 2.0 * FADDS_SYRK((double)(k_), (double)(n_)) );
}*/

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

double FLOPS_SCHUR(float t, char symm, int n, int p){
  if(symm == KBLAS_NonSymm)
    FLOPS_SCHUR_NonSYMM
  else
    FLOPS_SCHUR_SYMM
}
double FLOPS_SCHUR(double t, char symm, int n, int p){
  if(symm == KBLAS_NonSymm)
    FLOPS_SCHUR_NonSYMM
  else
    FLOPS_SCHUR_SYMM
}
double FLOPS_SCHUR(cuFloatComplex t, char symm, int n, int p){
  if(symm == KBLAS_NonSymm)
    FLOPS_SCHUR_NonSYMM
  else
    FLOPS_SCHUR_SYMM
}
double FLOPS_SCHUR(cuDoubleComplex t, char symm, int n, int p){
  if(symm == KBLAS_NonSymm)
    FLOPS_SCHUR_NonSYMM
  else
    FLOPS_SCHUR_SYMM
}*/
#endif
