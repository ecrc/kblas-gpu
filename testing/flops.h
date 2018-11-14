/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/

/**
 *
 * @copyright (c) 2009-2014 The University of Tennessee and The University
 *                          of Tennessee Research Foundation.
 *                          All rights reserved.
 * @copyright (c) 2012-2014 Inria. All rights reserved.
 * @copyright (c) 2012-2014 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria, Univ. Bordeaux. All rights reserved.
 *
 **/

/**
 *
 * @file testing/flops.h
 *
 *
 * @version 1.0.0
 * @author Mathieu Faverge
 * @date 2018-11-14
 *
 * @version 3.0.0
 * @author Ali Charara
 * @date 2018-11-14
 **/

/*
 * This file provide the flops formula for some of Level 3 BLAS and some
 * Lapack routines.  Each macro uses the same size parameters as the
 * function associated and provide one formula for additions and one
 * for multiplications. Example to use these macros:
 *
 *    FLOPS_ZGEMM( m, n, k )
 *
 * All the formula are reported in the LAPACK Lawn 41:
 *     http://www.netlib.org/lapack/lawns/lawn41.ps
 */

#ifndef _TESTING_FLOPS_
#define _TESTING_FLOPS_

#include <typeinfo>

#define is_complex(t) ( typeid(t) == typeid(cuFloatComplex) || typeid(t) == typeid(cuComplex) || typeid(t) == typeid(cuDoubleComplex) )
#define factorMul(t) (is_complex(t) ? 6. : 1.)
#define factorAdd(t) (is_complex(t) ? 2. : 1.)
//==============================================================================================
#define FMULS_GEMM(m_, n_, k_) ((m_) * (n_) * (k_))
#define FADDS_GEMM(m_, n_, k_) ((m_) * (n_) * (k_))

template<class T>
double FLOPS_GEMM(int m, int n, int k){
  return (is_complex(T) ? 6. : 1.) * FMULS_GEMM((double)(m), (double)(n), (double)(k))
       + (is_complex(T) ? 2. : 1.) * FADDS_GEMM((double)(m), (double)(n), (double)(k));
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
template<class T>
double FLOPS_POTRI(int n){
  return FLOPS_TRTRI<T>(n) + FLOPS_LAUUM<T>(n);
}

//==============================================================================================
template<class T>
double FLOPS_POTI(int n){
  return FLOPS_POTRF<T>(n) + FLOPS_POTRI<T>(n);
}

//==============================================================================================
template<class T>
double FLOPS_POSV(char side, int m, int n){
  return FLOPS_POTRF<T>(n) + FLOPS_POTRS<T>(side, m, n);
}

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
#define FMULS_GEQRF(m_, n_) (((m_) > (n_)) \
    ? ((n_) * ((n_) * (  0.5-(1./3.) * (n_) + (m_)) +    (m_) + 23. / 6.)) \
    : ((m_) * ((m_) * ( -0.5-(1./3.) * (m_) + (n_)) + 2.*(n_) + 23. / 6.)) )
#define FADDS_GEQRF(m_, n_) (((m_) > (n_)) \
    ? ((n_) * ((n_) * (  0.5-(1./3.) * (n_) + (m_))           +  5. / 6.)) \
    : ((m_) * ((m_) * ( -0.5-(1./3.) * (m_) + (n_)) +    (n_) +  5. / 6.)) )

template<class T>
double FLOPS_GEQRF(int m, int n){
  return factorMul(T) * FMULS_GEQRF((double)(m), (double)(n))
       + factorAdd(T) * FADDS_GEQRF((double)(m), (double)(n));
}

//==============================================================================================
#define FMULS_ORGQR(m_, n_, k_) ((k_) * (2.* (m_) * (n_) +   2. * (n_) - 5./3. + (k_) * ( 2./3. * (k_) - ((m_) + (n_)) - 1.)))
#define FADDS_ORGQR(m_, n_, k_) ((k_) * (2.* (m_) * (n_) + (n_) - (m_) + 1./3. + (k_) * ( 2./3. * (k_) - ((m_) + (n_))     )))

template<class T>
double FLOPS_ORGQR(int m, int n, int k){
  return factorMul(T) * FMULS_ORGQR((double)(m), (double)(n), (double)(k))
       + factorAdd(T) * FADDS_ORGQR((double)(m), (double)(n), (double)(k));
}

//==============================================================================================
template<class T>
double FLOPS_GEMM_LR_LLD(int m, int n, int k, int ra, int rb){
  return FLOPS_GEMM<T>(ra, rb, k) + FLOPS_GEMM<T>(ra, n, rb) + FLOPS_GEMM<T>(m, n, ra);
}
//==============================================================================================
template<class T>
double FLOPS_GEMM_LR_LLL(int m, int n, int k, int ra, int rb, int rc){
  int kApkC = ra+rc;
  return  FLOPS_GEMM<T>(ra, rb, m)      //P = Av^tBv
        + FLOPS_GEMM<T>(n, ra, rb)      //Bv * P^T
        + FLOPS_GEQRF<T>(m, kApkC)  //GEQRF(A)
        + factorMul(T) * n*rc           //Scale by Beta
        + FLOPS_GEQRF<T>(n, kApkC)  //GEQRF(B)
        + FLOPS_TRMM<T>(KBLAS_Right, kApkC, kApkC)    //rA = rA*rB^T
        + (kApkC*kApkC*kApkC) * (factorMul(T) + 2 * factorAdd(T)) //ACA
        + FLOPS_ORGQR<T>(m, kApkC, kApkC) //ORGQR(U)
        + FLOPS_ORGQR<T>(n, kApkC, kApkC) //ORGQR(V)
        + FLOPS_GEMM<T>(m, kApkC, kApkC)  //U=Qu*Tu
        + FLOPS_GEMM<T>(n, kApkC, kApkC)  //V=Qv*Tv
        ;
}

#endif //_TESTING_FLOPS_
