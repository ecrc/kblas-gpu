 /**
  - -* (C) Copyright 2013 King Abdullah University of Science and Technology
  Authors:
  Ali Charara (ali.charara@kaust.edu.sa)
  David Keyes (david.keyes@kaust.edu.sa)
  Hatem Ltaief (hatem.ltaief@kaust.edu.sa)
  
  Redistribution  and  use  in  source and binary forms, with or without
  modification,  are  permitted  provided  that the following conditions
  are met:
  
  * Redistributions  of  source  code  must  retain  the above copyright
  * notice,  this  list  of  conditions  and  the  following  disclaimer.
  * Redistributions  in  binary  form must reproduce the above copyright
  * notice,  this list of conditions and the following disclaimer in the
  * documentation  and/or other materials provided with the distribution.
  * Neither  the  name of the King Abdullah University of Science and
  * Technology nor the names of its contributors may be used to endorse
  * or promote products derived from this software without specific prior
  * written permission.
  * 
  T *HIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
  LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  **/
//#include "kblas.h"
//#include "operators.h"
//#include "Xtr_common.cuh"

//==============================================================================================

template<class T>
int kblasXtrmm(
  char side, char uplo, char transa, char diag,
  int m, int n,
  T alpha, const T *A, int incA,
  T *B, int incB,
  cudaStream_t    stream)
{
  T one = make_one<T>();
  //cublasSetKernelStream(stream);
  
  if(side == KBLAS_Left){
    int m1, m2;
    if(REG_SIZE(m))
      m1 = m2 = m/2;
    else{
      m1 = CLOSEST_REG_SIZE(m);
      m2 = m-m1;
    }
    
    if(uplo == KBLAS_Upper){

      if(SIMPLE_SIZE(m)){
        Xtrmm(side, uplo, transa, diag,
              m, n,
              alpha, A, incA,
                     B, incB );
        return 1;
      }
      //Left / Upper / NoTrans
      if(transa == KBLAS_NoTrans){
        if(!kblasXtrmm(side, uplo, transa, diag,
                       m1, n,
                       alpha, A, incA,
                              B, incB
                       , stream
                       )) return 0;

        Xgemm(transa, KBLAS_NoTrans,
              m1, n, m2,
              alpha, A+m1*incA, incA,
                     B+m1, incB,
              one,   B, incB);

        if(!kblasXtrmm(side, uplo, transa, diag,
                       m2, n,
                       alpha, A+m1+m1*incA, incA,
                              B+m1, incB
                       , stream
                       )) return 0;
      }
      //Left / Upper / [Conj]Trans
      else{
        if(!kblasXtrmm(side, uplo, transa, diag,
                       m2, n,
                       alpha, A+m1+m1*incA, incA,
                              B+m1, incB
                       , stream
                       )) return 0;

        Xgemm(transa, KBLAS_NoTrans,
              m2, n, m1,
              alpha, A+m1*incA, incA,
                     B, incB,
              one,   B+m1, incB);

        if(!kblasXtrmm(side, uplo, transa, diag,
                       m1, n,
                       alpha, A, incA,
                              B, incB
                       , stream
                       )) return 0;
      }

    }else{//uplo == KBLAS_Lower
      
      //Left / Lower / NoTrans
      if(transa == KBLAS_NoTrans){
        if(SIMPLE_SIZE(m)){
          Xtrmm(side, uplo, transa, diag,
                m, n,
                alpha, A, incA,
                       B, incB );
          return 1;
        }
        if(!kblasXtrmm(side, uplo, transa, diag,
                       m2, n,
                       alpha, A+m1+m1*incA, incA,
                              B+m1, incB
                       , stream
                       )) return 0;

        Xgemm(transa, KBLAS_NoTrans,
              m2, n, m1,
              alpha, A+m1, incA,
                     B, incB,
              one,   B+m1, incB);

        if(!kblasXtrmm(side, uplo, transa, diag,
                       m1, n,
                       alpha, A, incA,
                              B, incB
                       , stream
                       )) return 0;
      }
      //Left / Lower / [Conj]Trans
      else{//transa == KBLAS_Trans
#ifdef SUPPORT_CUBLAS
        if(kblas_trmm_use_custom){
          if(SIMPLE_SIZE_CUSTOM(m)){
            if(!trmm_custom(side, uplo, transa, diag,
                            m, n,
                            alpha, A, incA,
                                   B, incB
                            , stream
                            )) return 0;
            return 1;
          }
        }
        else//kblas_trmm_use_custom
#endif //SUPPORT_CUBLAS
        {
          if(SIMPLE_SIZE(m)){
            Xtrmm(side, uplo, transa, diag,
                  m, n,
                  alpha, A, incA,
                  B, incB );
            return 1;
          }
        }
        if(!kblasXtrmm(side, uplo, transa, diag,
                       m1, n,
                       alpha, A, incA,
                              B, incB
                       , stream
                       )) return 0;

        Xgemm(transa, KBLAS_NoTrans,
              m1, n, m2,
              alpha, A+m1, incA,
                     B+m1, incB,
              one,   B, incB);

        if(!kblasXtrmm(side, uplo, transa, diag,
                       m2, n,
                       alpha, A+m1+m1*incA, incA,
                              B+m1, incB
                       , stream
                       )) return 0;
      }//transa == KBLAS_Trans
      
    }
    
  }
  else{//side == KBLAS_Right
    int n1, n2;

    if(SIMPLE_SIZE(n)){
      Xtrmm(side, uplo, transa, diag,
            m, n,
            alpha, A, incA,
                   B, incB );
      return 1;
    }
    if(REG_SIZE(n))
      n1 = n2 = n/2;
    else{
      n1 = CLOSEST_REG_SIZE(n);
      n2 = n-n1;
    }

    if(uplo == KBLAS_Upper){
      //Right / Upper / NoTrans
      if(transa == KBLAS_NoTrans){
        if(!kblasXtrmm(side, uplo, transa, diag,
                       m, n2,
                       alpha, A+n1+n1*incA, incA,
                              B+n1*incB, incB
                       , stream
                      )) return 0;

        Xgemm(KBLAS_NoTrans, transa,
              m, n2, n1,
              alpha, B, incB,
                     A+n1*incA, incA,
              one,   B+n1*incB, incB);

        if(!kblasXtrmm(side, uplo, transa, diag,
                       m, n1,
                       alpha, A, incA,
                              B, incB
                       , stream
                      )) return 0;
      }
      //Right / Upper / [Conj]Trans
      else{
        if(!kblasXtrmm(side, uplo, transa, diag,
                       m, n1,
                       alpha, A, incA,
                              B, incB
                       , stream
                      )) return 0;

        Xgemm(KBLAS_NoTrans, transa,
              m, n1, n2,
              alpha, B+n1*incB, incB,
                     A+n1*incA, incA,
              one,   B, incB);

        if(!kblasXtrmm(side, uplo, transa, diag,
                       m, n2,
                       alpha, A+n1+n1*incA, incA,
                              B+n1*incB, incB
                       , stream
                       )) return 0;
      }
    }
    else{
      //Right / Lower / NoTrans
      if(transa == KBLAS_NoTrans){
        if(!kblasXtrmm(side, uplo, transa, diag,
                       m, n1,
                       alpha, A, incA,
                              B, incB
                       , stream
                       )) return 0;

        Xgemm(KBLAS_NoTrans, transa,
              m, n1, n2,
              alpha, B+n1*incB, incB,
                     A+n1, incA,
              one,   B, incB);

        if(!kblasXtrmm(side, uplo, transa, diag,
                       m, n2,
                       alpha, A+n1+n1*incA, incA,
                              B+n1*incB, incB
                       , stream
                      )) return 0;
      }
      //Right / Lower / [Conj]Trans
      else{
        if(!kblasXtrmm(side, uplo, transa, diag,
                       m, n2,
                       alpha, A+n1+n1*incA, incA,
                              B+n1*incB, incB
                       , stream
                      )) return 0;

        Xgemm(KBLAS_NoTrans, transa,
              m, n2, n1,
              alpha, B, incB,
                     A+n1, incA,
              one,   B+n1*incB, incB);

        if(!kblasXtrmm(side, uplo, transa, diag,
                       m, n1,
                       alpha, A, incA,
                              B, incB
                       , stream
                       )) return 0;
      }
    }
    
  }//side == KBLAS_Right

  return 1;
}
