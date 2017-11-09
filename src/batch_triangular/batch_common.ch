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
  THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
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
#ifndef __KBLAS_BATCH_COMMON_H__
#define __KBLAS_BATCH_COMMON_H__



//==============================================================================================
void gemm_batch_offset_wsquery_core(int batchCount,
                                    int A_row_off, int A_col_off,
                                    int B_row_off, int B_col_off,
                                    int C_row_off, int C_col_off,
                                    kblasWorkspaceState_t ws);

void gemm_batch_strided_wsquery_core(int batchCount, kblasWorkspaceState_t ws);

//==============================================================================================
void syrk_batch_wsquery_core(const int m, int batchCount, kblasWorkspaceState_t ws);

//==============================================================================================
template<bool STRIDED>
inline
void trsm_batch_wsquery_core( int batchCount,
                              char side, int m, int n,
                              kblasWorkspaceState_t wss)
{
  if( ( (side == KBLAS_Right) && (n > 16) ) ||
      ( (side == KBLAS_Left ) && (m > 16) ) ){
    if(STRIDED){
      gemm_batch_strided_wsquery_core(batchCount, wss);
    }else{
      gemm_batch_offset_wsquery_core( batchCount,
                                      1, 1, 1, 1, 1, 1,
                                      wss);
    }
  }else{
    wss->reset();
  }
}
//==============================================================================================
template<bool STRIDED>
inline
void trmm_batch_wsquery_core( int batchCount,
                              char side, int m, int n,
                              kblasWorkspaceState_t wss)
{
  if( ( (side == KBLAS_Right) && (n > 16) ) ||
      ( (side == KBLAS_Left ) && (m > 16) ) ){
    if(STRIDED){
      gemm_batch_strided_wsquery_core(batchCount, wss);
    }else{
      gemm_batch_offset_wsquery_core( batchCount,
                                      1, 1, 1, 1, 1, 1,
                                      wss);
    }
  }else{
    wss->reset();
  }
}

#endif //__KBLAS_BATCH_COMMON_H__