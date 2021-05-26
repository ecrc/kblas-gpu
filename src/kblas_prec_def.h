/**
 * @copyright (c) 2012- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/


/**
 * @file src/kblas_prec_def.h

 * KBLAS is a high performance CUDA library for subset of BLAS
 *    and LAPACK routines optimized for NVIDIA GPUs.
 * KBLAS is provided by KAUST.
 *
 * @version 4.0.0
 * @author Ali Charara
 * @date 2020-12-10
 **/

#ifndef __KBLAS_PREC_DEF__
#define __KBLAS_PREC_DEF__

#if defined   PREC_s
  #define TYPE float
  
  #define kblasXpotrf               kblasSpotrf
  
  #define kblasXtrsm                kblasStrsm
  #define kblasXtrmm                kblasStrmm

  #define kblasXgemm_batch          kblasSgemm_batch
  #define kblasXgemm_batch_strided  kblasSgemm_batch_strided

  #define kblasXsyrk_batch                  kblasSsyrk_batch
  #define kblasXsyrk_batch_wsquery          kblasSsyrk_batch_wsquery
  #define kblasXsyrk_batch_strided          kblasSsyrk_batch_strided
  #define kblasXsyrk_batch_strided_wsquery  kblasSsyrk_batch_strided_wsquery

  #define kblasXtrsm_batch          kblasStrsm_batch
  #define kblasXtrsm_batch_strided  kblasStrsm_batch_strided

  #define kblasXtrmm_batch          kblasStrmm_batch
  #define kblasXtrmm_batch_strided  kblasStrmm_batch_strided

  #define kblasXpotrf_batch         kblasSpotrf_batch
  #define kblasXpotrf_batch_strided kblasSpotrf_batch_strided

  #define kblasXlauum_batch         kblasSlauum_batch
  #define kblasXlauum_batch_strided kblasSlauum_batch_strided

  #define kblasXtrtri_batch         kblasStrtri_batch
  #define kblasXtrtri_batch_strided kblasStrtri_batch_strided

  #define kblasXpotrs_batch         kblasSpotrs_batch
  #define kblasXpotrs_batch_strided kblasSpotrs_batch_strided

  #define kblasXpotri_batch         kblasSpotri_batch
  #define kblasXpotri_batch_strided kblasSpotri_batch_strided

  #define kblasXpoti_batch          kblasSpoti_batch
  #define kblasXpoti_batch_strided  kblasSpoti_batch_strided

  #define kblasXposv_batch          kblasSposv_batch
  #define kblasXposv_batch_strided  kblasSposv_batch_strided

  #define kblasXgemm_lr_lld                          kblasSgemm_lr_lld
  #define kblasXgemm_lr_lld_wsquery                  kblasSgemm_lr_lld_wsquery
  #define kblasXgemm_lr_lld_batch                    kblasSgemm_lr_lld_batch
  #define kblasXgemm_lr_lld_batch_wsquery            kblasSgemm_lr_lld_batch_wsquery
  #define kblasXgemm_lr_lld_batch_strided            kblasSgemm_lr_lld_batch_strided
  #define kblasXgemm_lr_lld_batch_strided_wsquery    kblasSgemm_lr_lld_batch_strided_wsquery

  #define kblasXgemm_lr_lll                          kblasSgemm_lr_lll
  #define kblasXgemm_lr_lll_wsquery                  kblasSgemm_lr_lll_wsquery
  #define kblasXgemm_lr_lll_batch                    kblasSgemm_lr_lll_batch
  #define kblasXgemm_lr_lll_batch_wsquery            kblasSgemm_lr_lll_batch_wsquery
  #define kblasXgemm_lr_lll_batch_strided            kblasSgemm_lr_lll_batch_strided
  #define kblasXgemm_lr_lll_batch_strided_wsquery    kblasSgemm_lr_lll_batch_strided_wsquery
  
  #define kblasXgemm_tlr_lld                         kblasSgemm_tlr_lld
  #define kblasXgemm_tlr_lld_wsquery                 kblasSgemm_tlr_lld_wsquery
  #define kblasXgemm_plr_dev_tiled_wsquery           kblasSgemm_plr_dev_tiled_wsquery
  #define kblasXgemm_tlr_lll                         kblasSgemm_tlr_lll
  #define kblasXgemm_tlr_lll_wsquery                 kblasSgemm_tlr_lll_wsquery

  #define kblasXsvd_full_batch                    kblasSsvd_full_batch
  #define kblasXsvd_full_batch_wsquery            kblasSsvd_full_batch_wsquery
  #define kblasXsvd_full_batch_strided            kblasSsvd_full_batch_strided
  #define kblasXsvd_full_batch_strided_wsquery    kblasSsvd_full_batch_strided_wsquery
  #define kblasXsvd_full_batch_nonUniform_wsquery kblasSsvd_full_batch_nonUniform_wsquery

  #define cublasXsymm               cublasSsymm
  #define cublasXsyrk               cublasSsyrk
  #define cublasXtrsm               cublasStrsm
  #define cublasXgemm               cublasSgemm
  #define cublasXgeam               cublasSgeam
  #define cublasXscal               cublasSscal
  #define cublasXgemmBatched        cublasSgemmBatched
  #define cublasXgemmStridedBatched cublasSgemmStridedBatched

  #define magmablas_Xgemm_batched               magmablas_sgemm_batched
  #define magmablas_Xgemm_batched_core          magmablas_sgemm_batched_core
  #define magmablas_Xgemm_vbatched_core         magmablas_sgemm_vbatched_core
  #define magma_Xsymm_batched                   magmablas_ssymm_batched
  #define magmablas_Xsymm_vbatched_max_nocheck  magmablas_ssymm_vbatched_max_nocheck
  #define magmablas_Xtrmm_batched_core          magmablas_strmm_batched_core
  #define magmablas_Xtrsm_vbatched_max_nocheck  magmablas_strsm_vbatched_max_nocheck
  #define magmablas_Xsyrk_vbatched_max_nocheck  magmablas_ssyrk_vbatched_max_nocheck

#elif defined PREC_d
  #define TYPE double
  
  #define kblasXpotrf               kblasDpotrf
  
  #define kblasXtrsm                kblasDtrsm
  #define kblasXtrmm                kblasDtrmm

  #define kblasXgemm_batch          kblasDgemm_batch
  #define kblasXgemm_batch_strided  kblasDgemm_batch_strided

  #define kblasXsyrk_batch                  kblasDsyrk_batch
  #define kblasXsyrk_batch_wsquery          kblasDsyrk_batch_wsquery
  #define kblasXsyrk_batch_strided          kblasDsyrk_batch_strided
  #define kblasXsyrk_batch_strided_wsquery  kblasDsyrk_batch_strided_wsquery

  #define kblasXtrsm_batch          kblasDtrsm_batch
  #define kblasXtrsm_batch_strided  kblasDtrsm_batch_strided

  #define kblasXtrmm_batch          kblasDtrmm_batch
  #define kblasXtrmm_batch_strided  kblasDtrmm_batch_strided

  #define kblasXpotrf_batch         kblasDpotrf_batch
  #define kblasXpotrf_batch_strided kblasDpotrf_batch_strided

  #define kblasXlauum_batch         kblasDlauum_batch
  #define kblasXlauum_batch_strided kblasDlauum_batch_strided

  #define kblasXtrtri_batch         kblasDtrtri_batch
  #define kblasXtrtri_batch_strided kblasDtrtri_batch_strided

  #define kblasXpotrs_batch         kblasDpotrs_batch
  #define kblasXpotrs_batch_strided kblasDpotrs_batch_strided

  #define kblasXpotri_batch         kblasDpotri_batch
  #define kblasXpotri_batch_strided kblasDpotri_batch_strided

  #define kblasXpoti_batch          kblasDpoti_batch
  #define kblasXpoti_batch_strided  kblasDpoti_batch_strided

  #define kblasXposv_batch          kblasDposv_batch
  #define kblasXposv_batch_strided  kblasDposv_batch_strided

  #define kblasXgemm_lr_lld                          kblasDgemm_lr_lld
  #define kblasXgemm_lr_lld_wsquery                  kblasDgemm_lr_lld_wsquery
  #define kblasXgemm_lr_lld_batch                    kblasDgemm_lr_lld_batch
  #define kblasXgemm_lr_lld_batch_wsquery            kblasDgemm_lr_lld_batch_wsquery
  #define kblasXgemm_lr_lld_batch_strided            kblasDgemm_lr_lld_batch_strided
  #define kblasXgemm_lr_lld_batch_strided_wsquery    kblasDgemm_lr_lld_batch_strided_wsquery

  #define kblasXgemm_lr_lll                          kblasDgemm_lr_lll
  #define kblasXgemm_lr_lll_wsquery                  kblasDgemm_lr_lll_wsquery
  #define kblasXgemm_lr_lll_batch                    kblasDgemm_lr_lll_batch
  #define kblasXgemm_lr_lll_batch_wsquery            kblasDgemm_lr_lll_batch_wsquery
  #define kblasXgemm_lr_lll_batch_strided            kblasDgemm_lr_lll_batch_strided
  #define kblasXgemm_lr_lll_batch_strided_wsquery    kblasDgemm_lr_lll_batch_strided_wsquery
  
  #define kblasXgemm_tlr_lld                         kblasDgemm_tlr_lld
  #define kblasXgemm_tlr_lld_wsquery                 kblasDgemm_tlr_lld_wsquery
  #define kblasXgemm_plr_dev_tiled_wsquery           kblasDgemm_plr_dev_tiled_wsquery
  #define kblasXgemm_tlr_lll                         kblasDgemm_tlr_lll
  #define kblasXgemm_tlr_lll_wsquery                 kblasDgemm_tlr_lll_wsquery

  #define kblasXsvd_full_batch                    kblasDsvd_full_batch
  #define kblasXsvd_full_batch_wsquery            kblasDsvd_full_batch_wsquery
  #define kblasXsvd_full_batch_strided            kblasDsvd_full_batch_strided
  #define kblasXsvd_full_batch_strided_wsquery    kblasDsvd_full_batch_strided_wsquery
  #define kblasXsvd_full_batch_nonUniform_wsquery kblasDsvd_full_batch_nonUniform_wsquery

  #define cublasXsymm               cublasDsymm
  #define cublasXsyrk               cublasDsyrk
  #define cublasXtrsm               cublasDtrsm
  #define cublasXgemm               cublasDgemm
  #define cublasXgeam               cublasDgeam
  #define cublasXscal               cublasDscal
  #define cublasXgemmBatched        cublasDgemmBatched
  #define cublasXgemmStridedBatched cublasDgemmStridedBatched

  #define magmablas_Xgemm_batched               magmablas_dgemm_batched
  #define magmablas_Xgemm_batched_core          magmablas_dgemm_batched_core
  #define magmablas_Xgemm_vbatched_core         magmablas_dgemm_vbatched_core
  #define magma_Xsymm_batched                   magmablas_dsymm_batched
  #define magmablas_Xsymm_vbatched_max_nocheck  magmablas_dsymm_vbatched_max_nocheck
  #define magmablas_Xtrmm_batched_core          magmablas_dtrmm_batched_core
  #define magmablas_Xtrsm_vbatched_max_nocheck  magmablas_dtrsm_vbatched_max_nocheck
  #define magmablas_Xsyrk_vbatched_max_nocheck  magmablas_dsyrk_vbatched_max_nocheck

#elif defined PREC_c
  #define TYPE cuComplex

  #define kblasXtrsm                kblasCtrsm
  #define kblasXtrmm                kblasCtrmm

  #define kblasXgemm_batch          kblasCgemm_batch
  #define kblasXgemm_batch_strided  kblasCgemm_batch_strided
  // #define Xgemm_batch_strided_wsquery cgemm_batch_strided_wsquery

  #define kblasXsyrk_batch_wsquery          kblasCsyrk_batch_wsquery
  #define kblasXsyrk_batch                  kblasCsyrk_batch
  #define kblasXsyrk_batch_strided_wsquery  kblasCsyrk_batch_strided_wsquery
  #define kblasXsyrk_batch_strided          kblasCsyrk_batch_strided

  #define kblasXtrsm_batch          kblasCtrsm_batch
  #define kblasXtrsm_batch_strided  kblasCtrsm_batch_strided

  #define kblasXtrmm_batch          kblasCtrmm_batch
  #define kblasXtrmm_batch_strided  kblasCtrmm_batch_strided

  #define kblasXpotrf_batch         kblasCpotrf_batch
  #define kblasXpotrf_batch_strided kblasCpotrf_batch_strided

  #define kblasXlauum_batch         kblasClauum_batch
  #define kblasXlauum_batch_strided kblasClauum_batch_strided

  #define kblasXtrtri_batch         kblasCtrtri_batch
  #define kblasXtrtri_batch_strided kblasCtrtri_batch_strided

  #define kblasXpotrs_batch         kblasCpotrs_batch
  #define kblasXpotrs_batch_strided kblasCpotrs_batch_strided

  #define kblasXpotri_batch         kblasCpotri_batch
  #define kblasXpotri_batch_strided kblasCpotri_batch_strided

  #define kblasXpoti_batch          kblasCpoti_batch
  #define kblasXpoti_batch_strided  kblasCpoti_batch_strided

  #define kblasXposv_batch          kblasCposv_batch
  #define kblasXposv_batch_strided  kblasCposv_batch_strided

  #define kblasXgemm_lr_lld                          kblasCgemm_lr_lld
  #define kblasXgemm_lr_lld_wsquery                  kblasCgemm_lr_lld_wsquery
  #define kblasXgemm_lr_lld_batch                    kblasCgemm_lr_lld_batch
  #define kblasXgemm_lr_lld_batch_wsquery            kblasCgemm_lr_lld_batch_wsquery
  #define kblasXgemm_lr_lld_batch_strided            kblasCgemm_lr_lld_batch_strided
  #define kblasXgemm_lr_lld_batch_strided_wsquery    kblasCgemm_lr_lld_batch_strided_wsquery

  #define kblasXgemm_lr_lll                          kblasCgemm_lr_lll
  #define kblasXgemm_lr_lll_wsquery                  kblasCgemm_lr_lll_wsquery
  #define kblasXgemm_lr_lll_batch                    kblasCgemm_lr_lll_batch
  #define kblasXgemm_lr_lll_batch_wsquery            kblasCgemm_lr_lll_batch_wsquery
  #define kblasXgemm_lr_lll_batch_strided_wsquery    kblasCgemm_lr_lll_batch_strided_wsquery
  
  #define kblasXgemm_tlr_lld                         kblasCgemm_tlr_lld
  #define kblasXgemm_tlr_lld_wsquery                 kblasCgemm_tlr_lld_wsquery

  #define cublasXsymm               cublasCsymm
  #define cublasXsyrk               cublasCsyrk
  #define cublasXtrsm               cublasCtrsm
  #define cublasXgemm               cublasCgemm
  #define cublasXgeam               cublasCgeam
  #define cublasXscal               cublasCscal
  #define cublasXgemmBatched        cublasCgemmBatched
  #define cublasXgemmStridedBatched cublasCgemmStridedBatched

  #define magmablas_Xgemm_batched               magmablas_cgemm_batched
  #define magmablas_Xgemm_batched_core          magmablas_cgemm_batched_core
  #define magmablas_Xgemm_vbatched_core         magmablas_cgemm_vbatched_core
  #define magma_Xsymm_batched                   magmablas_csymm_batched
  #define magmablas_Xtrmm_batched_core          magmablas_ctrmm_batched_core
  #define magmablas_Xtrsm_vbatched_max_nocheck  magmablas_ctrsm_vbatched_max_nocheck
  #define magmablas_Xsyrk_vbatched_max_nocheck  magmablas_csyrk_vbatched_max_nocheck

#elif defined PREC_z
  #define TYPE cuDoubleComplex

  #define kblasXtrsm                kblasZtrsm
  #define kblasXtrmm                kblasZtrmm


  #define kblasXgemm_batch          kblasZgemm_batch
  #define kblasXgemm_batch_strided  kblasZgemm_batch_strided

  #define kblasXsyrk_batch                  kblasZsyrk_batch
  #define kblasXsyrk_batch_wsquery          kblasZsyrk_batch_wsquery
  #define kblasXsyrk_batch_strided          kblasZsyrk_batch_strided
  #define kblasXsyrk_batch_strided_wsquery  kblasZsyrk_batch_strided_wsquery

  #define kblasXtrsm_batch          kblasZtrsm_batch
  #define kblasXtrsm_batch_strided  kblasZtrsm_batch_strided

  #define kblasXtrmm_batch          kblasZtrmm_batch
  #define kblasXtrmm_batch_strided  kblasZtrmm_batch_strided

  #define kblasXpotrf_batch         kblasZpotrf_batch
  #define kblasXpotrf_batch_strided kblasZpotrf_batch_strided

  #define kblasXlauum_batch         kblasZlauum_batch
  #define kblasXlauum_batch_strided kblasZlauum_batch_strided

  #define kblasXtrtri_batch         kblasZtrtri_batch
  #define kblasXtrtri_batch_strided kblasZtrtri_batch_strided

  #define kblasXpotrs_batch         kblasZpotrs_batch
  #define kblasXpotrs_batch_strided kblasZpotrs_batch_strided

  #define kblasXpotri_batch         kblasZpotri_batch
  #define kblasXpotri_batch_strided kblasZpotri_batch_strided

  #define kblasXpoti_batch          kblasZpoti_batch
  #define kblasXpoti_batch_strided  kblasZpoti_batch_strided

  #define kblasXposv_batch          kblasZposv_batch
  #define kblasXposv_batch_strided  kblasZposv_batch_strided

  #define kblasXgemm_lr_lld                          kblasZgemm_lr_lld
  #define kblasXgemm_lr_lld_wsquery                  kblasZgemm_lr_lld_wsquery
  #define kblasXgemm_lr_lld_batch                    kblasZgemm_lr_lld_batch
  #define kblasXgemm_lr_lld_batch_wsquery            kblasZgemm_lr_lld_batch_wsquery
  #define kblasXgemm_lr_lld_batch_strided            kblasZgemm_lr_lld_batch_strided
  #define kblasXgemm_lr_lld_batch_strided_wsquery    kblasZgemm_lr_lld_batch_strided_wsquery

  #define kblasXgemm_lr_lll                          kblasZgemm_lr_lll
  #define kblasXgemm_lr_lll_wsquery                  kblasZgemm_lr_lll_wsquery
  #define kblasXgemm_lr_lll_batch                    kblasZgemm_lr_lll_batch
  #define kblasXgemm_lr_lll_batch_wsquery            kblasZgemm_lr_lll_batch_wsquery
  #define kblasXgemm_lr_lll_batch_strided            kblasZgemm_lr_lll_batch_strided
  #define kblasXgemm_lr_lll_batch_strided_wsquery    kblasZgemm_lr_lll_batch_strided_wsquery
  
  #define kblasXgemm_tlr_lld                         kblasZgemm_tlr_lld
  #define kblasXgemm_tlr_lld_wsquery                 kblasZgemm_tlr_lld_wsquery

  #define cublasXsymm               cublasZsymm
  #define cublasXsyrk               cublasZsyrk
  #define cublasXtrsm               cublasZtrsm
  #define cublasXgemm               cublasZgemm
  #define cublasXgeam               cublasZgeam
  #define cublasXscal               cublasZscal
  #define cublasXgemmBatched        cublasZgemmBatched
  #define cublasXgemmStridedBatched cublasZgemmStridedBatched

  #define magmablas_Xgemm_batched               magmablas_zgemm_batched
  #define magmablas_Xgemm_batched_core          magmablas_zgemm_batched_core
  #define magmablas_Xgemm_vbatched_core         magmablas_zgemm_vbatched_core
  #define magma_Xsymm_batched                   magmablas_zsymm_batched
  #define magmablas_Xtrmm_batched_core          magmablas_ztrmm_batched_core
  #define magmablas_Xtrsm_vbatched_max_nocheck  magmablas_ztrsm_vbatched_max_nocheck
  #define magmablas_Xsyrk_vbatched_max_nocheck  magmablas_zsyrk_vbatched_max_nocheck

#else
  #error "No precesion defined"
#endif

#endif //__KBLAS_PREC_DEF__

