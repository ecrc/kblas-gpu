set(tr_sources
  Xgemm_batch_core.cuh
  Xgemm_batch.cu
  Xhelper_funcs.cu
  Xhelper_funcs.cuh
  Xlauum_batch.cu
  Xlauum_batch_drivers.cuh
  Xlauum_batch_kernels.cuh
  Xposv_batch.cu
  Xposv_batch_drivers.cuh
  Xposv_batch_kernels.cuh
  Xpoti_batch.cu
  Xpoti_batch_drivers.cuh
  Xpoti_batch_kernels.cuh
  Xpotrf_batch.cu
  Xpotrf_batch_drivers.cuh
  Xpotrf_batch_kernels.cuh
  Xpotri_batch.cu
  Xpotri_batch_drivers.cuh
  Xpotri_batch_kernels.cuh
  Xpotrs_batch.cu
  Xpotrs_batch_drivers.cuh
  Xpotrs_batch_kernels.cuh
  Xsyrk_batch.cu
  Xsyrk_batch_drivers.cuh
  Xsyrk_batch_kernels.cuh
  Xtrmm_batch.cu
  Xtrmm_batch_drivers.cuh
  Xtrmm_batch_kernels.cuh
  Xtrsm_batch.cu
  Xtrsm_batch_drivers.cuh
  Xtrsm_batch_kernels.cuh
  Xtrtri_batch.cu
  Xtrtri_batch_drivers.cuh
  Xtrtri_batch_kernels.cuh)

target_sources(sblas_core PRIVATE ${tr_sources})
target_sources(dblas_core PRIVATE ${tr_sources})
target_sources(cblas_core PRIVATE ${tr_sources})
target_sources(zblas_core PRIVATE ${tr_sources})
