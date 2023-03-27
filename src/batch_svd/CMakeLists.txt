set(svd_sources
  Xaca_batch_core.cuh
  Xaca_batch.cu
  Xsvd_full_batch_core.cuh
  Xsvd_full_batch.cu)
target_sources(sblas_core PRIVATE ${svd_sources})
target_sources(dblas_core PRIVATE ${svd_sources})
# target_sources(cblas_core PRIVATE ${svd_sources})
# target_sources(zblas_core PRIVATE ${svd_sources})

target_sources(kblasgpu
  PRIVATE
  ara_util.cuh
  batch_ara.cu
  batch_block_copy.cu
  batch_geqp.cu
  batch_pstrf.cu
  batch_qr.cu
  batch_rand.cu
  batch_svd.cu
  batch_transpose.cu
  perf_counter.h
  qr_kernels.cuh
  svd_kernels.cuh
  thrust_wrappers.cu
  thrust_wrappers.h)
