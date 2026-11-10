set(lapack_sources
  Xpotrf_core.cuh
  Xpotrf.cu)
target_sources(sblas_core PRIVATE ${lapack_sources})
target_sources(dblas_core PRIVATE ${lapack_sources})
