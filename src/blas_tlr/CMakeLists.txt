set(gemm_tlr_sources
  Xgemm_tlr_core.cuh
  Xgemm_tlr.cu)
target_sources(sblas_core PRIVATE ${gemm_tlr_sources})
target_sources(dblas_core PRIVATE ${gemm_tlr_sources})
# target_sources(cblas_core PRIVATE ${gemm_tlr_sources})
# target_sources(zblas_core PRIVATE ${gemm_tlr_sources})
