set(blas_core_sources
  Xblas_core.ch
  Xblas_core.cu
  Xhelper_funcs.ch)
add_library(sblas_core OBJECT ${blas_core_sources})
add_library(dblas_core OBJECT ${blas_core_sources})
add_library(cblas_core OBJECT ${blas_core_sources})
add_library(zblas_core OBJECT ${blas_core_sources})

target_include_directories(sblas_core PRIVATE
  ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(dblas_core PRIVATE
  ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(cblas_core PRIVATE
  ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(zblas_core PRIVATE
  ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_LIST_DIR})

target_compile_definitions(sblas_core PUBLIC
  TARGET_SM=${CMAKE_CUDA_ARCHITECTURES})
target_compile_definitions(dblas_core PUBLIC
  TARGET_SM=${CMAKE_CUDA_ARCHITECTURES})
target_compile_definitions(cblas_core PUBLIC
  TARGET_SM=${CMAKE_CUDA_ARCHITECTURES})
target_compile_definitions(zblas_core PUBLIC
  TARGET_SM=${CMAKE_CUDA_ARCHITECTURES})

target_compile_definitions(sblas_core PRIVATE PREC_s)
target_compile_definitions(dblas_core PRIVATE PREC_d)
target_compile_definitions(cblas_core PRIVATE PREC_c)
target_compile_definitions(zblas_core PRIVATE PREC_z)

if(MAGMA_FOUND)
  target_compile_definitions(kblasgpu PRIVATE USE_MAGMA)
  target_compile_definitions(sblas_core PRIVATE USE_MAGMA)
  target_compile_definitions(dblas_core PRIVATE USE_MAGMA)
  target_compile_definitions(cblas_core PRIVATE USE_MAGMA)
  target_compile_definitions(zblas_core PRIVATE USE_MAGMA)

  # target_link_libraries(kblasgpu PUBLIC MAGMA::magma)
  target_link_libraries(sblas_core PUBLIC MAGMA::magma)
  target_link_libraries(dblas_core PUBLIC MAGMA::magma)
  target_link_libraries(cblas_core PUBLIC MAGMA::magma)
  target_link_libraries(zblas_core PUBLIC MAGMA::magma)
endif()

target_include_directories(kblasgpu PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(kblasgpu PRIVATE
  sblas_core dblas_core cblas_core zblas_core)
  # $<TARGET_OBJECTS:sblas_core>
  # $<TARGET_OBJECTS:dblas_core>
  # $<TARGET_OBJECTS:cblas_core>
  # $<TARGET_OBJECTS:zblas_core>)

target_sources(kblasgpu
  PRIVATE
  kblas_common.cu
  kblas_common.h
  kblas_error.h
  kblas_gpu_timer.h
  kblas_gpu_util.ch
  kblas_prec_def.h
  kblas_struct.h
  kblas_common.h
  workspace_queries.ch
  workspace_queries.cu)

if(_SUPPORT_BLAS2_)
  add_subdirectory(blas_l2)
endif()
if(_SUPPORT_BLAS3_)
  add_subdirectory(blas_l3)
endif()
if(_SUPPORT_BATCH_TR_)
  add_subdirectory(batch_triangular)
endif()
if(_SUPPORT_TLR_)
  add_subdirectory(blas_tlr)
endif()
if(_SUPPORT_SVD_)
  add_subdirectory(batch_svd)
endif()
