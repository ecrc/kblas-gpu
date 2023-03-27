install(FILES
  batch_ara.h
  batch_block_copy.h
  batch_geqp.h
  batch_pstrf.h
  batch_qr.h
  batch_rand.h
  batch_svd.h
  batch_transpose.h
  kblas_batch.h
  kblas_defs.h
  kblas.h
  kblas_l2.h
  kblas_l3.h
  kblas_operators.h
  kblas_tlr.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

target_include_directories(kblasgpu PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>)
