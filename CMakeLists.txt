cmake_minimum_required(VERSION 3.18)
project(kblasgpu VERSION 4.0.0 LANGUAGES CXX CUDA)

option(_SUPPORT_BLAS2_ "" ON)
option(_SUPPORT_BLAS3_ "" ON)
option(_SUPPORT_BATCH_TR_ "" ON)
option(_SUPPORT_TLR_ "" ON)
option(_SUPPORT_SVD_ "" ON)

option(_USE_MAGMA_ "Build with support for MAGMA" ON)

include(GNUInstallDirs)

find_package(BLAS)

if(_USE_MAGMA_)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")
  list(APPEND CMAKE_PREFIX_PATH $ENV{MAGMA_DIR} $ENV{MAGMA_ROOT})
  find_package(MAGMA)
endif()

add_library(kblasgpu "")

add_subdirectory(include)
add_subdirectory(src)

target_link_libraries(kblasgpu PUBLIC BLAS::BLAS)

if(MAGMA_FOUND)
  target_link_libraries(kblasgpu PUBLIC MAGMA::magma)
endif()

install(TARGETS
  kblasgpu sblas_core dblas_core cblas_core zblas_core
  EXPORT KBLASGPUTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT KBLASGPUTargets
  FILE kblasgpu-targets.cmake NAMESPACE KBLASGPU::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kblasgpu)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/kblasgpu-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/kblasgpu-config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kblasgpu)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/kblasgpu-config-version.cmake"
  COMPATIBILITY AnyNewerVersion)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/kblasgpu-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/kblasgpu-config-version.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/kblasgpu)
