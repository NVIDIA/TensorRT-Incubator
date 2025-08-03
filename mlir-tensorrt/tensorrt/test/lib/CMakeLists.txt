function(add_mlir_tensorrt_dialect_test_library target)
  mtrt_add_project_library(${target}
    PROJECT_NAME TensorRTDialect
    LIBRARY_TYPE TEST_LIBS
    DISABLE_INSTALL
    EXCLUDE_FROM_LIBMLIR
    PARTIAL_SOURCES_INTENDED
    ${ARGN}
  )
endfunction()

if(MLIR_TRT_TARGET_TENSORRT)
  add_subdirectory(Target)
endif()
add_subdirectory(TensorRT)
