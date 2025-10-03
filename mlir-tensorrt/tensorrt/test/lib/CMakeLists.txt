function(add_mlir_tensorrt_dialect_test_library target)
  mtrt_add_test_library(${target}
    PROJECT_NAME TensorRTDialect
    IGNORE_LINK_MTRT
    ${ARGN}
  )
endfunction()

if(MLIR_TRT_TARGET_TENSORRT)
  add_subdirectory(Target)
endif()
add_subdirectory(TensorRT)
