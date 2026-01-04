install(
  FILES
  MTRTRuntimeCore.cpp
  MTRTRuntimeCuda.cpp
  MTRTRuntimeTensorRT.cpp
  MTRTRuntimeStatus.cpp
  MTRTRuntimeCore.h
  MTRTRuntimeCuda.h
  MTRTRuntimeTensorRT.h
  MTRTRuntimeStatus.h

  DESTINATION src/mtrt-runtime

  COMPONENT
  MTRT_CPP_Runtime
)

add_library(MTRTStandaloneCPPRuntime STATIC
  MTRTRuntimeCore.cpp
  MTRTRuntimeStatus.cpp
)

target_include_directories(MTRTStandaloneCPPRuntime PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
)

if( MLIR_TRT_ENABLE_CUDA)
  target_sources(MTRTStandaloneCPPRuntime PRIVATE
    MTRTRuntimeCuda.cpp
  )
  target_link_libraries(MTRTStandaloneCPPRuntime PUBLIC
    ${MLIR_TRT_CUDA_TARGET}
  )
endif()

if( MLIR_TRT_TARGET_TENSORRT)
  target_sources(MTRTStandaloneCPPRuntime PRIVATE
    MTRTRuntimeTensorRT.cpp
  )
  target_link_libraries(MTRTStandaloneCPPRuntime PUBLIC
    TensorRTHeaderOnly
  )
endif()
