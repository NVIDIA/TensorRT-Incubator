set(LLVM_OPTIONAL_SOURCES
  StableHloInputPipelines.cpp
)
set(pipeline_deps_
  MLIRPass
  MLIRTransforms
  MLIRTensorRTTransforms
  MLIRTensorRTTensorRTRuntimeDialect
  MLIRTensorRTCUDADialect
  )

if(MLIR_TRT_ENABLE_HLO)
  list(APPEND pipeline_srcs_
    StableHloInputPipelines.cpp
  )
  list(APPEND pipeline_deps_
    MLIRTensorRTStableHloExtTransforms
    MLIRTensorRTStablehloToTensorRT
    MLIRTensorRTStablehloInputPreprocessing
  )
endif()

add_mlir_tensorrt_library(MLIRTensorRTPipelines
  ${pipeline_srcs_}
  LINK_LIBS PUBLIC ${pipeline_deps_}
)
