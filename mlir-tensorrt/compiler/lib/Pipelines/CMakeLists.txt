set(LLVM_OPTIONAL_SOURCES
  StableHloInputPipelines.cpp
)
set(pipeline_deps_
  MLIRPass
  MLIRTensorRTCUDADialect
  MLIRTensorRTTensorRTRuntimeDialect
  MLIRTensorRTTransforms
  MLIRTransforms
  )

if(MLIR_TRT_ENABLE_HLO)
  list(APPEND pipeline_srcs_
    StableHloInputPipelines.cpp
  )
  list(APPEND pipeline_deps_
    MLIRTensorRTStableHloExtTransforms
    MLIRTensorRTStablehloInputPreprocessing
    MLIRTensorRTStablehloRaiseQDQ
    MLIRTensorRTStablehloToSCF
    MLIRTensorRTStablehloToTensorRT
  )
endif()

add_mlir_tensorrt_library(MLIRTensorRTPipelines
  ${pipeline_srcs_}
  LINK_LIBS PUBLIC ${pipeline_deps_}
)
