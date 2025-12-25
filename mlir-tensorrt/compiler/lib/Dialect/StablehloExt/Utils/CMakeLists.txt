add_mlir_tensorrt_library(MLIRTensorRTStableHloExtUtils
  GatherScatterUtils.cpp
  StablehloBuilder.cpp
  StablehloMatchers.cpp
  Utils.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  ChloOps
  MLIRTensorRTStableHloExtIR
  MLIRTensorRTUtilsShapeInfo
  StablehloOps
  )

