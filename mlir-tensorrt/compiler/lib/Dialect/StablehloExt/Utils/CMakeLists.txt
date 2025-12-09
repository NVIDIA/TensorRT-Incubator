add_mlir_tensorrt_library(MLIRTensorRTStableHloExtUtils
  GatherScatterUtils.cpp
  StablehloBuilder.cpp
  StablehloMatchers.cpp
  Utils.cpp

  LINK_LIBS PUBLIC
  ChloOps
  MLIRTensorRTStableHloExtIR
  MLIRTensorRTUtilsShapeInfo
  StablehloOps
  )
