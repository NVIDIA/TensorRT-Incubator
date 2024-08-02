add_mlir_tensorrt_library(MLIRTensorRTStableHloExtUtils
  GatherScatterUtils.cpp
  StablehloBuilder.cpp
  Utils.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTStableHloExtIR
  )
