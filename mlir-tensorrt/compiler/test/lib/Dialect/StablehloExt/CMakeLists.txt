add_mlir_tensorrt_test_library(MLIRTensorRTTestStablehloMatchers
  StablehloMatchersTestPass.cpp
  LINK_LIBS PUBLIC
    MLIRTensorRTStableHloExtUtils
  MLIR_LIBS PUBLIC
    MLIRPass
    MLIRRewrite
    MLIRTransformUtils
)
