add_mlir_kernel_library(MLIRTensorRTKernelToLLVM
  KernelToLLVM.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRKernelConversionPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTKernelDialect
  MLIR_LIBS PUBLIC
    MLIREmitCDialect
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRPass
    MLIRRewrite
    MLIRTransforms
    MLIRTransformUtils
)
