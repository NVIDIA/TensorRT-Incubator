add_mlir_tensorrt_library(MLIRTensorRTHostToLLVM
  HostToLLVM.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTCUDAToLLVM
    MLIRTensorRTTensorRTRuntimeToLLVM
  MLIR_LIBS PUBLIC
    MLIRArithToLLVM
    MLIRComplexToLLVM
    MLIRControlFlowToLLVM
    MLIRConvertToLLVMInterface
    MLIRConvertToLLVMPass
    MLIRFuncToLLVM
    MLIRIndexToLLVM
    MLIRLLVMDialect
    MLIRMathToLLVM
    MLIRMemRefToLLVM
    MLIRNVVMToLLVM
    MLIRUBToLLVM
    MLIRVectorToLLVM
  )
