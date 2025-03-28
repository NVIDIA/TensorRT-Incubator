add_mlir_tensorrt_library(MLIRTensorRTHostToLLVM
  HostToLLVM.cpp

  DEPENDS
  MLIRTensorRTConversionPassIncGen

  LINK_LIBS PUBLIC
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
  MLIRTensorRTCUDAToLLVM
  MLIRTensorRTTensorRTRuntimeToLLVM
  MLIRUBToLLVM
  MLIRVectorToLLVM
  )
