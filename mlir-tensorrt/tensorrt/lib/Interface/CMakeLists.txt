add_mtrtd_library(MLIRTensorRTInterfaces
  TensorKindOpInterface.cpp

  DEPENDS
  MLIRTensorRTTensorKindOpInterfaceIncGen
  MLIRTensorRTTensorKindAttrInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAnalysis
  MLIRFuncDialect
)
