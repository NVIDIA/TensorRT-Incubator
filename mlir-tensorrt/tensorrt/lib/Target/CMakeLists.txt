add_subdirectory(PluginRegistry)
add_subdirectory(TensorRTEncodingOpInterface)

add_mlir_library(MLIRTensorRTTargetTensorRT
  TranslateToTensorRT.cpp
  Registration.cpp

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRTRTTensorRTDynamicLoader
  MLIRTensorRTTensorRTUtils
  MLIRTensorRTTensorRTEncodingOpInterface
  MLIRTensorRTTensorRTPluginRegistry
  MLIRTensorRTTensorRTEncodingImpl
  MLIRTensorRTDialect

  LINK_LIBS PRIVATE
  TensorRTHeaderOnly

  DEPENDS
  MLIRTensorRTTranslationPassIncGen
)
