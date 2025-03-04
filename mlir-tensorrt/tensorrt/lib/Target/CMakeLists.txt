add_subdirectory(PluginRegistry)
add_subdirectory(TensorRTEncodingOpInterface)

add_mtrtd_library(MLIRTensorRTTargetTensorRT
  Registration.cpp
  TranslateToTensorRT.cpp

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRPass
  MLIRTensorRTDialect
  MLIRTensorRTTensorRTEncodingImpl
  MLIRTensorRTTensorRTEncodingOpInterface
  MLIRTensorRTTensorRTPluginRegistry
  MLIRTensorRTTensorRTUtils
  MLIRTransformUtils
  MLIRTranslateLib
  MLIRTRTTensorRTDynamicLoader

  LINK_LIBS PRIVATE
  TensorRTHeaderOnly

  DEPENDS
  MLIRTensorRTTranslationPassIncGen
)
