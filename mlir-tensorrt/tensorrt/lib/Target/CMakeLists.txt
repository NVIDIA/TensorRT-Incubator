

if(MLIR_TRT_TARGET_TENSORRT)
  add_subdirectory(PluginRegistry)
  add_subdirectory(TensorRTEncodingOpInterface)
  set(translation_deps
    MLIRTensorRTTensorRTEncodingImpl
    MLIRTensorRTTensorRTEncodingOpInterface
    TensorRTHeaderOnly
    MLIRTRTTensorRTDynamicLoader
    MLIRTensorRTTensorRTPluginRegistry
    )
endif()

add_mtrtd_library(MLIRTensorRTTargetTensorRT
  Registration.cpp
  TranslateToTensorRT.cpp

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRPass
  MLIRTensorRTCommandLineExtras
  MLIRTensorRTTensorRTDialect
  MLIRTensorRTTensorRTUtils
  MLIRTransformUtils
  MLIRTranslateLib
  ${translation_deps}

  DEPENDS
  MLIRTensorRTTranslationPassIncGen
)
