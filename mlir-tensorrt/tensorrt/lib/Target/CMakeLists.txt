

if(MLIR_TRT_TARGET_TENSORRT)
  add_subdirectory(PluginRegistry)
  add_subdirectory(TensorRTEncodingOpInterface)
  set(translation_deps
    MLIRTensorRTTensorRTEncodingImpl
    MLIRTensorRTTensorRTEncodingOpInterface
    TensorRTHeaderOnly
    TensorRTDynamicLoader
    MLIRTensorRTTensorRTPluginRegistry
    )
endif()

add_mtrtd_library(MLIRTensorRTTargetTensorRT
  Registration.cpp
  TranslateToTensorRT.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTCommandLineExtras
    MLIRTensorRTTensorRTDialect
    MLIRTensorRTTensorRTUtils
    ${translation_deps}
  MLIR_LIBS PUBLIC
    MLIRDialect
    MLIRIR
    MLIRPass
    MLIRTransformUtils
    MLIRTranslateLib
  DEPENDS
  MLIRTensorRTTranslationPassIncGen
)
