if(MLIR_TRT_TARGET_TENSORRT)
  add_subdirectory(TensorRTDynamicLoader)
endif()


add_mlir_tensorrt_library(MLIRTensorRTOwningModuleRef
  OwningModuleRef.cpp

  PARTIAL_SOURCES_INTENDED

  MLIR_LIBS PUBLIC
    MLIRIR
  )

add_mlir_tensorrt_library(MLIRTensorRTRegionUtils
  RegionUtils.cpp

  PARTIAL_SOURCES_INTENDED

  MLIR_LIBS PUBLIC
    MLIRAnalysis
    MLIRIR
    MLIRTransformUtils
  )

add_mlir_tensorrt_library(MLIRTensorRTUtilsShapeInfo
  ShapeInfo.cpp
  ModuleUtils.cpp

  PARTIAL_SOURCES_INTENDED

  MLIR_LIBS PUBLIC
    MLIRAnalysis
    MLIRIR
    MLIRDialectUtils
    MLIRFuncDialect
  )

add_mlir_tensorrt_library(MLIRTensorRTModuleLikePass
  ModuleLikePass.cpp

  PARTIAL_SOURCES_INTENDED

  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRPass
  )
