add_mlir_tensorrt_library(MLIRTensorRTOutlineTensorROps
    OutlineTensorRTOps.cpp

    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRTensorRTPassesIncGenPassIncGen

    LINK_LIBS PUBLIC
      MLIRTensorRTExecutorPasses
      MLIRTensorRTDialectIncludes
      MLIRTensorRTRegionUtils
      MLIRTensorRTStablehloToPlan
      MLIRTensorRTTensorRTBackend

    MLIR_LIBS PUBLIC
      MLIRFuncDialect
      MLIRIR
      MLIRPass
    )
