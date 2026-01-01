add_mlir_tensorrt_library(MLIRTensorRTOutlineTensorROps
    OutlineTensorRTOps.cpp

    PARTIAL_SOURCES_INTENDED

    DEPENDS
    MLIRTensorRTGenericTransformPassIncGen

    LINK_LIBS PUBLIC
      MLIRExecutorGenericClustering
      MLIRTensorRTDialectIncludes
      MLIRTensorRTRegionUtils
      MLIRTensorRTStablehloToPlan
      MLIRTensorRTTensorRTBackend

    MLIR_LIBS PUBLIC
      MLIRFuncDialect
      MLIRIR
      MLIRPass
    )
