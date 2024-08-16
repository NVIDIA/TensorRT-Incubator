# Captures the MLIR upstream dialects/libraries that we depend on.
set(MLIR_TENSORRT_UPSTREAM_DEPENDENCIES
  # Dialects
  MLIRFuncDialect
  MLIRArithDialect
  MLIRPDLDialect
  MLIRShapeDialect
  MLIRFuncAllExtensions

  # Useful for PDLL debugging.
  MLIRPDLToPDLInterp

  # Other
  MLIRDialect
  MLIRSupport
)

# Cleanup the list of our libraries.
get_property(MLIR_TENSORRT_LIBS GLOBAL PROPERTY MLIR_TENSORRT_LIBS)
if(MLIR_TENSORRT_LIBS)
  list(REMOVE_DUPLICATES MLIR_TENSORRT_LIBS)
  list(SORT MLIR_TENSORRT_LIBS)
  message(STATUS "MLIR_TENSORRT_LIBS:")
  foreach(lib ${MLIR_TENSORRT_LIBS})
    message(STATUS "   ${lib}")
  endforeach()
endif()
get_property(MLIR_TENSORRT_DIALECT_LIBS GLOBAL PROPERTY MLIR_TENSORRT_DIALECT_LIBS)
if(MLIR_TENSORRT_DIALECT_LIBS)
  list(REMOVE_DUPLICATES MLIR_TENSORRT_DIALECT_LIBS)
  list(SORT MLIR_TENSORRT_DIALECT_LIBS)
  message(STATUS "MLIR_TENSORRT_DIALECT_LIBS:")
  foreach(lib ${MLIR_TENSORRT_DIALECT_LIBS})
    message(STATUS "   ${lib}")
  endforeach()
endif()

if(MLIR_TRT_TARGET_LUA)
  list(APPEND MLIR_TENSORRT_LIBS
    MLIRTensorRTTargetLua
  )
endif()

if(MLIR_TRT_TARGET_TENSORRT)
  list(APPEND MLIR_TENSORRT_LIBS
    MLIRTensorRTTensorRTEncodingImpl
    MLIRTensorRTTargetTensorRT
  )
endif()

if(MLIR_TRT_ENABLE_EXECUTOR)
  list(APPEND MLIR_TENSORRT_UPSTREAM_DEPENDENCIES
    MLIRLinalgTransforms
    MLIRMemRefTransforms
    MLIRSCFToControlFlow
    MLIRTensorInferTypeOpInterfaceImpl
    MLIRAffineDialect
    MLIRArithTransforms
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRBufferizationTransformOps
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRTensorDialect
    MLIRTensorTransformOps
    MLIRTransformDialect
    MLIRTensorRTStdToExecutor
    MLIRTensorRTExecutorTransforms
  )
endif()


if(MLIR_TRT_ENABLE_HLO)
  list(APPEND MLIR_TENSORRT_UPSTREAM_DEPENDENCIES
    StablehloPasses
    VhloOps
    VhloTypes
    ChloOps
  )
endif()

add_mlir_tensorrt_library(MLIRTensorRTRegistration
  Registration.cpp

  LINK_LIBS PUBLIC
  ${MLIR_TENSORRT_UPSTREAM_DEPENDENCIES}
  ${MLIR_TENSORRT_DIALECT_LIBS}
  ${MLIR_TENSORRT_LIBS}
)
