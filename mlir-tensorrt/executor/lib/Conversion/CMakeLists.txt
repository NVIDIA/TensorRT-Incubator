include_directories(${CMAKE_CURRENT_BINARY_DIR})
get_target_property(common_include_dirs MLIRTensorRTCommonIncludes INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${common_include_dirs})

add_mlir_executor_library(MLIRTensorRTConvertToExecutorCommon
  PARTIAL_SOURCES_INTENDED
  ConvertToExecutorCommon.cpp

  LINK_LIBS PUBLIC
    MLIRTensorRTExecutorDialect
    MLIRTensorRTExecutorUtils
    MLIRExecutorIncludes
  MLIR_LIBS PUBLIC
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRRewrite
    MLIRTransforms
    MLIRTransformUtils
    MLIRDLTIDialect
  )

add_mlir_pdll_library(MLIRExecutorStdToExectutorPatternsIncGen
  StdToExecutor.pdll
  StdToExecutor.pdll.h.inc
  )

add_mlir_executor_library(MLIRTensorRTStdToExecutor
  PARTIAL_SOURCES_INTENDED
  ExecutorToExecutor.cpp
  MemRefToExecutor.cpp
  StdToExecutor.cpp
  LinalgToExecutor.cpp
  TVMFFIUtils.cpp

  DEPENDS
  MLIRExecutorConversionPassIncGen
  MLIRExecutorStdToExectutorPatternsIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTConvertToExecutorCommon
    MLIRExecutorIncludes
    MLIRTensorRTRuntimeExecutableInterface
    DLPack::Headers
    tvm_ffi_header
  MLIR_LIBS PUBLIC
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRParser
    MLIRRewrite
    MLIRTransforms
    MLIRTransformUtils
)
