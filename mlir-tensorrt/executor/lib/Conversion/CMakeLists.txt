include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_executor_library(MLIRTensorRTConvertToExecutorCommon
  PARTIAL_SOURCES_INTENDED
  ConvertToExecutorCommon.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRRewrite
  MLIRTransforms
  MLIRTransformUtils
  MLIRDLTIDialect
  MLIRTensorRTExecutorDialect
  MLIRTensorRTExecutorUtils
  )

add_mlir_pdll_library(MLIRExecutorStdToExectutorPatternsIncGen
  MathToExecutor.pdll
  MathToExecutor.pdll.h.inc
  )

add_mlir_executor_library(MLIRTensorRTStdToExecutor
  PARTIAL_SOURCES_INTENDED
  ExecutorToExecutor.cpp
  MemRefToExecutor.cpp
  StdToExecutor.cpp
  LinalgToExecutor.cpp

  DEPENDS
  MLIRExecutorConversionPassIncGen
  MLIRExecutorStdToExectutorPatternsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRRewrite
  MLIRTensorRTConvertToExecutorCommon
  MLIRTransforms
  MLIRTransformUtils
)
