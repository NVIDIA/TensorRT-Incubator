set(extra_link_libs )

add_mlir_kernel_library(MLIRTensorRTKernelDialect
  Attributes.cpp
  Configuration.cpp
  Enums.cpp
  Ops.cpp
  Interfaces.cpp
  TilingInterfaceImpl.cpp
  TransformScheduleBase.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_KERNEL_SOURCE_DIR}/include

  DEPENDS
  MLIRTensorRTKernelIncGen

  LINK_LIBS PUBLIC
    ${extra_link_libs}
    ${MLIR_TRT_CUDA_TARGET}
    MLIRKernelLinalgUtils
    MLIRKernelScatterUtils
    MLIRTensorRTCommonToLoopsOpInterface
    MLIRTensorRTKernelIncludes
  MLIR_LIBS PUBLIC
    MLIRArithDialect
    MLIRArithValueBoundsOpInterfaceImpl
    MLIRControlFlowTransforms
    MLIRDialect
    MLIRDLTIDialect
    MLIRFuncDialect
    MLIRGPUDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRNVVMDialect
    MLIRPtrDialect
    MLIRSupport
    MLIRTensorUtils
    MLIRVectorDialect
)
