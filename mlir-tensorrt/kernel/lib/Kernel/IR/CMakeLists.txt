# Detect if we're using a pre-built distribution (imported targets) vs building from source
# Only use WHOLE_ARCHIVE with pre-built distributions to avoid duplicate symbols when building from source
if(NOT DEFINED _use_whole_archive)
  get_target_property(_mlir_tensor_dialect_imported MLIRTensorDialect IMPORTED)
  if(_mlir_tensor_dialect_imported)
    set(_use_whole_archive TRUE)
  else()
    set(_use_whole_archive FALSE)
  endif()
endif()

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
# Only use WHOLE_ARCHIVE with pre-built distributions to avoid duplicate symbols when building from source
if(_use_whole_archive)
  target_link_libraries(MLIRTensorRTKernelDialect PRIVATE
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRPtrDialect>
  )
  # Also link to object library if it exists
  if(TARGET obj.MLIRTensorRTKernelDialect)
    target_link_libraries(obj.MLIRTensorRTKernelDialect PRIVATE
      $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRPtrDialect>
    )
  endif()
endif()