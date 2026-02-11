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

add_mlir_kernel_library(MLIRTensorRTKernelTransformSchedules
  FallbackSchedule.cpp
  InitialTransformSchedule.cpp
  ScatterSchedule.cpp
  TransformBuilder.cpp
  TransformSchedules.cpp

  DEPENDS
  MLIRTensorRTKernelTransformSchedulesPassIncGen

  LINK_LIBS PUBLIC
    MLIRKernelTilingUtils
    MLIRTensorRTCommonToLoopsOpInterface
    MLIRTensorRTKernelDialect
    MLIRTensorRTKernelTransformOps
    MLIRTensorRTSupportDeviceInfo
    MLIRTensorRTSupportStatus
  MLIR_LIBS PUBLIC
    MLIRBufferizationTransformOps
    MLIRIR
    MLIRLinalgDialect
    MLIRPass
    MLIRTensorTransformOps
    MLIRTransformDialect
    MLIRTransformDialectTransforms
    MLIRTransformLoopExtension
    MLIRVectorTransformOps
)
target_include_directories(obj.MLIRTensorRTKernelTransformSchedules
  PRIVATE
  ${MLIR_TENSORRT_ROOT_DIR}/executor/include
  ${MLIR_TENSORRT_ROOT_BINARY_DIR}/executor/include)
# Only use WHOLE_ARCHIVE with pre-built distributions to avoid duplicate symbols when building from source
if(_use_whole_archive)
  target_link_libraries(MLIRTensorRTKernelTransformSchedules PRIVATE
    # Link transform libraries with WHOLE_ARCHIVE to ensure TypeID resolvers and build methods
    # for operations like ApplyFoldTensorEmptyPatternsOp are available.
    # Order: base dialect first, then operations, then utilities
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTransformDialect>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTensorTransformOps>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTransformDialectTransforms>
  )
  # Also link to object library if it exists
  if(TARGET obj.MLIRTensorRTKernelTransformSchedules)
    target_link_libraries(obj.MLIRTensorRTKernelTransformSchedules PRIVATE
      # Link transform libraries with WHOLE_ARCHIVE to ensure TypeID resolvers and build methods
      # for operations like ApplyFoldTensorEmptyPatternsOp are available.
      # Order: base dialect first, then operations, then utilities
      $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTransformDialect>
      $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTensorTransformOps>
      $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTransformDialectTransforms>
    )
  endif()
endif()
