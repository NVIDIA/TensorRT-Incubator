# This files builds a monolithic shared library containing
# all of the MLIR-TensorRT and MLIR C and CAPI libraries.
set(embedding_candidates MLIROptLib)
mtrt_append_project_targets(embedding_candidates MLIRTensorRT LIBRARY_TYPES CAPI LIBS)
mtrt_append_project_targets(embedding_candidates MLIRExecutor LIBRARY_TYPE CAPI LIBS)

if(MLIR_TRT_ENABLE_TESTING)
  mtrt_append_project_targets(embedding_candidates MLIRTensorRT LIBRARY_TYPE TEST_LIBS)
  mtrt_append_project_targets(embedding_candidates TensorRTDialect LIBRARY_TYPE TEST_LIBS)
endif()

# Get the link closure. Define the frontier as any target that does not have
# a corresponding `obj.${tgt}` target.
# TODO: when using MLIR from source, need to fix  this?
list(REMOVE_DUPLICATES embedding_candidates)
function(_frontier tgt)
  if(NOT TARGET "obj.${tgt}")
    set(_is_frontier TRUE PARENT_SCOPE)
    return()
  endif()
  set(_is_frontier FALSE PARENT_SCOPE)
endfunction()

mtrt_collect_interface_link_closure(embedding_candidates embedding_include_dirs
    ${embedding_candidates} STOP_AT _frontier INCLUDE_FRONTIER)

# Exclude libLLVMTableGen for the following reasons:
#  - it is only used by internal *-tblgen utilities;
#  - it pollutes the global options space.
list(REMOVE_ITEM embedding_candidates "LLVMTableGen")

# Build the aggregate library.
mtrt_add_aggregate_library(MTRT ${embedding_candidates})
target_include_directories(MTRT INTERFACE
  $<BUILD_INTERFACE:${embedding_include_dirs}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(MTRT PUBLIC
  FlatBuffers::FlatBuffers
  MLIRTensorRTCommonIncludes
  MLIRTensorRTDialectIncludes
  $<$<BOOL:${MLIR_TRT_TARGET_TENSORRT}>:TensorRTHeaderOnly>
  )
