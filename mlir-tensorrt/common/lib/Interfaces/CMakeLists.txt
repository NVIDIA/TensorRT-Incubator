# Directly include the common includes to satisfy `mlir_tablegen`.
function(mlir_tensorrt_common_interface interface)
  cmake_parse_arguments(
    "ARG" "" "KIND" "INCLUDE_DEPENDS" ${ARGN}
  )
  set(LLVM_TARGET_DEFINITIONS
    "${MLIR_TENSORRT_COMMON_SOURCE_DIR}/include/mlir-tensorrt-common/Interfaces/${interface}.td")
  set(OUTPUT_DIR
     "${MLIR_TENSORRT_COMMON_BINARY_DIR}/include/mlir-tensorrt-common/Interfaces")

  set(td_flags "")
  foreach(include_dep IN LISTS ARG_INCLUDE_DEPENDS)
    get_target_property(incDirs ${include_dep} INTERFACE_INCLUDE_DIRECTORIES)
    list(TRANSFORM incDirs PREPEND "-I")
    list(APPEND td_flags ${incDirs})
  endforeach()

  cmake_path(SET OUTPUT_DIR NORMALIZE "${OUTPUT_DIR}")
  cmake_path(RELATIVE_PATH OUTPUT_DIR BASE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  mlir_tablegen("${OUTPUT_DIR}/${interface}.h.inc" -gen-${ARG_KIND}-interface-decls ${td_flags})
  mlir_tablegen("${OUTPUT_DIR}/${interface}.cpp.inc" -gen-${ARG_KIND}-interface-defs ${td_flags})
  mtrt_add_public_tablegen_target(MLIRTensorRTCommon${interface}IncGen)
endfunction()

function(add_mlir_tensorrt_common_interface_library target)
   add_mlir_tensorrt_library("${target}"
    PARTIAL_SOURCES_INTENDED
    ${ARGN}
   )
endfunction()

mlir_tensorrt_common_interface(ToLoopsOpInterface
 KIND op
 INCLUDE_DEPENDS MLIRIR MLIRTensorRTCommonIncludes
)
mlir_tensorrt_common_interface(BoundsAttrInterface
 KIND attr
 INCLUDE_DEPENDS MLIRIR MLIRTensorRTCommonIncludes
)
mlir_tensorrt_common_interface(TensorKindOpInterface
  KIND op
  INCLUDE_DEPENDS MLIRIR MLIRTensorRTCommonIncludes
)
mlir_tensorrt_common_interface(TensorKindAttrInterface
  KIND attr
  INCLUDE_DEPENDS MLIRIR MLIRTensorRTCommonIncludes
)
mlir_tensorrt_common_interface(BufferizationScopeInterface
  KIND op
  INCLUDE_DEPENDS MLIRIR MLIRTensorRTCommonIncludes
)
mlir_tensorrt_common_interface(StreamSchedulableOpInterface
  KIND op
  INCLUDE_DEPENDS MLIRIR MLIRTensorRTCommonIncludes
)

add_mlir_tensorrt_common_interface_library(
  MLIRTensorRTCommonToLoopsOpInterface

  ToLoopsOpInterface.cpp

  DEPENDS
  MLIRTensorRTCommonToLoopsOpInterfaceIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTCommonIncludes
  MLIR_LIBS PUBLIC
    MLIRIR
  )

add_mlir_tensorrt_common_interface_library(
  MLIRTensorRTCommonBoundsAttrInterface

  BoundsAttrInterface.cpp

  DEPENDS
  MLIRTensorRTCommonBoundsAttrInterfaceIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTCommonIncludes
  MLIR_LIBS PUBLIC
    MLIRInferIntRangeInterface
    MLIRIR
  )

add_mlir_tensorrt_common_interface_library(
  MLIRTensorRTCommonTensorKindInterface
  TensorKindOpInterface.cpp
  TensorKindAttrInterface.cpp

  DEPENDS
  MLIRTensorRTCommonTensorKindOpInterfaceIncGen
  MLIRTensorRTCommonTensorKindAttrInterfaceIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTCommonIncludes
  MLIR_LIBS PUBLIC
    MLIRIR
    MLIRAnalysis
    MLIRFuncDialect
)

add_mlir_tensorrt_common_interface_library(
  MLIRTensorRTCommonBufferizationScopeInterface

  BufferizationScopeInterface.cpp

  DEPENDS
  MLIRTensorRTCommonBufferizationScopeInterfaceIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTCommonIncludes
  MLIR_LIBS PUBLIC
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
)

add_mlir_tensorrt_common_interface_library(
  MLIRTensorRTStreamSchedulableOpInterface

  StreamSchedulableOpInterface.cpp

  DEPENDS
  MLIRTensorRTCommonStreamSchedulableOpInterfaceIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTCommonIncludes
  MLIR_LIBS PUBLIC
    MLIRIR
)
