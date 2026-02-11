# This files builds a monolithic shared library containing
# all of the MLIR-TensorRT and MLIR C and CAPI libraries.
get_property(mtrt_libs GLOBAL PROPERTY MTRT_STATIC_LIBS)
get_property(mlir_libs GLOBAL PROPERTY MLIR_STATIC_LIBS)
mtrt_get_project_targets(MLIRTensorRT LIBRARY_TYPE CAPI OUT_VAR capi_libs)
mtrt_get_project_targets(MLIRTensorRT LIBRARY_TYPE MLIR_LIBS OUT_VAR mtrt_required_mlir_libs)

# Additional libraries for tools.
list(APPEND mlir_libs
  MLIRLspServerLib
  MLIROptLib
  MLIRTranslateLib
  ${mtrt_required_mlir_libs}
)
list(REMOVE_DUPLICATES mlir_libs)

list(APPEND mtrt_libs
  ${capi_libs}
  # Add additional CAPI libraries needed for Python consumers.
  MLIRCAPIDebug
  MLIRCAPIGPU
  MLIRCAPIIR
  MLIRCAPIInterfaces
  MLIRCAPILLVM
  MLIRCAPILinalg
  MLIRCAPIPDL
  MLIRCAPIQuant
  MLIRCAPITransformDialect
  MLIRCAPITransformDialectTransforms
  MLIRCAPITransforms
  )

list(APPEND mtrt_libs
  MLIRTransformDialectTransforms
  MLIRTransformDialect
  MLIRTransformUtils
  MLIRTransformDialectInterfaces
  MLIRControlFlowTransforms
  MLIREmitC            # provides emitc::translateToCpp
  # MLIRTransformInterpreter
  MLIRBufferDeallocationTransforms
)


if(MLIR_TRT_ENABLE_HLO)
  list(APPEND mtrt_libs
    ChloCAPI
    StablehloCAPI
    VhloCAPI
  )
endif()

# Only embed the MLIR libraries if we are building using static libraries, otherwise
# MLIR libraries only built as shared will lack OBJECT libraries. Using BUILD_SHARED_LIBS
# is only used for developer environments, never distribution.
if(NOT BUILD_SHARED_LIBS)
  list(APPEND mtrt_libs ${mlir_libs})
endif()

message(STATUS "LAN ADDED mtrt_libs: {mtrt_libs}")
mtrt_add_aggregate_library(MTRT ${mtrt_libs})

target_link_options(MTRT PRIVATE -Wl,--start-group -Wl,--end-group)

# This is only required for supporting test plugins like TensorRTTestPlugins that
# require LLVMSupport. If we are linking tools against MTRT and BUILD_SHARED_LIBS
# is OFF, then the plugin can't link LLVMSupport directly without ODR violations.
# Otherwise, we shouldn't need this.
if(MLIR_TRT_LINK_MTRT_DYLIB AND NOT BUILD_SHARED_LIBS)
  target_link_libraries(MTRT PRIVATE
    $<LINK_LIBRARY:WHOLE_ARCHIVE,LLVMSupport>
  )
endif()
