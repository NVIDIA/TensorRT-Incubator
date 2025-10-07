# This files builds a monolithic shared library containing
# all of the MLIR-TensorRT and MLIR C and CAPI libraries.
get_property(mtrt_libs GLOBAL PROPERTY MTRT_STATIC_LIBS)
get_property(mlir_libs GLOBAL PROPERTY MLIR_STATIC_LIBS)
mtrt_get_project_targets(MLIRTensorRT LIBRARY_TYPE CAPI OUT_VAR capi_libs)

# Additional libraries for tools.
list(APPEND mlir_libs
  MLIRLspServerLib
  MLIROptLib
  MLIRTranslateLib
)

list(APPEND mtrt_libs
  ${capi_libs}
  ${mlir_libs}
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

# Only embed the MLIR libraries if we are building using static libraries, otherwise
# MLIR libraries only built as shared will lack OBJECT libraries. Using BUILD_SHARED_LIBS
# is only used for developer environments, never distribution.
if(NOT BUILD_SHARED_LIBS)
  list(APPEND mtrt_libs ${mlir_libs})
endif()

mtrt_add_aggregate_library(MTRT ${mtrt_libs})
