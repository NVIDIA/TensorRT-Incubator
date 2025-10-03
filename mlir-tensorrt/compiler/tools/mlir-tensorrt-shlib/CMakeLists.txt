# This files builds a monolithic shared library containing
# all of the MLIR-TensorRT and MLIR C and CAPI libraries.
get_property(mtrt_libs GLOBAL PROPERTY MTRT_STATIC_LIBS)
get_property(mlir_libs GLOBAL PROPERTY MLIR_STATIC_LIBS)
mtrt_get_project_targets(MLIRTensorRT LIBRARY_TYPE CAPI OUT_VAR capi_libs)

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

mtrt_add_aggregate_library(MTRT ${mtrt_libs})
