mtrt_get_project_targets(Kernel LIBRARY_TYPE CAPI OUT_VAR capi_libs)
list(APPEND capi_libs
  MLIRCAPIDebug
  MLIRCAPIGPU
  MLIRCAPIIR
  MLIRCAPIInterfaces
  MLIRCAPILLVM
  MLIRCAPILinalg
  MLIRCAPIPDL
  MLIRCAPITransformDialect
  MLIRCAPITransformDialectTransforms
  MLIRCAPITransforms
  )

mtrt_add_aggregate_library(KernelCompiler ${capi_libs})
mtrt_check_incorrect_dylib_usage(KernelCompiler NO_MTRT)
