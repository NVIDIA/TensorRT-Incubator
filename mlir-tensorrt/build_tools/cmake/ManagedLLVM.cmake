macro(mtrt_llvm_project)
  CPMAddPackage(
    ${ARGN}
    )
  set(MLIR_CMAKE_DIR "${CMAKE_BINARY_DIR}/lib/cmake/mlir")
  set(LLVM_CMAKE_DIR "${llvm_project_BINARY_DIR}/lib/cmake/llvm")
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(LLVMConfig)
  include(MLIRConfig)

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

  # The 'MLIRPythonExtensions.Core' target upstream is missing an
  # EMBED_CAPI_LINK_LIBS argument on 'MLIRCAPITransforms'. Instead, it's
  # declared on the '_mlirRegisterEverything' extension, which appears to be wrong.
  # TODO: fix this upstream.
  if(MLIR_TRT_ENABLE_PYTHON)
    get_property(mlir_core_pybind_capi_embed
      TARGET MLIRPythonExtension.Core
      PROPERTY mlir_python_EMBED_CAPI_LINK_LIBS)
    list(FIND mlir_core_pybind_capi_embed MLIRCAPITransforms item_index)
    if(item_index EQUAL -1)
      set_property(TARGET MLIRPythonExtension.Core
        APPEND PROPERTY mlir_python_EMBED_CAPI_LINK_LIBS MLIRCAPITransforms)
    endif()
  endif()
endmacro()
