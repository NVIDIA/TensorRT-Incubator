get_property(MLIR_TENSORRT_TEST_LIBS GLOBAL PROPERTY MLIR_TENSORRT_TEST_LIBS)
get_property(MLIR_TENSORRT_LIBS GLOBAL PROPERTY MLIR_TENSORRT_LIBS)
get_property(MLIR_TENSORRT_DIALECT_LIBS GLOBAL PROPERTY MLIR_TENSORRT_DIALECT_LIBS)
set(MLIR_TENSORRT_LIBS
  ${MLIR_TENSORRT_LIBS} ${MLIR_TENSORRT_DIALECT_LIBS} ${MLIR_TENSORRT_TEST_LIBS}
  MLIRTensorRTTensorRTEncodingImpl
  MLIRTensorRTRegistration)
list(REMOVE_DUPLICATES MLIR_TENSORRT_LIBS)

include_directories(${MLIR_TENSORRT_ROOT_DIR}/compiler/include)
include_directories(${MLIR_TENSORRT_ROOT_BINARY_DIR}/compiler/include)
include_directories(${MLIR_TENSORRT_ROOT_DIR}/executor/include)
include_directories(${MLIR_TENSORRT_ROOT_BINARY_DIR}/executor/include)
include_directories(${MLIR_INCLUDE_DIRS})

set(LLVM_LINK_COMPONENTS
  Support
  )

# ----------------------------------------------------------------------------
# mlir-tensorrt-tblgen
# ----------------------------------------------------------------------------
add_mlir_tool(mlir-tensorrt-tblgen
  MlirTensorRtTblgen.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRTblgenLib
)
target_link_libraries(mlir-tensorrt-tblgen PRIVATE MLIRTblgenLib)
llvm_update_compile_flags(mlir-tensorrt-tblgen)
mlir_check_all_link_libraries(mlir-tensorrt-tblgen)
_mtrt_set_target_compile_defs(mlir-tensorrt-tblgen)

# ----------------------------------------------------------------------------
# mlir-tensorrt-translate
# ----------------------------------------------------------------------------
set(trt_translate_deps_
  MLIRTranslateLib
  ${MLIR_TENSORRT_LIBS}
  )
if(MLIR_TRT_TARGET_CPP)
  list(APPEND trt_translate_deps_ MLIRTargetCpp)
endif()

add_mlir_tool(mlir-tensorrt-translate
  MlirTensorRtTranslate.cpp
  PARTIAL_SOURCES_INTENDED
  )

target_link_libraries(mlir-tensorrt-translate PRIVATE
  ${trt_translate_deps_}
  )
llvm_update_compile_flags(mlir-tensorrt-translate)
mlir_check_link_libraries(mlir-tensorrt-translate)
_mtrt_set_target_compile_defs(mlir-tensorrt-translate)

# ----------------------------------------------------------------------------
# mlir-tensorrt-lsp-server
# ----------------------------------------------------------------------------
add_mlir_tool(mlir-tensorrt-lsp-server
  MlirTensorRtLspServer.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  ${MLIR_TENSORRT_LIBS}
  MLIRLspServerLib
)

target_link_libraries(mlir-tensorrt-lsp-server PRIVATE
  ${MLIR_TENSORRT_LIBS}
  MLIRLspServerLib
)
llvm_update_compile_flags(mlir-tensorrt-lsp-server)
mlir_check_link_libraries(mlir-tensorrt-lsp-server)
_mtrt_set_target_compile_defs(mlir-tensorrt-lsp-server)
set_target_properties(mlir-tensorrt-lsp-server PROPERTIES EXCLUDE_FROM_ALL TRUE)

# ----------------------------------------------------------------------------
# mlir-tensorrt-opt
# ----------------------------------------------------------------------------
add_mlir_tool(mlir-tensorrt-opt
  MlirTensorRtOpt.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRTensorRTConversionPassIncGen
  MLIRTensorRTTranslationPassIncGen
  MLIROptLib
  ${MLIR_TENSORRT_LIBS}

  SUPPORT_PLUGINS
)

target_link_libraries(mlir-tensorrt-opt PRIVATE
  MLIROptLib
  ${MLIR_TENSORRT_LIBS}
  )

llvm_update_compile_flags(mlir-tensorrt-opt)
mlir_check_link_libraries(mlir-tensorrt-opt)
_mtrt_set_target_compile_defs(mlir-tensorrt-opt)
export_executable_symbols_for_plugins(mlir-tensorrt-opt)

# ----------------------------------------------------------------------------
# mlir-tensorrt-runner
# ----------------------------------------------------------------------------
if(MLIR_TRT_TARGET_LUA)
  add_llvm_executable(mlir-tensorrt-runner
    PARTIAL_SOURCES_INTENDED
    MlirTensorRtRunner.cpp)
  _mtrt_set_target_compile_defs(mlir-tensorrt-runner)
  set(libs_
    MLIRExecutorRunnerLib
    )
  target_link_libraries(mlir-tensorrt-runner PRIVATE
    ${libs_}
  )
  target_include_directories(mlir-tensorrt-runner PRIVATE
    ${MLIR_TENSORRT_ROOT_DIR}/include
    ${MLIR_TENSORRT_ROOT_DIR}/executor/include)
  mlir_check_link_libraries(mlir-tensorrt-runner)
endif()
