add_subdirectory(TensorRTToEmitC)

# Embed the StandaloneCPP runtime sources/headers into a generated translation
# unit so the compiler can emit them without reading the source tree at runtime.
set(_mtrt_standalonecpp_dir "${CMAKE_SOURCE_DIR}/executor/lib/Runtime/StandaloneCPP")
set(_mtrt_standalonecpp_relpaths
  "MTRTRuntimeCore.cpp"
  "MTRTRuntimeCore.h"
  "MTRTRuntimeCuda.cpp"
  "MTRTRuntimeCuda.h"
  "MTRTRuntimeStatus.cpp"
  "MTRTRuntimeStatus.h"
  "MTRTRuntimeTensorRT.cpp"
  "MTRTRuntimeTensorRT.h"
)
# Derive absolute paths from the relative paths.
set(_mtrt_standalonecpp_inputs "${_mtrt_standalonecpp_relpaths}")
list(TRANSFORM _mtrt_standalonecpp_inputs PREPEND "${_mtrt_standalonecpp_dir}/")

# Escape list separators so the custom command passes a single -D...=<list>
# argument to the CMake script.
string(REPLACE ";" "\\;" _mtrt_standalonecpp_inputs_escaped "${_mtrt_standalonecpp_inputs}")
string(REPLACE ";" "\\;" _mtrt_standalonecpp_relpaths_escaped "${_mtrt_standalonecpp_relpaths}")
set(_mtrt_embedded_standalonecpp_cpp
  "${CMAKE_CURRENT_BINARY_DIR}/EmbeddedStandaloneCPP.cpp"
)
set(_mtrt_generate_embedded_file_group
  "${MLIR_TENSORRT_ROOT_DIR}/build_tools/cmake/MTRTGenerateEmbeddedFileGroup.cmake"
)
add_custom_command(
  OUTPUT "${_mtrt_embedded_standalonecpp_cpp}"
  COMMAND "${CMAKE_COMMAND}"
    -DOUTPUT=${_mtrt_embedded_standalonecpp_cpp}
    -DINPUT_FILES=${_mtrt_standalonecpp_inputs_escaped}
    -DREL_PATHS=${_mtrt_standalonecpp_relpaths_escaped}
    -DGROUP_NAME=StandaloneCPP
    -DHEADER_INCLUDE=mlir-tensorrt/Compiler/Passes/2_Host/HostToEmitC/HostToEmitC/EmbeddedStandaloneCPP.h
    -DNAMESPACE=mtrt::compiler::emitc_support
    -P "${_mtrt_generate_embedded_file_group}"
  DEPENDS
    "${_mtrt_generate_embedded_file_group}"
    ${_mtrt_standalonecpp_inputs}
  VERBATIM
)

add_mlir_tensorrt_library(MLIRTensorRTHostToEmitC
  HostToEmitC.cpp
  HostToEmitCGlobals.cpp
  HostToEmitCPatternsCuda.cpp
  HostToEmitCPatternsExecutor.cpp
  HostToEmitCPatternsMemRef.cpp
  HostToEmitCPatternsTensorRT.cpp
  EmitCppSupportFiles.cpp
  WrapModuleInEmitCClass.cpp
  "${_mtrt_embedded_standalonecpp_cpp}"

  DEPENDS
  MLIRTensorRTPassesIncGenPassIncGen

  LINK_LIBS PUBLIC
    MLIRTensorRTTensorRTToEmitC
    MLIRTensorRTExecutorDialect
    MLIRTensorRTCUDADialect
    MLIRTensorRTTensorRTRuntimeDialect
    MLIRTensorRTLLVMConversionUtils
    MLIRTensorRTSupportArtifactManager
  MLIR_LIBS PUBLIC
    MLIRArithToEmitC
    MLIRArithTransforms
    MLIREmitCDialect
    MLIRFuncToEmitC
    MLIRFuncTransforms
    MLIRMathDialect
    MLIRReconcileUnrealizedCasts
    MLIRSCFToEmitC
    MLIRTransforms
    MLIRMathToEmitC
  )
