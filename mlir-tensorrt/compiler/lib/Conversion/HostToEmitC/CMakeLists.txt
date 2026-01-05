# Embed the StandaloneCPP runtime sources/headers into a generated translation
# unit so the compiler can emit them without reading the source tree at runtime.
set(_mtrt_standalonecpp_dir "${CMAKE_SOURCE_DIR}/executor/lib/Runtime/StandaloneCPP")
set(_mtrt_standalonecpp_inputs
  "${_mtrt_standalonecpp_dir}/MTRTRuntimeCore.cpp"
  "${_mtrt_standalonecpp_dir}/MTRTRuntimeCore.h"
  "${_mtrt_standalonecpp_dir}/MTRTRuntimeCuda.cpp"
  "${_mtrt_standalonecpp_dir}/MTRTRuntimeCuda.h"
  "${_mtrt_standalonecpp_dir}/MTRTRuntimeStatus.cpp"
  "${_mtrt_standalonecpp_dir}/MTRTRuntimeStatus.h"
  "${_mtrt_standalonecpp_dir}/MTRTRuntimeTensorRT.cpp"
  "${_mtrt_standalonecpp_dir}/MTRTRuntimeTensorRT.h"
)
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

# Escape list separators so the custom command passes a single -D...=<list>
# argument to the CMake script.
string(REPLACE ";" "\\;" _mtrt_standalonecpp_inputs_escaped "${_mtrt_standalonecpp_inputs}")
string(REPLACE ";" "\\;" _mtrt_standalonecpp_relpaths_escaped "${_mtrt_standalonecpp_relpaths}")
set(_mtrt_embedded_standalonecpp_cpp
  "${CMAKE_CURRENT_BINARY_DIR}/EmbeddedStandaloneCPP.cpp"
)
add_custom_command(
  OUTPUT "${_mtrt_embedded_standalonecpp_cpp}"
  COMMAND "${CMAKE_COMMAND}"
    -DOUTPUT=${_mtrt_embedded_standalonecpp_cpp}
    -DINPUT_FILES=${_mtrt_standalonecpp_inputs_escaped}
    -DREL_PATHS=${_mtrt_standalonecpp_relpaths_escaped}
    -P "${CMAKE_SOURCE_DIR}/compiler/cmake/GenerateEmbeddedStandaloneCPP.cmake"
  DEPENDS
    "${CMAKE_SOURCE_DIR}/compiler/cmake/GenerateEmbeddedStandaloneCPP.cmake"
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
  MLIRTensorRTConversionPassIncGen

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
