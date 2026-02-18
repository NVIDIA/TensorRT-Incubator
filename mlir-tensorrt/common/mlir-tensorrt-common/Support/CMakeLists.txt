set(LLVM_TARGET_DEFINITIONS StatusEnums.td)
tablegen(MTRT StatusEnums.h.inc --gen-custom-enum-defs)
tablegen(MTRT StatusEnums.c.h.inc --gen-custom-enum-c-defs)
mtrt_add_public_tablegen_target(MLIRTensorRTStatusEnumIncGen)

add_library(MLIRTensorRTSupportStatus INTERFACE)
target_link_libraries(MLIRTensorRTSupportStatus INTERFACE
  MLIRTensorRTCommonIncludes
)
add_dependencies(MLIRTensorRTSupportStatus MLIRTensorRTStatusEnumIncGen)
mtrt_add_install(MLIRTensorRTSupportStatus UMBRELLA mtrt-libraries)

add_mlir_tensorrt_library(MLIRTensorRTCommandLineExtras
  PARTIAL_SOURCES_INTENDED
  CommandLineExtras.cpp
  Options.cpp

  LINK_LIBS PUBLIC
  MLIRTensorRTCommonIncludes
  )
