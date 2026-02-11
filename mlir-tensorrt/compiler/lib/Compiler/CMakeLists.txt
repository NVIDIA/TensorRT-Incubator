add_subdirectory(InputPipelines)

mtrt_get_project_targets(
    TensorRTDialect
    OUT_VAR MTRTD_LIBS
)

# Detect if we're using a pre-built distribution (imported targets) vs building from source
# Only use WHOLE_ARCHIVE with pre-built distributions to avoid duplicate symbols when building from source
get_target_property(_mlir_tensor_dialect_imported MLIRTensorDialect IMPORTED)
if(_mlir_tensor_dialect_imported)
  set(_use_whole_archive TRUE)
  message(STATUS "Detected pre-built MLIR distribution - using WHOLE_ARCHIVE for critical libraries")
else()
  set(_use_whole_archive FALSE)
  message(STATUS "Detected MLIR built from source - using normal linking")
endif()

# Link MLIRTargetCpp and its dependencies BEFORE creating the library
# to ensure translateToCpp symbol is available when Pipeline.cpp is linked.
# This must happen before add_mlir_tensorrt_library so the libraries are
# available when the source files are compiled/linked.
# Only use WHOLE_ARCHIVE with pre-built distributions to avoid duplicate symbols.
add_library(_MLIRTensorRTCompilerClientPrereqs INTERFACE)
if(_use_whole_archive)
  target_link_libraries(_MLIRTensorRTCompilerClientPrereqs INTERFACE
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTargetCpp>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTranslateLib>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIREmitCDialect>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRControlFlowDialect>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRFuncDialect>
  )
else()
  target_link_libraries(_MLIRTensorRTCompilerClientPrereqs INTERFACE
    MLIRTargetCpp
    MLIRTranslateLib
    MLIREmitCDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
  )
endif()

add_mlir_tensorrt_library(MLIRTensorRTCompilerClient
  Client.cpp
  Extension.cpp
  Options.cpp
  Pipeline.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    _MLIRTensorRTCompilerClientPrereqs
    ${MTRTD_LIBS}
    MLIRTensorRTChloToStablehloExt
    MLIRTensorRTCommonConvertToLoops
    MLIRTensorRTCompilerIncludes
    MLIRTensorRTCUDAToExecutor
    MLIRTensorRTCUDATransforms
    MLIRTensorRTDropNestedModules
    MLIRTensorRTExecutorTransforms
    MLIRTensorRTHostBackend
    MLIRTensorRTHostToEmitC
    MLIRTensorRTHostToLLVM
    MLIRTensorRTKernelBackend
    MLIRTensorRTKernelPipelines
    MLIRTensorRTKernelToCUDA
    MLIRTensorRTLinalgElementwiseFusion
    MLIRTensorRTLinalgSimplifyExtractSlice
    MLIRTensorRTLinalgToCUDA
    MLIRTensorRTMemRefCastElimination
    MLIRTensorRTMemRefToCUDA
    MLIRTensorRTOutlineTensorROps
    MLIRTensorRTPlanToExecutor
    MLIRTensorRTPlanTransforms
    MLIRTensorRTRegionUtils
    MLIRTensorRTRuntimeToExecutor
    MLIRTensorRTSCFDetensorizeLoops
    MLIRTensorRTStablehloInputPipeline
    MLIRTensorRTStablehloToArith
    MLIRTensorRTStablehloToLinalg
    MLIRTensorRTStablehloToPlan
    MLIRTensorRTStablehloToSCF
    MLIRTensorRTTargetLua
    MLIRTensorRTTargetLua
    MLIRTensorRTTensorExtPadToInsertSlice
    MLIRTensorRTTensorRTBackend
    MLIRTensorRTTensorRTToTensorRTRuntime
    MLIRTensorRTTransforms
    MLIRTensorRTTransformsSCFFloatStrengthReduce
    MLIRTensorRTTransformsUnrollForLoops
    StablehloLinalgTransforms

  MLIR_LIBS PUBLIC
    MLIRArithTransforms
    MLIREmitCTransforms
    MLIREmitCDialect
    MLIRIR
    MLIRLLVMIRTransforms
    MLIRLLVMDialect
    MLIRMathToEmitC
    MLIRTargetCpp
    MLIRTranslateLib
)
# Only use WHOLE_ARCHIVE with pre-built distributions to avoid duplicate symbols when building from source
if(_use_whole_archive)
  target_link_libraries(MLIRTensorRTCompilerClient PRIVATE
    # Link dependencies first, then MLIRTargetCpp
    # MLIRTargetCpp depends on MLIRTranslateLib, so link it as well
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIREmitCDialect>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRControlFlowDialect>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRFuncDialect>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTranslateLib>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTargetCpp>
  )
  # Also link to object library if it exists
  if(TARGET obj.MLIRTensorRTCompilerClient)
    target_link_libraries(obj.MLIRTensorRTCompilerClient PRIVATE
      # Link dependencies first, then MLIRTargetCpp
      # MLIRTargetCpp depends on MLIRTranslateLib, so link it as well
      $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIREmitCDialect>
      $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRControlFlowDialect>
      $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRFuncDialect>
      $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTranslateLib>
      $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTargetCpp>
    )
  endif()
endif()


add_subdirectory(Extensions)

# ------- Tasks go above this point ---------------------

mtrt_get_project_targets(MLIRTensorRT LIBRARY_TYPE DIALECT OUT_VAR MLIR_TENSORRT_DIALECT_LIBS)
mtrt_get_project_targets(MLIRTensorRT LIBRARY_TYPE LIBS OUT_VAR MLIR_TENSORRT_LIBS)
mtrt_get_project_targets(TensorRTDialect LIBRARY_TYPE LIBS OUT_VAR MTRTD_LIBS)
mtrt_get_project_targets(MLIRTensorRT LIBRARY_TYPE MLIR_LIBS OUT_VAR mlir_libs)
mtrt_get_project_targets(Kernel LIBRARY_TYPE LIBS OUT_VAR kernel_libs)

# Link critical MLIR libraries BEFORE creating the library
# to ensure symbols are available when pre-built object files are linked.
# This must happen before add_mlir_tensorrt_library so the libraries are
# available when kernel_libs (which contains pre-built object files) is processed.
# We create a temporary interface library to hold these dependencies.
# These libraries contain registration functions and TypeID resolvers that must
# be available when pre-built object files from the distribution are linked.
# Only use WHOLE_ARCHIVE with pre-built distributions to avoid duplicate symbols when building from source.
add_library(_MLIRTensorRTInitAllDialectsPrereqs INTERFACE)
if(_use_whole_archive)
  target_link_libraries(_MLIRTensorRTInitAllDialectsPrereqs INTERFACE
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTransformDialectTransforms>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRArithValueBoundsOpInterfaceImpl>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRControlFlowTransforms>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTensorTransformOps>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTensorInferTypeOpInterfaceImpl>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRFuncInlinerExtension>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIREmitCDialect>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRPtrDialect>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTensorDialect>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTransformDialect>
  )
else()
  target_link_libraries(_MLIRTensorRTInitAllDialectsPrereqs INTERFACE
    MLIRTransformDialectTransforms
    MLIRArithValueBoundsOpInterfaceImpl
    MLIRControlFlowTransforms
    MLIRTensorTransformOps
    MLIRTensorInferTypeOpInterfaceImpl
    MLIRFuncInlinerExtension
    MLIREmitCDialect
    MLIRPtrDialect
    MLIRTensorDialect
    MLIRTransformDialect
  )
endif()

add_mlir_tensorrt_library(MLIRTensorRTInitAllDialects
  InitAllDialects.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    MLIRTensorRTCompilerIncludes
    _MLIRTensorRTInitAllDialectsPrereqs
    ${MLIR_TENSORRT_DIALECT_LIBS}
    ${MLIR_TENSORRT_LIBS}
    ${MTRTD_LIBS}
    ${kernel_libs}

  MLIR_LIBS PUBLIC
    ${mlir_libs}
  )
# Link libraries directly (not through MLIR_LIBS)
# IMPORTANT: MLIRTransformDialectTransforms must come FIRST before any CAPI libraries
# that depend on it (like MLIRCAPITransformDialectTransforms). This is critical when
# using pre-built distributions where object files are linked directly.
# The pre-built object file obj.MLIRCAPITransformDialectTransforms/TransformInterpreter.cpp.o
# needs MLIRTransformDialectTransforms to be available, so we link it first.
# Only use WHOLE_ARCHIVE with pre-built distributions to avoid duplicate symbols when building from source.
if(_use_whole_archive)
  target_link_libraries(MLIRTensorRTInitAllDialects PRIVATE
    # Link MLIRTransformDialectTransforms FIRST to satisfy CAPI dependencies
    # This must come before any other libraries to ensure symbols are available
    # when pre-built object files (like obj.MLIRCAPITransformDialectTransforms) are linked
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTransformDialectTransforms>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRArithValueBoundsOpInterfaceImpl>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRControlFlowTransforms>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIREmitCDialect>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRFuncInlinerExtension>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRPtrDialect>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTensorDialect>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTensorInferTypeOpInterfaceImpl>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTensorTransformOps>
    $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTransformDialect>
  )
  # Also link to object library if it exists
  if(TARGET obj.MLIRTensorRTInitAllDialects)
    target_link_libraries(obj.MLIRTensorRTInitAllDialects PRIVATE
      # Link MLIRTransformDialectTransforms FIRST to satisfy CAPI dependencies
      $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTransformDialectTransforms>
      $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRArithValueBoundsOpInterfaceImpl>
      $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRControlFlowTransforms>
      $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIREmitCDialect>
      $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRFuncInlinerExtension>
      $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRPtrDialect>
      $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTensorDialect>
      $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTensorInferTypeOpInterfaceImpl>
      $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTensorTransformOps>
      $<LINK_LIBRARY:WHOLE_ARCHIVE,MLIRTransformDialect>
    )
  endif()
endif()

# Separate the pass registration from dialect registration.
# Pass registration is only needed by tools, so may not be required
# for certain clients.
add_mlir_tensorrt_library(MLIRTensorRTInitAllPasses
  InitAllPasses.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
    mtrt-headers

  LINK_LIBS PUBLIC
    MLIRTensorRTCompilerIncludes
    ${MLIR_TENSORRT_DIALECT_LIBS}
    ${MLIR_TENSORRT_LIBS}
    ${MTRTD_LIBS}
    ${kernel_libs}

  MLIR_LIBS PUBLIC
    ${mlir_libs}
  )
