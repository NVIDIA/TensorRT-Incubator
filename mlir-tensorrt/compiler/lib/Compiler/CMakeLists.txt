add_subdirectory(InputPipelines)

mtrt_get_project_targets(
    TensorRTDialect
    OUT_VAR MTRTD_LIBS
)

add_mlir_tensorrt_library(MLIRTensorRTCompilerClient
  Client.cpp
  Extension.cpp
  Options.cpp
  Pipeline.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
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
    MLIRIR
    MLIRLLVMIRTransforms
    MLIRLLVMDialect
    MLIRMathToEmitC
    MLIRTargetCpp
)


add_subdirectory(Extensions)

# ------- Tasks go above this point ---------------------

mtrt_get_project_targets(MLIRTensorRT LIBRARY_TYPE DIALECT OUT_VAR MLIR_TENSORRT_DIALECT_LIBS)
mtrt_get_project_targets(MLIRTensorRT LIBRARY_TYPE LIBS OUT_VAR MLIR_TENSORRT_LIBS)
mtrt_get_project_targets(TensorRTDialect LIBRARY_TYPE LIBS OUT_VAR MTRTD_LIBS)
mtrt_get_project_targets(MLIRTensorRT LIBRARY_TYPE MLIR_LIBS OUT_VAR mlir_libs)
mtrt_get_project_targets(Kernel LIBRARY_TYPE LIBS OUT_VAR kernel_libs)

add_mlir_tensorrt_library(MLIRTensorRTInitAllDialects
  InitAllDialects.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
    MLIRTensorRTCompilerIncludes
    ${MLIR_TENSORRT_DIALECT_LIBS}
    ${MLIR_TENSORRT_LIBS}
    ${MTRTD_LIBS}
    ${kernel_libs}

  MLIR_LIBS PUBLIC
    ${mlir_libs}
  )

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
