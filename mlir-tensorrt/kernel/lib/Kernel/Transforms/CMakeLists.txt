include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_kernel_library(MLIRTensorRTKernelTransforms
  AffineBoundsOptimization.cpp
  AnnotateKernelEntrypoints.cpp
  ApplyTransforms.cpp
  BufferizableOpInterfaceImpl.cpp
  BufferizationScopeOpInterfaceImpl.cpp
  SpecialFloatsTypeConversion.cpp
  DispatchGPUModuleCompilation.cpp
  ExpandMemRefArgs.cpp
  ExpandOps.cpp
  InferArgIndexingBounds.cpp
  KernelBufferize.cpp
  KernelToLLVMIRTranslation.cpp
  LowerKernelSort.cpp
  LowerToNVVM.cpp
  NormalizeQuantizedConversions.cpp
  PrepareLinalgForCodegen.cpp
  RefineArgumentLayouts.cpp
  SetGPUTarget.cpp
  SharedAllocToGlobal.cpp
  TranslateNVVMToPTX.cpp
  VectorTransforms.cpp
  GenerateSort.cpp

  DEPENDS
  MLIRTensorRTKernelTransformsPassIncGen

  LINK_LIBS PUBLIC
    MLIRKernelLinalgUtils
    MLIRTensorRTCommonBufferizationScopeInterface
    MLIRTensorRTCUDAToolkitUtils
    MLIRTensorRTKernelDialect
    MLIRTensorRTKernelIncludes
    MLIRTensorRTKernelToLLVM
    MLIRTensorRTSupportDeviceInfo
  LINK_COMPONENTS
    Analysis
    MC
    NVPTXCodeGen
    Passes
    BitReader
  MLIR_LIBS PUBLIC
    MLIRAffineDialect
    MLIRAffineToStandard
    MLIRAffineTransforms
    MLIRArithToEmitC
    MLIRArithToLLVM
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRBuiltinToLLVMIRTranslation
    MLIRComplexToLLVM
    MLIRComplexToStandard
    MLIRControlFlowToLLVM
    MLIREmitCTransforms
    MLIRFuncToEmitC
    MLIRFuncToLLVM
    MLIRGPUDialect
    MLIRGPUToLLVMIRTranslation
    MLIRGPUToNVVMTransforms
    MLIRIndexToLLVM
    MLIRIR
    MLIRLinalgTransformOps
    MLIRLLVMToLLVMIRTranslation
    MLIRMathToLLVM
    MLIRMemRefDialect
    MLIRMemRefToEmitC
    MLIRMemRefToLLVM
    MLIRMemRefTransforms
    MLIRNVGPUDialect
    MLIRNVGPUToNVVM
    MLIRNVGPUTransforms
    MLIRNVVMDialect
    MLIRNVVMToLLVM
    MLIRNVVMToLLVMIRTranslation
    MLIRReconcileUnrealizedCasts
    MLIRSCFToControlFlow
    MLIRSCFToEmitC
    MLIRSCFTransforms
    MLIRTensorDialect
    MLIRTensorInferTypeOpInterfaceImpl
    MLIRUBToLLVM
    MLIRVectorDialect
    MLIRVectorToLLVM
    MLIRVectorToSCF
)

if(MLIR_TRT_ENABLE_CUDA)
  if(MSVC)
    message(WARNING
      "MSVC toolchain detected; libdevice will not be embedded and will not be available."
    )
  else()
    enable_language(ASM)
    # Embed libdevice into the compiler so deployed environments do not need a CUDA
    # toolkit install at runtime.
    mtrt_embed_libdevice_into_target(MLIRTensorRTKernelTransforms)
  endif()
endif()

