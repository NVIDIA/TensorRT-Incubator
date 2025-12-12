include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(MLIR_TRT_ENABLE_CUDA)
  mtrt_find_libdevice_file(libDevicePath)
endif()

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
    MLIRKernelCUDAUtils
    MLIRKernelLinalgUtils
    MLIRTensorRTCommonBufferizationScopeInterface
    MLIRTensorRTKernelDialect
    MLIRTensorRTKernelIncludes
    MLIRTensorRTKernelToLLVM
    MLIRTensorRTSupportDeviceInfo
  LINK_COMPONENTS
    Analysis
    MC
    NVPTXCodeGen
    Passes
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

# Add the default libdevice path. This will point to the libdevice path in the
# CTK installation of the development machine. In the future we should bundle
# the libdevice binary and look for it relative to the compiler location.
target_compile_definitions(obj.MLIRTensorRTKernelTransforms PRIVATE
  "DEFAULT_LIBDEVICE_PATH=\"${libDevicePath}\""
)

