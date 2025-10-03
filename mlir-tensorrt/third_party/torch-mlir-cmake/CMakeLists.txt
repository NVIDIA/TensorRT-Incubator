#-------------------------------------------------------------------------------------
# This file is used to declare Torch-MLIR CMake targets in lieu of using the CMake
# code that comes with Torch-MLIR. We will do this until we have enough resources
# and support to fix the upstream Torch-MLIR CMake code.
#-------------------------------------------------------------------------------------

message(STATUS "Adding Torch-MLIR CMake targets")
message(STATUS "torch-mlir source dir: ${torch_mlir_SOURCE_DIR}")
message(STATUS "torch-mlir binary dir: ${torch_mlir_BINARY_DIR}")

find_package(MLIR REQUIRED CONFIG)
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

include_directories(
  "${torch_mlir_SOURCE_DIR}/include"
  "${torch_mlir_BINARY_DIR}/include"
  )

set(TORCH_MLIR_TABLEGEN_FLAGS "")
set(TORCH_MLIR_ENABLE_STABLEHLO "${MLIR_TRT_ENABLE_HLO}")
set(TORCH_MLIR_ENABLE_REFBACKEND OFF)
set(TORCH_MLIR_ENABLE_TOSA OFF)
set(TORCH_MLIR_ENABLE_ONNX_C_IMPORTER OFF)

if(TORCH_MLIR_ENABLE_STABLEHLO)
  add_definitions(-DTORCH_MLIR_ENABLE_STABLEHLO)
  list(APPEND TORCH_MLIR_TABLEGEN_FLAGS "-DTORCH_MLIR_ENABLE_STABLEHLO")
endif()

add_subdirectory(
  "${torch_mlir_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_BINARY_DIR}/include"
  )

set(torch_mlir_core_source_files
  lib/Conversion/TorchToSCF/TorchToSCF.cpp
  lib/Conversion/TorchToTMTensor/TorchToTMTensor.cpp
  lib/Conversion/TorchConversionToMLProgram/TorchConversionToMLProgram.cpp
  lib/Conversion/TorchToArith/TorchToArith.cpp
  lib/Conversion/TorchOnnxToTorch/DefaultDomainGtoP.cpp
  lib/Conversion/TorchOnnxToTorch/TorchOnnxToTorch.cpp
  lib/Conversion/TorchOnnxToTorch/Patterns.cpp
  lib/Conversion/TorchOnnxToTorch/DefaultDomainQtoZ.cpp
  lib/Conversion/TorchOnnxToTorch/OnnxRecurrentLayerOpExpanders.cpp
  lib/Conversion/TorchOnnxToTorch/ComMicrosoftDomain.cpp
  lib/Conversion/TorchOnnxToTorch/Passes.cpp
  lib/Conversion/TorchOnnxToTorch/Utils.cpp
  lib/Conversion/TorchOnnxToTorch/DefaultDomainAtoF.cpp
  lib/Conversion/Passes.cpp
  lib/Conversion/Utils/Utils.cpp
  lib/Conversion/TorchToLinalg/TorchToLinalg.cpp
  lib/Conversion/TorchToLinalg/Pooling.cpp
  lib/Conversion/TorchToLinalg/Reduction.cpp
  lib/Conversion/TorchToLinalg/Linear.cpp
  lib/Conversion/TorchToLinalg/Uncategorized.cpp
  lib/Conversion/TorchToLinalg/DataMovement.cpp
  lib/Conversion/TorchToLinalg/TensorConstructors.cpp
  lib/Conversion/TorchToLinalg/Utils.cpp
  lib/Conversion/TorchToLinalg/IndirectDataMovement.cpp
  lib/Conversion/TorchToLinalg/Random.cpp
  lib/Conversion/TorchToLinalg/TensorScalarInterop.cpp
  lib/Conversion/TorchToTensor/TorchToTensor.cpp
  lib/Dialect/TorchConversion/IR/TorchConversionDialect.cpp
  lib/Dialect/TorchConversion/IR/TorchConversionOps.cpp
  lib/Dialect/TorchConversion/Transforms/VerifyLinalgOnTensorsBackendContract.cpp
  lib/Dialect/TorchConversion/Transforms/UnpackQuantTensor.cpp
  lib/Dialect/TorchConversion/Transforms/BackendTypeConversion.cpp
  lib/Dialect/TorchConversion/Transforms/Passes.cpp
  lib/Dialect/TorchConversion/Transforms/ConvertCustomQuantOp.cpp
  lib/Dialect/TorchConversion/Transforms/BackendTypeConversionPasses.cpp
  lib/Dialect/TorchConversion/Transforms/VerifyTosaBackendContract.cpp
  lib/Dialect/Torch/IR/TorchOps.cpp
  lib/Dialect/Torch/IR/TorchTypes.cpp
  lib/Dialect/Torch/IR/TorchDialect.cpp
  lib/Dialect/Torch/IR/UtilsForODSGenerated.cpp
  lib/Dialect/Torch/IR/TorchOpsODSGenerated.cpp
  lib/Dialect/Torch/Transforms/AbstractInterpLibrary.cpp
  lib/Dialect/Torch/Transforms/ReifyDtypeCalculations.cpp
  lib/Dialect/Torch/Transforms/MaximizeValueSemantics.cpp
  lib/Dialect/Torch/Transforms/AdjustCallingConventions.cpp
  lib/Dialect/Torch/Transforms/SimplifyShapeCalculations.cpp
  lib/Dialect/Torch/Transforms/LowerToBackendContract.cpp
  lib/Dialect/Torch/Transforms/FuseQuantizedOps.cpp
  lib/Dialect/Torch/Transforms/RecomposeComplexOps.cpp
  lib/Dialect/Torch/Transforms/SimplifyDtypeCalculations.cpp
  lib/Dialect/Torch/Transforms/RestructureNonConstantAxes.cpp
  lib/Dialect/Torch/Transforms/DecomposeComplexOps.cpp
  lib/Dialect/Torch/Transforms/DropAbstractInterpCalculations.cpp
  lib/Dialect/Torch/Transforms/InlineGlobalSlots.cpp
  lib/Dialect/Torch/Transforms/ReifyAbstractInterpCalculationsUtils.cpp
  lib/Dialect/Torch/Transforms/ScalarizeShapes.cpp
  lib/Dialect/Torch/Transforms/PrepareForGlobalizeObjectGraph.cpp
  lib/Dialect/Torch/Transforms/Passes.cpp
  lib/Dialect/Torch/Transforms/ReifyShapeCalculations.cpp
  lib/Dialect/Torch/Transforms/EraseModuleInitializer.cpp
  lib/Dialect/Torch/Transforms/RefinePublicReturn.cpp
  lib/Dialect/Torch/Transforms/ReduceOpVariants.cpp
  lib/Dialect/Torch/Transforms/MatchQuantizedOps.cpp
  lib/Dialect/Torch/Transforms/GlobalizeObjectGraph.cpp
  lib/Dialect/Torch/Transforms/SimplifyAbstractInterpCalculationsUtils.cpp
  lib/Dialect/Torch/Utils/SparsityUtils.cpp
  lib/Dialect/Torch/Utils/TorchUpstream.cpp
  lib/Dialect/Torch/Utils/Utils.cpp
  lib/Dialect/TMTensor/IR/TMTensorInterfaces.cpp
  lib/Dialect/TMTensor/IR/TMTensorOps.cpp
  lib/Dialect/TMTensor/IR/TMTensorDialect.cpp
  lib/Dialect/TMTensor/IR/ScalarLoopOpInterface.cpp
  lib/Dialect/TMTensor/Transforms/Bufferize.cpp
  lib/Dialect/TMTensor/Transforms/ConvertToLoops.cpp
  lib/Dialect/TMTensor/Transforms/Passes.cpp
)

if(TORCH_MLIR_ENABLE_STABLEHLO)
  list(APPEND torch_mlir_core_source_files
    lib/Conversion/TorchToStablehlo/StablehloLegalizeUtils.cpp
    lib/Conversion/TorchToStablehlo/ViewLike.cpp
    lib/Conversion/TorchToStablehlo/Pooling.cpp
    lib/Conversion/TorchToStablehlo/Rng.cpp
    lib/Conversion/TorchToStablehlo/Reduction.cpp
    lib/Conversion/TorchToStablehlo/Linear.cpp
    lib/Conversion/TorchToStablehlo/Basic.cpp
    lib/Conversion/TorchToStablehlo/TorchToStablehlo.cpp
    lib/Conversion/TorchToStablehlo/GatherScatter.cpp
    lib/Conversion/TorchToStablehlo/Utils.cpp
    lib/Conversion/TorchToStablehlo/Uncategorized.cpp
    lib/Dialect/TorchConversion/Transforms/VerifyStablehloBackendContract.cpp
    )
endif()

list(TRANSFORM torch_mlir_core_source_files PREPEND ${torch_mlir_SOURCE_DIR}/)

set(torch_mlir_optional_deps)
if(TORCH_MLIR_ENABLE_STABLEHLO)
  list(APPEND torch_mlir_optional_deps
    ChloOps
    StablehloLinalgTransforms
    StablehloOps
    StablehloOptimizationPasses
    StablehloPasses
  )
endif()

mtrt_add_project_library(MLIRTensorRTTorchMLIR
  PROJECT_NAME MLIRTensorRT
  ${torch_mlir_core_source_files}

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRTorchConversionOpsIncGen
  MLIRTorchOpsIncGen
  MLIRTorchTypesIncGen
  TorchMLIRConversionPassIncGen
  TorchMLIRConversionTorchOnnxToTorchPassIncGen
  TorchMLIRTMTensorOpsIncGen
  TorchMLIRTMTensorTransformsPassesIncGen
  TorchMLIRTorchConversionPassIncGen
  TorchMLIRTorchPassIncGen

  LINK_LIBS PUBLIC
  ${torch_mlir_optional_deps}

  MLIR_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncInlinerExtension
  MLIRFuncTransforms
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRMLProgramDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTensorInferTypeOpInterfaceImpl
)

target_include_directories(
  obj.MLIRTensorRTTorchMLIR
  INTERFACE
  $<BUILD_INTERFACE:${torch_mlir_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${torch_mlir_BINARY_DIR}/include>
)

target_include_directories(
  MLIRTensorRTTorchMLIR
  INTERFACE
  $<BUILD_INTERFACE:${torch_mlir_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${torch_mlir_BINARY_DIR}/include>
)


mtrt_add_capi_library(TorchMLIRCAPI
  ${torch_mlir_SOURCE_DIR}/lib/CAPI/Dialects.cpp
  ${torch_mlir_SOURCE_DIR}/lib/CAPI/TorchOps.cpp
  ${torch_mlir_SOURCE_DIR}/lib/CAPI/TorchTypes.cpp
  ${torch_mlir_SOURCE_DIR}/lib/CAPI/Transforms.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${torch_mlir_SOURCE_DIR}/include/torch-mlir-c/

  ENABLE_AGGREGATION

  LINK_LIBS PUBLIC
  MLIRTensorRTTorchMLIR

  MLIR_LIBS PUBLIC
  MLIRCAPIIR
  MLIRIR
  MLIRSupport
)

declare_mlir_python_sources(TorchMLIRPythonSources)
declare_mlir_python_sources(TorchMLIRPythonExtensions)

set(TORCH_MLIR_PYTHON_ROOT_DIR "${torch_mlir_SOURCE_DIR}/python/torch_mlir")

declare_mlir_python_sources(TorchMLIRPythonSources.Dialects
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TorchMLIRPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TorchMLIRPythonSources.Dialects
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/TorchBinding.td
  SOURCES dialects/torch/__init__.py
  DIALECT_NAME torch
)

declare_mlir_python_sources(TorchMLIRPythonSources.Importers
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TorchMLIRPythonSources
  SOURCES
    extras/fx_importer.py
)

declare_mlir_python_sources(TorchMLIRPythonSources.PublicAPI
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TorchMLIRPythonSources
  SOURCES
    compiler_utils.py
    fx.py
    extras/fx_decomp_util.py
)

declare_mlir_python_sources(TorchMLIRPythonSources.Tools
  ROOT_DIR "${TORCH_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TorchMLIRPythonSources
  SOURCES
    tools/opt/__main__.py
)

declare_mlir_python_extension(TorchMLIRPythonExtensions.Main
  MODULE_NAME _torchMlir
  ADD_TO_PARENT TorchMLIRPythonExtensions
  SOURCES TorchMLIRModule.cpp
  EMBED_CAPI_LINK_LIBS
    TorchMLIRCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)
