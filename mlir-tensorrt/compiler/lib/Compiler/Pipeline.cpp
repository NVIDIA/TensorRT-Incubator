//===- Pipeline.cpp -------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// Implementation for the Pipeline.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Compiler/Pipeline.h"
#include "mlir-executor/Conversion/Passes.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir-executor/Target/Lua/TranslateToRuntimeExecutable.h"
#include "mlir-kernel/Kernel/Pipelines/Pipelines.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "mlir-tensorrt-dialect/TensorRT/IR/TensorRTDialect.h"
#include "mlir-tensorrt/Backends/Host/Passes.h"
#include "mlir-tensorrt/Backends/Kernel/KernelBackend.h"
#include "mlir-tensorrt/Backends/Kernel/Passes.h"
#include "mlir-tensorrt/Compiler/Extension.h"
#include "mlir-tensorrt/Compiler/InputPipelines/LinalgInputPipeline.h"
#include "mlir-tensorrt/Compiler/InputPipelines/StablehloInputPipeline.h"
#include "mlir-tensorrt/Compiler/Options.h"
#include "mlir-tensorrt/Compiler/Pipeline.h"
#include "mlir-tensorrt/Conversion/CUDAToExecutor/CUDAToExecutor.h"
#include "mlir-tensorrt/Conversion/HostToEmitC/HostToEmitC.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/Plan/IR/PlanEnums.h"
#include "mlir-tensorrt/Dialect/Plan/Transforms/Passes.h"
#include "mlir-tensorrt/Dialect/StablehloExt/Transforms/Passes.h"
#include "mlir-tensorrt/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <functional>

using namespace mlir;
using namespace mtrt;
using namespace mtrt::compiler;

//===----------------------------------------------------------------------===//
// Pipeline
//===----------------------------------------------------------------------===//
void Pipeline::setupPassManagerInstrumentation() {
  // TODO: add API in upstream to detect whether this PM already has
  // instrumentation attached.
  if (options->hasDebugOptions()) {
    options->getDebugOptions().applyToPassManager(*this);
    return;
  }
  // Populate from global CL options.
  // TODO: we may want to consider making this a non-error.
  if (failed(applyPassManagerCLOptions(*this)))
    llvm::report_fatal_error("failed to populate pass manager "
                             "instrumentation from global CL options");
  applyDefaultTimingPassManagerCLOptions(*this);
}

Pipeline::Pipeline(mlir::MLIRContext *context,
                   llvm::IntrusiveRefCntPtr<MainOptions> options)
    : mlir::PassManager(context, mlir::ModuleOp::getOperationName()),
      options(std::move(options)) {
  mtrt::cantFail(initialize());
}

Pipeline::~Pipeline() {}

Status Pipeline::initialize() {
  if (initialized)
    return getInternalErrorStatus("attempted to reinitialize a pipeline");

  assert(this->getPasses().empty() && "expected empty pass manager");

  // Ensure options are finalized before they drive instrumentation and pass
  // population. This is important for options that default differently based on
  // other option values (e.g. input kind).
  MTRT_RETURN_IF_ERROR(options->finalize());

  // Now that options are parsed, we can setup instrumentation.
  setupPassManagerInstrumentation();

  // Now we can populate the pass manager.
  populatePassManager();

  initialized = true;

  return getOkStatus();
}

/// When the output path is a directory or empty, this function is used to
/// create the output file name with an extension guessed from the host target.
static std::string
createMainOutputFileName(llvm::StringRef initialName, HostTarget hostTarget,
                         std::optional<llvm::StringRef> overrideExtension) {
  initialName = initialName.trim();
  if (!initialName.empty() &&
      (initialName == "-" || !llvm::sys::fs::is_directory(initialName)))
    return initialName.str();

  llvm::SmallString<128> result = initialName;
  std::string name = "output";
  if (overrideExtension)
    name += overrideExtension->str();
  else if (hostTarget == HostTarget::EmitC)
    name += ".cpp";
  else if (hostTarget == HostTarget::LLVM)
    name += ".mlir";
  else if (hostTarget == HostTarget::Executor)
    name += ".rtexe";
  else
    llvm_unreachable("unknown output kind");

  llvm::sys::path::append(result, name);
  return result.str().str();
}

std::unique_ptr<llvm::ToolOutputFile>
Pipeline::openOutputFile(llvm::StringRef outputFileName,
                         std::string &errorMessage,
                         std::optional<llvm::StringRef> overrideExtension) {
  if (!initialized) {
    errorMessage = "attempted to open an output file before initialization";
    return nullptr;
  }
  std::string processedOutputName = createMainOutputFileName(
      outputFileName, options->hostTarget, overrideExtension);
  if (outputFileName != "-" && !options->artifactsDirectory.empty() &&
      // Only try to combine if we are sure that the artifacts directory is
      // present. Sometimes it is unused. Note: the artifacts directory is not
      // automatically created by the Pipeline; it must already exist.
      llvm::sys::fs::is_directory(options->artifactsDirectory) &&
      // This is an absolute path - don't combine.
      !llvm::sys::path::is_absolute(processedOutputName) &&
      // This is relative, but it has a parent path - don't combine.
      !llvm::sys::path::has_parent_path(processedOutputName)) {
    llvm::SmallString<128> path;
    llvm::sys::path::append(path, options->artifactsDirectory,
                            processedOutputName);
    return mlir::openOutputFile(path, &errorMessage);
  }
  return mlir::openOutputFile(processedOutputName, &errorMessage);
}

LogicalResult Pipeline::translateToTargetFormat(mlir::ModuleOp module,
                                                llvm::raw_ostream &os) {
  const HostTarget hostTarget = getOptions().hostTarget;
  if (hostTarget == HostTarget::EmitC) {
    if (failed(emitc::translateToCpp(module, os)))
      return failure();
    return success();
  }

  if (hostTarget == HostTarget::LLVM) {
    module->print(os);
    return success();
  }

  if (hostTarget == HostTarget::Executor) {
    if (failed(mlir::translateToRuntimeExecutable(module, os)))
      return failure();
    return success();
  }

  return emitError(module->getLoc()) << "unknown host target format";
}

static plan::PlanBufferizationOptions
convertBufferizationOptions(const MainOptions &pipelineOpts) {
  const BufferizationOptions &opts = pipelineOpts.get<BufferizationOptions>();
  plan::PlanBufferizationOptions bufferizationOpts{};
  bufferizationOpts.forceEntrypointsReturnAllocs =
      opts.forceEntrypointsReturnAllocs;
  bufferizationOpts.deallocationPrivateFuncDynamicOwnership =
      opts.deallocationPrivateFuncDynamicOwnership;
  bufferizationOpts.enableBufferLoopHoisting = opts.enableBufferLoopHoisting;
  bufferizationOpts.enableBufferHoisting = opts.enableBufferHoisting;
  bufferizationOpts.enablePinnedMemoryPromotion =
      opts.enablePinnedMemoryPromotion;
  return bufferizationOpts;
}

static void addCleanupPasses(OpPassManager &pm) {
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

static void populateExtensionPasses(mlir::OpPassManager &pm,
                                    const MainOptions &options, Phase phase,
                                    const ExtensionList &extensions) {
  (void)options;
  for (auto &[key, ext] : extensions)
    ext->populatePasses(pm, phase);
}

static void populateSetupPipeline(OpPassManager &pm,
                                  const MainOptions &options) {

  pm.addPass(plan::createPopulateDefaultBackendMetadataPass(
      plan::PopulateDefaultBackendMetadataPassOptions{
          /*defaultBackends=*/SmallVector<std::string>(
              options.defaultBackends.begin(), options.defaultBackends.end()),
          /*defaultMemorySpaceString=*/options.defaultMemorySpace.getValue()}));

  if (options.inputKind != plan::InputKind::TensorRT)
    pm.addPass(plan::createLegalizeIOBoundsAttributesPass());

  pm.addPass(plan::createVerifyInputAndAssignSlotsPass());

  if (options.runtimeABIVersion >= 1 &&
      options.hostTarget != HostTarget::LLVM) {
    pm.addPass(executor::createExecutorGenerateABIWrappersPass(
        executor::ExecutorGenerateABIWrappersPassOptions{
            /*forceUndefOutputArgs=*/options.get<BufferizationOptions>()
                .forceEntrypointsReturnAllocs,
        }));
  }
}

static void populateInputPipeline(OpPassManager &pm, const MainOptions &options,
                                  const ExtensionList &extensions) {
  if (options.inputKind == plan::InputKind::Stablehlo) {
    mtrt::compiler::buildStablehloInputPipeline(
        pm, options.get<StablehloInputOptions>(),
        [&](mlir::OpPassManager &pm, const StablehloInputOptions &opts) {
          pm.addNestedPass<func::FuncOp>(
              stablehlo_ext::createConstantFoldingPass(
                  stablehlo_ext::ConstantFoldingPassOptions{
                      opts.constantFoldSizeLimit}));
          populateExtensionPasses(pm, options, Phase::ConstantFolding,
                                  extensions);
        });
    return;
  }

  if (options.inputKind == plan::InputKind::Linalg) {
    mtrt::compiler::buildLinalgInputPipeline(pm,
                                             options.get<LinalgInputOptions>());
  }
}

static void addExecutorLoweringTail(OpPassManager &pm,
                                    const MainOptions &options,
                                    const ExtensionList &extensions) {
  if (options.runtimeABIVersion < 1) {
    pm.addNestedPass<func::FuncOp>(
        executor::createExecutorPopulateFunctionMetadataPass());
  }

  populateExtensionPasses(pm, options, Phase::ExecutorLowering, extensions);

  ConvertCUDAToExecutorPassOptions cudaToExecutorOpts;
  cudaToExecutorOpts.indexBitwidth =
      options.get<ExecutorOptions>().indexBitwidth;
  pm.addPass(createConvertCUDAToExecutorPass(cudaToExecutorOpts));

  mlir::executor::ConvertStdToExecutorPassOptions stdToExecOpts;
  stdToExecOpts.indexBitwidth = options.get<ExecutorOptions>().indexBitwidth;
  mlir::executor::buildExecutorLoweringPipeline(
      pm, stdToExecOpts, [](mlir::TypeConverter &typeConverter) {
        mlir::populateCUDAToExecutorTypeConversions(typeConverter);
      });
}

static void addLLVMOrEmitCTail(OpPassManager &pm, const MainOptions &options,
                               bool requestCWrappers) {
  const HostTarget hostTarget = options.hostTarget;
  assert((hostTarget == HostTarget::LLVM || hostTarget == HostTarget::EmitC) &&
         "expected LLVM or EmitC host target");

  pm.addPass(createConvertComplexToStandardPass());

  // For EmitC lowering, preserve control flow for readability.
  if (hostTarget != HostTarget::EmitC)
    pm.addPass(mlir::createSCFToControlFlowPass());

  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(memref::createExpandOpsPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  addCleanupPasses(pm);

  pm.addPass(affine::createAffineExpandIndexOpsPass());
  pm.addPass(mlir::createLowerAffinePass());
  addCleanupPasses(pm);

  populateExtensionPasses(pm, options, Phase::ExecutorLowering,
                          options.getExtensions());

  if (hostTarget == HostTarget::LLVM) {
    if (requestCWrappers)
      pm.addNestedPass<func::FuncOp>(LLVM::createLLVMRequestCWrappersPass());
    pm.addPass(createConvertCUDAToLLVMPass());
    pm.addPass(createConvertHostToLLVMPass());
    addCleanupPasses(pm);
  }

  if (hostTarget == HostTarget::EmitC) {
    mtrt::compiler::applyEmitCLoweringPipeline(
        pm, options.get<EmitCOptions>().wrapModuleInEmitCClass);
  }

  pm.addPass(mlir::executor::createExecutorSerializeArtifactsPass(
      mlir::executor::ExecutorSerializeArtifactsPassOptions{
          /*artifactsDirectory=*/options.artifactsDirectory,
          /*createManifest=*/true,
      }));
}

void Pipeline::populatePassManager() {
  OpPassManager &pm = getPassManager();
  const MainOptions &options = getOptions();

  assert(pm.getPasses().empty() && "expected empty pass manager");

  populateSetupPipeline(pm, options);
  populateInputPipeline(pm, options, options.getExtensions());

  // Dispatch based on input kind.
  switch (options.inputKind) {
  case mlir::plan::InputKind::Stablehlo: {
    // Add pre-clustering extension passes
    populateExtensionPasses(pm, options, Phase::PreClustering,
                            options.getExtensions());

    plan::buildPlanSegmentationPipeline(
        pm, options.runtimeABIVersion, options.inputKind,
        options.get<BufferizationOptions>().forceEntrypointsReturnAllocs,
        options.entrypoint, options.get<plan::PlanClusteringOptions>());

    // Compile outlined scalarizable host clusters.
    pm.addNestedPass<func::FuncOp>(
        mtrt::compiler::createProcessHostClustersPass());
    pm.addNestedPass<func::FuncOp>(mlir::createConvertStablehloToArithPass());

    populateExtensionPasses(pm, options, Phase::PostClustering,
                            options.getExtensions());

    break;
  }
  case mlir::plan::InputKind::TensorRT: {
    pm.addPass(createOutlineTensorRTOpsPass());
    break;
  }
  case mlir::plan::InputKind::Linalg: {
    const auto &deviceOpts = options.get<DeviceOptions>();
    const auto &kernelGenOpts = options.get<KernelGenOptions>();

    pm.addPass(mtrt::compiler::createKernelSegmentationPass());

    // Kernel Generation
    kernel::buildTransformIRPipeline(
        pm, mtrt::compiler::getKernelGenClusterAttrName(),
        deviceOpts.computeCapability, deviceOpts.maxSharedMemoryPerBlockKb,
        deviceOpts.maxRegistersPerBlock,
        /*generatorBenefit=*/kernelGenOpts.generatorBenefit);

    // Populate target information.
    kernel::SetGPUTargetPassOptions setTargetOptions{};
    setTargetOptions.inferTargetFromHost = deviceOpts.shouldInferFromHost;
    setTargetOptions.chip =
        "sm_" + std::to_string(deviceOpts.computeCapability);
    setTargetOptions.maxRegistersPerBlock = deviceOpts.maxRegistersPerBlock;
    setTargetOptions.maxSharedMemoryPerBlockKb =
        deviceOpts.maxSharedMemoryPerBlockKb;

    pm.addPass(kernel::createLowerKernelSortPass());
    pm.addPass(kernel::createSetGPUTargetPass(setTargetOptions));
    pm.addPass(kernel::createAnnotateKernelEntrypointsPass());

    pm.addNestedPass<gpu::GPUModuleOp>(
        kernel::createDispatchGPUModuleCompilationPass(
            kernel::DispatchGPUModuleCompilationPassOptions{
                kernel::GPUModuleLoweringPhase::PreBufferization}));
    break;
  }
  }

  pm.addNestedPass<func::FuncOp>(plan::createPostClusteringValidationPass());
  pm.addNestedPass<func::FuncOp>(mtrt::createSCFDetensorizePass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  populateExtensionPasses(pm, options, Phase::PreBufferization,
                          options.getExtensions());

  // Bufferization
  plan::buildPlanBufferizationPipeline(pm,
                                       convertBufferizationOptions(options));

  populateExtensionPasses(pm, options, Phase::PostBufferization,
                          options.getExtensions());

  pm.addPass(mtrt::createDropNestedModulesPass());

  if (options.hostTarget == HostTarget::Executor &&
      options.get<OptimizationOptions>().hoistAllocsToGlobals) {
    pm.addPass(executor::createExecutorAllocsToGlobalsPass());
  }

  pm.addPass(createConvertMemRefToCUDAPass());

  if (options.hostTarget == HostTarget::Executor) {
    pm.addPass(createConvertPlanToExecutorPass());
    addExecutorLoweringTail(pm, options, options.getExtensions());

  } else if (options.hostTarget == HostTarget::LLVM ||
             options.hostTarget == HostTarget::EmitC) {
    addLLVMOrEmitCTail(pm, options,
                       /*requestCWrappers=*/true);
  }
}
