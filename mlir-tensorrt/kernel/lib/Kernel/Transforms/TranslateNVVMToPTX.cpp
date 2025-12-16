//===- TranslateNVVMToPTX.cpp ---------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2023-2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
// Implementation of pass to translate LLVM/NVVM Dialect to PTX.
//===----------------------------------------------------------------------===//
#include "mlir-kernel/Kernel/IR/Configuration.h"
#include "mlir-kernel/Kernel/IR/Ops.h"
#include "mlir-kernel/Kernel/Transforms/KernelToLLVMIRTranslation.h"
#include "mlir-kernel/Kernel/Transforms/Passes.h"
#include "mlir-kernel/Utils/CUDAUtils.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/Internalize.h"

#define DEBUG_TYPE "translate-nvvm-to-ptx"
#define DBGS() llvm::dbgs() << "[" DEBUG_TYPE "]"

namespace mlir {
namespace kernel {
#define GEN_PASS_DEF_TRANSLATENVVMTOPTXPASS
#include "mlir-kernel/Kernel/Transforms/Passes.h.inc"
} // namespace kernel
} // namespace mlir

using namespace mlir;
using namespace mlir::kernel;

static constexpr StringRef kPTXDataAttrName = "kernel.ptx_data";
static constexpr StringRef kBlobKey = "gpu.module.kernels.ptx_data";

/// Create all directories in `path`, ignoring those that already exist. Failure
/// to create a directory results in emitting a diagnostic and returning
/// failure.
static LogicalResult createDirectories(StringRef path) {
  if (std::error_code EC =
          llvm::sys::fs::create_directories(path, /*IgnoreExisting=*/true)) {
    llvm::errs() << "could not create directory '" << path
                 << "': " << EC.message();
    return failure();
  }
  return success();
}

/// Return the symbol names of parent symbol tables and ending with the symbol
/// name of `op`.
static SmallVector<StringRef> getSymbolNames(Operation *op) {
  SmallVector<StringRef> pathElements;
  while (op) {
    if (!op->hasTrait<OpTrait::SymbolTable>() &&
        !op->hasAttr(SymbolTable::getSymbolAttrName())) {
      op = op->getParentOp();
      continue;
    }
    StringAttr symbolName =
        op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    pathElements.push_back(symbolName ? symbolName.strref() : "no-symbol-name");
    op = op->getParentOp();
  }
  // Return in the order of top level (module) down to `op`.
  std::reverse(pathElements.begin(), pathElements.end());
  return pathElements;
}

/// Returns the PTX file name as a sequence of concatenated symbol names
/// starting from the highest level SymbolTable parent and ending with the
/// function name.
static llvm::SmallString<128> getKernelModulePTXFileName(Operation *op,
                                                         StringRef basePath) {
  llvm::SmallString<128> path = basePath;
  SmallVector<StringRef> symbolNames = getSymbolNames(op);
  llvm::sys::path::append(
      path, llvm::formatv("{0:$[_]}.ptx", llvm::make_range(symbolNames.begin(),
                                                           symbolNames.end())));
  return path;
}

/// Write the PTX data to a file at the given directory.
static void savePTXToFile(Operation *op, StringRef path, StringRef data) {
  const bool userSpecifiedFile = path.ends_with(".ptx");
  llvm::SmallString<128> finalPath;
  if (!userSpecifiedFile) {
    if (failed(createDirectories(path)))
      return;
    finalPath = getKernelModulePTXFileName(op, path);
  } else {
    finalPath = path;
  }
  std::string error;
  std::unique_ptr<llvm::ToolOutputFile> of =
      mlir::openOutputFile(finalPath, &error);
  if (!of) {
    llvm::errs() << "failed to open " << finalPath << ": " << error << "\n";
    return;
  }
  of->os() << data;
  if (of->os().has_error()) {
    llvm::errs() << "failed to write TensorRT engine to " << finalPath << ": "
                 << of->os().error().message() << "\n";
    return;
  }
  of->keep();
}

/// Create the TargetMachine, which is LLVM's primary interface for describing
/// the compilation target.
static FailureOr<std::unique_ptr<llvm::TargetMachine>>
createTargetMachine(Location loc, StringRef gpuArchitecture,
                    StringRef features) {
  std::string error;
  llvm::Triple fullTriple("nvptx64-nvidia-cuda");
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget("", fullTriple, error);
  if (target == nullptr)
    return emitError(loc, "failed to lookup target: " + error);
  llvm::TargetMachine *machine = target->createTargetMachine(
      fullTriple, gpuArchitecture, features, {}, {});
  if (machine == nullptr)
    return emitError(loc, "failed to create target machine");
  return std::unique_ptr<llvm::TargetMachine>(machine);
}

/// Link in `libdevice.10.bc` that is shipped from CUDA installation.
/// TODO: use libdevice shipped with redistributable package, then fallback to
/// CTX installation.
static LogicalResult linkLibdevice(Location loc, llvm::Module &module,
                                   llvm::TargetMachine &targetMachine) {

  llvm::Linker linker(module);
  unsigned linkerFlags =
      llvm::Linker::LinkOnlyNeeded | llvm::Linker::OverrideFromSrc;

  StringRef filename = DEFAULT_LIBDEVICE_PATH;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code error = fileOrErr.getError())
    return emitError(loc, "could not open libdevice bitcode file " + filename);

  llvm::MemoryBufferRef libdeviceCode((*fileOrErr)->getBuffer(),
                                      "libdevice.10.bc");

  LLVM_DEBUG(DBGS() << "libdevice bitcode data size: "
                    << libdeviceCode.getBufferSize() << "\n");

  llvm::Expected<std::unique_ptr<llvm::Module>> libdeviceModule =
      llvm::parseBitcodeFile(libdeviceCode, module.getContext());
  if (!libdeviceModule)
    return emitError(loc, "could not parse libdevice bitcode: ")
           << llvm::toString(libdeviceModule.takeError());

  (*libdeviceModule)->setDataLayout(targetMachine.createDataLayout());
  (*libdeviceModule)->setTargetTriple(targetMachine.getTargetTriple());
  if (linker.linkInModule(
          std::move(*libdeviceModule), linkerFlags,
          [](llvm::Module &module, const StringSet<> &ss) {
            llvm::internalizeModule(module, [&ss](const llvm::GlobalValue &gv) {
              return !gv.hasName() || (ss.count(gv.getName()) == 0);
            });
          }))
    return emitError(loc) << "failed to link libdevice";

  return success();
}

/// Convert the given llvm::Module into PTX. The given TargetMachine indicates
/// the chip features (e.g. sm_80). The final PTX is emitted to the given
/// output stream.
static FailureOr<std::string>
optimizeModuleAndEmitPtx(llvm::Module &llvmModule,
                         llvm::TargetMachine &targetMachine) {
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  // Run function inlining and optimization.
  llvm::PipelineTuningOptions PTO;
  PTO.LoopVectorization = false;
  PTO.SLPVectorization = false;

  llvm::PassBuilder PB(&targetMachine, PTO); //, std::nullopt, &pic);
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  llvm::ModulePassManager MPM;

  StringRef nnvmReflectPassName = "nvvm-reflect";
  if (PB.parsePassPipeline(MPM, nnvmReflectPassName)) {
    llvm::errs() << "Could not parse -" << nnvmReflectPassName << "\n";
  }

  llvm::FunctionPassManager FPM;
  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
  llvm::OptimizationLevel level = llvm::OptimizationLevel::O1;
  MPM.addPass(PB.buildPerModuleDefaultPipeline(level));

  MPM.run(llvmModule, MAM);

  // Emit PTX.
  std::string result;
  {
    llvm::raw_string_ostream ss(result);
    llvm::buffer_ostream pstream(ss);
    llvm::legacy::PassManager codegenPasses;
    if (targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                          llvm::CodeGenFileType::AssemblyFile))
      return failure();
    codegenPasses.run(llvmModule);
  }
  return result;
}

/// Translate the given module to PTX code. The gpuArchitecture indicates
/// the compute capability (e.g. sm_80).
static FailureOr<std::string> translateToPTXCode(Operation *module,
                                                 StringRef gpuArchitecture,
                                                 StringRef features) {

  LLVM_DEBUG(DBGS() << "Translating MLIR module to PTX:\n" << *module << "\n");

  // Translate the LLVM IR module to target ISA (PTX) using NVPTX backend.
  auto targetMachine =
      createTargetMachine(module->getLoc(), gpuArchitecture, features);
  if (failed(targetMachine))
    return failure();

  // Lower the module to an LLVM IR module using a separate context to
  // enable multi-threaded processing.
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    module->emitOpError() << "failed to compile to LLVM IR";
    return failure();
  }

  LLVM_DEBUG(DBGS() << "After translating:\n"; (*llvmModule).dump();
             llvm::dbgs() << "\n");

  // Set the module's data layout and triple.
  const llvm::Triple &targetTriple = (*targetMachine)->getTargetTriple();
  llvmModule->setDataLayout((*targetMachine)->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);

  // Link libdevice.10.bc if required. Per NV docs, this should go before all
  // the LLVM optimizations that appear in the function below.
  bool requiresLibDevice =
      llvm::any_of(llvmModule->functions(), [](const llvm::Function &func) {
        return !func.isIntrinsic() && func.isDeclaration() &&
               func.getName().starts_with("__nv");
      });
  if (requiresLibDevice &&
      failed(linkLibdevice(module->getLoc(), *llvmModule, **targetMachine)))
    return failure();

  FailureOr<std::string> ptx =
      optimizeModuleAndEmitPtx(*llvmModule, **targetMachine);
  if (failed(ptx)) {
    module->emitOpError() << "failed to compile LLVM module and emit PTX";
    return failure();
  }

  LLVM_DEBUG(DBGS() << "Emitted PTX:\n" << *ptx << "\n");

  return ptx;
}

namespace {
//===----------------------------------------------------------------------===//
// Translate NVVM to PTX Pass
//===----------------------------------------------------------------------===//
struct TranslateNVVMToPTXPass
    : public kernel::impl::TranslateNVVMToPTXPassBase<TranslateNVVMToPTXPass> {
  using Base::Base;
  void runOnOperation() override {
    // Insert transform IR at the end
    gpu::GPUModuleOp module = getOperation(); // E.g. a gpu.module
    std::optional<StringRef> chipName = getUniqueTargetChip(module);
    if (!chipName) {
      emitError(module->getLoc()) << "could not determine target chip name";
      return signalPassFailure();
    }

    // Starting Compute Capability 9.0, there is a baseline feature set,
    // architecture specific feature set (selected by using suffix `a` in the
    // compilation target name), and family specific feature set (selected by
    // using suffix `f` in the compilation target name).
    // To enable using wider PTX instructions set, we set compilation target to
    // use architecture specific feature set if Compute Capability is found to
    // be greater than 9.0.
    std::string archAndFeatureSetVariant = chipName->str();
    if (chipName->starts_with("sm_") &&
        !(chipName->ends_with("a") || chipName->ends_with("f"))) {
      unsigned smVersion;
      if (!chipName->drop_front(3).getAsInteger(10, smVersion) &&
          smVersion >= 90)
        archAndFeatureSetVariant += "a";
    }

    // Translate all `llvm.func` operations to PTX code.
    std::string features = "+ptx" + std::to_string(getHighestPTXVersion());
    auto ptx = translateToPTXCode(module, archAndFeatureSetVariant, features);
    if (failed(ptx))
      return signalPassFailure();
    StringRef ptxStr = *ptx;

    // Save the PTX code if requested.
    if (!dumpPtxPath.empty())
      savePTXToFile(module, dumpPtxPath, ptxStr);

    // Set PTX code (in Hex) as an attribute of gpu.module
    OpBuilder b = OpBuilder(module);

    auto data = DenseI8ResourceElementsAttr::get(
        RankedTensorType::get({static_cast<int64_t>(ptxStr.size())},
                              IntegerType::get(b.getContext(), 8)),
        kBlobKey,
        HeapAsmResourceBlob::allocateAndCopyWithAlign(
            ArrayRef<char>(ptxStr.begin(), ptxStr.end()), alignof(char),
            /*dataIsMutable=*/true));

    module->setAttr(b.getStringAttr(kPTXDataAttrName), data);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // Initialize LLVM NVPTX backend.
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
    registerLLVMDialectTranslation(registry);
    registerNVVMDialectTranslation(registry);
    registerGPUDialectTranslation(registry);
    kernel::impl::TranslateNVVMToPTXPassBase<
        TranslateNVVMToPTXPass>::getDependentDialects(registry);
  }
};
} // namespace
