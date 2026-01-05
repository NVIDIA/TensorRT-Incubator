//===- HostToEmitC.cpp ----------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2025 NVIDIA CORPORATION & AFFILIATES.
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
///
/// Implementation of the `convert-host-to-emitc` pass.
///
/// The goal of this pass is to transform "host-side" MLIR (Executor, CUDA,
/// TensorRTRuntime, MemRef, etc.) into EmitC ops such that `mlir-to-cpp`
/// produces compilable C++ that calls into the StandaloneCPP runtime.
///
/// Roughly, the generated C++ has the form:
///   - `#include <...>` plus `#include "MTRTRuntime*.h"`
///   - module-scope resources materialized as C++ globals and init/destroy
///     helper functions (see `HostToEmitCGlobals.cpp`)
///   - function bodies lowered to straight-line C++ expressions/statements
///     calling `mtrt::...` runtime helpers (see `HostToEmitCPatterns*.cpp`)
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Conversion/HostToEmitC/HostToEmitC.h"
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/IR/ExecutorAttributes.h"
#include "mlir-executor/Executor/Transforms/Passes.h"
#include "mlir-executor/Support/ArtifactManager.h"
#include "mlir-tensorrt/Compiler/Options.h"
#include "mlir-tensorrt/Conversion/Passes.h"
#include "mlir-tensorrt/Dialect/CUDA/IR/CUDADialect.h"
#include "mlir-tensorrt/Dialect/Plan/IR/Plan.h"
#include "mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntime.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Conversion/FuncToEmitC/FuncToEmitC.h"
#include "mlir/Conversion/MathToEmitC/MathToEmitCPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "HostToEmitCDetail.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTHOSTTOEMITCPASS
#include "mlir-tensorrt/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::host_to_emitc;

//===----------------------------------------------------------------------===//
// HostToEmitC Pass/Patterns API
//===----------------------------------------------------------------------===//

static Value materializeConversion(OpBuilder &builder, Type resultType,
                                   ValueRange inputs, Location loc) {
  if (inputs.size() == 1 &&
      emitc::CastOp::areCastCompatible(inputs.front().getType(), resultType))
    return builder.create<emitc::CastOp>(loc, resultType, inputs[0]);
  return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
      .getResult(0);
}

/// Populate EmitC type conversions and op conversion patterns.
static void populateEmitCConversionPatternsAndLegality(
    const DataLayout &dataLayout, TypeConverter &typeConverter,
    ConversionTarget &target, RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  Type cuEngineType = emitc::OpaqueType::get(ctx, "nvinfer1::ICudaEngine");
  Type cuEnginePtrType = emitc::PointerType::get(cuEngineType);
  Type trtExecCtxType =
      emitc::OpaqueType::get(ctx, "nvinfer1::IExecutionContext");
  Type trtExecCtxPtrType = emitc::PointerType::get(trtExecCtxType);
  Type voidPtrType{
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"))};

  // Type conversion defines the C++ ABI we intend to emit.
  //
  // Key mappings:
  //   - `memref<...xT>` -> `mtrt::RankedMemRef<Rank>` (opaque C++ struct)
  //   - `!executor.ptr` -> `void*`
  //   - `!trtrt.engine` -> `nvinfer1::ICudaEngine*`
  //   - `!trtrt.context` -> `nvinfer1::IExecutionContext*`
  //   - CUDA opaque types -> `CUmodule` / `CUfunction` / `CUstream`
  typeConverter.addConversion([](Type t) { return t; });
  typeConverter.addConversion(
      [&](BaseMemRefType memrefType) -> std::optional<Type> {
        if (memrefType.hasRank())
          return getMemRefDescriptorType(memrefType.getContext(),
                                         memrefType.getRank());
        return {};
      });

  typeConverter.addConversion(
      [=](executor::PointerType pointerType) { return voidPtrType; });

  typeConverter.addConversion([=](Type t) -> std::optional<Type> {
    if (isa<trtrt::EngineType>(t))
      return cuEnginePtrType;
    if (isa<trtrt::ExecutionContextType>(t))
      return trtExecCtxPtrType;
    return {};
  });

  typeConverter.addConversion([&](Type t) -> std::optional<Type> {
    StringRef name =
        llvm::TypeSwitch<Type, StringRef>(t)
            .Case<cuda::ModuleType>(
                [](cuda::ModuleType) -> StringRef { return "CUmodule"; })
            .Case<cuda::FunctionType>(
                [](cuda::FunctionType) -> StringRef { return "CUfunction"; })
            .Case<cuda::StreamType>(
                [](cuda::StreamType) -> StringRef { return "CUstream"; })
            .Case<cuda::StreamType>(
                [](cuda::StreamType) -> StringRef { return "CUstream"; })
            .Default([](Type) -> StringRef { return ""; });
    if (name.empty())
      return {};
    return emitc::OpaqueType::get(t.getContext(), name);
  });

  // Setup conversion materialization functions.
  typeConverter.addSourceMaterialization(materializeConversion);
  typeConverter.addTargetMaterialization(materializeConversion);

  // Setup legality constraints.
  target.addLegalOp<executor::FileArtifactOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addLegalDialect<emitc::EmitCDialect>();
  target.addIllegalDialect<trtrt::TensorRTRuntimeDialect, cuda::CUDADialect,
                           executor::ExecutorDialect>();
  target.addDynamicallyLegalOp<func::FuncOp>([&typeConverter](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<func::CallOp>([&typeConverter](func::CallOp op) {
    return typeConverter.isLegal(op->getResultTypes()) &&
           typeConverter.isLegal(op->getOperandTypes());
  });
  target.addDynamicallyLegalOp<func::ReturnOp>(
      [&typeConverter](func::ReturnOp op) {
        return mlir::isLegalForReturnOpTypeConversionPattern(op, typeConverter);
      });

  populateHostToEmitCExecutorPatterns(patterns, typeConverter, dataLayout);
  populateHostToEmitCMemRefPatterns(patterns, typeConverter, dataLayout);
  populateHostToEmitCCudaPatterns(patterns, typeConverter, dataLayout);
  populateHostToEmitCTensorRTPatterns(patterns, typeConverter, dataLayout);
  mlir::populateSCFToEmitCConversionPatterns(patterns, typeConverter);
  mlir::populateArithToEmitCPatterns(typeConverter, patterns);
  mlir::populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      patterns, typeConverter);
  mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
  mlir::populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);
}

namespace {
class HostToEmitCPass
    : public mlir::impl::ConvertHostToEmitCPassBase<HostToEmitCPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    Location loc = moduleOp->getLoc();

    // Detect which runtime headers are required before conversion.
    // We prefer to only include TensorRT / CUDA headers when actually used so
    // that non-TRT EmitC outputs can compile in more environments.
    bool needsCudaRuntime = false;
    bool needsTensorRTRuntime = false;
    moduleOp.walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (!dialect)
        return;
      needsCudaRuntime |= isa<cuda::CUDADialect>(dialect);
      needsTensorRTRuntime |= isa<trtrt::TensorRTRuntimeDialect>(dialect);
    });

    if (!moduleOp.getSymName())
      moduleOp.setSymName("unnamed_module");

    // Before running pattern-based conversion driver, handle ctor/dtor.
    //
    // Intended C++ shape:
    //   - module-scope globals for resources (engines, modules, constants, ...)
    //   - per-resource `<name>_initialize()` / `<name>_destroy()` helpers
    //   - (optionally) aggregated `<module>_initialize_all()` /
    //   `_destroy_all()`
    IRRewriter rewriter(moduleOp->getContext());

    if (failed(convertHostToEmitCGlobals(moduleOp, emitAggregateInitDestroy)))
      return signalPassFailure();

    // Now run the pattern-based conversion.
    const DataLayoutAnalysis &dataLayoutAnalysis =
        getAnalysis<DataLayoutAnalysis>();
    const DataLayout &dataLayout = dataLayoutAnalysis.getAtOrAbove(moduleOp);

    TypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    populateEmitCConversionPatternsAndLegality(dataLayout, typeConverter,
                                               target, patterns);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      emitError(getOperation()->getLoc())
          << "failed to apply conversion in " << getArgument();
      return signalPassFailure();
    }

    //===----------------------------------------------------------------------===//
    // Insert includes and comments at the top of the module.
    //===----------------------------------------------------------------------===//
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    // clang-format off
    auto verbatim = [&](StringRef content) { rewriter.create<emitc::VerbatimOp>(loc, content); };
    verbatim("//===----------------------------------------------------------------------===//");
    verbatim("// Generated by the mlir-tensorrt compiler.");
    verbatim("//===----------------------------------------------------------------------===//");
    verbatim("#pragma once");
    // clang-format on
    rewriter.create<emitc::IncludeOp>(loc, "cstdio", true);
    rewriter.create<emitc::IncludeOp>(loc, "cstdint", true);
    rewriter.create<emitc::IncludeOp>(loc, "cstddef", true);
    rewriter.create<emitc::IncludeOp>(loc, "cstdlib", true);
    rewriter.create<emitc::IncludeOp>(loc, "cstring", true);
    rewriter.create<emitc::IncludeOp>(loc, "cmath", true);
    rewriter.create<emitc::IncludeOp>(loc, "cassert", true);
    rewriter.create<emitc::IncludeOp>(loc, "MTRTRuntimeCore.h", false);
    if (needsCudaRuntime)
      rewriter.create<emitc::IncludeOp>(loc, "MTRTRuntimeCuda.h", false);
    if (needsTensorRTRuntime)
      rewriter.create<emitc::IncludeOp>(loc, "MTRTRuntimeTensorRT.h", false);

    //===----------------------------------------------------------------------===//
    // cleanup
    //===----------------------------------------------------------------------===//
    RewritePatternSet cleanupPatterns(moduleOp->getContext());
    cleanupPatterns.add(
        +[](emitc::CastOp op, PatternRewriter &rewriter) -> LogicalResult {
          // Eliminate useless casts.
          if (op.getType() == op.getOperand().getType()) {
            rewriter.replaceOp(op, op.getOperand());
            return llvm::success();
          }
          if (auto parent = op.getSource().getDefiningOp<emitc::CastOp>()) {
            if (isa<emitc::PointerType, IndexType, IntegerType, FloatType>(
                    parent.getType())) {
              rewriter.modifyOpInPlace(op, [&]() {
                op.getSourceMutable().assign(parent.getSource());
              });
              if (parent->use_empty())
                rewriter.eraseOp(parent);
              return llvm::success();
            }
          }
          if (auto parent = op.getSource().getDefiningOp<emitc::VariableOp>()) {
            if (!parent->hasOneUse())
              return failure();
            rewriter.modifyOpInPlace(
                parent, [&]() { parent.getResult().setType(op.getType()); });
            rewriter.replaceOp(op, parent);
            return llvm::success();
          }
          if (auto parent = op.getSource().getDefiningOp<emitc::ApplyOp>()) {
            if (!parent->hasOneUse())
              return failure();
            rewriter.modifyOpInPlace(
                parent, [&]() { parent.getResult().setType(op.getType()); });
            rewriter.replaceOp(op, parent);
            return llvm::success();
          }
          return failure();
        });

    if (failed(mlir::applyPatternsGreedily(moduleOp,
                                           std::move(cleanupPatterns)))) {
      emitError(moduleOp->getLoc())
          << "failed to apply cleanup patterns in " << getArgument();
      return signalPassFailure();
    }

    // For functions, we convert `executor.ptr` arguments to void-pointers in
    // the rewrite patterns. This is because the function argument value types
    // are in function ABI attributes and can't be used as part of the type
    // conversion. At this step, we look at all arguments which have ABI
    // information. If there is a more refiend ABI value type available for
    // poitner arguments, update the pointer type. This is always legal since
    // users of `void*` values are always `emitc.cast` operations.
    // In addition, remove any other attributes which are no longer
    // relevant and will fail verification after this type conversion.
    for (auto funcOp : getOperation().getOps<func::FuncOp>()) {
      if (executor::abi::isABIWrapperFunction(funcOp))
        funcOp->removeAttr(executor::ExecutorDialect::kFuncABIAttrName);

      FunctionType funcType = funcOp.getFunctionType();
      SmallVector<Type> updatedArgTypes(funcType.getInputs());
      SmallVector<Type> updatedResultTypes(funcType.getResults());

      for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
        if (auto abiAttr = funcOp.getArgAttrOfType<executor::ArgumentABIAttr>(
                i, executor::ExecutorDialect::kArgABIAttrName)) {
          if (Type convertedType =
                  typeConverter.convertType(abiAttr.getValueType())) {
            Type newPointerType = getPointerType(convertedType);
            updatedArgTypes[i] = newPointerType;
            funcOp.getArgument(i).setType(newPointerType);
          }
          funcOp.removeArgAttr(i, executor::ExecutorDialect::kArgABIAttrName);
        }
        if (funcOp.getArgAttr(i, plan::PlanDialect::kShapeBoundsAttrName))
          funcOp.removeArgAttr(i, plan::PlanDialect::kShapeBoundsAttrName);
        if (funcOp.getArgAttr(i, plan::PlanDialect::kValueBoundsAttrName))
          funcOp.removeArgAttr(i, plan::PlanDialect::kValueBoundsAttrName);
      }

      funcOp.setFunctionType(FunctionType::get(
          funcOp.getContext(), updatedArgTypes, updatedResultTypes));
    }
  }
};
} // namespace

static std::string createEmitCOutputFileName(llvm::StringRef initialName) {
  initialName = initialName.trim();
  if (!initialName.empty() &&
      (initialName == "-" || !llvm::sys::fs::is_directory(initialName)))
    return initialName.str();

  llvm::SmallString<128> result = initialName;
  llvm::sys::path::append(result, "output.cpp");
  return result.str().str();
}

void mtrt::compiler::applyEmitCLoweringPipeline(mlir::OpPassManager &pm,
                                                const EmitCOptions &opts,
                                                llvm::StringRef outputPath,
                                                llvm::StringRef entrypoint) {
  const bool wrapModuleInEmitCClass = opts.wrapModuleInEmitCClass;

  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertMathToEmitC());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::arith::createArithExpandOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());

  // If we are going to wrap the module in an EmitC class, suppress emission of
  // aggregate module-scope init/destroy helpers (`<module>_initialize_all` /
  // `<module>_destroy_all`). The wrapper pass (`wrap-module-in-emitc-class`)
  // creates its own consolidated lifecycle methods.
  mlir::ConvertHostToEmitCPassOptions hostToEmitCOpts{};
  hostToEmitCOpts.emitAggregateInitDestroy = !wrapModuleInEmitCClass;
  pm.addPass(mlir::createConvertHostToEmitCPass(hostToEmitCOpts));

  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  if (wrapModuleInEmitCClass)
    pm.addPass(mlir::createWrapModuleInEmitCClassPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(emitc::createFormExpressionsPass());

  // Optionally emit C++ support files (runtime, CMake, driver) as artifacts.
  const bool emitRuntimeFiles = opts.emitSupportFiles || opts.emitRuntimeFiles;
  const bool emitCMakeFile = opts.emitSupportFiles || opts.emitCMakeFile;
  const bool emitTestDriver = opts.emitSupportFiles || opts.emitTestDriver;
  if (emitRuntimeFiles || emitCMakeFile || emitTestDriver) {
    mlir::EmitCppSupportFilesPassOptions supportOpts;
    supportOpts.emitRuntimeFiles = emitRuntimeFiles;
    supportOpts.emitCMakeFile = emitCMakeFile;
    supportOpts.emitTestDriver = emitTestDriver;
    // Pass the output file path relative to the artifacts directory so that
    // emitted CMake files can use ${CMAKE_CURRENT_LIST_DIR} to reference it.
    supportOpts.outputFile =
        llvm::sys::path::filename(createEmitCOutputFileName(outputPath)).str();
    supportOpts.entrypoint = entrypoint.str();
    supportOpts.supportSubdir = "emitc_support";
    pm.addPass(createEmitCppSupportFilesPass(supportOpts));
  }
}
