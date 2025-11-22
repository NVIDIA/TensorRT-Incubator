//===- ExecutorToExecutor.cpp ---------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
/// Implementation of executor-to-executor lowerings.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Conversion/ConvertToExecutorCommon.h"
#include "mlir-executor/Conversion/Passes.h"
#include "mlir-executor/Conversion/TVMFFIUtils.h"
#include "mlir-executor/Executor/IR/ExecutorAttributes.h"
#include "mlir-executor/Executor/Utils/Utils.h"
#include "mlir-executor/Runtime/API/Executable.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringExtras.h"

namespace mlir::executor {
#define GEN_PASS_DEF_CONVERTEXECUTORTOEXECUTORPASS
#include "mlir-executor/Conversion/Passes.h.inc"
} // namespace mlir::executor

using namespace mlir;
using namespace mlir::executor;

//===----------------------------------------------------------------------===//
// Executor-to-Executor Structural Conversions
//===----------------------------------------------------------------------===//

namespace {
/// Rewrite `executor.constant` if the type is illegal.
struct RewriteExecutorConst
    : public ConvertOpToExecutorPattern<executor::ConstantOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(executor::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter = getTypeConverter();
    Type resultType = typeConverter->convertType(op.getType());
    if (!resultType || resultType == op.getType())
      return failure();

    // We need to ensure that index type attributes are converted correctly.
    auto convertAttr = [&](Attribute val) -> Attribute {
      if (auto intAttr = dyn_cast<IntegerAttr>(val)) {
        return rewriter.getIntegerAttr(
            typeConverter->convertType(intAttr.getType()),
            intAttr.getValue().getSExtValue());
      };
      return val;
    };

    rewriter.replaceOpWithNewOp<executor::ConstantOp>(
        op, cast<TypedAttr>(convertAttr(adaptor.getValue())));
    return success();
  }
};

/// Structural conversion for the `executor.global` operation.
struct ConvertExecutorGlobalOp
    : public ConvertOpToExecutorPattern<executor::GlobalOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(executor::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType = typeConverter->convertType(op.getType());
    if (!convertedType || convertedType == op.getType())
      return failure();
    rewriter.modifyOpInPlace(op, [&]() { op.setType(convertedType); });
    return success();
  }
};

/// Convert `executor.func` by converting the function type signature using the
/// type converter.
struct ConvertExecutorFunc
    : public ConvertOpToExecutorPattern<executor::FuncOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(executor::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    executor::ExecutorFunctionType oldType = op.getFunctionType();
    ExecutorTypeConverter::SignatureConversion conversion(
        oldType.getArgs().size());
    executor::ExecutorFunctionType newType =
        getTypeConverter()->convertExecutorFunctionSignature(oldType,
                                                             conversion);
    if (!newType || newType == oldType)
      return failure();
    auto newFuncOp =
        rewriter.create<executor::FuncOp>(op.getLoc(), op.getName(), newType);
    newFuncOp.setSymVisibilityAttr(op.getSymVisibilityAttr());
    rewriter.replaceOp(op, newFuncOp->getResults());
    return success();
  }
};
} // namespace

namespace {
/// Convert `executor.call`.
struct ConvertExecutorCall
    : public ConvertOpToExecutorPattern<executor::CallOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(executor::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(), resultTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<executor::CallOp>(
        op, resultTypes, op.getCallee(), adaptor.getOperands());
    return success();
  }
};

/// Rewrite `executor` arithmetic ops if the types  are illegal.
struct LegalizeExecutorOperands : public ConversionPattern {
  LegalizeExecutorOperands(ExecutorTypeConverter &typeConverter,
                           MLIRContext *ctx, PatternBenefit benefit = 1)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag{}, benefit, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!llvm::isa<ExecutorDialect>(op->getDialect()) ||
        isa<executor::ConstantOp, executor::FuncOp, executor::CallOp,
            executor::CallPluginOp, executor::PluginOp,
            executor::BufferBitcastOp>(op) ||
        op->getNumRegions() > 0 ||
        (op->getNumResults() == 0 && op->getNumOperands() == 0))
      return failure();
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op->getResultTypes(), resultTypes)))
      return failure();
    OperationState state(op->getLoc(), op->getName(), operands, resultTypes,
                         llvm::to_vector(op->getAttrDictionary()));
    rewriter.replaceOp(op, rewriter.create(state)->getResults());
    return success();
  }
};

/// Handle conversion of ConstantResource attribute initializer values. This
/// should really only be invoked when the initializer value has "index" element
/// type , which needs to be converted to the target index type for
/// future serialization.
struct ConstantResourceConversionPattern
    : public ConvertOpToExecutorPattern<executor::DataSegmentOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;

  LogicalResult
  matchAndRewrite(executor::DataSegmentOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto attr = dyn_cast<DenseIntElementsAttr>(op.getValue());
    if (!attr || attr.getElementType() != rewriter.getIndexType())
      return failure();

    Type srcType = attr.getType().getElementType();
    Type dstType = getTypeConverter()->convertType(srcType);
    assert(srcType != dstType && dstType.isSignlessInteger() &&
           "index type should be converted to a signless integer type");
    DenseElementsAttr newAttr = attr.mapValues(dstType, [&](const APInt &src) {
      return src.sextOrTrunc(dstType.getIntOrFloatBitWidth());
    });
    rewriter.modifyOpInPlace(op, [&]() { op.setValueAttr(newAttr); });
    return success();
  }
};

struct LowerBufferBitcastOp
    : public ConvertOpToExecutorPattern<executor::BufferBitcastOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;

  LogicalResult
  matchAndRewrite(executor::BufferBitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

/// Lower `executor.abi.recv` to `executor.load`.
struct LowerABIRecvOp : public ConvertOpToExecutorPattern<executor::ABIRecvOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;

  LogicalResult
  matchAndRewrite(executor::ABIRecvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = typeConverter->convertType(op.getType());
    if (!resultType)
      return failure();

    auto ptr = cast<BlockArgument>(op.getPtr());
    auto func = op->getParentOfType<FunctionOpInterface>();
    assert(func && "must be inside a function");
    auto abiAttr = executor::abi::getArgumentABIAttr(func, ptr);
    assert(abiAttr && "must have an abi attribute");
    if (abiAttr.getUndef())
      return failure();

    // Create offset = 0 for the load operation
    Value offset = rewriter.create<executor::ConstantOp>(
        op.getLoc(),
        rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), 0));

    // Replace abi.recv with load ptr + 0
    rewriter.replaceOpWithNewOp<executor::LoadOp>(op, resultType,
                                                  adaptor.getPtr(), offset);
    return success();
  }
};

/// Lower `executor.abi.send` to `executor.store`.
struct LowerABISendOp : public ConvertOpToExecutorPattern<executor::ABISendOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;

  LogicalResult
  matchAndRewrite(executor::ABISendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type valueType = op.getValue().getType();

    // Check if value is a scalar type (FloatType, IndexType, IntegerType) or
    // ComplexType
    bool isScalar =
        isa<FloatType, IndexType, IntegerType, ComplexType>(valueType);

    if (isScalar) {
      // Scalar types can always be lowered to store
      Value offset = rewriter.create<executor::ConstantOp>(
          op.getLoc(),
          rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), 0));

      rewriter.create<executor::StoreOp>(op.getLoc(), adaptor.getPtr(), offset,
                                         adaptor.getValue());
      rewriter.replaceOp(op, adaptor.getValue());
      return success();
    }

    // Check if value is a MemRef type
    if (auto memrefType = dyn_cast<MemRefType>(valueType)) {
      // For MemRef types, check additional conditions:
      // 1. The ABIArgumentAttr must be marked as undef
      // 2. The ownership value must be statically known to be true

      // Get the function containing this operation
      auto func = op->getParentOfType<FunctionOpInterface>();
      if (!func)
        return failure();

      // Get the block argument for the ptr operand
      auto blockArg = dyn_cast<BlockArgument>(op.getPtr());
      if (!blockArg)
        return failure();

      // Get the ABI attribute for this argument
      executor::ArgumentABIAttr abiAttr =
          executor::abi::getArgumentABIAttr(func, blockArg);
      if (!abiAttr)
        return failure();

      // Check if the argument has 'undef' parameter set
      if (!abiAttr.getUndef())
        return failure();

      // Check if ownership is statically known to be true
      Value ownership = op.getOwnership();
      if (!ownership)
        return failure();

      // Use matchPattern to check if ownership is statically true
      if (!mlir::matchPattern(ownership, mlir::m_One()))
        return failure();

      // All conditions met, lower to store
      Value offset = rewriter.create<executor::ConstantOp>(
          op.getLoc(),
          rewriter.getIntegerAttr(getTypeConverter()->getIndexType(), 0));

      rewriter.create<executor::StoreOp>(op.getLoc(), adaptor.getPtr(), offset,
                                         adaptor.getValue());
      rewriter.replaceOp(op, adaptor.getValue());
      return success();
    }

    // For other types, return failure
    return failure();
  }
};

/// Lowers `executor.plugin` operations to runtime callable handles.
///
/// For TVM FFI backend, this conversion:
/// 1. Creates a global variable to hold the plugin callable handle
/// 2. Encodes the plugin name and function name as constant string resources
/// 3. Calls the runtime function `_create_plugin_callable_tvm_ffi` which:
///    - Loads the TVM FFI library (if not already loaded)
///    - Resolves the function symbol from the library
///    - Returns an opaque callable handle
///
/// The callable handle is stored in a global variable and can be retrieved
/// later when lowering `executor.call_plugin` operations. This handle wraps
/// a `tvm::ffi::Function` object that implements the `TVMFFISafeCallType`
/// calling convention.
struct ConvertPluginOp : public ConvertOpToExecutorPattern<executor::PluginOp> {
  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;

  /// Encodes a string as a constant resource in the module and returns a
  /// pointer to it. The string is stored in a data segment and can be
  /// accessed at runtime by the TVM FFI runtime.
  Value encodeString(OpBuilder &builder, Location loc, ModuleOp module,
                     StringRef data) const {
    DataSegmentOp resourceOp =
        getOrCreateStringConstant(builder, loc, module, "str_literal", data);
    return builder.create<executor::ConstantResourceLoadOp>(
        loc, hostPointerType, resourceOp.getSymNameAttr());
  }

  LogicalResult
  matchAndRewrite(executor::PluginOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    executor::FFIBackend ffiBackend = op.getFfiBackend();
    Location loc = op.getLoc();

    if (ffiBackend == executor::FFIBackend::tvm_ffi) {
      // Create a global variable to hold the plugin callable handle. This
      // handle is initialized at module load time and can be reused across
      // multiple calls.
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto globalOp = rewriter.create<executor::GlobalOp>(loc, op.getName(),
                                                          hostPointerType);
      Block &body = globalOp.getBodyRegion().emplaceBlock();
      rewriter.setInsertionPointToStart(&body);
      // Encode plugin and function names as constant string resources.
      // These strings are used by the runtime to locate and load the FFI
      // library and resolve the function symbol.
      Value pluginName =
          encodeString(rewriter, loc, module, op.getPluginName());
      Value functionName =
          encodeString(rewriter, loc, module, op.getFunctionName());
      // Call the runtime function to create the callable handle. This will:
      // - Load the TVM FFI library from the plugin name (if not already loaded)
      // - Resolve the function symbol from the library
      // - Return an opaque handle wrapping the TVM FFI function
      auto callOp =
          createTVMFFIPluginCallableBuilder.create(rewriter, loc, module,
                                                   {
                                                       pluginName,
                                                       functionName,
                                                   });
      // Store the callable handle in the global variable's initializer.
      rewriter.create<executor::ReturnOp>(loc, callOp->getResult(0));
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }

private:
  Type hostPointerType = executor::PointerType::get(this->getContext(),
                                                    executor::MemoryType::host);
  Type i64Type = IntegerType::get(this->getContext(), 64);
  /// Runtime function builder for `_create_plugin_callable_tvm_ffi` with
  /// signature: void* _create_plugin_callable_tvm_ffi(lib_name, function_name)
  ///
  /// This runtime function:
  /// - Loads the TVM FFI library from `lib_name` (caching loaded libraries)
  /// - Resolves the function symbol `function_name` from the library
  /// - Returns an opaque callable handle wrapping the `tvm::ffi::Function`
  ExecutorCallBuilder createTVMFFIPluginCallableBuilder = {
      this->getContext(),
      "_create_plugin_callable_tvm_ffi",
      {hostPointerType},
      {/*lib_name*/ hostPointerType,
       /*function_name*/ hostPointerType}};
};

/// Lowers `executor.call_plugin` operations to TVM FFI runtime calls.
///
/// This conversion implements the TVM FFI calling convention by:
/// 1. Parsing the argument specification (`arg_spec`) to determine how to
///    arrange arguments according to the decode spec
/// 2. Converting MLIR values to `TVMFFIAny` structures:
///    - POD values (integers, floats) are stored directly in the 16-byte
///      `TVMFFIAny` union
///    - Tensor values are converted to DLPack `DLTensor` structures and
///      stored as pointers (`kTVMFFIDLTensorPtr`)
///    - Attributes are converted to appropriate `TVMFFIAny` values
/// 3. Packing all `TVMFFIAny` values into a contiguous array according to
///    the decode spec ordering (args, rets, attrs, optional None values)
/// 4. Promoting the array to host-allocated memory so it's accessible to
///    the TVM FFI runtime
/// 5. Calling the runtime function `_call_plugin_tvm_ffi` which:
///    - Sets CUDA streams for DLTensor arguments (if applicable)
///    - Invokes the TVM FFI function using `TVMFFISafeCallType` signature:
///      `int (*TVMFFISafeCallType)(void* handle, const TVMFFIAny* args,
///                                 int32_t num_args, TVMFFIAny* result)`
///
/// Memory management:
/// - DLTensor structures are allocated on the stack (via alloca) and contain
///   pointers to the actual tensor data, shape, and strides arrays
/// - Shape and strides arrays are promoted to host memory for runtime access
/// - The `TVMFFIAny` array is allocated on the host stack for passing to FFI
///
/// Stream handling:
/// - CUDA streams are set per-tensor based on device information in DLTensor
/// - This allows proper synchronization for multi-stream CUDA operations
struct CallPluginConversionPattern
    : public ConvertOpToExecutorPattern<executor::CallPluginOp> {

  Type hostPointerType = this->getHostPointerType();
  Type i32Type = IntegerType::get(this->getContext(), 32);
  /// Runtime function builder for `_call_plugin_tvm_ffi` with signature:
  /// void _call_plugin_tvm_ffi(callable_handle, stream, args_array_ptr,
  /// num_args)
  ExecutorCallBuilder tvmffiCallBuilder = {this->getContext(),
                                           "_call_plugin_tvm_ffi",
                                           {},
                                           {/*callee*/ hostPointerType,
                                            /*stream*/ hostPointerType,
                                            /*args*/ hostPointerType,
                                            /*args_count*/ i32Type}};

  using ConvertOpToExecutorPattern::ConvertOpToExecutorPattern;
  LogicalResult
  matchAndRewrite(executor::CallPluginOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TVM FFI functions currently do not return values through the result
    // parameter. All outputs must be provided as input arguments with I/O
    // aliasing.
    if (op->getNumResults() > 0)
      return failure();

    // Get the CUDA device ID for DLTensor construction. This identifies which
    // GPU device the tensors reside on.
    Value deviceId = this->getCUDADeviceId(rewriter, op->getLoc(),
                                           op->getParentOfType<ModuleOp>());
    // Get or create a CUDA stream for async operations. If not provided,
    // use the default stream (index 0).
    Value stream = adaptor.getStream();
    if (!stream) {
      stream = this->getGlobalCudaStream(rewriter, op->getLoc(),
                                         op->getParentOfType<ModuleOp>(), 0);
    }

    // Parse the argument specification string to determine how to arrange
    // arguments. The spec string encodes the calling convention (e.g.,
    // "args.0,rets.0,attrs.my_attr").
    auto decodeSpecStrs =
        llvm::to_vector(op.getArgSpec().getAsValueRange<StringAttr>());
    auto immediateArgsDict = op.getImmediateArgsAttr();
    if (!immediateArgsDict)
      immediateArgsDict = DictionaryAttr::get(op.getContext(), {});

    llvm::SmallVector<NamedAttribute> immediateArgs;
    // Parse the arg spec into a structured DecodeSpec that tells us:
    // - Which input arguments to use (DecodeArg)
    // - Which output arguments to use (DecodeRet)
    // - Which attributes to use (DecodeAttr)
    // - Where to insert optional None values (OptionalNoneTag)
    FailureOr<abi::plugin::DecodeSpec> decodeSpec =
        executor::abi::plugin::ParseArgSpec(
            op, op.getArgs().size(), op.getOutputs().size(), decodeSpecStrs,
            immediateArgsDict, immediateArgs);
    if (failed(decodeSpec))
      return rewriter.notifyMatchFailure(op, "failed to parse arg spec");

    // Create a helper to build TVMFFIAny structures and DLTensor conversions.
    executor::TVMFFIArgsCallHelper callHelper(
        rewriter, this->getTypeConverter()->getIndexType());

    // Convert all arguments/outputs to TVMFFIAny values and arrange them
    // according to the decode spec. This creates a table containing the
    // packed TVMFFIAny structures ready for the FFI call.
    mtrt::StatusOr<Value> argsArray =
        callHelper.createTVMFFIAnyArrayForPluginCall(
            op->getLoc(), *decodeSpec, op.getArgs(), adaptor.getArgs(),
            op.getOutputs(), adaptor.getOutputs(), immediateArgsDict, deviceId);
    if (!argsArray.isOk())
      return rewriter.notifyMatchFailure(
          op, "failed to create TVMFFIAny array: " +
                  argsArray.getStatus().getMessage());

    // Promote the TVMFFIAny array to host-allocated memory so the runtime
    // can access it. The array contains pointers to DLTensor structures and
    // POD values embedded directly.
    Value argsArrayPtr = callHelper.promoteToAlloca(op->getLoc(), *argsArray);

    // Load the callable handle from the global variable created by
    // ConvertPluginOp. This handle wraps the TVM FFI function.
    Value callee = rewriter.create<executor::GetGlobalOp>(
        op->getLoc(), hostPointerType, op.getCallee());
    // Pass the number of arguments in the array (determined by decode spec).
    Value numArgs = rewriter.create<executor::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(decodeSpec->items.size()));
    // Call the runtime function which will:
    // 1. Set CUDA streams for DLTensor arguments
    // 2. Invoke the TVM FFI function using the safe call signature
    // 3. Handle errors and propagate them appropriately
    this->tvmffiCallBuilder.create(rewriter, op->getLoc(),
                                   op->getParentOfType<ModuleOp>(),
                                   {callee, stream, argsArrayPtr, numArgs});
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void executor::populateExecutorDialectLegality(
    ExecutorTypeConverter &typeConverter, ConversionTarget &target) {}

void executor::populateExecutorStructuralConversionPatternsAndLegality(
    RewritePatternSet &patterns, ExecutorTypeConverter &typeConverter,
    ConversionTarget &target) {
  // Mark ABIRecvOp as always illegal so it gets lowered
  target.addIllegalOp<executor::ABIRecvOp, executor::ABISendOp,
                      executor::PluginOp, executor::CallPluginOp>();

  target.addDynamicallyLegalDialect<executor::ExecutorDialect>(
      [&](Operation *op) {
        if (auto funcOp = dyn_cast<executor::FuncOp>(op)) {
          auto type = funcOp.getFunctionType();
          return typeConverter.isLegal(type.getArgs()) &&
                 typeConverter.isLegal(type.getResults());
        }
        if (auto globalOp = dyn_cast<executor::GlobalOp>(op))
          return typeConverter.isLegal(globalOp.getType());

        if (auto dataSegmentOp = dyn_cast<DataSegmentOp>(op))
          return dataSegmentOp.getElementType() !=
                 IndexType::get(op->getContext());

        return typeConverter.isLegal(op->getOperandTypes()) &&
               typeConverter.isLegal(op->getResultTypes());
      });

  // TODO: move more func lowerings from `executor-expand-ops` to here.
  patterns
      .add<RewriteExecutorConst, LegalizeExecutorOperands,
           ConvertExecutorGlobalOp, ConvertExecutorFunc, ConvertExecutorCall,
           ConstantResourceConversionPattern, LowerABIRecvOp, LowerABISendOp,
           ConvertPluginOp, CallPluginConversionPattern, LowerBufferBitcastOp>(
          typeConverter, patterns.getContext());
}

static LogicalResult convertExecutorFunctionMetadataAttrs(
    Operation *module, const ExecutorTypeConverter &typeConverter) {
  // Convert any `executor.function_metadata` attributes on the function
  // types.
  SmallVector<FunctionOpInterface> funcs;
  module->walk([&](FunctionOpInterface func) {
    if (!func->hasAttr(ExecutorDialect::kFunctionMetadataAttrName))
      return;
    funcs.push_back(func);
  });

  // Converts the type in the metadata signature. We allow most types
  // since they can carry important high-level information. But we disallow
  // 'index type', since we expect the backend to be specialized to i32 or i64.
  auto convertType = [&](Type t) -> Type {
    if (isa<IndexType>(t))
      return typeConverter.getIndexType();
    if (isa<MemRefType>(t)) {
      auto mT = llvm::cast<MemRefType>(t);
      if (llvm::isa<IndexType>(mT.getElementType())) {
        return MemRefType::get(mT.getShape(), typeConverter.getIndexType(),
                               mT.getLayout(), mT.getMemorySpace());
      }
    }
    return t;
  };

  for (FunctionOpInterface func : funcs) {
    auto metadata = func->getAttrOfType<FunctionMetadataAttr>(
        ExecutorDialect::kFunctionMetadataAttrName);
    SmallVector<Type> argTypes =
        llvm::map_to_vector(metadata.getArgs(), convertType);
    SmallVector<Type> resultTypes =
        llvm::map_to_vector(metadata.getResults(), convertType);

    auto attr = FunctionMetadataAttr::get(
        module->getContext(), argTypes, resultTypes,
        metadata.getNumOutputArgs(), metadata.getArgBounds(),
        metadata.getResultBounds(), metadata.getShapeFunc(),
        metadata.getCconv());

    func->setAttr(ExecutorDialect::kFunctionMetadataAttrName, attr);
  }

  return success();
}

namespace {

/// Pass to convert `executor` to `executor` dialect operations.
class ConvertExecutorToExecutorPass
    : public mlir::executor::impl::ConvertExecutorToExecutorPassBase<
          ConvertExecutorToExecutorPass> {
public:
  using Base::Base;

  ConvertExecutorToExecutorPass(
      const executor::ConvertExecutorToExecutorPassOptions
          &executorToExecutorOpts,
      const std::function<void(TypeConverter &)>
          &populateAdditionalTypeConversions)
      : Base(executorToExecutorOpts),
        populateAdditionalTypeConversions(populateAdditionalTypeConversions) {}

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ConversionTarget target(*ctx);
    LowerToExecutorOptions opts;
    opts.indexType = IntegerType::get(ctx, indexBitwidth);
    Operation *op = getOperation();
    FailureOr<DataLayout> dataLayout =
        executor::setDataLayoutSpec(op, indexBitwidth, 64);
    if (failed(dataLayout)) {
      emitError(op->getLoc())
          << "failed to set DataLayout; op has DLTI spec that is "
             "inconsistent with provided options";
      return signalPassFailure();
    }

    ExecutorTypeConverter typeConverter(ctx, opts, std::move(*dataLayout));
    if (populateAdditionalTypeConversions)
      populateAdditionalTypeConversions(typeConverter);

    RewritePatternSet patterns(ctx);
    executor::populateExecutorStructuralConversionPatternsAndLegality(
        patterns, typeConverter, target);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();

    if (failed(convertExecutorFunctionMetadataAttrs(op, typeConverter)))
      return signalPassFailure();
  }

private:
  std::function<void(TypeConverter &)> populateAdditionalTypeConversions{};
};
} // namespace

std::unique_ptr<Pass> mlir::executor::createConvertExecutorToExecutorPass(
    const ConvertExecutorToExecutorPassOptions &executorToExecutorOpts,
    const std::function<void(TypeConverter &)>
        &populateAdditionalTypeConversions) {
  return std::make_unique<ConvertExecutorToExecutorPass>(
      executorToExecutorOpts, populateAdditionalTypeConversions);
}
