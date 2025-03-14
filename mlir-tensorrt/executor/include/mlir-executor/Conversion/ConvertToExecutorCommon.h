//===- ConvertToExecutorCommon.h --------------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright 2024 NVIDIA CORPORATION & AFFILIATES.
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
/// Declarations for code common to all "Convert X-to-Executor" passes.
///
//===----------------------------------------------------------------------===//a
#ifndef INCLUDE_MLIR_EXECUTOR_CONVERSION_CONVERTTOEXECUTORCOMMON
#define INCLUDE_MLIR_EXECUTOR_CONVERSION_CONVERTTOEXECUTORCOMMON

#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Utils/MemRefDescriptorAdaptor.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace executor {
//===----------------------------------------------------------------------===//
// Executor Conversion Options
//===----------------------------------------------------------------------===//

/// Determines how `memref` typed arguments in function signatures are
/// converted.
enum class MemRefArgPassingConvention {
  /// MemRef arguments are passed as unpacked individual scalars.
  /// Unpacking/repacking of the struct occurs at function call boundaries and
  /// at the start of function bodies respectively.
  Unpacked,
  /// MemRef arguments are passed as structs by value.
  Packed
};

/// Encapsulates options that can be chosen by the caller.
struct LowerToExecutorOptions {
  LowerToExecutorOptions() = default;

  explicit LowerToExecutorOptions(Type indexType) : indexType(indexType) {}

  Type indexType;

  MemRefArgPassingConvention memrefArgPassingConvention;
};

//===----------------------------------------------------------------------===//
// Helpers for checking for or setting the DataLayout specification
// on a module.
//===----------------------------------------------------------------------===//

/// Retrieve the module's dlti.dl_spec attribute, but return nullptr if the
/// Module does not have one explicitly set (don't return default layout).
std::optional<DataLayoutSpecInterface> getDataLayoutSpec(ModuleOp op);

/// If no `dlti.dl_spec` attribute is set on the module, then set the spec
/// using the provided options. Return the DataLayout for the module. If the
/// `dlti.dl_spec` was already set, then return failure if the layout is
/// incompatible with the options.
FailureOr<DataLayout> setDataLayoutSpec(Operation *op, uint64_t indexBitwidth,
                                        uint64_t pointerBitWidth);

//===----------------------------------------------------------------------===//
// Executor Conversion Target
//===----------------------------------------------------------------------===//
class ExecutorTypeConverter;

/// A conversion target that populates functions and other information useful
/// for converting from different sources.
class ExecutorConversionTarget : public ConversionTarget {
public:
  explicit ExecutorConversionTarget(MLIRContext &ctx);
};

//===----------------------------------------------------------------------===//
// Executor Type Converter
//===----------------------------------------------------------------------===//

/// Supports type conversion during conversion to the Executor dialect.
class ExecutorTypeConverter : public TypeConverter {
public:
  /// Create an TypeConverter using custom options.
  ExecutorTypeConverter(MLIRContext *ctx, const LowerToExecutorOptions &options,
                        DataLayout dataLayout);
  /// Inherit conversion functions.
  using TypeConverter::convertType;

  /// Return a const handle to options.
  const LowerToExecutorOptions &getOptions() const { return options; }
  /// Return a converted function signature.
  Type convertFunctionSignature(FunctionType funcType,
                                SignatureConversion &result) const;

  /// Return a converted `!executor.func` type signature.
  ExecutorFunctionType
  convertExecutorFunctionSignature(ExecutorFunctionType funcType,
                                   SignatureConversion &result) const;

  /// Return the index integer type.
  Type getIndexType() const;
  /// Return an opaque function type.
  Type getOpaquePointerType(MemoryType type) const;

  /// Return the descriptor fields associated with a memref struct.
  FailureOr<SmallVector<Type>> getMemRefDescriptorFields(
      MemRefType type, std::optional<MemoryType> space = std::nullopt) const;

  /// Returns the data layout to use during and after conversion.
  const mlir::DataLayout &getDataLayout() const { return dataLayout; }

  /// Returns backend builtin function name, given a op name.
  std::string convertOpNameToBackendBuiltinFuncName(StringRef opName) const;

  /// Return the bytes required by a single element in a memref of type
  /// `memrefType`. We need indirection here in case the memref element type is
  /// non-POD. This allows for handling "memref<...xmemref<...>>"
  uint64_t getMemRefElementTypeByteSize(MemRefType memrefType) const;

  LowerToExecutorOptions options;

  MLIRContext *getContext() const { return dialect->getContext(); }

protected:
  executor::ExecutorDialect *dialect{nullptr};

  /// Data layout analysis mapping scopes to layouts active in them.
  DataLayout dataLayout;
};

//===----------------------------------------------------------------------===//
// Executor Derived Conversion Pattern Rewriters
//===----------------------------------------------------------------------===//
class MemRefDescriptor;

/// A derived ConversionPattern that also allows a variety of helper methods to
/// be accessed from within `matchAndRewrite` functions.
class ConvertToExecutorPattern : public ConversionPattern {
protected:
  /// Construct a conversion pattern with the given Executor-specific type
  /// converter, and forward the remaining arguments to ConversionPattern.
  template <typename... Args>
  explicit ConvertToExecutorPattern(ExecutorTypeConverter &typeConverter,
                                    Args &&...args)
      : ConversionPattern(typeConverter, std::forward<Args>(args)...) {}

  // The below methods are for convenience use within the `matchAndRewrite`
  // function of derived patterns.

  /// Overides base class templated method to get Executor type converter.
  const ExecutorTypeConverter *getTypeConverter() const {
    return ConversionPattern::getTypeConverter<ExecutorTypeConverter>();
  }

  //===----------------------------------------------------------------------===//
  // Common helpers for conversion patterns
  //===----------------------------------------------------------------------===//

  /// Return the CUDA device ID as an integer. Based on the program model, this
  /// is MPI ran if the program is in a SPMD context.
  Value getCUDADeviceId(OpBuilder &builder, Location loc,
                        ModuleOp module) const;

  /// Create a constant operation of the correct integer type and return the
  /// result.
  Value createIndexConstant(ImplicitLocOpBuilder &b, int64_t value) const;

  struct MemRefAllocationInformation {
    SmallVector<Value> sizes;
    SmallVector<Value> strides;
    Value sizeBytes;
    MemoryType memorySpace;
  };

  /// Returns the information necessary to allocate a memref of the given type.
  /// If the memref has any unknown dims, then the dynamic dims must also be
  /// provided.
  FailureOr<MemRefAllocationInformation>
  getMemRefAllocationInformation(ImplicitLocOpBuilder &b, MemRefType memRefType,
                                 ValueRange dynamicSizes) const;

  /// Return the offset (in units of "# memref element type") from the
  /// descriptors aligned ptr to the element indexed by `indices`. Assumes
  /// the indices are in bounds, so this could be potentially unsafe in
  /// `indices` is incorrect.
  Value getLinearizedOffset(ImplicitLocOpBuilder &b,
                            const MemRefDescriptor &descriptor,
                            ValueRange indices) const;

  /// Return the specified linear offset value in units of bytes. It is assumed
  /// that the input value is in units of "elements" where the element type is
  /// the same as the element type of `memrefType`.
  Value convertOffsetInElementsToBytes(ImplicitLocOpBuilder &b,
                                       Value offsetInElements,
                                       MemRefType memRefType) const;

  /// Return a  type of `!executor.ptr<host>`.
  PointerType getHostPointerType() const;

  /// Return a  type of `!executor.ptr<host_pinned>`.
  PointerType getHostPinnedPointerType() const;

  /// Return a  type of `!executor.ptr<device>`.
  PointerType getDevicePointerType() const;

  /// Convert the operands a function call. This treats values that originally
  /// had type `memref` specially depending on the type converter's calling
  /// convention options.
  SmallVector<Value>
  convertFuncCallOperands(RewriterBase &rewriter, Location loc,
                          ValueRange originalOperands,
                          ValueRange adaptorOperands,
                          executor::MemRefArgPassingConvention cconv) const;

  /// Return the byte type size of `t` or failure if `t` does not have a known
  /// fixed type size.
  FailureOr<uint64_t> getTypeSizeInBytes(Type t) const;

  /// Return the DataLayout object attached to the type converter.
  const DataLayout &getDataLayout() const {
    return getTypeConverter()->getDataLayout();
  }

  /// Returns `true` if `t` represents a contiguous area of memory. This is true
  /// if (1) `t` has canonical strides (identity layout) or (2) `t` has
  /// non-identity strides but is size `1` in all dimensions where the stride is
  /// non-canonical.
  static bool isContiguous(MemRefType t);

  /// Returns `true` if a copy from `srcMemRefType` to `dstMemRefType` requires
  /// a strided copy, `false` if it can be expressed as a single memcpy. If it
  /// can be, return the offset for the destination memref, otherwise return
  /// failure.
  static bool isCopyStrided(MemRefType srcMemRefType, MemRefType dstMemRefType);

  /// Return the executor::MemoryType space of a memref type, if it has such
  /// an address space attribute.
  static std::optional<MemoryType> getMemorySpace(MemRefType type);

  /// Returns true if the MemRef address space is host-visible.
  static bool isHostVisibleMemoryType(MemRefType type);

  /// Returns true if the MemRef address space is host-visible only.
  static bool isHostVisibleOnlyMemoryType(MemRefType type);

  /// Returns true if the MemRef address space is device-visible.
  static bool isDeviceVisibleMemoryType(MemRefType type);

public:
  /// Gets or inserts a GlobalOp that creates a CudaStream and inserts a
  /// `getGlobal` operation and returns the Value. The index specifies different
  /// global streams and is used to create the symbol name, e.g. `@stream0`,
  /// `@stream1` and so on.
  static Value getGlobalCudaStream(RewriterBase &rewriter, Location loc,
                                   ModuleOp module, unsigned index);
};

/// Wrapper that allows declaring a specific source operation in the template
/// parameter. Same as `OpConversionPattern` but specific to Executor dialect.
/// When instantiated, a Executor type converter must be passed.
template <typename SourceOp>
class ConvertOpToExecutorPattern : public ConvertToExecutorPattern {
public:
  using OpAdaptor = typename SourceOp::Adaptor;
  using OneToNOpAdaptor =
      typename SourceOp::template GenericAdaptor<ArrayRef<ValueRange>>;

  ConvertOpToExecutorPattern(ExecutorTypeConverter &typeConverter,
                             MLIRContext *context, PatternBenefit benefit = 1)
      : ConvertToExecutorPattern(typeConverter, SourceOp::getOperationName(),
                                 benefit, context) {}

  /// Wrappers around the ConversionPattern methods that pass the derived op
  /// type.
  LogicalResult match(Operation *op) const final {
    return match(cast<SourceOp>(op));
  }
  void rewrite(Operation *op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const final {
    if constexpr (SourceOp::hasProperties())
      return rewrite(cast<SourceOp>(op),
                     OpAdaptor(operands, op->getAttrDictionary(),
                               cast<SourceOp>(op).getProperties()),
                     rewriter);
    rewrite(cast<SourceOp>(op), OpAdaptor(operands, op->getAttrDictionary()),
            rewriter);
  }
  void rewrite(Operation *op, ArrayRef<ValueRange> operands,
               ConversionPatternRewriter &rewriter) const final {
    auto sourceOp = cast<SourceOp>(op);
    rewrite(sourceOp, OneToNOpAdaptor(operands, sourceOp), rewriter);
  }
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if constexpr (SourceOp::hasProperties())
      return matchAndRewrite(cast<SourceOp>(op),
                             OpAdaptor(operands, op->getAttrDictionary(),
                                       cast<SourceOp>(op).getProperties()),
                             rewriter);
    return matchAndRewrite(cast<SourceOp>(op),
                           OpAdaptor(operands, op->getAttrDictionary()),
                           rewriter);
  }
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<ValueRange> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceOp = cast<SourceOp>(op);
    return matchAndRewrite(sourceOp, OneToNOpAdaptor(operands, sourceOp),
                           rewriter);
  }

  /// Rewrite and Match methods that operate on the SourceOp type. These must be
  /// overridden by the derived pattern class.
  virtual LogicalResult match(SourceOp op) const {
    (void)op;
    llvm_unreachable("must override match or matchAndRewrite");
  }
  virtual void rewrite(SourceOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    (void)op;
    (void)adaptor;
    (void)rewriter;
    llvm_unreachable("must override matchAndRewrite or a rewrite method");
  }
  virtual void rewrite(SourceOp op, OneToNOpAdaptor adaptor,
                       ConversionPatternRewriter &rewriter) const {
    SmallVector<Value> oneToOneOperands =
        getOneToOneAdaptorOperands(adaptor.getOperands());
    rewrite(op, OpAdaptor(oneToOneOperands, adaptor), rewriter);
  }
  virtual LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
    if (failed(match(op)))
      return failure();
    rewrite(op, adaptor, rewriter);
    return success();
  }
  virtual LogicalResult
  matchAndRewrite(SourceOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
    SmallVector<Value> oneToOneOperands =
        getOneToOneAdaptorOperands(adaptor.getOperands());
    return matchAndRewrite(op, OpAdaptor(oneToOneOperands, adaptor), rewriter);
  }

private:
  using ConversionPattern::matchAndRewrite;
};

//===----------------------------------------------------------------------===//
// ExecutorMemRefBuilder
//===----------------------------------------------------------------------===//

/// Helper class to produce Executor dialect operations extracting or inserting
/// elements of a MemRef table/struct descriptor. Wraps a Value pointing to the
/// descriptor.
class MemRefDescriptor : public MemRefDescriptorAdaptor {
public:
  /// Construct a helper for the given descriptor value.
  MemRefDescriptor(Value descriptor, MemRefType type);

  static MemRefDescriptor
  fromComponents(ImplicitLocOpBuilder &b,
                 const ExecutorTypeConverter &typeConverter, MemRefType type,
                 Value allocatedPtr, Value alignedPtr, Value offset,
                 ValueRange sizes, ValueRange strides);
  static MemRefDescriptor
  fromComponents(ImplicitLocOpBuilder &b,
                 const ExecutorTypeConverter &typeConverter, MemRefType type,
                 Value allocatedPtr, Value alignedPtr, OpFoldResult offset,
                 ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides);

  /// Builds IR inserting the offset into the descriptor.
  void setConstantOffset(ImplicitLocOpBuilder &b, uint64_t offset);

  /// Builds IR inserting the pos-th size into the descriptor
  void setConstantSize(ImplicitLocOpBuilder &b, unsigned pos, uint64_t size);

  /// Builds IR inserting the pos-th stride into the descriptor
  void setConstantStride(ImplicitLocOpBuilder &b, unsigned pos,
                         uint64_t stride);

  SmallVector<Value> sizes(ImplicitLocOpBuilder &b) const;
  SmallVector<Value> strides(ImplicitLocOpBuilder &b) const;

  SmallVector<Value> unpack(ImplicitLocOpBuilder &b) {
    SmallVector<Value> result{allocatedPtr(b), alignedPtr(b), offset(b)};
    llvm::append_range(result, sizes(b));
    llvm::append_range(result, strides(b));
    return result;
  }

  PointerType getPtrType() {
    return cast<PointerType>(cast<TableType>(getType()).getBody()[0]);
  }

  /// Get the number of elements in the memref. Note that this only gives the
  /// shape volume and not the buffer volume (for strided types).
  Value shapeVolumeInElements(ImplicitLocOpBuilder &b) const;

  /// Get the number of elements in the memref. Note that this only gives the
  /// shape volume and not the buffer volume (for strided types).
  Value subShapeVolumeInElements(ImplicitLocOpBuilder &b, unsigned start,
                                 unsigned size) const;

  /// Get the number of bytes in the memref. Note that this only gives the
  /// shape volume and not the buffer volume (for strided types).
  Value shapeVolumeInBytes(ImplicitLocOpBuilder &b) const;

  static bool isMemRefDescriptorFieldTypes(MemRefType originalType,
                                           Type indexType, TypeRange types);

private:
  // Cached index type.
  Type indexType;
};

//===----------------------------------------------------------------------===//
// ExecutorCallBuilder
//===----------------------------------------------------------------------===//

struct ExecutorCallBuilder {
  ExecutorCallBuilder(MLIRContext *ctx, StringRef functionName,
                      ArrayRef<Type> returnType, ArrayRef<Type> argumentTypes,
                      bool trailingVarArgs = false)
      : functionName(functionName),
        functionType(executor::ExecutorFunctionType::get(
            ctx, argumentTypes, returnType,
            trailingVarArgs ? UnitAttr::get(ctx) : nullptr)) {}

  executor::CallOp create(OpBuilder &builder, Location loc, ModuleOp module,
                          ArrayRef<Value> arguments) const;

  std::string functionName;
  executor::ExecutorFunctionType functionType;
};

} // namespace executor
} // namespace mlir

#endif // INCLUDE_MLIR_EXECUTOR_CONVERSION_CONVERTTOEXECUTORCOMMON
