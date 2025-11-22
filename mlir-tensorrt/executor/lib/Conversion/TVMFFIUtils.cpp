//===- TVMFFIUtils.cpp ------------------------------------------*- C++ -*-===//
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
/// Utilities for generating IR that sets up and executes TVM-FFI C call
/// when lowering the `executor.call_plugin` operation.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Conversion/TVMFFIUtils.h"
#include "dlpack/dlpack.h"
#include "mlir-executor/Conversion/ConvertToExecutorCommon.h"
#include "mlir-executor/Executor/IR/Executor.h"
#include "mlir-executor/Executor/IR/ExecutorAttributes.h"
#include "mlir-executor/Executor/Utils/Utils.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "tvm/ffi/c_api.h"

using namespace mlir;
using namespace mlir::executor;
using namespace mtrt;

TVMFFIArgsCallHelper::TVMFFIArgsCallHelper(OpBuilder &builder, Type indexType)
    : impl(std::make_unique<Impl>(builder, indexType)) {}

TVMFFIArgsCallHelper::~TVMFFIArgsCallHelper() = default;

struct TVMFFIArgsCallHelper::Impl {

  Impl(OpBuilder &builder, Type indexType)
      : builder(builder), indexType(indexType) {}

private:
  OpBuilder &builder;
  Type indexType;
  MLIRContext *ctx = builder.getContext();
  Type hostPointerType =
      builder.getType<executor::PointerType>(MemoryType::host);
  Type i8Type = IntegerType::get(ctx, 8);
  Type i16Type = IntegerType::get(ctx, 16);
  Type i32Type = IntegerType::get(ctx, 32);
  Type i64Type = IntegerType::get(ctx, 64);

  std::array<Type, 2> dlDeviceStructElementTypes = {/*device_type*/ i32Type,
                                                    /*device_id*/ i32Type};
  Type dlDeviceStructType =
      builder.getType<executor::TableType>(dlDeviceStructElementTypes);

  std::array<Type, 3> dlDataTypeStructElementTypes = {
      /*code*/ i8Type, /*bits*/ i8Type, /*lanes*/ i16Type};
  Type dlDataTypeStructType =
      builder.getType<executor::TableType>(dlDataTypeStructElementTypes);

  Type dlTensorStructType(executor::MemoryType addrSpace) const {
    const std::array<Type, 7> dlTensorStructElementTypes = {
        /*data*/ builder.getType<executor::PointerType>(addrSpace),
        /*device*/ dlDeviceStructType,
        /*ndim*/ i32Type,
        /*dtype*/ dlDataTypeStructType,
        /*shape*/ hostPointerType,
        /*strides*/ hostPointerType,
        /*byte_offset*/ i64Type};
    return builder.getType<executor::TableType>(dlTensorStructElementTypes);
  }

  Value constI64(Location loc, int64_t value) const {
    return builder.create<executor::ConstantOp>(
        loc, builder.getI64IntegerAttr(value));
  }
  Value constI16(Location loc, int16_t value) const {
    return builder.create<executor::ConstantOp>(
        loc, builder.getI16IntegerAttr(value));
  }
  Value constI32(Location loc, int32_t value) const {
    return builder.create<executor::ConstantOp>(
        loc, builder.getI32IntegerAttr(value));
  }
  Value constIndex(Location loc, int64_t value) const {
    return builder.create<executor::ConstantOp>(
        loc, builder.getIntegerAttr(indexType, value));
  }
  Value constI8(Location loc, int8_t value) const {
    return builder.create<executor::ConstantOp>(
        loc, builder.getI8IntegerAttr(value));
  }
  Value createTable(Location loc, ValueRange elements) const {
    auto tableType = builder.getType<executor::TableType>(
        llvm::to_vector(elements.getTypes()));
    return builder.create<executor::CreateTableOp>(loc, tableType, elements);
  }
  Value getOffset(Location loc, Type elementType, OpFoldResult offset) const {
    return builder.create<executor::GetOffsetOp>(
        loc, indexType, elementType, ArrayRef<OpFoldResult>{offset});
  }
  Value zextToI64(Location loc, Value value) const {
    if (value.getType().isInteger(64))
      return value;
    return builder.create<executor::ZExtOp>(loc, i64Type, value);
  }
  Value ptrToI64(Location loc, Value value) const {
    return builder.create<executor::PtrToIntOp>(loc, i64Type, value);
  }

  Value bitcastAndZextToI64(Location loc, Value value) const {
    Type integerType =
        IntegerType::get(ctx, value.getType().getIntOrFloatBitWidth());
    return zextToI64(
        loc, builder.create<executor::BitcastOp>(loc, integerType, value));
  }

  /// Creates an i64 array in host memory from static and dynamic values.
  ///
  /// For static shapes, creates a constant resource in the module and returns
  /// a pointer to it. For dynamic shapes, creates a table on the stack
  /// containing a mix of constant and dynamic values, then promotes it to an
  /// alloca.
  ///
  /// This is used to create shape and stride arrays for DLTensor structures,
  /// which must be accessible from host memory for TVM FFI runtime access.
  Value createI64Array(Location loc, ArrayRef<int64_t> values,
                       ArrayRef<Value> dynamicVals,
                       StringRef namePrefix) const {
    if (!ShapedType::isDynamicShape(values)) {
      ModuleOp module = builder.getInsertionBlock()
                            ->getParentOp()
                            ->getParentOfType<ModuleOp>();
      llvm::SmallString<16> name = getUniqueSymbolName(module, namePrefix);
      auto attr = builder.getI64TensorAttr(values);
      auto dataOp = executor::createConstantResourceDeclaration(
          builder, loc, module, name, attr);
      return builder.create<executor::ConstantResourceLoadOp>(
          loc, hostPointerType, dataOp.getName());
    }

    // Dynamic case.
    SmallVector<Value> elements;
    unsigned dynamicIdx = 0;

    for (unsigned i = 0; i < values.size(); i++) {
      if (ShapedType::isDynamic(values[i])) {
        assert(dynamicIdx < dynamicVals.size() && "Expected dynamic value");
        elements.push_back(dynamicVals[dynamicIdx++]);
      } else {
        elements.push_back(constI64(loc, values[i]));
      }
    }
    Value tableValue = createTable(loc, elements);
    return promoteToAlloca(loc, tableValue);
  }

  /// Promotes shape and strides arrays to host-allocated memory.
  ///
  /// DLTensor requires shape and strides arrays to be accessible from the TVM
  /// FFI runtime. This function:
  /// 1. Extracts static and dynamic dimensions/strides from the memref
  /// 2. Creates i64 arrays containing shape and stride values
  /// 3. Promotes them to host memory (via alloca or constant resources)
  /// 4. Calculates the byte offset from element offset
  ///
  /// Returns a tuple of (shape_ptr, strides_ptr, bytes_offset) where the
  /// pointers point to host-allocated arrays that can be safely accessed by TVM
  /// FFI.
  std::tuple<Value, Value, Value>
  promoteShapeAndStridesToAlloca(Location loc,
                                 executor::MemRefDescriptor desc) const {
    MemRefType memrefType = desc.getMemRefType();
    SmallVector<Value> dynamicSizes;
    SmallVector<Value> dynamicStrides;
    auto [strides, offset] = memrefType.getStridesAndOffset();
    for (unsigned i = 0; i < memrefType.getRank(); i++) {
      if (ShapedType::isDynamic(memrefType.getShape()[i]))
        dynamicSizes.push_back(desc.size(builder, loc, i));
      if (ShapedType::isDynamic(strides[i]))
        dynamicStrides.push_back(desc.stride(builder, loc, i));
    }
    Value shapePtr =
        createI64Array(loc, memrefType.getShape(), dynamicSizes, "shape_array");
    Value stridesPtr =
        createI64Array(loc, strides, dynamicStrides, "strides_array");
    Value elementsOffset = [&] {
      if (!ShapedType::isDynamic(offset))
        return constI64(loc, offset);
      return desc.offset(builder, loc);
    }();
    Value bytesOffset =
        getOffset(loc, memrefType.getElementType(), elementsOffset);
    return std::make_tuple(shapePtr, stridesPtr, bytesOffset);
  }

  /// Maps executor memory space attributes to DLPack device types.
  ///
  /// Converts MLIR memory space types to DLPack device type enum values:
  /// - `device` -> kDLCUDA
  /// - `host` -> kDLCPU
  /// - `host_pinned` -> kDLCUDAHost
  /// - `unified` -> kDLCUDAManaged
  StatusOr<DLDeviceType> getDLDeviceType(MemRefType memrefType) const {
    auto memorySpace =
        dyn_cast_or_null<executor::MemoryTypeAttr>(memrefType.getMemorySpace());
    if (!memorySpace)
      return getInvalidArgStatus("missing memory space attribute");
    switch (memorySpace.getValue()) {
    case MemoryType::device:
      return DLDeviceType::kDLCUDA;
    case MemoryType::host:
      return DLDeviceType::kDLCPU;
    case MemoryType::host_pinned:
      return DLDeviceType::kDLCUDAHost;
    case MemoryType::unified:
      return DLDeviceType::kDLCUDAManaged;
    default:
      return getInvalidArgStatus("Unsupported memory space: {0}",
                                 memorySpace.getValue());
    }
  }
  /// Maps MLIR element types to DLPack data type structures.
  ///
  /// Converts MLIR types to DLPack `DLDataType` format:
  /// - Integer types -> kDLInt with appropriate bit width (rounded up to bytes)
  /// - Complex types -> kDLComplex with double the element bit width
  /// - BF16 -> kDLBfloat (16 bits)
  /// - F16/F32 -> kDLFloat with corresponding bit width
  /// - Float8E4M3FN -> kDLFloat8_e4m3fn (8 bits)
  /// - Float4E2M1FN -> kDLFloat4_e2m1fn (8 bits)
  ///
  /// The `DLDataType` structure contains `code` (type code), `bits` (bit
  /// width), and `lanes` (vectorization lanes, typically 1 for scalars).
  StatusOr<DLDataType> getDLDataType(Type elementType) const {
    DLDataType dtype;
    if (auto integerType = dyn_cast<IntegerType>(elementType)) {
      dtype.code = DLDataTypeCode::kDLInt;
      dtype.bits = llvm::divideCeil(integerType.getWidth(), 8) * 8;
      dtype.lanes = 1;
      return dtype;
    }
    if (auto complexType = dyn_cast<ComplexType>(elementType)) {
      dtype.code = DLDataTypeCode::kDLComplex;
      dtype.bits = complexType.getElementType().getIntOrFloatBitWidth() * 2;
      dtype.lanes = 1;
      return dtype;
    }
    if (elementType.isBF16()) {
      dtype.code = DLDataTypeCode::kDLBfloat;
      dtype.bits = 16;
      dtype.lanes = 1;
      return dtype;
    }
    if (elementType.isF16() || elementType.isF32()) {
      dtype.code = DLDataTypeCode::kDLFloat;
      dtype.bits = elementType.getIntOrFloatBitWidth();
      dtype.lanes = 1;
      return dtype;
    }
    if (isa<Float8E4M3FNType>(elementType)) {
      dtype.code = DLDataTypeCode::kDLFloat8_e4m3fn;
      dtype.bits = 8;
      dtype.lanes = 1;
      return dtype;
    }

    if (isa<Float4E2M1FNType>(elementType)) {
      dtype.code = DLDataTypeCode::kDLFloat4_e2m1fn;
      dtype.bits = 8;
      dtype.lanes = 1;
      return dtype;
    }
    return getInvalidArgStatus(
        "Unsupported MLIR->DLDataType conversion for element type: {0}",
        elementType);
  }

  /// Creates a DLPack `DLDevice` structure as an executor table.
  ///
  /// The `DLDevice` structure contains:
  /// - `device_type` (i32): The device type enum (CUDA, CPU, etc.)
  /// - `device_id` (i32): The device ID/index
  ///
  /// This is embedded as part of the DLTensor structure to identify where
  /// the tensor data resides.
  StatusOr<Value> createDLDeviceStruct(Location loc, Value deviceId,
                                       MemRefType memrefType) const {
    MTRT_ASSIGN_OR_RETURN(DLDeviceType deviceType, getDLDeviceType(memrefType));
    Value deviceTypeValue = constI32(loc, static_cast<int32_t>(deviceType));
    assert(deviceId.getType() == builder.getI32Type() &&
           "deviceId must be i32");
    return createTable(loc, {deviceTypeValue, deviceId});
  }

  /// Creates a DLPack `DLDataType` structure as an executor table.
  ///
  /// The `DLDataType` structure contains:
  /// - `code` (i8): Type code enum (kDLInt, kDLFloat, etc.)
  /// - `bits` (i8): Bit width of the data type
  /// - `lanes` (i16): Vectorization lanes (typically 1 for scalars)
  ///
  /// This is embedded as part of the DLTensor structure to describe the
  /// element type of the tensor.
  StatusOr<Value> createDLDataTypeStruct(Location loc, Type elementType) const {
    MTRT_ASSIGN_OR_RETURN(DLDataType dtype, getDLDataType(elementType));
    return createTable(loc, ValueRange{
                                constI8(loc, static_cast<int8_t>(dtype.code)),
                                constI8(loc, dtype.bits),
                                constI16(loc, dtype.lanes),
                            });
  }

  /// Maps MLIR types to TVM FFI type indices.
  ///
  /// The type index identifies the type stored in a `TVMFFIAny` structure.
  /// Supported types include:
  /// - Boolean (i1) -> kTVMFFIBool
  /// - Integer types -> kTVMFFIInt
  /// - Float types -> kTVMFFIFloat
  /// - MemRef types -> kTVMFFIDLTensorPtr (as pointers to DLTensor)
  StatusOr<TVMFFITypeIndex> getTVMFFITypeIndex(Location loc, Type type) const {
    if (type.isInteger(1)) {
      return TVMFFITypeIndex::kTVMFFIBool;
    }
    if (auto integerType = dyn_cast<IntegerType>(type)) {
      return TVMFFITypeIndex::kTVMFFIInt;
    }
    if (auto floatType = dyn_cast<FloatType>(type)) {
      return TVMFFITypeIndex::kTVMFFIFloat;
    }
    if (isa<MemRefType>(type)) {
      return TVMFFITypeIndex::kTVMFFIDLTensorPtr;
    }
    return getInvalidArgStatus("Unsupported type: {0}", type);
  }

  /// Creates a `TVMFFIAny` structure for a POD (Plain Old Data) value.
  ///
  /// POD values (integers, floats) are stored directly in the `TVMFFIAny` value
  /// field without heap allocation. The structure layout is:
  /// - `type_index` (i32): The TVM FFI type index
  /// - `zero_padding` (i32): Set to 0 for POD values
  /// - `v_int64`/`v_float64` (i64): The actual value (extended/bitcast as
  /// needed)
  ///
  /// Integer values wider than 64 bits are not supported. Float values are
  /// bitcast to i64 to fit in the union field.
  StatusOr<Value> createTVMFFIAnyPODValue(Location loc, Value value) const {
    MTRT_ASSIGN_OR_RETURN(TVMFFITypeIndex typeIndex,
                          getTVMFFITypeIndex(loc, value.getType()));
    if (auto intValue = dyn_cast<TypedValue<IntegerType>>(value)) {
      if (intValue.getType().getIntOrFloatBitWidth() > 64)
        return getInvalidArgStatus("Integer type too wide for TVMFFIAny: {0}",
                                   intValue.getType());

      return createTable(loc, {constI32(loc, typeIndex), constI32(loc, 0),
                               zextToI64(loc, intValue)});
    }
    if (auto floatValue = dyn_cast<TypedValue<FloatType>>(value)) {
      return createTable(loc, {constI32(loc, typeIndex), constI32(loc, 0),
                               bitcastAndZextToI64(loc, floatValue)});
    }
    return getInvalidArgStatus("Unsupported type: {0}", value.getType());
  }

  /// Creates a `TVMFFIAny` structure containing a pointer to a DLTensor.
  ///
  /// For tensor/array values, TVM FFI uses DLPack-compatible DLTensor
  /// structures. This function:
  /// 1. Creates a DLTensor structure from the memref descriptor
  /// 2. Allocates it on the stack (via alloca)
  /// 3. Wraps the pointer in a `TVMFFIAny` with `type_index =
  /// kTVMFFIDLTensorPtr`
  ///
  /// The DLTensor structure contains all metadata needed by TVM FFI to access
  /// the tensor data, including shape, strides, data type, and device
  /// information.
  StatusOr<Value> createTVMFFIAnyDLTensorPtrValue(Location loc,
                                                  MemRefDescriptor desc,
                                                  Value deviceId) const {
    MTRT_ASSIGN_OR_RETURN(Value dltensor, createDLTensor(loc, deviceId, desc));
    Value dlTensorPtr = promoteToAlloca(loc, dltensor);
    return createTable(loc, {constI32(loc, TVMFFITypeIndex::kTVMFFIDLTensorPtr),
                             constI32(loc, 0), ptrToI64(loc, dlTensorPtr)});
  }

  /// Creates a `TVMFFIAny` structure representing a None/optional value.
  ///
  /// The structure has:
  /// - `type_index = kTVMFFINone`
  /// - `zero_padding = 0`
  /// - `v_int64 = 0`
  Value createTVMFFIAnyNone(Location loc) const {
    return createTable(loc, {constI32(loc, TVMFFITypeIndex::kTVMFFINone),
                             constI32(loc, 0), constI64(loc, 0)});
  }

  /// Creates a `TVMFFIAny` structure for a string literal.
  ///
  /// Strings in TVM FFI can be represented in multiple formats. This function
  /// creates a `kTVMFFIRawStr` format, which stores:
  /// - `type_index = kTVMFFIRawStr`
  /// - `v_c_str` (i64 as pointer): Pointer to a null-terminated C string
  ///
  /// The string data is stored as a constant resource in the module, and a
  /// pointer to it is embedded in the `TVMFFIAny` structure.
  Value createTVMFFIAnyStrLiteral(Location loc, StringRef str) const {
    ModuleOp module =
        builder.getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
    DataSegmentOp dataOp = executor::getOrCreateStringConstant(
        builder, loc, module, "str_literal", str);
    Value ptr = builder.create<executor::ConstantResourceLoadOp>(
        loc, hostPointerType, dataOp.getName());
    return createTable(loc, {constI32(loc, TVMFFITypeIndex::kTVMFFIRawStr),
                             constI32(loc, static_cast<int32_t>(str.size())),
                             ptrToI64(loc, ptr)});
  }

  /// Creates a `TVMFFIAny` structure from an MLIR value.
  ///
  /// Dispatches to the appropriate creator based on the value type:
  /// - POD types (integers, floats, index) -> `createTVMFFIAnyPODValue`
  /// - MemRef types -> `createTVMFFIAnyDLTensorPtrValue`
  ///
  /// `originalValue` is the original tensor value before conversion,
  /// `convertedValue` is the bufferized memref value, and `deviceID` identifies
  /// the device for tensor operations.
  StatusOr<Value> createTVMFFIAnyValue(Location loc, Value originalValue,
                                       Value convertedValue,
                                       Value deviceID) const {
    if (isa<IntegerType, FloatType, IndexType>(originalValue.getType())) {
      return createTVMFFIAnyPODValue(loc, convertedValue);
    }
    if (auto memrefType = dyn_cast<MemRefType>(originalValue.getType())) {
      return createTVMFFIAnyDLTensorPtrValue(
          loc, executor::MemRefDescriptor(convertedValue, memrefType),
          deviceID);
    }
    return getInvalidArgStatus("unsupported type: {0}",
                               originalValue.getType());
  }

  /// Creates a `TVMFFIAny` structure from an MLIR attribute.
  ///
  /// Converts compile-time constant attributes to runtime `TVMFFIAny` values:
  /// - Integer/Float attributes -> POD values
  /// - String attributes -> String literals (kTVMFFIRawStr format)
  ///
  /// This is used for immediate arguments specified in the plugin call
  /// configuration.
  StatusOr<Value> createTVMFFIAny(Location loc, Attribute attr) const {
    if (isa<IntegerAttr, FloatAttr>(attr)) {
      Value constVal =
          builder.create<executor::ConstantOp>(loc, cast<TypedAttr>(attr));
      return createTVMFFIAnyPODValue(loc, constVal);
    }
    if (auto strAttr = dyn_cast<StringAttr>(attr)) {
      return createTVMFFIAnyStrLiteral(loc, strAttr.getValue());
    }
    return getInvalidArgStatus("unsupported attribute type: {0}", attr);
  }

public:
  /// Constructs an `!executor.table` value representing a `DLTensor` struct.
  ///
  /// Creates a DLPack-compatible tensor structure following the DLTensor
  /// layout:
  /// - `data` (pointer): Pointer to the tensor data in the specified memory
  /// space
  /// - `device` (DLDevice): Device type and ID (CUDA, CPU, etc.)
  /// - `ndim` (i32): Number of dimensions
  /// - `dtype` (DLDataType): Data type information (code, bits, lanes)
  /// - `shape` (i64*): Pointer to array of dimension sizes
  /// - `strides` (i64*): Pointer to array of strides (or nullptr for row-major)
  /// - `byte_offset` (i64): Offset in bytes from the start of the data buffer
  ///
  /// The shape and strides arrays are promoted to host-allocated memory since
  /// they need to be accessible from the TVM FFI runtime. The returned table
  /// represents the complete DLTensor structure as a flat structure suitable
  /// for passing to TVM FFI functions.
  StatusOr<Value> createDLTensor(Location loc, Value deviceId,
                                 executor::MemRefDescriptor desc) const {
    MemRefType memrefType = desc.getMemRefType();
    Value dataPtr = desc.alignedPtr(builder, loc);
    executor::MemoryTypeAttr addrSpace =
        dyn_cast_or_null<executor::MemoryTypeAttr>(memrefType.getMemorySpace());
    if (!addrSpace)
      return getInvalidArgStatus("missing memory space attribute required for "
                                 "MemRef->DLTensor encoding");
    MTRT_ASSIGN_OR_RETURN(Value deviceValue,
                          createDLDeviceStruct(loc, deviceId, memrefType));
    Value ndim = constI32(loc, memrefType.getRank());
    MTRT_ASSIGN_OR_RETURN(
        Value dtypeValue,
        createDLDataTypeStruct(loc, memrefType.getElementType()));
    auto [shape, strides, bytesOffset] =
        promoteShapeAndStridesToAlloca(loc, desc);
    return createTable(loc, {dataPtr, deviceValue, ndim, dtypeValue, shape,
                             strides, bytesOffset});
  }

  StatusOr<Value> createTVMFFIAnyArrayForPluginCall(
      Location loc, const abi::plugin::DecodeSpec &decodeSpec,
      ValueRange originalArgs, ValueRange convertedArgs,
      ValueRange originalOutputs, ValueRange convertedOutputs,
      DictionaryAttr attrDict, Value deviceID) const {

    SmallVector<Value> finalArgumentElements(decodeSpec.items.size());
    llvm::SmallDenseMap<Value, Value> anyValues;

    auto getOrCreateAnyArg = [&](unsigned argIndex) -> StatusOr<Value> {
      if (Value anyVal = anyValues.lookup(convertedArgs[argIndex]))
        return anyVal;
      MTRT_ASSIGN_OR_RETURN(Value anyValue,
                            createTVMFFIAnyValue(loc, originalArgs[argIndex],
                                                 convertedArgs[argIndex],
                                                 deviceID));
      anyValues.insert(std::make_pair(convertedArgs[argIndex], anyValue));
      return anyValue;
    };
    auto getOrCreateAnyRet = [&](unsigned retIndex) -> StatusOr<Value> {
      if (Value anyVal = anyValues.lookup(convertedOutputs[retIndex]))
        return anyVal;
      MTRT_ASSIGN_OR_RETURN(Value anyValue,
                            createTVMFFIAnyValue(loc, originalOutputs[retIndex],
                                                 convertedOutputs[retIndex],
                                                 deviceID));
      anyValues.insert(std::make_pair(convertedOutputs[retIndex], anyValue));
      return anyValue;
    };

    for (auto [idx, item] : llvm::enumerate(decodeSpec.items)) {
      if (std::holds_alternative<abi::plugin::DecodeArg>(item.kind)) {
        unsigned argIndex = std::get<abi::plugin::DecodeArg>(item.kind).index;
        MTRT_ASSIGN_OR_RETURN(Value anyValue, getOrCreateAnyArg(argIndex));
        finalArgumentElements[idx] = anyValue;
        continue;
      }
      if (std::holds_alternative<abi::plugin::DecodeRet>(item.kind)) {
        unsigned retIndex = std::get<abi::plugin::DecodeRet>(item.kind).index;
        MTRT_ASSIGN_OR_RETURN(Value anyValue, getOrCreateAnyRet(retIndex));
        finalArgumentElements[idx] = anyValue;
        continue;
      }
      if (std::holds_alternative<abi::plugin::DecodeAttr>(item.kind)) {
        llvm::StringRef attrKey =
            std::get<abi::plugin::DecodeAttr>(item.kind).name;
        Attribute attr = attrDict.get(attrKey);
        if (!attr)
          return getInvalidArgStatus(
              "attribute key {0} not found in immediate arguments dictionary",
              attrKey);
        MTRT_ASSIGN_OR_RETURN(Value anyValue, createTVMFFIAny(loc, attr));
        finalArgumentElements[idx] = anyValue;
        continue;
      }
      if (std::holds_alternative<abi::plugin::OptionalNoneTag>(item.kind)) {
        finalArgumentElements[idx] = createTVMFFIAnyNone(loc);
        continue;
      }
      llvm_unreachable("unknown decode item kind");
    }

    return createTable(loc, finalArgumentElements);
  }

  Value promoteToAlloca(Location loc, Value value) const {
    Type valueType = value.getType();
    Value one = constIndex(loc, 1);
    auto ptr = builder.create<executor::AllocaOp>(loc, hostPointerType, one,
                                                  IntegerAttr{}, valueType);
    Value zero = constIndex(loc, 0);
    builder.create<executor::StoreOp>(loc, ptr, zero, value);
    return ptr;
  }
};

StatusOr<Value>
TVMFFIArgsCallHelper::createDLTensor(Location loc, Value deviceId,
                                     executor::MemRefDescriptor desc) const {
  return impl->createDLTensor(loc, deviceId, desc);
}

Value TVMFFIArgsCallHelper::promoteToAlloca(Location loc, Value value) const {
  return impl->promoteToAlloca(loc, value);
}

StatusOr<Value> TVMFFIArgsCallHelper::createTVMFFIAnyArrayForPluginCall(
    Location loc, const abi::plugin::DecodeSpec &decodeSpec,
    ValueRange originalArgs, ValueRange convertedArgs,
    ValueRange originalOutputs, ValueRange convertedOutputs,
    DictionaryAttr attrDict, Value deviceID) const {
  return impl->createTVMFFIAnyArrayForPluginCall(
      loc, decodeSpec, originalArgs, convertedArgs, originalOutputs,
      convertedOutputs, attrDict, deviceID);
}
