//===- LuaRuntimeCompat.cpp -----------------------------------------------===//
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
/// Implementation of compatibility layer for converting between MTRT Runtime
/// values and Lua objects across different ABI versions.
///
//===----------------------------------------------------------------------===//
#include "LuaRuntimeCompat.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/API/Executable.h"
#include "mlir-executor/Runtime/API/MemRefABI.h"
#include "mlir-executor/Runtime/Backend/Common/DataTypes.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaRuntime.h"
#include "mlir-executor/Runtime/Support/CUDAHelpers.h"
#include "mlir-executor/Runtime/Support/Support.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"

using namespace mtrt;

namespace {

/// A wrapper around a bump pointer allocator that allocates storage for
/// MLIR-ABI compatible MemRef descriptors.
class DescriptorAllocator {
public:
  DescriptorAllocator(uint64_t capacity) {
    descriptorPointers.reserve(capacity);
    ranks.reserve(capacity);
  }

  /// Return the number of descriptors created.
  uint64_t size() const { return descriptorPointers.size(); }

  /// Return the UnrankedMemRefDescriptor for the given index.
  UnrankedMemRefDescriptor getDescriptorAsUnranked(unsigned index) const;

  /// Push a new descriptor with the given rank.
  UnrankedMemRefDescriptor pushDescriptor(int64_t rank);

private:
  llvm::SmallVector<uintptr_t> descriptorPointers;
  llvm::SmallVector<int64_t> ranks;
  llvm::BumpPtrAllocator allocator;
};

} // namespace

UnrankedMemRefDescriptor
DescriptorAllocator::getDescriptorAsUnranked(unsigned index) const {
  assert(index < size() && "index out of bounds");
  return UnrankedMemRefDescriptor{ranks[index], descriptorPointers[index]};
}

template <int64_t Rank>
uintptr_t allocateDescriptorAndZeroInit(llvm::BumpPtrAllocator &allocator) {
  auto *ptr = allocator.Allocate<MemRefDescriptor<Rank>>(1);
  memset(ptr, 0, sizeof(MemRefDescriptor<Rank>));
  return reinterpret_cast<uintptr_t>(ptr);
}

template <int64_t... Ranks>
uintptr_t dispatchAllocateDescriptorAndZeroInit(
    llvm::BumpPtrAllocator &allocator, int64_t rank,
    std::integer_sequence<int64_t, Ranks...>) {
  uintptr_t result = 0;
  (void)((rank == Ranks
              ? (result = allocateDescriptorAndZeroInit<Ranks>(allocator), true)
              : false) ||
         ...);
  return result;
}

UnrankedMemRefDescriptor DescriptorAllocator::pushDescriptor(int64_t rank) {
  uintptr_t ptr = dispatchAllocateDescriptorAndZeroInit(
      allocator, rank, std::make_integer_sequence<int64_t, 17>());
  descriptorPointers.push_back(ptr);
  ranks.push_back(rank);
  return UnrankedMemRefDescriptor{rank, ptr};
}

//===----------------------------------------------------------------------===//
// ABI Version 0 Implementation
//===----------------------------------------------------------------------===//

namespace mtrt::abi_v0 {

namespace {
/// Helper class for reading MemRef data from a Lua table
class MemRefTableReader {
public:
  explicit MemRefTableReader(const sol::object &obj) : mIndex(1) {
    assert(obj.is<sol::table>() && "Expected a table for MemRefValue");
    mMemRefTable = obj.as<sol::table>();
  }

  template <typename T>
  T getNextValue() {
    return mMemRefTable.get<T>(mIndex++);
  }

private:
  sol::table mMemRefTable;
  int mIndex;
};
} // namespace

/// Box a MemRef value to Lua representation (ABI v0)
/// Format: {allocated_ptr, aligned_ptr, offset, shape..., strides...}
static StatusOr<sol::object> boxMemRefToLua(const MemRefValue &value,
                                            sol::state_view &lua) {
  uintptr_t ptr = value.getMemory();
  assert(ptr != 0 && "expected non-null pointer");
  MTRT_DBG("pushing memref argument ptr={0} shape=({1:$[,]}) "
           "strides=({2:$[,]}) bitwidth={3} size={4}",
           value.getVoidPtr(), value.getShape(), value.getStrides(),
           value.getElementBitWidth(), value.getTotalFootprintInBytes());

  std::vector<sol::object> memrefTable;
  memrefTable.reserve(3 + 2 * value.getRank());
  llvm::append_range(memrefTable,
                     llvm::ArrayRef<sol::object>{
                         sol::make_object(lua, ptr),
                         sol::make_object(lua, ptr),
                         sol::make_object(lua, value.getLayout().getOffset()),
                     });

  // Push shape/strides.
  for (int64_t dim : value.getShape())
    memrefTable.push_back(sol::make_object(lua, dim));
  for (int64_t dim : value.getStrides())
    memrefTable.push_back(sol::make_object(lua, dim));

  return sol::make_object(lua, std::move(memrefTable));
}

/// Box a Scalar value to Lua representation (ABI v0)
static StatusOr<sol::object> boxScalarToLua(const ScalarValue &value,
                                            sol::state_view &lua) {
  ScalarType type = value.getType();
  sol::object obj;
  switch (type.getCode()) {
  case ScalarTypeCode::f8e4m3fn:
    obj = sol::make_object(lua, value.get<F8E4M3FN>());
    break;
  case ScalarTypeCode::f16:
    obj = sol::make_object(lua, value.get<Float16>());
    break;
  case ScalarTypeCode::bf16:
    obj = sol::make_object(lua, value.get<BFloat16>());
    break;
  case ScalarTypeCode::f32:
    obj = sol::make_object(lua, value.get<float>());
    break;
  case ScalarTypeCode::f64:
    obj = sol::make_object(lua, value.get<double>());
    break;
  case ScalarTypeCode::i1:
    obj = sol::make_object(lua, value.get<int8_t>());
    break;
  case ScalarTypeCode::i4:
    obj = sol::make_object(lua, value.get<int8_t>());
    break;
  case ScalarTypeCode::i8:
    obj = sol::make_object(lua, value.get<int8_t>());
    break;
  case ScalarTypeCode::ui8:
    obj = sol::make_object(lua, value.get<int8_t>());
    break;
  case ScalarTypeCode::i16:
    obj = sol::make_object(lua, value.get<int16_t>());
    break;
  case ScalarTypeCode::i32:
    obj = sol::make_object(lua, value.get<int32_t>());
    break;
  case ScalarTypeCode::i64:
    obj = sol::make_object(lua, value.get<int64_t>());
    break;
  case ScalarTypeCode::f4e2m1fn:
    obj = sol::make_object(lua, value.get<uint8_t>());
    break;
  case ScalarTypeCode::unknown:
  case ScalarTypeCode::complex32:
  case ScalarTypeCode::complex64:
    return getInvalidArgStatus(
        "function input argument with scalar type {0} is unsupported",
        mtrt::flat::EnumNameScalarTypeCode(type.getCode()));
  }
  return obj;
}

/// Unbox a Scalar value from Lua representation (ABI v0)
static StatusOr<std::unique_ptr<ScalarValue>>
unboxScalarFromLua(const sol::object &obj, ScalarTypeCode code) {
  switch (code) {
  case ScalarTypeCode::i1:
    return std::make_unique<ScalarValue>(obj.as<int8_t>(), code);
  case ScalarTypeCode::i4:
    return std::make_unique<ScalarValue>(obj.as<int8_t>(), code);
  case ScalarTypeCode::i8:
    return std::make_unique<ScalarValue>(obj.as<int8_t>(), code);
  case ScalarTypeCode::ui8:
    return std::make_unique<ScalarValue>(obj.as<int8_t>(), code);
  case ScalarTypeCode::i16:
    return std::make_unique<ScalarValue>(obj.as<int16_t>(), code);
  case ScalarTypeCode::i32:
    return std::make_unique<ScalarValue>(obj.as<int32_t>(), code);
  case ScalarTypeCode::i64:
    return std::make_unique<ScalarValue>(obj.as<int64_t>(), code);
  case ScalarTypeCode::f8e4m3fn:
    return std::make_unique<ScalarValue>(obj.as<F8E4M3FN>(), code);
  case ScalarTypeCode::f16:
    return std::make_unique<ScalarValue>(obj.as<Float16>(), code);
  case ScalarTypeCode::bf16:
    return std::make_unique<ScalarValue>(obj.as<BFloat16>(), code);
  case ScalarTypeCode::f32:
    return std::make_unique<ScalarValue>(obj.as<float>(), code);
  case ScalarTypeCode::f64:
    return std::make_unique<ScalarValue>(obj.as<double>(), code);
  case ScalarTypeCode::complex32:
    return std::make_unique<ScalarValue>(
        static_cast<float>(obj.as<sol::table>()[1]),
        static_cast<float>(obj.as<sol::table>()[2]), code);
  case ScalarTypeCode::complex64:
    return std::make_unique<ScalarValue>(
        static_cast<double>(obj.as<sol::table>()[1]),
        static_cast<double>(obj.as<sol::table>()[2]), code);
  case ScalarTypeCode::f4e2m1fn:
    return std::make_unique<ScalarValue>(obj.as<uint8_t>(), code);
  case mtrt::ScalarTypeCode::unknown:
    return getInvalidArgStatus("Unsupported scalar type code: {0}",
                               mtrt::flat::EnumNameScalarTypeCode(code));
  }
  llvm_unreachable("unknown scalar type code");
}

/// Unbox a MemRef value from Lua representation (ABI v0)
static StatusOr<std::unique_ptr<RuntimeValue>>
unboxMemRefFromLua(const sol::object &obj, const MemRefTypeView &memRefView,
                   LuaRuntimeSession &session) {
  RuntimeClient &client = *session.getClient();
  MemRefTableReader reader(obj);

  // Extract MemRef metadata
  uintptr_t allocPtr = reader.getNextValue<uintptr_t>();
  [[maybe_unused]] uintptr_t alignedPtr = reader.getNextValue<uintptr_t>();
  int64_t offset = reader.getNextValue<int64_t>();
  assert(offset == 0 && "expected offset to be 0");

  unsigned rank = memRefView.getRank();
  llvm::SmallVector<int64_t, 4> shape(rank);
  llvm::SmallVector<int64_t, 4> strides(rank);

  // Extract shape and strides
  for (unsigned dim = 0; dim < rank; ++dim)
    shape[dim] = reader.getNextValue<int64_t>();
  for (unsigned dim = 0; dim < rank; ++dim)
    strides[dim] = reader.getNextValue<int64_t>();

  MTRT_DBGF("Transferring ownership of returned MemRefValue [ptr %p] to client",
            reinterpret_cast<void *>(allocPtr));

  // Create a memref so that client now tracks it.
  Ref<Stream> stream = session.getStream();
  StatusOr<Ref<MemRefStorage>> storage = client.getAllocator().takeOwnership(
      allocPtr, memRefView.getAddressSpace(),
      stream ? stream->getDevice() : nullptr, stream);
  if (!storage.isOk())
    return storage.getStatus();

  auto memref =
      MemRefValue::create(memRefView.getAddressSpace(),
                          memRefView.getElementType(), std::move(*storage),
                          offset, shape, strides, client.getDevices()[0].get());
  if (!memref.isOk())
    return memref.getStatus();

  return std::unique_ptr<RuntimeValue>(std::move(*memref));
}

static StatusOr<sol::object> mtrtBoxValueToLua(RuntimeValue *value,
                                               sol::state_view &lua) {
  if (!value)
    return getInvalidArgStatus("value cannot be null");
  if (MemRefValue *memref = llvm::dyn_cast<MemRefValue>(value))
    return abi_v0::boxMemRefToLua(*memref, lua);
  if (ScalarValue *scalar = llvm::dyn_cast<ScalarValue>(value))
    return abi_v0::boxScalarToLua(*scalar, lua);
  return getInvalidArgStatus(
      "value must be either MemRef or Scalar for boxing to Lua");
}

static StatusOr<std::unique_ptr<RuntimeValue>>
luaUnboxToMTRT(const sol::object &obj, const TypeUnionView &type,
               LuaRuntimeSession &session) {
  if (type.isa<ScalarTypeView>()) {
    ScalarTypeCode code = type.get<ScalarTypeView>();
    MTRT_ASSIGN_OR_RETURN(std::unique_ptr<ScalarValue> scalar,
                          abi_v0::unboxScalarFromLua(obj, code));
    return std::unique_ptr<RuntimeValue>(std::move(scalar));
  }
  if (type.isa<MemRefTypeView>()) {
    const auto &memRefView = type.get<MemRefTypeView>();
    return abi_v0::unboxMemRefFromLua(obj, memRefView, session);
  }
  return getInvalidArgStatus(
      "type must be either ScalarTypeView or MemRefTypeView for unboxing "
      "from Lua");
}

namespace {
class LuaInvocation {
public:
  LuaInvocation(LuaRuntimeSession &session, FunctionView func);

  StatusOr<llvm::SmallVector<std::unique_ptr<RuntimeValue>>>
  invoke(llvm::ArrayRef<RuntimeValue *> inputArgs,
         llvm::ArrayRef<RuntimeValue *> outputArgs);

private:
  LuaRuntimeSession &session;
  FunctionView func;
  llvm::SmallVector<sol::object> args;
};
} // namespace

LuaInvocation::LuaInvocation(LuaRuntimeSession &session, FunctionView func)
    : session(session), func(std::move(func)) {
  FunctionSignatureView sig = func.getSignature();
  args.reserve(sig.getNumInputArgs() + sig.getNumOutputArgs());
}

StatusOr<llvm::SmallVector<std::unique_ptr<RuntimeValue>>>
LuaInvocation::invoke(llvm::ArrayRef<RuntimeValue *> inputArgs,
                      llvm::ArrayRef<RuntimeValue *> outputArgs) {
  sol::state_view lua = session.getLuaState();
  AllocTracker &tracker = session.getAllocTracker();
  Ref<Stream> stream = session.getStream();
  FunctionSignatureView sig = func.getSignature();

  for (auto [idx, rv] : llvm::enumerate(inputArgs)) {
    MTRT_ASSIGN_OR_RETURN(sol::object arg, abi_v0::mtrtBoxValueToLua(rv, lua));
    args.emplace_back(std::move(arg));

    // Track MemRef pointers as external (managed outside the session)
    if (MemRefValue *memref = llvm::dyn_cast<MemRefValue>(rv)) {
      PointerInfo pointerInfo = memref->getPointerInfo(PointerOwner::external);
      tracker.track(pointerInfo);
    }
  }
  for (auto [idx, rv] : llvm::enumerate(outputArgs)) {
    if (!llvm::isa<MemRefValue>(rv))
      return getInvalidArgStatus(
          "output (destination) argument #{0} to function "
          "{1} has an unsupported type; "
          "destination arguments must be MemRefs",
          idx + 1, func.getName());
    MemRefValue *memref = llvm::cast<MemRefValue>(rv);
    MTRT_ASSIGN_OR_RETURN(sol::object arg,
                          abi_v0::boxMemRefToLua(*memref, lua));
    args.emplace_back(std::move(arg));

    // Track MemRef pointers as external (managed outside the session)
    PointerInfo pointerInfo = memref->getPointerInfo(PointerOwner::external);
    tracker.track(pointerInfo);
  }

  std::unique_ptr<DeviceGuard> deviceGuard;
  // Set stream if provided and create a device guard.
  if (stream) {
    RETURN_STATUS_IF_ERROR(session.setStream(stream));
    if (Device *device = stream->getDevice()) {
      MTRT_ASSIGN_OR_RETURN(deviceGuard, device->createDeviceGuard());
    }
  }

  // Call the function, passing the arguments either as a table or unpacked as
  // determined by the calling convention.
  sol::protected_function funcObj = lua[func.getName()];
  if (!funcObj.valid() || !funcObj.is<sol::function>())
    return getStatusWithMsg(StatusCode::InternalError, "no function named \"",
                            std::string(func.getName()), "\" found");

  sol::protected_function_result pfr =
      sig.getCConv() == CallingConvention::unpacked
          ? funcObj(sol::as_args(args))
          : funcObj(args);

  if (!pfr.valid()) {
    sol::error err(pfr);
    return getInternalErrorStatus("failed to run function \"{0}\": {1}",
                                  func.getName(), err.what());
  }

  llvm::SmallVector<std::unique_ptr<RuntimeValue>> results;
  results.reserve(sig.getNumResults());

  for (unsigned i = 0; i < sig.getNumResults(); ++i) {
    const auto &resultType = sig.getResult(i);
    sol::object obj = pfr[i];
    MTRT_ASSIGN_OR_RETURN(std::unique_ptr<RuntimeValue> result,
                          abi_v0::luaUnboxToMTRT(obj, resultType, session));

    // For MemRef results, transfer ownership from session to client by
    // untracking from the session's AllocTracker. Checking existence is
    // important to avoid calling `untrack` multiple times when different views
    // of the same buffer are returned in result.
    if (MemRefValue *memref = llvm::dyn_cast<MemRefValue>(result.get())) {
      uintptr_t allocPtr = memref->getMemory();
      if (session.getAllocTracker().contains(allocPtr)) {
        session.getAllocTracker().untrack(allocPtr);
        session.getPinnedMemoryAllocator().untrack(allocPtr);
      }
    }

    results.push_back(std::move(result));
  }

  return results;
}

static Status validateArgTypeAgainstSpec(const RuntimeValue *runArg,
                                         const TypeUnionView &sigArg) {
  if (sigArg.isa<MemRefTypeView>()) {
    if (runArg->getKind() != RuntimeValue::Kind::MemRef)
      return getInvalidArgStatus(
          "function expects a memref type but received scalar type");
    auto view = sigArg.get<MemRefTypeView>();
    auto value = static_cast<const MemRefValue *>(runArg);

    if (view.getElementType() != value->getScalarType())
      return getInvalidArgStatus(
          "function expects a memref type with element type {0} but "
          "received {1}",
          view.getElementType().getStrRef(),
          value->getScalarType().getStrRef());

    if (view.getRank() != value->getRank())
      return getInvalidArgStatus(
          "function expects a memref type with rank {0} but received {1}",
          view.getRank(), value->getRank());

    if (view.getShape() != value->getShape()) {
      for (unsigned i = 0; i < view.getShape().size(); ++i) {
        if (value->getShape()[i] < 0)
          return getInvalidArgStatus("all shape dimensions extents must be "
                                     "non-negative but received shape [{0:, }]",
                                     mtrt::format_shape(value->getShape()));
        if (view.getShape()[i] != kDynamicSize &&
            view.getShape()[i] != value->getShape()[i])
          return getInvalidArgStatus(
              "Runtime shape mismatch. Expected [{0:, }] "
              "but received [{1:, }]",
              mtrt::format_shape(view.getShape()),
              mtrt::format_shape(value->getShape()));
      }
    }

    if (view.getStrides() != value->getStrides()) {
      bool isEmpty = llvm::is_contained(view.getShape(), 0) ||
                     llvm::is_contained(value->getShape(), 0);
      if (!isEmpty) { // Allow any non-canonical stride for empty tensor
        for (unsigned i = 0; i < view.getStrides().size(); ++i) {
          if (value->getStrides()[i] < 0)
            return getInvalidArgStatus("all strides must be non-negative but "
                                       "received strides [{0:, }]",
                                       mtrt::format_shape(value->getStrides()));
          if (view.getStrides()[i] != kDynamicSize &&
              view.getStrides()[i] != value->getStrides()[i]) {
            // Allow the special case of non-canonical stride for unit
            // dimensions See https://github.com/pytorch/pytorch/issues/99803
            // for more detail.
            if ((value->getShape()[i] == 1 && value->getStrides()[i] == 1))
              continue;

            return getInvalidArgStatus(
                "Runtime stride mismatch. Expected [{0:, }] "
                "but received [{1:, }]",
                mtrt::format_shape(view.getStrides()),
                mtrt::format_shape(value->getStrides()));
          }
        }
      }
    }

    if (view.getAddressSpace() != value->getAddressSpace())
      return getInvalidArgStatus("function expects a memref type with "
                                 "address space {0} but received {1}",
                                 EnumNamePointerType(view.getAddressSpace()),
                                 EnumNamePointerType(value->getAddressSpace()));

  } else {
    assert(sigArg.isa<ScalarTypeView>());
    if (runArg->getKind() != RuntimeValue::Kind::Scalar)
      return getInvalidArgStatus(
          "function expects a scalar type but received memref type");
    auto view = sigArg.get<ScalarTypeView>();
    auto value = static_cast<const ScalarValue *>(runArg);

    if (view != value->getType().getCode())
      return getInvalidArgStatus(
          "function expects a scalar type with element type {0} but "
          "received {1}",
          mtrt::flat::EnumNameScalarTypeCode(view),
          mtrt::flat::EnumNameScalarTypeCode(value->getType().getCode()));
  }
  return getOkStatus();
}

} // namespace mtrt::abi_v0

//===----------------------------------------------------------------------===//
// ABI v1 Implementation
//===----------------------------------------------------------------------===//

namespace mtrt::abi_v1 {

/// Box a MemRefValue to a Lua object for a function input. This just allocates
/// a descriptor, populates it with the MemRefValue's information, and puts the
/// pointer to the ranked descriptor in the Lua object.
/// NOTE: the descriptor storage is owned by the DescriptorAllocator and will be
/// deallocated when the DescriptorAllocator is destroyed. DescriptorAllocator
/// must outlive all uses.
static StatusOr<sol::object> boxMemRefToLua(const MemRefValue &value,
                                            sol::state_view &lua,
                                            DescriptorAllocator &allocator) {
  UnrankedMemRefDescriptor desc = allocator.pushDescriptor(value.getRank());
  MTRT_RETURN_IF_ERROR(populateMemRefDescriptor(desc, value));
  return sol::make_object(lua, desc.rankedDescriptorPtr);
}

/// Same as above, but used in caes where we don't have a MemRefValue but just a
/// raw pointer and buffer type.
static StatusOr<sol::object> boxMemRefToLua(uintptr_t data,
                                            const BufferType &bufferType,
                                            sol::state_view &lua,
                                            DescriptorAllocator &allocator) {
  UnrankedMemRefDescriptor desc =
      allocator.pushDescriptor(bufferType.getRank());
  MTRT_RETURN_IF_ERROR(populateMemRefDescriptor(
      desc, data, data, bufferType.getLayout().getOffset(),
      bufferType.getShape(), bufferType.getLayout().getStrides()));
  return sol::make_object(lua, desc.rankedDescriptorPtr);
}

/// Box a MemRefValue to a Lua object for a function output argument that is
/// marked 'undef', meaning that a new memref allocation will be returned and
/// the caller must take ownership. This just allocates a descriptor, zero
/// initializes it, and puts the pointer to the ranked descriptor in the Lua
/// object.
/// NOTE: the descriptor storage is owned by the DescriptorAllocator and will be
/// deallocated when the DescriptorAllocator is destroyed. DescriptorAllocator
/// must outlive all uses.
static StatusOr<sol::object>
boxUndefMemRefToLua(int64_t rank, sol::state_view &lua,
                    DescriptorAllocator &allocator) {
  UnrankedMemRefDescriptor desc = allocator.pushDescriptor(rank);
  return sol::make_object(lua, desc.rankedDescriptorPtr);
}

static Status addPackedArg(RuntimeValue *value, DescriptorAllocator &allocator,
                           std::vector<uintptr_t> &packedArgStorage) {
  if (MemRefValue *memref = llvm::dyn_cast<MemRefValue>(value)) {
    UnrankedMemRefDescriptor desc = allocator.pushDescriptor(memref->getRank());
    MTRT_RETURN_IF_ERROR(populateMemRefDescriptor(desc, *memref));
    packedArgStorage.push_back(desc.rankedDescriptorPtr);
    return getOkStatus();
  }
  if (ScalarValue *scalar = llvm::dyn_cast<ScalarValue>(value)) {
    packedArgStorage.push_back(reinterpret_cast<uintptr_t>(scalar->getRaw()));
    return getOkStatus();
  }
  return getInvalidArgStatus(
      "value must be either MemRef or Scalar for boxing to Lua");
}

static Status addPackedArg(uintptr_t data, const BufferType &bufferType,
                           DescriptorAllocator &allocator,
                           std::vector<uintptr_t> &packedArgStorage) {
  UnrankedMemRefDescriptor desc =
      allocator.pushDescriptor(bufferType.getRank());
  MTRT_RETURN_IF_ERROR(populateMemRefDescriptor(
      desc, data, data, bufferType.getLayout().getOffset(),
      bufferType.getShape(), bufferType.getLayout().getStrides()));
  packedArgStorage.push_back(desc.rankedDescriptorPtr);
  return getOkStatus();
}

static Status addUndefPackedArg(int64_t rank, DescriptorAllocator &allocator,
                                std::vector<uintptr_t> &packedArgStorage) {
  UnrankedMemRefDescriptor desc = allocator.pushDescriptor(rank);
  packedArgStorage.push_back(desc.rankedDescriptorPtr);
  return getOkStatus();
}

/// Construct a MemRefValue from an output descriptor. Corectly handles cases
/// where callee allocated and where calleer allocated. The `isUndef` flag
/// indicates that the output argument is marked 'undef' in the signature and
/// the caller must take ownership of the memref. Otherwise, we just return a
/// reference to the existing storage, which is provided in
/// `existingStorageRef`.
///
/// NOTE: The `existingStorageRef` is only non-null when `isUndef` is false.
///
/// NOTE: technically, based on what `abi_v1` currently supports, we could just
/// forward the RuntimeValue in the invocation function when `undef=false` and
/// not call this method at all. However, returning a new RuntimeValue here with
/// a reference to the storage allows for the possibility that we could use
/// `realloc` in the future internal to the function and resize the output
/// buffer. In that case, the executed function code would also populate new
/// shape/strides on the descriptor, which would require this copy to occur.
static StatusOr<std::unique_ptr<MemRefValue>>
createResultMemRefValueFromDescriptor(uintptr_t rankedDescriptorPtr,
                                      const MemRefTypeView &memRefView,
                                      bool isUndef,
                                      Ref<MemRefStorage> existingStorageRef,
                                      RuntimeSession &session) {
  MTRT_ASSIGN_OR_RETURN(MemRefDescriptorView desc,
                        getMemRefDescriptorInfo(UnrankedMemRefDescriptor{
                            memRefView.getRank(), rankedDescriptorPtr}));

  RuntimeClient &client = *session.getClient();
  Ref<Stream> stream = session.getStream();

  // If this is a new allocation (undef-marked output argument), take ownership.
  if (isUndef) {
    MTRT_ASSIGN_OR_RETURN(Ref<MemRefStorage> storage,
                          client.getAllocator().takeOwnership(
                              desc.basePtr, memRefView.getAddressSpace(),
                              stream ? stream->getDevice() : nullptr, stream));
    MTRT_ASSIGN_OR_RETURN(
        auto memref,
        MemRefValue::create(memRefView.getAddressSpace(),
                            memRefView.getElementType(), std::move(storage),
                            desc.offset,
                            llvm::ArrayRef<int64_t>(desc.shape, desc.rank),
                            llvm::ArrayRef<int64_t>(desc.strides, desc.rank),
                            stream ? stream->getDevice() : nullptr));
    return memref;
  }

  // Otherwise, return a reference to the existing storage.
  assert(existingStorageRef && "expected existing storage ref");
  MTRT_ASSIGN_OR_RETURN(
      auto memref,
      MemRefValue::create(memRefView.getAddressSpace(),
                          memRefView.getElementType(), existingStorageRef,
                          desc.offset,
                          llvm::ArrayRef<int64_t>(desc.shape, desc.rank),
                          llvm::ArrayRef<int64_t>(desc.strides, desc.rank),
                          stream ? stream->getDevice() : nullptr));
  return memref;
}

/// Unbox a MemRefValue from a Lua object that is returned via an output
/// argument. See above `createResultMemRefValueFromDescriptor` for more
/// details.
static StatusOr<std::unique_ptr<MemRefValue>>
unboxMemRefFromLua(const sol::object &obj, const MemRefTypeView &memRefView,
                   bool isUndef, Ref<MemRefStorage> existingStorageRef,
                   RuntimeSession &session) {
  uintptr_t rankedDescriptorPtr = obj.as<uintptr_t>();
  return createResultMemRefValueFromDescriptor(
      rankedDescriptorPtr, memRefView, isUndef, existingStorageRef, session);
}

static StatusOr<sol::object>
mtrtBoxInputValueToLua(RuntimeValue *value, sol::state_view &lua,
                       DescriptorAllocator &allocator) {
  if (MemRefValue *memref = llvm::dyn_cast<MemRefValue>(value))
    return abi_v1::boxMemRefToLua(*memref, lua, allocator);
  if (ScalarValue *scalar = llvm::dyn_cast<ScalarValue>(value)) {
    /// Scalar value boxing is the same for V0.
    return abi_v0::boxScalarToLua(*scalar, lua);
  }
  return getInvalidArgStatus(
      "value must be either MemRef or Scalar for boxing to Lua");
}

static StatusOr<sol::object> mtrtBoxOutputMemRefArgToLua(
    MemRefValue *value, bool isUndef, MemRefTypeView type, sol::state_view &lua,
    DescriptorAllocator &descAllocator, RuntimeClient &client,
    Ref<Stream> stream, AllocTracker &tracker, Ref<MemRefStorage> &storageRef) {
  assert(storageRef == nullptr && "expected storage ref to be nullptr");
  if (!isUndef) {
    if (value != nullptr) {
      PointerInfo pointerInfo = value->getPointerInfo(PointerOwner::external);
      tracker.track(pointerInfo);
      storageRef = value->getStorageRef();
      return abi_v1::boxMemRefToLua(*value, lua, descAllocator);
    }

    // We need to allocate the memref ourselves.
    BufferType bufferType = BufferType::getFromSerializedType(type);
    if (!bufferType.hasStaticShape()) {
      // TODO: actually, they are supported as long as we specify in the ABI the
      // protocol for the shape function.
      return getInvalidArgStatus("dynamic shape memrefs are not supported when "
                                 "automatically allocating output buffers");
    }

    MTRT_ASSIGN_OR_RETURN(Ref<MemRefStorage> storage,
                          client.getAllocator().allocate(
                              bufferType.getAddressSpace(),
                              bufferType.getFootprintSizeInBytes(),
                              /*alignment=*/std::nullopt,
                              stream ? stream->getDevice() : nullptr, stream));
    return abi_v1::boxMemRefToLua(storage->getPtr(), bufferType, lua,
                                  descAllocator);
  }

  if (value != nullptr)
    return getInvalidArgStatus("output argument value must be nullptr for "
                               "undef-marked output argument");
  return abi_v1::boxUndefMemRefToLua(type.getRank(), lua, descAllocator);
}

static Status addPackedOutputArg(MemRefValue *value, bool isUndef,
                                 MemRefTypeView type, sol::state_view &lua,
                                 DescriptorAllocator &descAllocator,
                                 RuntimeClient &client, Ref<Stream> stream,
                                 AllocTracker &tracker,
                                 Ref<MemRefStorage> &storageRef,
                                 std::vector<uintptr_t> &packedArgStorage) {
  assert(storageRef == nullptr && "expected storage ref to be nullptr");
  if (!isUndef) {
    if (value != nullptr) {
      PointerInfo pointerInfo = value->getPointerInfo(PointerOwner::external);
      tracker.track(pointerInfo);
      storageRef = value->getStorageRef();
      return abi_v1::addPackedArg(value, descAllocator, packedArgStorage);
    }

    // We need to allocate the memref ourselves.
    BufferType bufferType = BufferType::getFromSerializedType(type);
    if (!bufferType.hasStaticShape()) {
      // TODO: actually, they are supported as long as we specify in the ABI the
      // protocol for the shape function.
      return getInvalidArgStatus("dynamic shape memrefs are not supported when "
                                 "automatically allocating output buffers");
    }

    MTRT_ASSIGN_OR_RETURN(Ref<MemRefStorage> storage,
                          client.getAllocator().allocate(
                              bufferType.getAddressSpace(),
                              bufferType.getFootprintSizeInBytes(),
                              /*alignment=*/std::nullopt,
                              stream ? stream->getDevice() : nullptr, stream));
    return abi_v1::addPackedArg(storage->getPtr(), bufferType, descAllocator,
                                packedArgStorage);
  }

  if (value != nullptr)
    return getInvalidArgStatus("output argument value must be nullptr for "
                               "undef-marked output argument");
  return abi_v1::addUndefPackedArg(type.getRank(), descAllocator,
                                   packedArgStorage);
}

static StatusOr<std::pair<sol::object, std::unique_ptr<ScalarValue>>>
mtrtBoxOutputScalarArgToLua(ScalarType type, sol::state_view &lua) {
  std::unique_ptr<ScalarValue> scalar = ScalarValue::createUndef(type);
  sol::object obj =
      sol::make_object(lua, reinterpret_cast<uintptr_t>(scalar->getRaw()));
  return std::make_pair(std::move(obj), std::move(scalar));
}

static StatusOr<std::unique_ptr<ScalarValue>>
addPackedOutputArg(ScalarType type, std::vector<uintptr_t> &packedArgStorage) {
  std::unique_ptr<ScalarValue> scalar = ScalarValue::createUndef(type);
  packedArgStorage.push_back(reinterpret_cast<uintptr_t>(scalar->getRaw()));
  return scalar;
}

namespace {
class LuaInvocation {
public:
  LuaInvocation() = delete;
  LuaInvocation(LuaRuntimeSession &session, FunctionView func);
  ~LuaInvocation() = default;

  StatusOr<llvm::SmallVector<std::unique_ptr<RuntimeValue>>>
  invoke(llvm::ArrayRef<RuntimeValue *> inputArgs,
         llvm::ArrayRef<RuntimeValue *> outputArgs);

private:
  LuaRuntimeSession &session;
  FunctionView func;
  DescriptorAllocator allocator;
  llvm::SmallVector<sol::object> luaArgs;
  llvm::SmallVector<Ref<MemRefStorage>> outputArgStorageRefs;
  llvm::SmallVector<std::unique_ptr<ScalarValue>> scalarResultStorage;
};
} // namespace

LuaInvocation::LuaInvocation(LuaRuntimeSession &session, FunctionView func)
    : session(session), func(std::move(func)),
      allocator(this->func.getSignature().getNumResults() +
                this->func.getSignature().getNumOutputArgs()),
      outputArgStorageRefs(this->func.getSignature().getNumOutputArgs(),
                           nullptr),
      scalarResultStorage(this->func.getSignature().getNumOutputArgs()) {
  FunctionSignatureView sig = func.getSignature();
  luaArgs.reserve(
      sig.hasPackedArgs() ? 1 : sig.getNumInputArgs() + sig.getNumOutputArgs());
}

StatusOr<llvm::SmallVector<std::unique_ptr<RuntimeValue>>>
LuaInvocation::invoke(llvm::ArrayRef<RuntimeValue *> inputArgs,
                      llvm::ArrayRef<RuntimeValue *> outputArgs) {
  sol::state_view lua = session.getLuaState();
  AllocTracker &tracker = session.getAllocTracker();
  Ref<Stream> stream = session.getStream();
  FunctionSignatureView sig = func.getSignature();
  const bool isPacked = sig.hasPackedArgs();
  std::vector<uintptr_t> packedArgStorage;
  packedArgStorage.reserve(sig.getNumInputArgs() + sig.getNumOutputArgs());

  // In ABIv1, if the user passes an empty `outputArgs` array, all required
  // storage will be automatically allocated for the caller. Otherwise, the size
  // of the output args needs to match the number of results in the signature.
  if (outputArgs.size() != sig.getNumResults() && !outputArgs.empty()) {
    return getInvalidArgStatus("function expects {0} output args "
                               "but received {1}",
                               sig.getNumResults(), outputArgs.size());
  }

  // To handle the empty output args case, define this lambda.
  auto getCallerOutputArg = [&](unsigned idx) -> RuntimeValue * {
    assert(idx < sig.getNumResults() && "index out of bounds");
    if (outputArgs.empty())
      return nullptr;
    return outputArgs[idx];
  };

  for (auto [idx, rv] : llvm::enumerate(inputArgs)) {
    if (!isPacked) {
      MTRT_ASSIGN_OR_RETURN(sol::object arg,
                            abi_v1::mtrtBoxInputValueToLua(rv, lua, allocator));
      luaArgs.emplace_back(std::move(arg));
    } else {
      MTRT_RETURN_IF_ERROR(
          abi_v1::addPackedArg(rv, allocator, packedArgStorage));
    }

    // Track MemRef pointers as external (managed outside the session)
    if (MemRefValue *memref = llvm::dyn_cast<MemRefValue>(rv)) {
      PointerInfo pointerInfo = memref->getPointerInfo(PointerOwner::external);
      tracker.track(pointerInfo);
    }
  }
  for (unsigned i = 0, numResults = sig.getNumResults(); i < numResults; ++i) {
    if (sig.getOutputArg(i).isa<MemRefTypeView>()) {
      auto memRefType = sig.getOutputArg(i).get<MemRefTypeView>();
      bool isUndef = sig.getUndef()[i];

      if (!isPacked) {
        MTRT_ASSIGN_OR_RETURN(
            sol::object arg,
            abi_v1::mtrtBoxOutputMemRefArgToLua(
                llvm::dyn_cast_if_present<MemRefValue>(getCallerOutputArg(i)),
                isUndef, memRefType, lua, allocator, *session.getClient(),
                stream, tracker, outputArgStorageRefs[i]));
        luaArgs.emplace_back(std::move(arg));
      } else {
        MTRT_RETURN_IF_ERROR(abi_v1::addPackedOutputArg(
            llvm::dyn_cast_if_present<MemRefValue>(getCallerOutputArg(i)),
            isUndef, memRefType, lua, allocator, *session.getClient(), stream,
            tracker, outputArgStorageRefs[i], packedArgStorage));
      }
      continue;
    }

    auto scalarType = sig.getOutputArg(i).get<ScalarTypeView>();
    if (!isPacked) {
      using ScalarBoxingResult =
          std::pair<sol::object, std::unique_ptr<ScalarValue>>;
      MTRT_ASSIGN_OR_RETURN(
          ScalarBoxingResult result,
          abi_v1::mtrtBoxOutputScalarArgToLua(scalarType.view->type(), lua));
      luaArgs.emplace_back(std::move(result.first));
      scalarResultStorage[i] = std::move(result.second);
    } else {
      MTRT_ASSIGN_OR_RETURN(scalarResultStorage[i],
                            abi_v1::addPackedOutputArg(scalarType.view->type(),
                                                       packedArgStorage));
    }
    continue;
  }

  std::unique_ptr<DeviceGuard> deviceGuard;
  // Set stream if provided and create a device guard.
  if (stream) {
    RETURN_STATUS_IF_ERROR(session.setStream(stream));
    if (Device *device = stream->getDevice()) {
      MTRT_ASSIGN_OR_RETURN(deviceGuard, device->createDeviceGuard());
    }
  }

  // Call the function, passing the arguments either as a table or unpacked as
  // determined by the calling convention.
  sol::protected_function funcObj = lua[func.getName()];
  if (!funcObj.valid() || !funcObj.is<sol::function>())
    return getStatusWithMsg(StatusCode::InternalError, "no function named \"",
                            std::string(func.getName()), "\" found");

  sol::protected_function_result pfr =
      !isPacked ? funcObj(sol::as_args(luaArgs))
                : funcObj(reinterpret_cast<uintptr_t>(packedArgStorage.data()));

  if (!pfr.valid()) {
    sol::error err(pfr);
    return getInternalErrorStatus("failed to run function \"{0}\": {1}",
                                  func.getName(), err.what());
  }

  llvm::SmallVector<std::unique_ptr<RuntimeValue>> results;
  results.reserve(sig.getNumResults());
  const unsigned numInputArgs = sig.getNumInputArgs();
  for (unsigned i = 0, numResults = sig.getNumResults(); i < numResults; ++i) {
    const auto &resultType = sig.getResult(i);
    const bool isUndef = sig.getUndef()[i];
    if (resultType.isa<MemRefTypeView>()) {
      auto memrefType = resultType.get<MemRefTypeView>();
      Ref<MemRefStorage> storageRef = outputArgStorageRefs[i];
      uintptr_t memory;
      std::unique_ptr<MemRefValue> result;
      if (!isPacked) {
        sol::object obj = luaArgs[i + numInputArgs];
        MTRT_ASSIGN_OR_RETURN(
            result, abi_v1::unboxMemRefFromLua(obj, memrefType, isUndef,
                                               storageRef, session));
        memory = result->getMemory();
      } else {
        uintptr_t rankedDescriptorPtr = packedArgStorage[i + numInputArgs];
        MTRT_ASSIGN_OR_RETURN(
            result,
            abi_v1::createResultMemRefValueFromDescriptor(
                rankedDescriptorPtr, memrefType, isUndef, storageRef, session));
        memory = result->getMemory();
      }
      if (session.getAllocTracker().contains(memory)) {
        session.getAllocTracker().untrack(memory);
        session.getPinnedMemoryAllocator().untrack(memory);
      }
      results.emplace_back(std::move(result));
    } else {
      assert(scalarResultStorage[i] && "expected scalar result storage");
      results.emplace_back(std::move(scalarResultStorage[i]));
    }
  }
  return results;
}

} // namespace mtrt::abi_v1

//===----------------------------------------------------------------------===//
// Invocation API
//===----------------------------------------------------------------------===//

static Status
validateArgsAgainstSignature(const FunctionSignatureView &sig,
                             llvm::ArrayRef<RuntimeValue *> args,
                             llvm::ArrayRef<RuntimeValue *> outArgs) {

  if (sig.getNumInputArgs() != args.size())
    return getInvalidArgStatus("function expects {0} input args "
                               "but received {1}",
                               sig.getNumInputArgs(), args.size());

  if (sig.getNumOutputArgs() != outArgs.size()) {
    // Starting in ABIv1, we allow having empty output arguments.
    if (sig.getAbiVersion() < 1 || !outArgs.empty())
      return getInvalidArgStatus("function expects {0} output args "
                                 "(destination args) but received {1}",
                                 sig.getNumOutputArgs(), outArgs.size());
  }

  // Validate the inferred Lua function type here against the signature.
  for (unsigned i = 0, e = sig.getNumInputArgs(); i < e; ++i) {
    auto status = abi_v0::validateArgTypeAgainstSpec(args[i], sig.getArg(i));
    if (!status.isOk())
      return getInvalidArgStatus("Input argument {0} validation failed: {1}", i,
                                 status.getMessage());
  }
  for (unsigned i = 0, e = outArgs.size(); i < e; ++i) {
    // TODO: In ABI v0, we must provide all output arguments, but in ABI v1 we
    // allow having 'nullptr' for output arguments which indicates that we
    // should automatically allocate for the caller.
    auto status =
        abi_v0::validateArgTypeAgainstSpec(outArgs[i], sig.getOutputArg(i));
    if (!status.isOk())
      return getInvalidArgStatus(
          "Output argument {0} validation failed against "
          "corresponding function signature arg {1}. Reason: {2}",
          i, i + args.size(), status.getMessage());
  }
  return getOkStatus();
}

StatusOr<llvm::SmallVector<std::unique_ptr<RuntimeValue>>>
mtrt::invokeLuaFunction(LuaRuntimeSession &session, FunctionView func,
                        llvm::ArrayRef<RuntimeValue *> args,
                        llvm::ArrayRef<RuntimeValue *> outArgs,
                        uint32_t abiVersion) {
  AllocTracker &tracker = session.getAllocTracker();
  FunctionSignatureView sig = func.getSignature();

  MTRT_RETURN_IF_ERROR(validateArgsAgainstSignature(sig, args, outArgs));

  Ref<Stream> sessionStream = session.getStream();
  sol::object mainProgramStream =
      sol::state_view(session.getLuaState())["stream0"];
  const bool requiresStreamJoin = sessionStream && mainProgramStream.valid() &&
                                  mainProgramStream.is<uintptr_t>();
  // Forward declare an event pointer for the stream join, if required. It will
  // be re-used for the beginning join and end join. Re-use is fine since
  // cudaStreamWaitEvent captures necessary state "by value".
  uintptr_t joinEvent = -1;

  if (requiresStreamJoin) {
    // "Join" the program stream with the session stream by making sure future
    // work submitted on the program stream occurs after current work on the
    // session stream is done.
    // Currently it is the caller's responsibility for ensuring that readiness
    // of the arguments is tied to the session stream.
    MTRT_ASSIGN_OR_RETURN(joinEvent, mtrt::createCUDAEvent());
    MTRT_RETURN_IF_ERROR(
        mtrt::recordCUDAEvent(joinEvent, sessionStream->getCUDAHandle()));
    MTRT_RETURN_IF_ERROR(mtrt::waitCUDAEventOnStream(
        mainProgramStream.as<uintptr_t>(), joinEvent));
  }

  // Track the pointers for internal debugging.
  for (auto *rv : args) {
    if (MemRefValue *memref = llvm::dyn_cast<MemRefValue>(rv)) {
      PointerInfo pointerInfo = memref->getPointerInfo(PointerOwner::external);
      tracker.track(pointerInfo);
    }
  }
  for (auto *rv : outArgs) {
    if (MemRefValue *memref = llvm::dyn_cast_if_present<MemRefValue>(rv)) {
      PointerInfo pointerInfo = memref->getPointerInfo(PointerOwner::external);
      tracker.track(pointerInfo);
    }
  }

  llvm::SmallVector<std::unique_ptr<RuntimeValue>> results;
  if (abiVersion == 0) {
    abi_v0::LuaInvocation invocation(session, func);
    MTRT_ASSIGN_OR_RETURN(results, invocation.invoke(args, outArgs));
  } else if (abiVersion == 1) {
    abi_v1::LuaInvocation invocation(session, func);
    MTRT_ASSIGN_OR_RETURN(results, invocation.invoke(args, outArgs));
  } else {
    return getInvalidArgStatus("Unsupported ABI version: {0}", abiVersion);
  }

  // Forget the input pointers.
  for (auto *rv : args) {
    if (MemRefValue *memref = llvm::dyn_cast<MemRefValue>(rv)) {
      auto it = session.getAllocTracker().find(memref->getMemory());
      if (it != session.getAllocTracker().end())
        session.getAllocTracker().erase(it);
    }
  }
  for (auto *rv : outArgs) {
    if (MemRefValue *memref = llvm::dyn_cast_if_present<MemRefValue>(rv)) {
      auto it = session.getAllocTracker().find(memref->getMemory());
      if (it != session.getAllocTracker().end())
        session.getAllocTracker().erase(it);
    }
  }

  // Make session stream wait on program stream.
  if (requiresStreamJoin) {
    MTRT_RETURN_IF_ERROR(
        mtrt::recordCUDAEvent(joinEvent, mainProgramStream.as<uintptr_t>()));
    MTRT_RETURN_IF_ERROR(
        mtrt::waitCUDAEventOnStream(sessionStream->getCUDAHandle(), joinEvent));
    MTRT_RETURN_IF_ERROR(mtrt::destroyCUDAEvent(joinEvent));
  }
  return results;
}
