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
#include "mlir-executor/Runtime/Backend/Common/DataTypes.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaRuntime.h"
#include "mlir-executor/Runtime/Support/Support.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "llvm/ADT/STLExtras.h"

using namespace mtrt;

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
  StatusOr<Ref<MemRefStorage>> storage = client.getAllocator().takeOwnership(
      allocPtr, memRefView.getAddressSpace(),
      session.getCudaStream()->getDevice(), session.getCudaStream());
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
  Ref<Stream> stream = session.getCudaStream();
  FunctionSignatureView sig = func.getSignature();

  for (auto [idx, rv] : llvm::enumerate(inputArgs)) {
    MTRT_ASSIGN_OR_RETURN(sol::object arg, mtrtBoxValueToLua(rv, lua, 0));
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
    RETURN_STATUS_IF_ERROR(session.setCudaStream(stream));
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
    MTRT_ASSIGN_OR_RETURN(
        std::unique_ptr<RuntimeValue> result,
        luaUnboxToMTRT(obj, resultType, session, /*abiVersion=*/0));

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
// Public API Implementation
//===----------------------------------------------------------------------===//

StatusOr<sol::object> mtrt::mtrtBoxValueToLua(RuntimeValue *value,
                                              sol::state_view &lua,
                                              uint32_t abiVersion) {
  if (!value)
    return getInvalidArgStatus("value cannot be null");

  switch (abiVersion) {
  case 0: {
    if (MemRefValue *memref = llvm::dyn_cast<MemRefValue>(value))
      return abi_v0::boxMemRefToLua(*memref, lua);

    if (ScalarValue *scalar = llvm::dyn_cast<ScalarValue>(value))
      return abi_v0::boxScalarToLua(*scalar, lua);

    return getInvalidArgStatus(
        "value must be either MemRef or Scalar for boxing to Lua");
  }
  default:
    return getInvalidArgStatus("Unsupported ABI version: {0}", abiVersion);
  }
}

StatusOr<std::unique_ptr<RuntimeValue>>
mtrt::luaUnboxToMTRT(const sol::object &obj, const TypeUnionView &type,
                     LuaRuntimeSession &session, uint32_t abiVersion) {
  switch (abiVersion) {
  case 0: {
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
  default:
    return getInvalidArgStatus("Unsupported ABI version: {0}", abiVersion);
  }
}

static Status validateArgsAgainstSignature(
    const FunctionSignatureView &sig, llvm::ArrayRef<RuntimeValue *> args,
    llvm::ArrayRef<RuntimeValue *> outArgs, uint32_t abiVersion) {

  if (sig.getNumInputArgs() != args.size())
    return getInvalidArgStatus("function expects {0} input args "
                               "but received {1}",
                               sig.getNumInputArgs(), args.size());

  if (sig.getNumOutputArgs() != outArgs.size()) {
    return getInvalidArgStatus("function expects {0} output args "
                               "(destination args) but received {1}",
                               sig.getNumOutputArgs(), outArgs.size());
  }

  // Validate the inferred Lua function type here against the signature.
  for (unsigned i = 0, e = sig.getNumInputArgs(); i < e; ++i) {
    auto status = abi_v0::validateArgTypeAgainstSpec(args[i], sig.getArg(i));
    if (!status.isOk())
      return getInvalidArgStatus("Input argument {0} validation failed: {1}", i,
                                 status.getString());
  }
  for (unsigned i = 0, e = sig.getNumOutputArgs(); i < e; ++i) {
    // TODO: In ABI v0, we must provide all output arguments, but in ABI v1 we
    // allow having 'nullptr' for output arguments which indicates that we
    // should automatically allocate for the caller.
    auto status =
        abi_v0::validateArgTypeAgainstSpec(outArgs[i], sig.getOutputArg(i));
    if (!status.isOk())
      return getInvalidArgStatus(
          "Output argument {0} validation failed against "
          "corresponding function signature arg {1}. Reason: {2}",
          i, i + args.size(), status.getString());
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

  MTRT_RETURN_IF_ERROR(
      validateArgsAgainstSignature(sig, args, outArgs, abiVersion));

  // Track the pointers for internal debugging.
  for (auto *rv : args) {
    if (MemRefValue *memref = llvm::dyn_cast<MemRefValue>(rv)) {
      PointerInfo pointerInfo = memref->getPointerInfo(PointerOwner::external);
      tracker.track(pointerInfo);
    }
  }
  for (auto *rv : outArgs) {
    if (MemRefValue *memref = llvm::dyn_cast<MemRefValue>(rv)) {
      PointerInfo pointerInfo = memref->getPointerInfo(PointerOwner::external);
      tracker.track(pointerInfo);
    }
  }

  llvm::SmallVector<std::unique_ptr<RuntimeValue>> results;
  if (abiVersion == 0) {
    abi_v0::LuaInvocation invocation(session, func);
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
    if (MemRefValue *memref = llvm::dyn_cast<MemRefValue>(rv)) {
      auto it = session.getAllocTracker().find(memref->getMemory());
      if (it != session.getAllocTracker().end())
        session.getAllocTracker().erase(it);
    }
  }
  return results;
}
