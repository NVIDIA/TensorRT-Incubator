//===- LuaRuntime.cpp -----------------------------------------------------===//
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
/// Implementation of Lua runtime and high-level entrypoints for Lua code
/// execution.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Runtime/Backend/Lua/LuaRuntime.h"
#include "lua.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/API/ExecutableFlatbuffer.h"
#include "mlir-executor/Runtime/Backend/Common/CUDACommon.h"
#include "mlir-executor/Runtime/Backend/Common/CommonRuntime.h"
#include "mlir-executor/Runtime/Backend/Common/DataTypes.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaExtensionRegistry.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaExtensions.h"
#include "mlir-executor/Runtime/Backend/Lua/Modules/Utils/MemRefUtils.h"
#include "mlir-executor/Runtime/Backend/Utils/NvtxUtils.h"
#include "mlir-executor/Runtime/Support/Support.h"
#include "mlir-executor/Support/Allocators.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/ManagedStatic.h"
#include <memory>

#ifdef MLIR_TRT_ENABLE_CUDA
#include "cuda_runtime_api.h"
#endif // MLIR_TRT_ENABLE_CUDA

using namespace mlirtrt;
using namespace mlirtrt::runtime;

static constexpr uint64_t kMinConstantBufferByteAlignment = 8;

/// This function registers default implementations for the required SPMD
/// functions, reflecting that the executable is expected to run against a
/// single fixed CUDA device and is not part of a larger device grid.
/// These functions are only used if the runtime session is created with the
/// "single-device" feature enabled.
static void registerDefaultDeviceDependentMethods(lua_State *state,
                                                  int32_t numDevices,
                                                  int32_t deviceIdx) {
  sol::state_view lua(state);
  lua["__spmd_global_num_ranks"] = [numDevices](sol::this_state state) {
    return numDevices;
  };
  lua["__spmd_global_rank"] = [deviceIdx](sol::this_state state) {
    return deviceIdx;
  };
}

namespace mlirtrt::runtime {
void registerLuaCoreRuntimeExtension();
#ifdef MLIR_TRT_ENABLE_CUDA
void registerLuaCudaRuntimeExtension();
#endif
#ifdef MLIR_TRT_ENABLE_CUBLAS
void registerLuaCublasRuntimeExtension();
#endif
#ifdef MLIR_TRT_TARGET_TENSORRT
void registerLuaTensorRTRuntimeExtension();
#endif
#ifdef MLIR_TRT_ENABLE_NCCL
void registerLuaNcclRuntimeExtension();
#endif
} // namespace mlirtrt::runtime

void runtime::registerLuaRuntimeExtensions() {
  registerLuaCoreRuntimeExtension();
#ifdef MLIR_TRT_ENABLE_CUDA
  registerLuaCudaRuntimeExtension();
#endif
#ifdef MLIR_TRT_ENABLE_CUBLAS
  registerLuaCublasRuntimeExtension();
#endif
#ifdef MLIR_TRT_TARGET_TENSORRT
  registerLuaTensorRTRuntimeExtension();
#endif
#ifdef MLIR_TRT_ENABLE_NCCL
  registerLuaNcclRuntimeExtension();
#endif

  // The "single-device" module provides default implementation for the SPMD
  // device rank/num rank functions which just map to the one enabled device .
  registerLuaRuntimeExtension(
      "single-device",
      LuaRuntimeExtension{
          [](const RuntimeSessionOptions &options, lua_State *state,
             PinnedMemoryAllocator *pinnedMemoryAllocator,
             AllocTracker *allocTracker, ResourceTracker *resourceTracker) {
            registerDefaultDeviceDependentMethods(
                state, options.getNumDevices(), options.getDeviceId());
          }});
}

/// If the program was compiled with NCCL enabled, then check for the
/// NCCL uuid if the system has multiple GPUs.
static Status maybeCheckForValidNcclUuid(const RuntimeSessionOptions &options) {
#if MLIR_TRT_ENABLE_NCCL

  if (options.getNumDevices() > 1 && options.getNcclUuid().empty())
    return getInternalErrorStatus(
        "number of devices is {0} but the NCCL UUID is empty",
        options.getNumDevices());

  MTRT_DBG("creating session with DeviceID={0}/{1} UUID={2}",
           options.getDeviceId(), options.getNumDevices(),
           options.getNcclUuid());

#endif
  return getOkStatus();
}

//===----------------------------------------------------------------------===//
// LuaRuntimeSession
//===----------------------------------------------------------------------===//

class LuaRuntimeSession::Impl {
public:
  lua_State *getLuaState() { return luaState.lua_state(); }

private:
  sol::state luaState;
};

/// Load a data segment into the Lua state.
///
/// If the data segment is not initialized, then we return an error.
/// If the data segment is not aligned to the minimum alignment, then we
/// allocate a new buffer and copy the data into it.
/// Otherwise, we use an pointer to the data segment in the executable memory.
static Status loadHostDataSegment(sol::state_view &lua,
                                  const DataSegmentInfo &segment,
                                  RuntimeSession *session) {
  assert((segment.getAddressSpace() == PointerType::host ||
          segment.getAddressSpace() == PointerType::pinned_host) &&
         "expected host address space");
  const size_t bytes = segment.size();
  if (segment.isUninitialized()) {
    MTRT_ASSIGN_OR_RETURN(
        StatusOr<PointerInfo> buffer,
        mlirtrt::runtime::allocate(session->getAllocTracker(),
                                   segment.getAddressSpace(), bytes,
                                   segment.getAlignment(), {}));
    lua[segment.getName()] = buffer->ptr;
    return getOkStatus();
  }

  if (!llvm::isAddrAligned(llvm::Align(segment.getAlignment()),
                           segment.data())) {
    MTRT_WARNV("constant (name={0}, size={1}) is not aligned to minimum "
               "{2} bytes; copying into aligned buffer",
               segment.getName(), segment.size(), segment.getAlignment());
    MTRT_ASSIGN_OR_RETURN(
        StatusOr<PointerInfo> buffer,
        mlirtrt::runtime::allocate(session->getAllocTracker(),
                                   segment.getAddressSpace(), bytes,
                                   segment.getAlignment(), {}));
    std::memcpy(reinterpret_cast<void *>(buffer->ptr),
                reinterpret_cast<const void *>(segment.data()), bytes);
    lua[segment.getName()] = buffer->ptr;
    return getOkStatus();
  }

  // Otherwise, just use an external view.
  lua[segment.getName()] = reinterpret_cast<uintptr_t>(segment.data());
  session->getAllocTracker().track(
      PointerInfo(reinterpret_cast<uintptr_t>(segment.data()), segment.size(),
                  PointerType::host, PointerOwner::external));

  return getOkStatus();
}

/// Load a device data segment into the Lua state.
static Status loadDeviceDataSegment(sol::state_view &lua,
                                    const DataSegmentInfo &segment,
                                    RuntimeSession *session) {
#ifdef MLIR_TRT_ENABLE_CUDA
  assert(segment.getAddressSpace() == PointerType::device &&
         "expected host address space");
  const size_t bytes = segment.size();

  MTRT_ASSIGN_OR_RETURN(
      StatusOr<PointerInfo> buffer,
      mlirtrt::runtime::allocate(session->getAllocTracker(),
                                 segment.getAddressSpace(), bytes,
                                 kMinConstantBufferByteAlignment, {}));

  lua[segment.getName()] = buffer->ptr;

  // No initialization data.
  if (segment.isUninitialized())
    return getOkStatus();

  // Copy initial data into buffer.
  RETURN_ERROR_IF_CUDART_ERROR(
      cudaMemcpy(reinterpret_cast<void *>(buffer->ptr),
                 reinterpret_cast<const void *>(segment.data()), bytes,
                 cudaMemcpyHostToDevice));
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA support");
#endif
}

static Status loadDataSegment(sol::state_view &lua,
                              const DataSegmentInfo &segment,
                              RuntimeSession *session) {
  if (segment.getAddressSpace() == PointerType::host ||
      segment.getAddressSpace() == PointerType::pinned_host)
    return loadHostDataSegment(lua, segment, session);

  if (segment.getAddressSpace() == PointerType::device)
    return loadDeviceDataSegment(lua, segment, session);

  return getInternalErrorStatus(
      "global {0} has unsupported address space {1}", segment.getName(),
      mtrt::flat::EnumNamePointerType(segment.getAddressSpace()));
}

LuaRuntimeSession::LuaRuntimeSession(RuntimeSessionOptions options,
                                     ExecutableView executable)
    : RuntimeSession(std::move(options), std::move(executable)),
      impl(std::unique_ptr<Impl>(new Impl())) {}

LuaRuntimeSession::~LuaRuntimeSession() = default;

lua_State *LuaRuntimeSession::getLuaState() { return impl->getLuaState(); }

StatusOr<std::unique_ptr<LuaRuntimeSession>>
LuaRuntimeSession::create(RuntimeSessionOptions options,
                          ExecutableView executable,
                          LuaModuleRegistrationFunc registerExtraLuaFuncs) {
  MTRT_RETURN_IF_ERROR(maybeCheckForValidNcclUuid(options));

  auto session = std::unique_ptr<LuaRuntimeSession>(
      new LuaRuntimeSession(std::move(options), executable));
  sol::state_view lua = session->getLuaState();
  lua.open_libraries(sol::lib::base, sol::lib::string, sol::lib::coroutine);

  // Register builtin methods.
  MTRT_RETURN_IF_ERROR(populateRuntimeExtensions(
      session->getOptions(), lua.lua_state(),
      &session->getPinnedMemoryAllocator(), &session->getAllocTracker(),
      &session->getResourceTracker()));

  // Register user-provided methods.
  if (registerExtraLuaFuncs)
    registerExtraLuaFuncs(session->getLuaState(), &session->getAllocTracker(),
                          &session->getResourceTracker());

  // Load globals into the context.
  // TODO: eliminate this copy, we already own the executable.
  if (session->getExecutable()) {
    ExecutableView executable = session->getExecutable();
    MTRT_DBGF("loading %lu constants", executable.getDataSegments().size());
    for (DataSegmentInfo segment : executable.getDataSegments())
      MTRT_RETURN_IF_ERROR(loadDataSegment(lua, segment, session.get()));

    // Load the main Lua script.
    sol::protected_function_result result =
        lua.script(executable.getCode(), sol::script_pass_on_error);
    if (!result.valid()) {
      sol::error err = result;
      return getStatusWithMsg(StatusCode::InternalError,
                              "failed to load lua script: ", err.what());
    }
  }

  // Call the executor_init_globals function, if present.
  sol::protected_function initGlobals = lua["executor_init_globals"];
  if (initGlobals.get_type() == sol::type::function) {
    if (!initGlobals.is<std::function<void()>>())
      return getStatusWithMsg(StatusCode::InternalError,
                              "executor_init_globals function should have "
                              "signature function<void()>");
    sol::protected_function_result result = initGlobals();
    if (!result.valid()) {
      sol::error err(result);
      return getStatusWithMsg(StatusCode::InternalError,
                              "failed to initialize globals: ", err.what());
    }
  }

  return session;
}

/// Get the primary stream for the loaded executable to use.
CudaStream LuaRuntimeSession::getCudaStream() {
#ifdef MLIR_TRT_ENABLE_CUDA
  auto stream = sol::state_view(getLuaState())["stream0"].get<CudaStream>();
  return stream;
#else
  llvm::report_fatal_error("runtime not compiled with CUDA support");
#endif
}

/// Set the primary stream for the loaded executable to use.
Status LuaRuntimeSession::setCudaStream(CudaStream stream) {
#ifdef MLIR_TRT_ENABLE_CUDA
  sol::state_view lua = getLuaState();
  lua["stream0"] = stream;
  return getOkStatus();
#else
  return getInternalErrorStatus("runtime not compiled with CUDA support");
#endif
}

//===----------------------------------------------------------------------===//
// Convenience Functions
//===----------------------------------------------------------------------===//

StatusOr<int64_t> mlirtrt::runtime::runExecutorLuaScript(
    RuntimeSessionOptions options, std::string_view luaScript,
    LuaRuntimeSession::LuaModuleRegistrationFunc registerExtraLuaFuncs) {
  ADD_RUNTIME_MODULE_RANGE("runtime_runExecutorLuaScript");

  StatusOr<Ref<RuntimeClient>> client = RuntimeClient::create();
  if (!client.isOk())
    return client.getStatus();

  MTRT_ASSIGN_OR_RETURN(
      std::unique_ptr<LuaRuntimeSession> session,
      LuaRuntimeSession::create(std::move(options), ExecutableView(nullptr),
                                std::move(registerExtraLuaFuncs)));

  sol::state_view lua = session->getLuaState();
  sol::protected_function_result result = lua.safe_script(luaScript);
  if (!result.valid()) {
    sol::error err = result;
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to load lua script: ", err.what());
  }

  // Call the main function, if present.
  sol::protected_function mainObj = lua["main"];
  if (mainObj.get_type() != sol::type::function)
    return getStatusWithMsg(StatusCode::InternalError,
                            "no main function present");
  if (!mainObj.is<std::function<int()>>())
    return getStatusWithMsg(
        StatusCode::InternalError,
        "main function should have signature function<int()>");
  result = mainObj();

  if (!result.valid()) {
    sol::error err = result;
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to run main function: ", err.what());
  }

  if (result.return_count() != 1 || result.get_type(0) != sol::type::number)
    return getStatusWithMsg(
        StatusCode::InternalError,
        "main function did not return an integer return code");

  return result[0].get<int64_t>();
}

StatusOr<int64_t> mlirtrt::runtime::runExecutorExecutable(
    RuntimeSessionOptions options, std::unique_ptr<Executable> executable,
    LuaRuntimeSession::LuaModuleRegistrationFunc registerExtraLuaFuncs) {

  StatusOr<Ref<RuntimeClient>> client = RuntimeClient::create();
  if (!client.isOk())
    return client.getStatus();

  MTRT_ASSIGN_OR_RETURN(
      std::unique_ptr<LuaRuntimeSession> session,
      LuaRuntimeSession::create(std::move(options), executable->getView(),
                                std::move(registerExtraLuaFuncs)));

  // Call the main function, if present.
  sol::state_view lua = session->getLuaState();
  sol::protected_function mainObj = lua["main"];
  if (mainObj.get_type() != sol::type::function)
    return getStatusWithMsg(StatusCode::InternalError,
                            std::string("no main function present"));
  if (!mainObj.is<std::function<int()>>())
    return getStatusWithMsg(
        StatusCode::InternalError,
        "main function should have signature function<int()>");

  sol::protected_function_result result = mainObj();
  if (!result.valid()) {
    sol::error err(result);
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to run main function: ", err.what());
  }

  if (result.return_count() != 1 || result.get_type(0) != sol::type::number)
    return getStatusWithMsg(
        StatusCode::InternalError,
        "main function did not return an integer return code");

  return result[0].get<int64_t>();
}

/// A "memref" in executor IR is packed into a table of
/// (allocated_ptr, aligned_ptr, offset, [shape list], [stride list]) when
/// passed across the function I/O
/// We also need to inform the RuntimeSession's `tracker` about metadata
/// for the input memref, including that it is managed outside the session.
static Status pushMemRefTableArg(sol::state_view &lua, AllocTracker &tracker,
                                 llvm::SmallVector<sol::object> &args,
                                 const MemRefValue &value) {
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

  args.emplace_back(sol::make_object(lua, std::move(memrefTable)));

  PointerInfo pointerInfo = value.getPointerInfo(PointerOwner::external);

  // Track the pointer in the current AllocTracker
  tracker.track(pointerInfo);

  return getOkStatus();
}

static Status pushScalarArgument(sol::state_view &lua,
                                 llvm::SmallVector<sol::object> &args,
                                 const ScalarValue &value) {
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
  default:
    return getInvalidArgStatus(
        "function input argument with scalar type {0} is unsupported",
        mtrt::flat::EnumNameScalarTypeCode(type.getCode()));
  }
  args.push_back(obj);
  return getOkStatus();
}

static Status validateArgsTypesAgainstFuncArgs(const RuntimeValue *runArg,
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
          "receieved {1}",
          view.getElementType().getStrRef(),
          value->getScalarType().getStrRef());

    if (view.getRank() != value->getRank())
      return getInvalidArgStatus(
          "function expects a memref type with rank {0} but receieved {1}",
          view.getRank(), value->getRank());

    if (view.getShape() != value->getShape()) {
      for (unsigned i = 0; i < view.getShape().size(); ++i) {
        if (value->getShape()[i] < 0)
          return getInvalidArgStatus(
              "all shape dimensions extents must be "
              "non-negative but received shape [{0:$[, ]}]",
              value->getShape());
        if (view.getShape()[i] >= 0 &&
            view.getShape()[i] != value->getShape()[i])
          return getInvalidArgStatus(
              "Runtime shape mismatch. Expected [{0:$[, ]}] "
              "but received [{1:$[, ]}]",
              view.getShape(), value->getShape());
      }
    }

    if (view.getStrides() != value->getStrides()) {
      bool isEmpty = llvm::is_contained(view.getShape(), 0);
      if (!isEmpty) { // Allow any non-canonical stride for empty tensor
        for (unsigned i = 0; i < view.getStrides().size(); ++i) {
          if (value->getStrides()[i] < 0)
            return getInvalidArgStatus("all strides must be non-negative but "
                                       "received shape [{0:$[, ]}]",
                                       value->getStrides());
          if (view.getStrides()[i] >= 0 &&
              view.getStrides()[i] != value->getStrides()[i])
            // Allow the special case of non-canonical stride for unit
            // dimensions See https://github.com/pytorch/pytorch/issues/99803
            // for more detail
            if (value->getShape()[i] != 1 || value->getStrides()[i] != 1)
              return getInvalidArgStatus(
                  "Runtime stride mismatch. Expected [{0:$[, ]}] "
                  "but received [{1:$[, ]}]",
                  view.getStrides(), value->getStrides());
        }
      }
    }

    if (view.getAddressSpace() != value->getAddressSpace())
      return getInvalidArgStatus("function expects a memref type with "
                                 "address space {0} but receieved {1}",
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
          "receieved {1}",
          mtrt::flat::EnumNameScalarTypeCode(view),
          mtrt::flat::EnumNameScalarTypeCode(value->getType().getCode()));
  }
  return getOkStatus();
}

// MemRefTableReader encapsulates the logic for reading result MemRef data from
// a Lua table. For now assume MemRef value is encoded as a table.
class MemRefTableReader {
public:
  MemRefTableReader(const sol::protected_function_result &pfr, int resultIndex)
      : mPfr(pfr), mIndex(1) {
    // Assume result is always a memref.
    sol::object obj = mPfr[resultIndex];
    assert(obj.is<sol::table>() && "Expected a table for MemRefValue");
    mMemRefTable = obj.as<sol::table>();
  }

  // Retrieves the next value of type T from the MemRef table
  // This method advances the internal index automatically
  template <typename T>
  T getNextValue() {
    return mMemRefTable.get<T>(mIndex++);
  }

private:
  const sol::protected_function_result &mPfr;
  sol::table mMemRefTable;
  int mIndex;
};

// Extracts a scalar value from the function result
// This handles different integer types represented by ScalarTypeCode
StatusOr<std::unique_ptr<ScalarValue>>
getScalarValue(const sol::protected_function_result &pfr, int index,
               const FunctionSignatureView &sig) {
  assert(sig.getCConv() == CallingConvention::unpacked);
  ScalarTypeCode code = sig.getResult(index).get<ScalarTypeView>();
  switch (code) {
  case ScalarTypeCode::i1:
    return std::make_unique<ScalarValue>(pfr[index].get<int8_t>(), code);
  case ScalarTypeCode::i4:
    return std::make_unique<ScalarValue>(pfr[index].get<int8_t>(), code);
  case ScalarTypeCode::i8:
    return std::make_unique<ScalarValue>(pfr[index].get<int8_t>(), code);
  case ScalarTypeCode::ui8:
    return std::make_unique<ScalarValue>(pfr[index].get<int8_t>(), code);
  case ScalarTypeCode::i16:
    return std::make_unique<ScalarValue>(pfr[index].get<int16_t>(), code);
  case ScalarTypeCode::i32:
    return std::make_unique<ScalarValue>(pfr[index].get<int32_t>(), code);
  case ScalarTypeCode::i64:
    return std::make_unique<ScalarValue>(pfr[index].get<int64_t>(), code);
  case ScalarTypeCode::f8e4m3fn:
    return std::make_unique<ScalarValue>(pfr[index].get<F8E4M3FN>(), code);
  case ScalarTypeCode::f16:
    return std::make_unique<ScalarValue>(pfr[index].get<Float16>(), code);
  case ScalarTypeCode::bf16:
    return std::make_unique<ScalarValue>(pfr[index].get<BFloat16>(), code);
  case ScalarTypeCode::f32:
    return std::make_unique<ScalarValue>(pfr[index].get<float>(), code);
  case ScalarTypeCode::f64:
    return std::make_unique<ScalarValue>(pfr[index].get<double>(), code);
  case ScalarTypeCode::complex32:
    return std::make_unique<ScalarValue>(
        static_cast<float>(static_cast<sol::table>(pfr[index])[1]),
        static_cast<float>(static_cast<sol::table>(pfr[index])[2]), code);
  case ScalarTypeCode::complex64:
    return std::make_unique<ScalarValue>(
        static_cast<double>(static_cast<sol::table>(pfr[index])[1]),
        static_cast<double>(static_cast<sol::table>(pfr[index])[2]), code);
  default:
    return getInvalidArgStatus("Unsupported scalar type code: {0}",
                               mtrt::flat::EnumNameScalarTypeCode(code));
  }
}

/// Parses the results of a function call, handling both scalar and MemRef
/// return types.
static StatusOr<llvm::SmallVector<std::unique_ptr<RuntimeValue>>>
parseResults(const sol::protected_function_result &pfr,
             const FunctionSignatureView &sig, LuaRuntimeSession &session,
             RuntimeClient &client) {
  llvm::SmallVector<std::unique_ptr<RuntimeValue>> results;
  results.reserve(sig.getNumResults());

  for (unsigned i = 0; i < sig.getNumResults(); ++i) {
    const auto &resultType = sig.getResult(i);

    if (resultType.isa<ScalarTypeView>()) {
      auto scalarValue = getScalarValue(pfr, i, sig);
      if (!scalarValue.isOk())
        return scalarValue.getStatus();
      results.push_back(std::move(*scalarValue));
      continue;
    }

    if (!resultType.isa<MemRefTypeView>())
      return getInvalidArgStatus("Result can only be a memref or scalar");

    // Handle MemRef return values
    const auto &memRefView = resultType.get<MemRefTypeView>();
    MemRefTableReader reader(pfr, i);

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

    // Create an external MemRef and track it in both session and client
    // allocation trackers
    MTRT_DBGF(
        "Transferring ownership of returned MemRefValue [ptr %p] to client",
        reinterpret_cast<void *>(allocPtr));

    // We are returning the memref, so transfer ownership to the client.
    // Since ownership is transferred, we can stop tracking `allocPtr` if it
    // exists in `AllocTracker`. Checking existence is important to avoid
    // calling `untrack` multiple times when different views of the same buffer
    // are returned in result.
    if (session.getAllocTracker().contains(allocPtr)) {
      session.getAllocTracker().untrack(allocPtr);
      session.getPinnedMemoryAllocator().untrack(allocPtr);
    }

    // Create a memref so that client now tracks it.
    StatusOr<Ref<MemRefStorage>> storage = client.getAllocator().takeOwnership(
        allocPtr, memRefView.getAddressSpace(), session.getCudaStream());
    if (!storage.isOk())
      return storage.getStatus();

    auto memref = MemRefValue::create(memRefView.getAddressSpace(),
                                      memRefView.getElementType(),
                                      std::move(*storage), offset, shape,
                                      strides, client.getDevices()[0].get());
    if (!memref.isOk())
      return memref.getStatus();

    // Add the memref to the results vector
    results.push_back(std::move(*memref));
  }

  return results;
}

StatusOr<llvm::SmallVector<std::unique_ptr<RuntimeValue>>>
runtime::executeFunctionWithLuaBackend(
    LuaRuntimeSession &session, std::string_view name,
    llvm::ArrayRef<RuntimeValue *> inputArgs,
    llvm::ArrayRef<RuntimeValue *> outputArgs, std::optional<CudaStream> stream,
    std::optional<RuntimeClient *> client) {

  StatusOr<FunctionView> func = session.getExecutable().getFunction(name);
  if (func.isError())
    return func.getStatus();

  FunctionSignatureView sig = (*func).getSignature();

  // Call the main function, if present.
  sol::state_view lua = session.getLuaState();
  AllocTracker &tracker = session.getAllocTracker();
  sol::protected_function funcObj = lua[name];
  if (funcObj.get_type() != sol::type::function)
    return getStatusWithMsg(StatusCode::InternalError, "no function named \"",
                            std::string(name), "\" found");

  // Validate the number of arguments against the signature.
  if (sig.getNumOutputArgs() != outputArgs.size())
    return getInvalidArgStatus(
        "function expects {0} output args (destination args) but received {1}",
        sig.getNumOutputArgs(), outputArgs.size());
  if (sig.getNumInputArgs() != inputArgs.size())
    return getInvalidArgStatus("function expects {0} input args "
                               "(non-destination args) but received {1}",
                               sig.getNumInputArgs(), inputArgs.size());

  // Validate the inferred Lua function type here against the signature.
  for (unsigned i = 0; i < inputArgs.size(); ++i) {
    auto status = validateArgsTypesAgainstFuncArgs(inputArgs[i], sig.getArg(i));
    if (!status.isOk())
      return getInvalidArgStatus(
          "Input argument {0} validation failed against "
          "corresponding function signature arg {0}. Reason: {1}",
          i, status.getString());
  }
  for (unsigned i = 0; i < outputArgs.size(); ++i) {
    auto status =
        validateArgsTypesAgainstFuncArgs(outputArgs[i], sig.getOutputArg(i));
    if (!status.isOk())
      return getInvalidArgStatus(
          "Output argument {0} validation failed against "
          "corresponding function signature arg {1}. Reason: {2}",
          i, i + inputArgs.size(), status.getString());
  }

  // Create the arguments.
  llvm::SmallVector<sol::object> args;
  args.reserve(inputArgs.size() + outputArgs.size());
  for (auto [idx, rv] : llvm::enumerate(inputArgs)) {
    if (MemRefValue *memref = llvm::dyn_cast<MemRefValue>(rv)) {
      MTRT_RETURN_IF_ERROR(pushMemRefTableArg(lua, tracker, args, *memref));
      continue;
    }
    if (ScalarValue *scalar = llvm::dyn_cast<ScalarValue>(rv)) {
      MTRT_RETURN_IF_ERROR(pushScalarArgument(lua, args, *scalar));
      continue;
    }
    return getInvalidArgStatus(
        "input argument #{0} to function {1} has an unsupported type; "
        "arguments must be either MemRefs or scalars",
        idx + 1, name);
  }
  for (auto [idx, rv] : llvm::enumerate(outputArgs)) {
    if (MemRefValue *memref = llvm::dyn_cast<MemRefValue>(rv)) {
      MTRT_RETURN_IF_ERROR(pushMemRefTableArg(lua, tracker, args, *memref));
      continue;
    }
    return getInvalidArgStatus("output (destination) argument #{0} to function "
                               "{1} has an unsupported type; "
                               "destination arguments must be MemRefs",
                               idx + 1, name);
  }

#ifdef MLIR_TRT_ENABLE_CUDA
  int32_t requiredDevice = 0, callerDevice = 0;
  if (session.getOptions().isFeatureEnabled("cuda")) {
    // TODO: This is a temporary hack to ensure that the current device
    // context is set correctly.
    RETURN_ERROR_IF_CUDART_ERROR(cudaGetDevice(&callerDevice));
    requiredDevice = session.getOptions().getDeviceId();
    if (requiredDevice != callerDevice)
      RETURN_ERROR_IF_CUDART_ERROR(cudaSetDevice(requiredDevice));

    // Set stream if provided.
    if (stream)
      RETURN_STATUS_IF_ERROR(session.setCudaStream(*stream));
  }
#endif // MLIR_TRT_ENABLE_CUDA

  // Call the function, passing the arguments either as a table or unpacked as
  // determined by the calling convention.
  sol::protected_function_result pfr =
      sig.getCConv() == CallingConvention::unpacked
          ? funcObj(sol::as_args(args))
          : funcObj(args);

  if (!pfr.valid()) {
    sol::error err(pfr);
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to run function \"", std::string(name),
                            "\": ", err.what());
  }

  // Forget the input pointers.
  for (auto [idx, rv] : llvm::enumerate(inputArgs)) {
    if (MemRefValue *memref = llvm::dyn_cast<MemRefValue>(rv)) {
      auto it = session.getAllocTracker().find(memref->getMemory());
      if (it != session.getAllocTracker().end())
        session.getAllocTracker().erase(it);
    }
  }
  for (auto [idx, rv] : llvm::enumerate(outputArgs)) {
    if (MemRefValue *memref = llvm::dyn_cast<MemRefValue>(rv)) {
      auto it = session.getAllocTracker().find(memref->getMemory());
      if (it != session.getAllocTracker().end())
        session.getAllocTracker().erase(it);
    }
  }

#ifdef MLIR_TRT_ENABLE_CUDA
  if (session.getOptions().isFeatureEnabled("cuda")) {
    if (requiredDevice != callerDevice)
      RETURN_ERROR_IF_CUDART_ERROR(cudaSetDevice(callerDevice));
  }
#endif // MLIR_TRT_ENABLE_CUDA

  if (sig.getNumResults() == 0)
    return llvm::SmallVector<std::unique_ptr<RuntimeValue>>();

  if (!client)
    return getInvalidArgStatus(
        "runtime client cannot be nullptr if results are returned");

  return parseResults(pfr, sig, session, **client);
}
