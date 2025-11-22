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
#include "LuaRuntimeCompat.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/API/ExecutableFlatbuffer.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaExtensionRegistry.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaExtensions.h"
#include "mlir-executor/Runtime/Backend/Utils/NvtxUtils.h"
#include "mlir-executor/Runtime/Support/Support.h"
#include "mlir-executor/Support/Allocators.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Alignment.h"
#include <memory>

#ifdef MLIR_TRT_ENABLE_CUDA
#include "cuda_runtime_api.h"
#endif // MLIR_TRT_ENABLE_CUDA

using namespace mtrt;
using namespace mtrt;

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

namespace mtrt {
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
} // namespace mtrt

void mtrt::registerLuaRuntimeExtensions() {
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
      LuaRuntimeExtension{[](const LuaRuntimeExtensionInitArgs &args) {
        registerDefaultDeviceDependentMethods(args.state,
                                              args.options.getNumDevices(),
                                              args.options.getDeviceId());
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
    MTRT_ASSIGN_OR_RETURN(StatusOr<PointerInfo> buffer,
                          mtrt::allocate(session->getAllocTracker(),
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
    MTRT_ASSIGN_OR_RETURN(StatusOr<PointerInfo> buffer,
                          mtrt::allocate(session->getAllocTracker(),
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

  MTRT_ASSIGN_OR_RETURN(StatusOr<PointerInfo> buffer,
                        mtrt::allocate(session->getAllocTracker(),
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

LuaRuntimeSession::LuaRuntimeSession(Ref<RuntimeClient> client,
                                     RuntimeSessionOptions options,
                                     ExecutableView executable)
    : RuntimeSession(std::move(options), std::move(executable),
                     std::move(client)),
      impl(std::unique_ptr<Impl>(new Impl())) {}

LuaRuntimeSession::~LuaRuntimeSession() = default;

lua_State *LuaRuntimeSession::getLuaState() { return impl->getLuaState(); }

StatusOr<std::unique_ptr<LuaRuntimeSession>>
LuaRuntimeSession::create(Ref<RuntimeClient> client_,
                          RuntimeSessionOptions options,
                          ExecutableView executable,
                          LuaModuleRegistrationFunc registerExtraLuaFuncs) {
  MTRT_RETURN_IF_ERROR(maybeCheckForValidNcclUuid(options));

  auto session = std::unique_ptr<LuaRuntimeSession>(new LuaRuntimeSession(
      std::move(client_), std::move(options), std::move(executable)));
  sol::state_view lua = session->getLuaState();
  lua.open_libraries(sol::lib::base, sol::lib::string, sol::lib::coroutine);

  // Register builtin methods.
  MTRT_RETURN_IF_ERROR(populateRuntimeExtensions(LuaRuntimeExtensionInitArgs{
      session->getOptions(), lua.lua_state(),
      &session->getPinnedMemoryAllocator(), &session->getAllocTracker(),
      &session->getResourceTracker(),
      session->getClient()->getPluginRegistry()}));

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

/// Set the primary stream for the loaded executable to use.
Status LuaRuntimeSession::onStreamChanged(Ref<Stream> oldStream,
                                          Ref<Stream> newStream) {
  sol::state_view lua = getLuaState();
  lua["stream0"] = newStream->getCUDAHandle();
  return getOkStatus();
}

//===----------------------------------------------------------------------===//
// Convenience Functions
//===----------------------------------------------------------------------===//

StatusOr<int64_t> mtrt::runExecutorLuaScript(
    RuntimeSessionOptions options, std::string_view luaScript,
    LuaRuntimeSession::LuaModuleRegistrationFunc registerExtraLuaFuncs) {
  ADD_RUNTIME_MODULE_RANGE("runtime_runExecutorLuaScript");

  MTRT_ASSIGN_OR_RETURN(Ref<RuntimeClient> client, RuntimeClient::create());

  MTRT_ASSIGN_OR_RETURN(std::unique_ptr<LuaRuntimeSession> session,
                        LuaRuntimeSession::create(
                            client, std::move(options), ExecutableView(nullptr),
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

static Status validateMainFunctionSignature(FunctionView mainFunc) {
  auto returnErr = [&]() -> Status {
    return getInvalidArgStatus("unsupported function signature {0}", mainFunc);
  };
  if (mainFunc.getSignature().getNumResults() > 1 ||
      mainFunc.getSignature().getNumInputArgs() != 0)
    return returnErr();
  if (mainFunc.getSignature().getNumResults() == 1) {
    if (!mainFunc.getSignature().getResult(0).isa<ScalarTypeView>())
      return returnErr();
    auto resultType =
        mainFunc.getSignature().getResult(0).get<ScalarTypeView>();
    if (resultType != ScalarTypeCode::i64 && resultType != ScalarTypeCode::i32)
      return returnErr();
    return getOkStatus();
  }
  return getOkStatus();
}

StatusOr<int64_t> mtrt::runExecutorExecutable(
    RuntimeSessionOptions options, std::unique_ptr<Executable> executable,
    LuaRuntimeSession::LuaModuleRegistrationFunc registerExtraLuaFuncs) {

  MTRT_ASSIGN_OR_RETURN(Ref<RuntimeClient> client, RuntimeClient::create());

  MTRT_ASSIGN_OR_RETURN(std::unique_ptr<LuaRuntimeSession> session,
                        LuaRuntimeSession::create(
                            client, std::move(options), executable->getView(),
                            std::move(registerExtraLuaFuncs)));

  MTRT_ASSIGN_OR_RETURN(FunctionView mainFunc, executable->getFunction("main"));

  MTRT_RETURN_IF_ERROR(validateMainFunctionSignature(mainFunc));

  MTRT_ASSIGN_OR_RETURN(
      llvm::SmallVector<std::unique_ptr<RuntimeValue>> resultValues,
      session->executeFunction(mainFunc.getName(), {}, {}));

  if (resultValues.empty())
    return 0;

  ScalarValue *scalar = llvm::cast<ScalarValue>(resultValues.front().get());
  if (scalar->getType() == ScalarTypeCode::i64)
    return scalar->get<int64_t>();
  if (scalar->getType() == ScalarTypeCode::i32)
    return scalar->get<int32_t>();
  // Conditions on the return type are checked in validateMainFunctionSignature.
  llvm_unreachable("unsupported return type");
}

StatusOr<llvm::SmallVector<std::unique_ptr<RuntimeValue>>>
LuaRuntimeSession::executeFunction(llvm::StringRef name,
                                   llvm::ArrayRef<RuntimeValue *> inputs,
                                   llvm::ArrayRef<RuntimeValue *> outArgs) {
  MTRT_ASSIGN_OR_RETURN(FunctionView func,
                        this->getExecutable().getFunction(name));
  uint32_t abiVersion = func.getAbiVersion();
  MTRT_ASSIGN_OR_RETURN(
      llvm::SmallVector<std::unique_ptr<RuntimeValue>> resultValues,
      mtrt::invokeLuaFunction(*this, func, inputs, outArgs, abiVersion));
  return resultValues;
}
