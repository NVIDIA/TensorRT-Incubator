//===- LuaRuntime.cpp ------ ----------------------------------------------===//
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
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-executor/Runtime/API/ExecutableFlatbuffer.h"
#include "mlir-executor/Runtime/Backend/Common/CommonRuntime.h"
#include "mlir-executor/Runtime/Backend/Lua/LuaRegistration.h"
#include "mlir-executor/Runtime/Backend/Lua/Modules/CUDA/CudaModule.h"
#include "mlir-executor/Runtime/Backend/Lua/Modules/Core/CoreModule.h"
#include "mlir-executor/Runtime/Backend/Lua/Modules/CuBLAS/CuBLASModule.h"
#include "mlir-executor/Runtime/Backend/Lua/Modules/NCCL/NcclModule.h"
#include "mlir-executor/Runtime/Backend/Lua/Modules/TensorRT/TensorRTModule.h"
#include "mlir-executor/Runtime/Backend/Lua/Modules/Utils/MemRefUtils.h"
#include "mlir-executor/Runtime/Backend/Utils/NvtxUtils.h"
#include "mlir-executor/Support/Allocators.h"
#include "mlir-executor/Support/Status.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include "cuda_fp16.h"
#include "cuda_fp8.h"
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

using namespace mlirtrt;
using namespace mlirtrt::runtime;

static void registerLuaRuntimeMethodsCommon(
    lua_State *state, PinnedMemoryAllocator *pinnedMemoryAllocator,
    AllocTracker *allocTracker, ResourceTracker *resourceTracker) {
  registerExecutorCoreModuleLuaRuntimeMethods(state, pinnedMemoryAllocator,
                                              allocTracker);
  registerExecutorCUDAModuleLuaRuntimeMethods(
      state, allocTracker, pinnedMemoryAllocator, resourceTracker);
  registerExecutorCuBLASModuleLuaRuntimeMethods(state, allocTracker,
                                                resourceTracker);
  registerExecutorTensorRTModuleLuaRuntimeMethods(
      state, pinnedMemoryAllocator, allocTracker, resourceTracker);
}

void mlirtrt::runtime::registerLuaRuntimeMethods(
    lua_State *state, const RuntimeSessionOptions &options,
    PinnedMemoryAllocator *pinnedMemoryAllocator, AllocTracker *allocTracker,
    ResourceTracker *resourceTracker) {
  registerLuaRuntimeMethodsCommon(state, pinnedMemoryAllocator, allocTracker,
                                  resourceTracker);
#ifdef MLIR_TRT_ENABLE_NCCL
  registerExecutorNCCLModuleLuaRuntimeMethods(state, resourceTracker);
  registerDeviceDependentNCCLMethods(state, options.getNumDevices(),
                                     options.getDeviceId(),
                                     options.getNcclUuid());
#else
  // MpiCommSizeOp/MpiCommRankOp are used to get the device count and id. When
  // not building with NCCL, always use device 0.
  registerDeviceDependentNCCLMethods(state, /*numDevices=*/1, /*deviceIdx=*/0,
                                     "");
#endif
}

StatusOr<int64_t>
mlirtrt::runtime::runExecutorLuaScript(std::string_view luaScript) {
  ADD_RUNTIME_MODULE_RANGE("runtime_runExecutorLuaScript");

  StatusOr<std::unique_ptr<RuntimeClient>> client = RuntimeClient::create();
  if (!client.isOk())
    return client.getStatus();

  sol::state lua;
  lua.open_libraries(sol::lib::base, sol::lib::string);
  registerLuaRuntimeMethods(lua.lua_state(), RuntimeSessionOptions(),
                            &(*client)->getPinnedMemorAllocator(),
                            &(*client)->getAllocTracker(),
                            &(*client)->getResourceTracker());

  sol::protected_function_result result = lua.script(luaScript);
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
  int returnCode = result;
  return returnCode;
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

/// Create an execution state. This will setup a Lua environment and invoke
/// global initialization.
StatusOr<std::unique_ptr<RuntimeSession>>
mlirtrt::runtime::createRuntimeSessionWithLuaBackend(
    ExecutableView executable, const RuntimeSessionOptions &options) {
  ADD_RUNTIME_MODULE_RANGE("runtime_loadExecutable");

  MTRT_RETURN_IF_ERROR(maybeCheckForValidNcclUuid(options));

  auto pinnedMemoryAllocator = std::make_unique<PinnedMemoryAllocator>();
  auto allocTracker = std::make_unique<AllocTracker>();
  auto resourceTracker = std::make_unique<ResourceTracker>();

  sol::state lua;
  lua.open_libraries(sol::lib::base, sol::lib::string);
  registerLuaRuntimeMethods(lua.lua_state(), options,
                            pinnedMemoryAllocator.get(), allocTracker.get(),
                            resourceTracker.get());

  // Load globals into the context.
  // TODO: eliminate this copy, we already own the executable.
  MTRT_DBGF("loading %lu constants", executable.getConstants().size());
  for (ConstantView constant : executable.getConstants()) {
    size_t bytes = constant.size();

    MTRT_ASSIGN_OR_RETURN(StatusOr<PointerInfo> buffer,
                          mlirtrt::runtime::allocate(*allocTracker,
                                                     PointerType::host, bytes,
                                                     alignof(char), {}));
    std::memcpy(reinterpret_cast<void *>(buffer->ptr),
                reinterpret_cast<const void *>(constant.data()), bytes);
    lua[constant.getName()] = buffer->ptr;
  }
  // Load the main Lua script.
  sol::protected_function_result result =
      lua.script(executable.getCode(), sol::script_pass_on_error);
  if (!result.valid()) {
    sol::error err = result;
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to load lua script: ", err.what());
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
  return std::make_unique<RuntimeSession>(
      options, executable, std::move(lua), std::move(pinnedMemoryAllocator),
      std::move(allocTracker), std::move(resourceTracker));
}

StatusOr<int64_t> mlirtrt::runtime::runExecutorExecutable(
    std::unique_ptr<Executable> executable) {

  StatusOr<std::unique_ptr<RuntimeClient>> client = RuntimeClient::create();
  if (!client.isOk())
    return client.getStatus();

#ifdef MLIR_TRT_ENABLE_NCCL
  StatusOr<RuntimeSessionOptions> options =
      RuntimeSessionOptions::createUsingSingleHostMpi();
#else
  StatusOr<RuntimeSessionOptions> options = RuntimeSessionOptions();
#endif
  if (!options.isOk())
    return options.getStatus();

  StatusOr<std::unique_ptr<RuntimeSession>> session =
      createRuntimeSessionWithLuaBackend(executable->getView(), *options);
  if (!session.isOk())
    return session.getStatus();

  // Call the main function, if present.
  sol::state_view lua((*session)->getLuaState());
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
  return result.get<int64_t>();
}

/// Set the primary stream for the loaded executable to use.
Status mlirtrt::runtime::setRuntimeSessionCudaStream(RuntimeSession &session,
                                                     cudaStream_t stream) {
  sol::state_view state(session.getLuaState());
  state["stream0"] = CudaStreamPtr(stream);
  return getOkStatus();
}

/// Get the primary stream for the loaded executable to use.
cudaStream_t
mlirtrt::runtime::getRuntimeSessionCudaStream(RuntimeSession &session) {
  sol::state_view state(session.getLuaState());
  auto stream = state["stream0"].get<CudaStreamPtr>();
  return stream;
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
           value.getVoidPtr(), fmtRange(value.getShape()),
           fmtRange(value.getStrides()), value.getElementBitWidth(),
           value.getTotalFootprintInBytes());

  std::vector<sol::object> memrefTable;
  memrefTable.reserve(3 + 2 * value.getRank());
  llvm::append_range(memrefTable, llvm::ArrayRef<sol::object>{
                                      sol::make_object(lua, ptr),
                                      sol::make_object(lua, ptr),
                                      sol::make_object(lua, value.getOffset()),
                                  });

  // Push shape/strides.
  for (int64_t dim : value.getShape())
    memrefTable.push_back(sol::make_object(lua, dim));
  for (int64_t dim : value.getStrides())
    memrefTable.push_back(sol::make_object(lua, dim));

  args.emplace_back(sol::make_object(lua, std::move(memrefTable)));

  PointerInfo pointerInfo = value.getPointerInfo(PointerOwner::external);
  tracker.track(pointerInfo);

  return getOkStatus();
}

static Status pushScalarArgument(sol::state_view &lua,
                                 llvm::SmallVector<sol::object> &args,
                                 const ScalarValue &value) {
  ScalarType type = value.getType();
  sol::object obj(nullptr);
  switch (type.getCode()) {
  case ScalarTypeCode::f8e4m3fn:
    obj = sol::make_object(lua, value.get<__nv_fp8_e4m3>());
    break;
  case ScalarTypeCode::f16:
    obj = sol::make_object(lua, value.get<__half>());
    break;
  case ScalarTypeCode::bf16:
    obj = sol::make_object(lua, value.get<nv_bfloat16>());
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
        impl::EnumNameScalarTypeCode(type.getCode()));
  }
  args.push_back(obj);
  return getOkStatus();
}

static Status valiateArgsTypesAgainstFuncArgs(const RuntimeValue *runArg,
                                              const TypeUnionView &sigArg) {
  if (sigArg.isa<MemrefTypeView>()) {
    if (runArg->getKind() != RuntimeValue::Kind::MemRef)
      return getInvalidArgStatus(
          "function expects a memref type but received scalar type");
    auto view = sigArg.get<MemrefTypeView>();
    auto value = static_cast<const MemRefValue *>(runArg);

    if (view.getElementType() != *value->getScalarType())
      return getInvalidArgStatus(
          "function expects a memref type with element type {0} but "
          "receieved {1}",
          view.getElementType().getStrRef(),
          value->getScalarType()->getStrRef());

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
              mlirtrt::fmtRange(value->getShape()));
        if (view.getShape()[i] >= 0 &&
            view.getShape()[i] != value->getShape()[i])
          return getInvalidArgStatus(
              "Runtime shape mismatch. Expected [{0:$[, ]}] "
              "but received [{1:$[, ]}]",
              mlirtrt::fmtRange(view.getShape()),
              mlirtrt::fmtRange(value->getShape()));
      }
    }

    if (view.getStrides() != value->getStrides()) {
      for (unsigned i = 0; i < view.getStrides().size(); ++i) {
        if (value->getStrides()[i] < 0)
          return getInvalidArgStatus(
              "all strides must be non-negative but received shape [{0:$[, ]}]",
              mlirtrt::fmtRange(value->getStrides()));
        if (view.getStrides()[i] >= 0 &&
            view.getStrides()[i] != value->getStrides()[i])
          return getInvalidArgStatus(
              "Runtime stride mismatch. Expected [{0:$[, ]}] "
              "but received [{1:$[, ]}]",
              mlirtrt::fmtRange(view.getStrides()),
              mlirtrt::fmtRange(value->getStrides()));
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
          impl::EnumNameScalarTypeCode(view),
          impl::EnumNameScalarTypeCode(value->getType().getCode()));
  }
  return getOkStatus();
}

StatusOr<llvm::SmallVector<std::unique_ptr<RuntimeValue>>>
runtime::executeFunctionWithLuaBackend(
    RuntimeSession &session, std::string_view name,
    llvm::ArrayRef<RuntimeValue *> inputArgs,
    llvm::ArrayRef<RuntimeValue *> outputArgs,
    std::optional<cudaStream_t> stream) {

  FunctionView meta = session.getExecutable().getFunction(name);
  FunctionSignatureView sig = meta.getSignature();

  // Call the main function, if present.
  sol::state_view lua(session.getLuaState());
  AllocTracker &tracker = session.getAllocTracker();
  sol::protected_function funcObj = lua[name];
  if (funcObj.get_type() != sol::type::function)
    return getStatusWithMsg(StatusCode::InternalError, "no function named \"",
                            std::string(name), "\" found");

  if (sig.getNumResults() > 0)
    return getInvalidArgStatus("functions with {0} results are not supported",
                               sig.getNumResults());

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
    auto status = valiateArgsTypesAgainstFuncArgs(inputArgs[i], sig.getArg(i));
    if (!status.isOk())
      return getInvalidArgStatus(
          "Input argument {0} validation failed against "
          "corresponding function signature arg {0}. Reason: {1}",
          i, status.getString());
  }
  for (unsigned i = 0; i < outputArgs.size(); ++i) {
    auto status =
        valiateArgsTypesAgainstFuncArgs(outputArgs[i], sig.getOutputArg(i));
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

  if (stream)
    RETURN_STATUS_IF_ERROR(setRuntimeSessionCudaStream(session, *stream));

  // If the number of arguments exceed a particular threshold, then
  // we pass arguments packed into a table, otherwise we pass as arguments.
  sol::protected_function_result result =
      sig.getCConv() == CallingConvention::unpacked
          ? funcObj(sol::as_args(args))
          : funcObj(args);

  if (!result.valid()) {
    sol::error err(result);
    return getStatusWithMsg(StatusCode::InternalError,
                            "failed to run function \"", std::string(name),
                            "\": ", err.what());
  }

  return llvm::SmallVector<std::unique_ptr<RuntimeValue>>{};
}
