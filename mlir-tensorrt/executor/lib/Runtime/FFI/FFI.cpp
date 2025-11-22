//===- FFI.cpp ------------------------------------------------------------===//
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
/// Implementation of the FFI host API, including registration of plugins,
/// enumeration of plugins and dispatch utilities.
///
//===----------------------------------------------------------------------===//
#include "mlir-executor/Runtime/FFI/FFI.h"
#include "dlpack/dlpack.h"
#include "mlir-executor/Runtime/Support/Support.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "tvm/ffi/c_api.h"
#include "tvm/ffi/extra/c_env_api.h"
#include "tvm/ffi/extra/module.h"
#include "tvm/ffi/function.h"
#include "tvm/ffi/object.h"
#include <new>
#include <string>
#include <vector>

using namespace mtrt;

//===----------------------------------------------------------------------===//
// TVMFFI-related utilities
//===----------------------------------------------------------------------===//

namespace mtrt {
struct TVMFFILibraryHandle {
  std::string name;
  tvm::ffi::Module module;
};

struct TVMFFICallableHandle {
  tvm::ffi::Function function;
};

} // namespace mtrt

Status mtrt::invokeTVMFFICallable(TVMFFICallableHandle *callable,
                                  uintptr_t stream, uintptr_t argsArrayPtr,
                                  int64_t numArgs) noexcept {
  try {
    MTRT_DBG(
        "invokeTVMFFICallable: callable={0} stream={1:x} argsArrayPtr={2:x} "
        "args.size={3}",
        callable, stream, argsArrayPtr, numArgs);
    auto *func = reinterpret_cast<const tvm::ffi::FunctionObj *>(
        callable->function.get());

    const auto *args = reinterpret_cast<const TVMFFIAny *>(argsArrayPtr);

    // Technically only certain devices might need to have the stream set, but
    // it can't hurt to populate it always every time we call. It is a bit
    // wasteful, but these arrays are small.
    for (int64_t i = 0; i < numArgs; ++i) {
      if (args[i].type_index != kTVMFFIDLTensorPtr)
        continue;

      const DLTensor *tensor =
          reinterpret_cast<const DLTensor *>(args[i].v_ptr);
      void *prevStream{nullptr};
      if (int errc = TVMFFIEnvSetStream(
              tensor->device.device_type, tensor->device.device_id,
              reinterpret_cast<TVMFFIStreamHandle>(stream), &prevStream);
          errc != 0) {
        tvm::ffi::Error err = tvm::ffi::details::MoveFromSafeCallRaised();
        return getInternalErrorStatus(
            "TVMFFIEnvSetStream failed; error code: {0}", err.what());
      }
    }

    tvm::ffi::Any result;
    if (int errc =
            func->safe_call(const_cast<tvm::ffi::FunctionObj *>(func), args,
                            numArgs, reinterpret_cast<TVMFFIAny *>(&result));
        errc != 0) {
      tvm::ffi::Error err = tvm::ffi::details::MoveFromSafeCallRaised();
      return getInternalErrorStatus("failed to invoke TVM-FFI callable: {0}",
                                    err.what());
    }

    return getOkStatus();
  } catch (const std::exception &e) {
    return getInternalErrorStatus("failed to invoke TVM-FFI callable: {0}",
                                  e.what());
  }
  return getOkStatus();
}

//===----------------------------------------------------------------------===//
// PluginRegistry
//===----------------------------------------------------------------------===//

PluginRegistry::PluginRegistry() {}
PluginRegistry::~PluginRegistry() {}

StatusOr<TVMFFILibraryHandle *>
PluginRegistry::loadTVMFFILibrary(const std::string &path) {
  try {
    for (const auto &lib : tvmLibRefs) {
      if (lib->name == path)
        return lib.get();
    }
    tvm::ffi::Module module = tvm::ffi::Module::LoadFromFile(path);
    tvmLibRefs.emplace_back(new TVMFFILibraryHandle{path, std::move(module)});
    return tvmLibRefs.back().get();
  } catch (const tvm::ffi::Error &e) {
    return getInternalErrorStatus(
        "failed to load plugin library using TVM-FFI: {0}", e.what());
  }
}

StatusOr<TVMFFICallableHandle *>
PluginRegistry::createTVMFFICallable(const std::string &libName,
                                     const std::string &funcName) {
  try {
    MTRT_ASSIGN_OR_RETURN(TVMFFILibraryHandle * lib,
                          loadTVMFFILibrary(libName));
    tvm::ffi::Optional<tvm::ffi::Function> func =
        lib->module->GetFunction(funcName);
    if (!func.defined()) {
      return getInvalidArgStatus("function {0} not found in library {1}",
                                 funcName, libName);
    }
    tvmFuncRefs.emplace_back(new TVMFFICallableHandle{func.value()});
    return tvmFuncRefs.back().get();
  } catch (const tvm::ffi::Error &e) {
    return getInternalErrorStatus("failed to create TVM-FFI callable: {0}",
                                  e.what());
  }
}
