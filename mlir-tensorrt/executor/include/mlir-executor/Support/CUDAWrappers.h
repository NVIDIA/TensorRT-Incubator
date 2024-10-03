//===- CUDAWrappers.h -------------------------------------------*- C++ -*-===//
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
/// RAII Wrappers for CUDA C API Objects
///
//===----------------------------------------------------------------------===//
#ifndef MLIR_TENSORRT_SUPPORT_CUDAWRAPPERS_H
#define MLIR_TENSORRT_SUPPORT_CUDAWRAPPERS_H
#ifdef MLIR_EXECUTOR_ENABLE_CUDA

#include "cuda_runtime_api.h"
#include "mlir-executor/Support/Status.h"
#include <functional>
#include <memory>

namespace mlirtrt {

/// Wraps a CUDA C API object (`ObjType`, e.g. `cudaEvent_t`). The `ObjType` is
/// usually an aliased pointer (e.g. `cudaEvent_t` isa `CUevent_st *`). The
/// `Derived` CRTP class implements the `create` and `destroy` methods.
template <typename Derived, typename ObjType>
class CUDAObjectWrapper {
public:
  using ObjDeleter = std::function<void(ObjType *)>;

  /// Default construct the null object.
  CUDAObjectWrapper()
      : obj(nullptr, [](ObjType *o) {
          if (o)
            Derived::destroy(*o);
          delete o;
        }) {}

  /// Create the object or return failure status.
  static StatusOr<Derived> get() {
    ObjType *obj = new ObjType;
    Status result = Derived::create(obj);
    if (!result.isOk())
      return result;
    return Derived(std::unique_ptr<ObjType, ObjDeleter>{obj, [](ObjType *o) {
                                                          if (o)
                                                            Derived::destroy(
                                                                *o);
                                                          delete o;
                                                        }});
  }

  /// Move-construct the wrapped object.
  CUDAObjectWrapper(CUDAObjectWrapper<Derived, ObjType> &&other) = default;

  // Copy is disallowed.
  CUDAObjectWrapper(const CUDAObjectWrapper<Derived, ObjType> &other) = delete;
  CUDAObjectWrapper
  operator=(const CUDAObjectWrapper<Derived, ObjType> &other) = delete;

  /// Return the underlying object pointer.
  ObjType value() const { return obj ? *obj : nullptr; }

  /// Implicit cast to the underlying object poitner.
  operator ObjType() { return obj ? *obj : nullptr; }

private:
  /// Create the object from a pointer.
  explicit CUDAObjectWrapper(std::unique_ptr<ObjType, ObjDeleter> ptr)
      : obj(std::move(ptr)) {}

protected:
  std::unique_ptr<ObjType, std::function<void(ObjType *)>> obj;
};

/// RAII wrapper around `cudaEvent_t.
class CUDAEvent : public CUDAObjectWrapper<CUDAEvent, cudaEvent_t> {
public:
  /// Construct a default (nullptr) event using `CUDAEvent()`.
  using CUDAObjectWrapper<CUDAEvent, cudaEvent_t>::CUDAObjectWrapper;

  /// Construct a new event using `CUDAEvent::get`.
  using CUDAObjectWrapper<CUDAEvent, cudaEvent_t>::get;

  /// Creates CUDA at `event` or returns error status.
  static Status create(cudaEvent_t *event) {
    RETURN_ERROR_IF_CUDART_ERROR(cudaEventCreate(event));
    return getOkStatus();
  }

  /// Destroys the CUDA event.
  static void destroy(cudaEvent_t event) { cudaEventDestroy(event); }
};

} // namespace mlirtrt

#endif // MLIR_EXECUTOR_ENABLE_CUDA
#endif // MLIR_TENSORRT_SUPPORT_CUDAWRAPPERS_H
