//===- Status.cpp -----------------------------------------------*- C++ -*-===//
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
/// MLIR-TensorRT runtime status implementation.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt-common-c/Support/Status.h"
#include "mlir-tensorrt-common/Support/Status.h"

//===----------------------------------------------------------------------===//
// MTRT_Status
//===----------------------------------------------------------------------===//

struct MTRT_StatusImpl {
public:
  MTRT_StatusImpl(mtrt::StatusCode code, const char *msg)
      : s(code, msg), msgStorage(s.getString()) {}
  const char *getMessage() { return msgStorage.c_str(); }

private:
  mtrt::Status s;
  std::string msgStorage;
};

static mtrt::StatusCode getStatusCodeFromMTRTStatusCode(MTRT_StatusCode code) {
  return static_cast<mtrt::StatusCode>(code);
}

MTRT_Status mtrtStatusCreate(MTRT_StatusCode code, const char *msg) {
  return MTRT_Status{std::make_unique<MTRT_StatusImpl>(
                         getStatusCodeFromMTRTStatusCode(code), msg)
                         .release()};
}

void mtrtStatusGetMessage(MTRT_Status error, const char **dest) {
  auto errorImpl = reinterpret_cast<MTRT_StatusImpl *>(error.ptr);
  *dest = errorImpl->getMessage();
}

void mtrtStatusDestroy(MTRT_Status error) {
  if (error.ptr)
    delete reinterpret_cast<MTRT_StatusImpl *>(error.ptr);
}
