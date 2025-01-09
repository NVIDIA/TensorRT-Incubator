//===- CoreModule.cpp -----------------------------------------------------===//
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
/// Core C runtime function definitions
///
//===----------------------------------------------------------------------===//
#include "./CoreModule.h"
#include "llvm/ADT/ArrayRef.h"

void __memset_32(uintptr_t pointer, size_t offset, size_t numBytes,
                 uint32_t fillInt) {
  llvm::MutableArrayRef<uint32_t> buffer(reinterpret_cast<uint32_t *>(pointer),
                                         numBytes / sizeof(fillInt));
  std::fill(buffer.begin(), buffer.end(), fillInt);
}

void __memset_16(uintptr_t pointer, size_t offset, size_t numBytes,
                 uint16_t fillInt) {
  llvm::MutableArrayRef<uint16_t> buffer(reinterpret_cast<uint16_t *>(pointer),
                                         numBytes / sizeof(fillInt));
  std::fill(buffer.begin(), buffer.end(), fillInt);
}

void __memset_8(uintptr_t pointer, size_t offset, size_t numBytes,
                uint8_t fillInt) {
  llvm::MutableArrayRef<uint8_t> buffer(reinterpret_cast<uint8_t *>(pointer),
                                        numBytes / sizeof(fillInt));
  std::fill(buffer.begin(), buffer.end(), fillInt);
}
