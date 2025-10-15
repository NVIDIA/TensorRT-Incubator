//===- MemRefABI.cpp ------------------------------------------------------===//
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
#include "mlir-executor/Runtime/API/MemRefABI.h"
#include "mlir-executor/Runtime/API/API.h"
#include "mlir-tensorrt-common/Support/Status.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mtrt;

//===----------------------------------------------------------------------===//
// MemRefDescriptor
//===----------------------------------------------------------------------===//

llvm::raw_ostream &mtrt::operator<<(llvm::raw_ostream &os,
                                    const MemRefDescriptorView &desc) {
  os << llvm::formatv("MemRefDescriptorView {{ rank = {0}, basePtr = {1:x}, "
                      "data = {2:x}, offset = {3}",
                      desc.rank, desc.basePtr, desc.data, desc.offset);

  if (desc.rank > 0) {
    llvm::ArrayRef<int64_t> shape(desc.shape, desc.rank);
    llvm::ArrayRef<int64_t> strides(desc.strides, desc.rank);
    os << llvm::formatv(", shape = [{0:$[, ]}], strides = [{1:$[, ]}]",
                        llvm::make_range(shape.begin(), shape.end()),
                        llvm::make_range(strides.begin(), strides.end()));
  }

  os << " }";
  return os;
}

llvm::raw_ostream &mtrt::operator<<(llvm::raw_ostream &os,
                                    const UnrankedMemRefDescriptor &desc) {
  os << llvm::formatv("UnrankedMemRefDescriptor {{ rank = {0}, "
                      "rankedDescriptorPtr = {1:x}",
                      desc.rank, desc.rankedDescriptorPtr);

  // Try to get detailed information about the descriptor
  StatusOr<MemRefDescriptorView> info = mtrt::getMemRefDescriptorInfo(desc);
  if (info.isOk()) {
    os << ", view = " << *info;
  } else {
    os << llvm::formatv(", error = \"{0}\"", info.getStatus().getString());
  }

  os << " }";
  return os;
}

template <unsigned... Ranks>
Status
dispatchPopulateMemRefDescriptor(UnrankedMemRefDescriptor desc,
                                 const MemRefValue &memref,
                                 std::integer_sequence<unsigned, Ranks...>) {
  bool success = false;
  ((desc.rank == Ranks ? (populateMemRefDescriptor<Ranks>(
                              reinterpret_cast<MemRefDescriptor<Ranks> *>(
                                  desc.rankedDescriptorPtr),
                              memref),
                          success = true)
                       : false) ||
   ...);
  return success
             ? getOkStatus()
             : getInvalidArgStatus("unsupported memref rank {0}", desc.rank);
}

Status mtrt::populateMemRefDescriptor(UnrankedMemRefDescriptor desc,
                                      const MemRefValue &memref) {
  if (desc.rank != memref.getRank())
    return getInvalidArgStatus(
        "descriptor rank {0} does not match memref rank {1}", desc.rank,
        memref.getRank());

  return dispatchPopulateMemRefDescriptor(
      desc, memref, std::make_integer_sequence<unsigned, 17>());
}

template <unsigned Rank>
MemRefDescriptorView getMemRefDescriptorInfoImpl(MemRefDescriptor<Rank> *desc) {
  if constexpr (Rank == 0) {
    return MemRefDescriptorView{0,       desc->ptr, desc->aligned, desc->offset,
                                nullptr, nullptr};
  } else {
    return MemRefDescriptorView{Rank,         desc->ptr,   desc->aligned,
                                desc->offset, desc->shape, desc->strides};
  }
}

template <unsigned... Ranks>
StatusOr<MemRefDescriptorView>
dispatchGetMemRefDescriptorInfo(UnrankedMemRefDescriptor desc,
                                std::integer_sequence<unsigned, Ranks...>) {
  StatusOr<MemRefDescriptorView> result =
      getInvalidArgStatus("unsupported memref rank {0}", desc.rank);
  (void)((desc.rank == Ranks ? (result = getMemRefDescriptorInfoImpl<Ranks>(
                                    reinterpret_cast<MemRefDescriptor<Ranks> *>(
                                        desc.rankedDescriptorPtr)),
                                true)
                             : false) ||
         ...);
  return result;
}

StatusOr<MemRefDescriptorView>
mtrt::getMemRefDescriptorInfo(UnrankedMemRefDescriptor desc) {
  return dispatchGetMemRefDescriptorInfo(
      desc, std::make_integer_sequence<unsigned, 17>());
}
