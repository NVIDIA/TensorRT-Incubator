//===- Registration.cpp =--- ----------------------------------------------===//
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
/// Implementation for registration methods. This file is currently just a
/// stub. It helps clangd correctly parse the registration headers and forces
/// compilation of the headers.
///
//===----------------------------------------------------------------------===//
#include "mlir-tensorrt/Registration/RegisterMlirTensorRtDialects.h"
#include "mlir-tensorrt/Registration/RegisterMlirTensorRtPasses.h"
#include "mlir-tensorrt/Registration/RegisterMlirTensorRtTranslations.h"
