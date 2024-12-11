# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tensorrt10",
    srcs = [
        # This dep throws an error:
        # bazel-bin/mlir-tensorrt-opt: error while loading shared libraries: do_not_link_against_nvinfer_builder_resource: cannot open shared object file: No such file or directory
        # "lib/libnvinfer_builder_resource.so.10.2.0",
        "lib/libnvinfer.so.10.2.0",
        "lib/libnvinfer_plugin.so.10.2.0",
        "lib/libnvinfer_dispatch.so.10.2.0",
        "lib/libnvinfer_lean.so.10.2.0",
        "lib/libnvinfer_vc_plugin.so.10.2.0",
        "lib/libnvonnxparser.so.10.2.0",
    ],
    hdrs = [
        "include/NvInfer.h",
        "include/NvInferConsistency.h",
        "include/NvInferConsistencyImpl.h",
        "include/NvInferImpl.h",
        "include/NvInferLegacyDims.h",
        "include/NvInferPlugin.h",
        "include/NvInferPluginUtils.h",
        "include/NvInferRuntime.h",
        "include/NvInferRuntimeBase.h",
        "include/NvInferRuntimeCommon.h",
        "include/NvInferRuntimePlugin.h",
        "include/NvInferSafeRuntime.h",
        "include/NvInferVersion.h",
    ],
    includes = ["include"],
)
