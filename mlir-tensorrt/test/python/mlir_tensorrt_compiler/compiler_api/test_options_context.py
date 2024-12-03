# REQUIRES: host-has-at-least-1-gpus
# RUN: %PYTHON %s 2>&1 | FileCheck %s

import mlir_tensorrt.compiler.api as api
from mlir_tensorrt.compiler.ir import *


with Context() as context:
    client = api.CompilerClient(context)
    # Try to create a non-existent option type
    try:
        opts = api.OptionsContext(client, "non-existent-options-type", [])
    except Exception as err:
        print(err)

    opts = api.OptionsContext(
        client,
        "stable-hlo-to-executable",
        [
            "--tensorrt-builder-opt-level=3",
            "--tensorrt-strongly-typed=false",
            "--tensorrt-workspace-memory-pool-limit=1gb",
        ],
    )

    print(opts)


# CHECK: InvalidArgument: InvalidArgument: non-existent-options-type is not a valid option type. Valid options were: stable-hlo-to-executable
# CHECK: --tensorrt-timing-cache-path= --device-infer-from-host=true --debug-only= --executor-index-bitwidth=64 --entrypoint=main --plan-clustering-disallow-host-tensors-in-tensorrt-clusters=false --tensorrt-workspace-memory-pool-limit=1073741824 --device-max-registers-per-block=65536 --tensorrt-strongly-typed=false --tensorrt-layer-info-dir= --device-compute-capability=86 --debug=false --mlir-print-ir-tree-dir= --disable-tensorrt-extension=false --tensorrt-builder-opt-level=3 --tensorrt-engines-dir=
