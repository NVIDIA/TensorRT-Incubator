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
        # Set some options explicitly so we can spot check the `print` output.
        [
            "--tensorrt-builder-opt-level=3",
            "--tensorrt-strongly-typed=false",
            "--tensorrt-workspace-memory-pool-limit=1gb",
        ],
    )

    print(opts)


# CHECK: InvalidArgument: InvalidArgument: non-existent-options-type is not a valid option type. Valid options were: stable-hlo-to-executable
# CHECK: Options[{{.*--tensorrt-workspace-memory-pool-limit=1073741824.*--tensorrt-strongly-typed=false.*--tensorrt-builder-opt-level=3.*}}]
