# RUN: %PYTHON %s

import mlir_tensorrt.compiler.api as compiler
import mlir_tensorrt.compiler.ir as ir
import mlir_tensorrt.compiler.torch_bridge as torch_bridge
import mlir_tensorrt.runtime.api as runtime
import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        y = x1 + x2
        return y


def compile():
    x1 = torch.randn(2, 2).to(torch.float32)
    torch_input = torch_bridge.TorchInput(Model(), (x1, x1))

    with ir.Context() as ctx:
        shlo_module = torch_bridge.get_mlir_module_from_torch_module(
            ctx, torch_input, torch_bridge.OutputType.STABLEHLO
        )
        client = compiler.CompilerClient(ctx)
        task = client.get_compilation_task(
            "stablehlo-to-executable",
            [
                "--tensorrt-builder-opt-level=3",
                "--tensorrt-strongly-typed=false",
                "--tensorrt-workspace-memory-pool-limit=1gb",
            ],
        )
        task.run(shlo_module.operation)
        return compiler.translate_mlir_to_executable(shlo_module.operation)


def torch_add():
    exe = compile()
    client = runtime.RuntimeClient()
    stream = client.create_stream()
    devices = client.get_devices()

    if len(devices) == 0:
        return

    session_options = runtime.RuntimeSessionOptions(num_devices=1, device_id=0)
    session = runtime.RuntimeSession(session_options, exe)

    torch_arg0 = torch.arange(0.0, 4.0, dtype=torch.float32).reshape(2, 2)
    arg0 = client.create_memref_view_from_dlpack(torch_arg0.__dlpack__())
    arg0 = client.copy_to_device(arg0, device=devices[0])

    torch_arg1 = torch.zeros(2, 2, dtype=torch.float32)
    arg1 = client.create_memref_view_from_dlpack(torch_arg1.__dlpack__())
    arg1 = client.copy_to_device(arg1, device=devices[0])

    session.execute_function(
        "main", in_args=[arg0, arg0], out_args=[arg1], stream=stream
    )

    data = np.asarray(client.copy_to_host(arg1, stream=stream))
    stream.sync()

    np.testing.assert_equal(data, np.arange(0.0, 8.0, 2.0).reshape(2, 2))


if __name__ == "__main__":
    torch_add()
