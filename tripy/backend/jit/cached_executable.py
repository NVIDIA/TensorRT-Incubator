import base64
import tempfile
from typing import Any, Dict, List

import mlir_tensorrt.compiler.api as compiler

from tripy.backend.utils import TensorInfo
from tripy.utils.json import Decoder, Encoder

from tripy.common.device import device


class CachedExecutable:
    def __init__(
        self,
        executable: compiler.Executable,
        input_info: List[TensorInfo],
        output_devices: List[device],
    ):
        self.executable = executable
        self.input_info = input_info
        # HACK (#155): Store output devices as executable device does not match Trace IR device.
        self.output_devices = output_devices

    def is_compatible(self, new_input_info: List[TensorInfo]) -> bool:
        for inp, new_inp in zip(self.input_info, new_input_info):
            if new_inp.dtype != inp.dtype:
                return False

            # TODO: This may be too strict in multi-GPU cases
            if new_inp.device != inp.device:
                return False

            def is_shape_compatible(shape, new_shape):
                return all(new_dim.min >= dim.min and new_dim.max <= dim.max for dim, new_dim in zip(shape, new_shape))

            if not is_shape_compatible(inp.shape, new_inp.shape):
                return False
        return True


@Encoder.register(CachedExecutable)
def encode(cached_executable: CachedExecutable) -> Dict[str, Any]:
    # TODO: Add an MLIR-TRT API to save an executable directly to a string. For now, we WAR
    # this with a temporary intermediate file.
    with tempfile.NamedTemporaryFile("wb+") as f:
        raise NotImplementedError(f"Not yet implemented. Need to figure out how to serialize MLIR-TRT executables")
        mlir_backend.save(cached_executable.executable, f.name)

        f.flush()
        f.seek(0)
        data = f.read()

        # Encode the executable to base 64 which is very compact.
        executable = base64.b64encode(data).decode()
        return {
            "data": executable,
            "input_info": cached_executable.input_info,
            "output_devices": cached_executable.output_devices,
        }


@Decoder.register(CachedExecutable)
def decode(dct: Dict[str, str]) -> CachedExecutable:
    raise NotImplementedError(f"Not yet implemented. Need to figure out how to serialize MLIR-TRT executables")
    data = base64.b64decode(dct["data"].encode(), validate=True)
    executable = mlir_backend.load(data=data)

    return CachedExecutable(executable, dct["input_info"], dct["output_devices"])
