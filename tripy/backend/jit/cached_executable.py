import base64
import tempfile
from typing import Any, Dict, List

from tripy.backend.mlir.mlir import mlir_wrapper, void_ptr
from tripy.utils.json import Decoder, Encoder
from tripy.backend.jit.utils import TensorInfo


class CachedExecutable:
    def __init__(self, executable: void_ptr, input_info: List[TensorInfo], output_info: List[TensorInfo]):
        self.executable = executable
        self.input_info = input_info
        # HACK (#109): Store output information so we don't need to go all the way to the FlatIR each time.
        # This is needed only because the MLIR executor currently needs the output buffers ahead of time.
        self.output_info = output_info

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
        mlir_backend = mlir_wrapper()
        mlir_backend.save(cached_executable.executable, f.name)

        f.flush()
        f.seek(0)
        data = f.read()

        # Encode the executable to base 64 which is very compact.
        executable = base64.b64encode(data).decode()
        return {
            "data": executable,
            "input_info": cached_executable.input_info,
            "output_info": cached_executable.output_info,
        }


@Decoder.register(CachedExecutable)
def decode(dct: Dict[str, str]) -> CachedExecutable:
    mlir_backend = mlir_wrapper()
    data = base64.b64decode(dct["data"].encode(), validate=True)
    executable = mlir_backend.load(data=data)

    return CachedExecutable(executable, dct["input_info"], dct["output_info"])
