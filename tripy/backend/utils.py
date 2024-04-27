from dataclasses import dataclass
from typing import Any, Dict, List

from tripy.common.types import ShapeInfo
from tripy.utils.json import Decoder, Encoder


# HACK (#109): This is a temporary class which we need in order to convey output information
# to MLIR-TRT. Eventually, it should just call back into Tripy when it needs memory allocated.
@dataclass
class TensorInfo:
    shape: ShapeInfo
    dtype: "tripy.dtype"
    device: "tripy.device"


@Encoder.register(TensorInfo)
def encode(tensor_info: TensorInfo) -> Dict[str, Any]:
    return {"shape": tensor_info.shape, "dtype": tensor_info.dtype, "device": tensor_info.device}


@Decoder.register(TensorInfo)
def decode(dct: Dict[str, Any]) -> TensorInfo:
    return TensorInfo(dct["shape"], dct["dtype"], dct["device"])


def get_tensor_info(tensors) -> List[TensorInfo]:
    return [TensorInfo(tensor.shape, tensor.dtype, tensor.device) for tensor in tensors]


def get_devices(tensor_info):
    return [info.device for info in tensor_info]
