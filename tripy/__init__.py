__version__ = "0.1.0"
import tripy.common.datatype
from tripy.backend import jit
from tripy.common import TripyException, device
from tripy.common.datatype import *
from tripy.frontend import (
    Dim,
    Tensor,
    arange,
    full,
    full_like,
    iota,
    iota_like,
    nn,
    ones,
    ones_like,
    rand,
    randn,
    where,
    zeros,
    zeros_like,
)
from tripy.utils.json import load, save

__all__ = [
    "arange",
    "device",
    "Dim",
    "full_like",
    "full",
    "iota_like",
    "iota",
    "jit",
    "load",
    "nn",
    "where",
    "ones",
    "ones_like",
    "rand",
    "randn",
    "save",
    "Tensor",
    "zeros_like",
    "zeros",
] + tripy.common.datatype.__all__


def __getattr__(name: str):
    from tripy.common.exception import search_for_missing_attr

    look_in = [(Tensor, "tripy.Tensor"), (nn, "tripy.nn")]
    search_for_missing_attr("tripy", name, look_in)
