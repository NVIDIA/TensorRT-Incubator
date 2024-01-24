from tripy.frontend.nn.module import Module
from tripy.frontend.nn.parameter import Parameter
from tripy.frontend.nn.linear import Linear
from tripy.frontend.nn.layernorm import LayerNorm
from tripy.frontend.nn.embedding import Embedding
from tripy.frontend.nn.functional import softmax

__all__ = ["Parameter", "Module", "Linear", "LayerNorm", "Embedding", "softmax"]


def __getattr__(name: str):
    if name in __all__:
        return locals()[name]

    from tripy.common.exception import search_for_missing_attr
    import tripy as tp

    look_in = [(tp.Tensor, "tripy.Tensor"), (tp, "tripy")]
    search_for_missing_attr("tripy", name, look_in)
