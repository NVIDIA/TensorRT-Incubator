from typing import Optional, Union, Dict, Tuple, Any
from torch import nn
from torch.export import ExportedProgram
from dataclasses import dataclass, field
from .compiler_utils import OutputType
from . import fx
from .extras.fx_importer import FxImporter
from .dialects import torch as torch_d

__all__ = ["TorchInput", "get_mlir_module_from_torch_module"]


@dataclass
class TorchInput:
    """
    Dataclass that holds Torch module and corresponding args.
    Args:
        f: Torch NN Module or exported program (e.g. torch.fx.GraphModule).
        args: Input arguments to Torch module
        dynamic_shapes: Dynamic shape in the form of dictionary.
            For example, consider the subgraph below and input with all dynamic dimensions.

            class Model(nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(x)

            Dynamic shapes in this case are given as follows,
            x = torch.randn(3, 4)
            dim_n = Dim("n", min=1, max=10)
            dim_x1 = Dim("x1", min=1, max=100)
            dynamic_shapes = {"x": {0: dim_n, 1: dim_x1}}

            Its important to note that variable name and `key` name that
            represnets dynamic shape for that variable must match.
        kwargs: Keyword args to Torch module, if any.
    """

    f: Union[nn.Module, ExportedProgram]
    args: Tuple[Any, ...] = field(default_factory=tuple)
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


def get_mlir_module_from_torch_module(
    context, torch_input: TorchInput, output_type: OutputType
):
    """
    Returns MLIR `ModuleOp` created by importing Torch module.
    Args:
        context: MLIR context
        torch_input: Instance of `TorchInput` dataclass that specifies input torch module
            and related arguments.
        output_type: Dialect of operations in the returned MLIR module. Useful ones for MLIR-TensorRT
            are `OutputType.STABLEHLO` and `OutputType.LINALG_ON_TENSORS`.
    Returns:
        MLIR `ModuleOp`
    """
    torch_d.register_dialect(context)
    fx_importer = FxImporter(context=context)
    return fx.export_and_import(
        torch_input.f,
        *torch_input.args,
        output_type=output_type,
        fx_importer=fx_importer,
        dynamic_shapes=torch_input.dynamic_shapes,
        # Symbolic shape expressions are supported only in Torch-IR
        import_symbolic_shape_expressions=False,
        func_name="main",
        **torch_input.kwargs,
    )
