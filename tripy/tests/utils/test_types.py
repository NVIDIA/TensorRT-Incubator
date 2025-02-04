from nvtripy.utils.types import str_from_type_annotation, type_str_from_arg
import pytest
from typing import Callable, Dict, List, Optional, Tuple

import nvtripy as tp
import pytest
import torch


@pytest.mark.parametrize(
    "typ, expected",
    [
        (tp.types.IntLike, "int | nvtripy.DimensionSize"),
        (Tuple[tp.types.IntLike], "Tuple[int | nvtripy.DimensionSize]"),
        (List[tp.types.IntLike], "List[int | nvtripy.DimensionSize]"),
        (Dict[str, tp.Tensor], "Dict[str, nvtripy.Tensor]"),
        (tp.types.TensorLike, "nvtripy.Tensor | numbers.Number"),
        (tp.types.ShapeLike, "Sequence[int | nvtripy.DimensionSize]"),
        (tp.Tensor, "nvtripy.Tensor"),
        (torch.Tensor, "torch.Tensor"),
        (int, "int"),
        (Optional[int], "int | None"),
        (Callable[[int], int], "Callable[[int], int]"),
    ],
)
def test_str_from_type_annotation(typ, expected):
    assert str_from_type_annotation(typ) == expected


@pytest.mark.parametrize(
    "typ, expected",
    [
        (tp.Tensor([1, 2, 3]), "nvtripy.Tensor"),
        (torch.tensor([1, 2, 3]), "torch.Tensor"),
        (0, "int"),
        ("hi", "str"),
        ([0, 1, 2], "List[int]"),
        ([0, "1", 2], "List[int | str]"),
        ({0: 1}, "Dict[int, int]"),
        ({0: 1, "a": "b"}, "Dict[int | str, int | str]"),
    ],
)
def test_type_str_from_arg(typ, expected):
    assert type_str_from_arg(typ) == expected
