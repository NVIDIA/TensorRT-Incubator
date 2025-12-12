# Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
"""Utility functions for MLIR-TensorRT JAX primitives."""
from typing import List
from jax import __version_info__ as jax_version


def _compare_versions(pa, pb):
    return (pa > pb) - (pa < pb)


JAX_VERSION_0_6_0_OR_GREATER = _compare_versions(jax_version, (0, 6, 0)) >= 0


def declare_primitive(name: str):
    # Declare a new JAX primitive. Provides logic to support both JAX 0.5.x and 0.6.x.
    if JAX_VERSION_0_6_0_OR_GREATER:
        from jax.extend.core import Primitive

        return Primitive(name)
    else:
        from jax.core import Primitive

        return Primitive(name)


def default_layouts(*shapes) -> List[range]:
    # Create row-major layouts for given shapes.
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


__all__ = [
    "declare_primitive",
    "default_layouts",
    "JAX_VERSION_0_6_0_OR_GREATER",
]
