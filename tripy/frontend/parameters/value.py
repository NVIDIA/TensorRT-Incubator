from dataclasses import dataclass
from typing import Any

from tripy.frontend.parameters.base import BaseParameters


@dataclass
class ValueParameters(BaseParameters):
    """
    Represents values stored in a tensor.
    """

    values: Any  # TODO: This should be a GPU-backed tensor.
