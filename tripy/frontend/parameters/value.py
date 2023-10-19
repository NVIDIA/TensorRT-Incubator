from dataclasses import dataclass
from typing import Any
import numpy as np

from tripy.frontend.parameters.base import BaseParameters


@dataclass
class ValueParameters(BaseParameters):
    """
    Represents values stored in a tensor.
    """

    values: Any  # TODO: This should be a GPU-backed tensor.

    def shape(self):
        assert isinstance(self.values, list) or isinstance(self.values, np.ndarray)
        if isinstance(self.values, list):
            return [len(self.values)]
        elif isinstance(self.values, np.ndarray):
            return self.values.shape
        return None
