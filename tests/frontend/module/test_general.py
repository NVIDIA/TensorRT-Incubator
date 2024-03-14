import dataclasses
import inspect

import pytest

from tests import helper
from tripy.frontend.module import Module

MODULE_TYPES = {
    obj
    for obj in helper.discover_tripy_objects()
    if inspect.isclass(obj) and issubclass(obj, Module) and obj is not Module
}


@pytest.mark.parametrize("ModuleType", MODULE_TYPES)
class TestModules:
    def test_is_dataclass(self, ModuleType):
        assert dataclasses.is_dataclass(
            ModuleType
        ), f"Modules must be data classes so that we can ensure attributes have type annotations and documentation"
