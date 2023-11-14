from typing import Any
import ctypes
import pytest
import atexit
import importlib
from enum import Enum


class FakeLibMemory(Enum):
    INVALID = -1
    ALLOC = 1
    DEALLOC = 2


G_TEST_VAL = FakeLibMemory.INVALID


class MockFunc:
    def __init__(self, name) -> None:
        self.name = name
        self._argtypes = None
        self._restype = None

    @property
    def argtypes(self):
        return self._argtypes

    @argtypes.setter
    def argtypes(self, types):
        self._argtypes = types

    @property
    def restypes(self):
        return self._argtypes

    @restypes.setter
    def restypes(self, types):
        self._restype = types

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        global G_TEST_VAL
        if self.name == "initialize":
            G_TEST_VAL = FakeLibMemory.ALLOC

        if self.name == "destroy":
            G_TEST_VAL = FakeLibMemory.DEALLOC

        return True


class MockLib:
    def __init__(self, file_name) -> None:
        pass

    def __getattr__(self, func_name):
        return MockFunc(func_name)


class InitMockLib(ctypes.CDLL):
    def __init__(self, name: str | None) -> None:
        super().__init__(name)

    def __getattr__(self, name: str):
        if name == "initialize" or name == "destroy":
            return MockFunc(name)
        else:
            return super().__getattr__(name)


# Test that the backend compiler library exports the functions that are required.
def test_function_availability_from_backend(monkeypatch):
    global G_TEST_VAL
    from tripy.backend.mlir.mlir import _MlirCompiler
    import ctypes

    monkeypatch.setattr(ctypes, "CDLL", InitMockLib)

    try:
        _MlirCompiler()
        assert G_TEST_VAL == FakeLibMemory.ALLOC
    except AttributeError as e:
        pytest.fail(f"_MlirCompiler backend library should have functions exported correctly. Exception {e} raised.")


# Test that destroy function for MLIRCompiler is correctly called when the module closes via atexit.
def test_mlir_resource_freed(monkeypatch):
    global G_TEST_VAL

    # Register functions from atexit.register
    registered_functions = []

    def mock_register(func, *args, **kwargs):
        registered_functions.append((func, args, kwargs))

    monkeypatch.setattr(atexit, "register", mock_register)
    monkeypatch.setattr(ctypes, "CDLL", MockLib)

    from tripy.backend.mlir import mlir

    old_compiler = mlir.G_COMPILER_BACKEND

    importlib.reload(mlir)
    mlir.G_COMPILER_BACKEND = mlir._MlirCompiler()
    for func, args, kwargs in registered_functions:
        func(*args, **kwargs)

    mlir.G_COMPILER_BACKEND = old_compiler
    assert G_TEST_VAL == FakeLibMemory.DEALLOC
