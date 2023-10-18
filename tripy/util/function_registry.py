from typing import Dict, Any


class FunctionRegistry:
    """
    Maintains a mapping of keys to functions.

    Example:
    ::

        from tripy.util import FunctionRegistry
        from typing import Callable, Any

        example_registry = FunctionRegistry(Callable[[], str])

        @example_registry(str)
        def example_str() -> str:
            return "called example_str!"

        @example_registry(int)
        def example_int() -> str:
            return "called example_int!"

        assert example_registry[str]() == "called example_str!"
        assert example_registry[int]() == "called example_int!"
    """

    def __init__(self, FuncType: type) -> None:
        """
        Args:
            FuncType: The type of the function.

        Example:
        ::

            from tripy.util import FunctionRegistry
            from typing import Callable, Any

            example_registry = FunctionRegistry(Callable[[int, int], int])
        """
        self.FuncType = FuncType
        self.registry: Dict[type, FuncType] = {}

    def __call__(self, key: Any):
        """
        Registers a function with this function registry.
        This function allows instances of the class to be used as decorators.

        Args:
            key: The key under which to register the function.

        Example:
        ::

            from tripy.util import FunctionRegistry
            from typing import Callable, Any

            example_registry = FunctionRegistry(Callable[[], None])

            @example_registry("my_key")
            def func() -> None:
                pass

            assert example_registry["my_key"] == func
        """

        def impl(func: self.FuncType):
            self.registry[key] = func
            return func

        return impl

    def __getitem__(self, key: Any):
        assert (
            key in self.registry
        ), f"Key: {key} was not found in the registry.\nNote: Available keys: {list(self.registry.keys())}"
        return self.registry[key]
