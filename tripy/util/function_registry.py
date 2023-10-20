from typing import Any


class FunctionRegistry(dict):
    """
    Maintains a mapping of keys to functions.

    Besides a specialized __call__ function, this behaves exactly like a dictionary.

    Example:
    ::

        from tripy.util import FunctionRegistry

        example_registry = FunctionRegistry()

        @example_registry(str)
        def example_str() -> str:
            return "called example_str!"

        @example_registry(int)
        def example_int() -> str:
            return "called example_int!"

        assert example_registry[str]() == "called example_str!"
        assert example_registry[int]() == "called example_int!"
    """

    def __call__(self, key: Any):
        """
        Registers a function with this function registry.
        This function allows instances of the class to be used as decorators.

        Args:
            key: The key under which to register the function.

        Example:
        ::

            from tripy.util import FunctionRegistry

            example_registry = FunctionRegistry()

            @example_registry("my_key")
            def func() -> None:
                pass

            assert example_registry["my_key"] == func
        """

        def impl(func):
            self[key] = func
            return func

        return impl
