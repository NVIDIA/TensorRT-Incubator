import contextlib
import copy
import os
import sys
from functools import partial
from textwrap import indent
from typing import Any, Callable, Dict, Union

from colored import Style

from tripy.logging.verbosity import VerbosityConfig, make_verbosity_map
from tripy.logging.verbosity_trie import VerbosityTrie


class _Logger:
    VERBOSITY_CONFIGS: Dict[str, VerbosityConfig] = make_verbosity_map()

    def __init__(self) -> None:
        self._indentation = 0
        self.verbosity = "info"
        """
        Verbosity to use for the logger. The verbosity may be either a single string
        or set of strings and can be provided in a per-path dictionary to set per-module/submodule
        verbosities.

        Possible values for this come from the keys of VERBOSITY_CONFIGS.

        For example:
        ::

            # Enable `info` logging in all modules
            logger.verbosity = "info"

            # Enable `info` and `trace` logging in all modules
            logger.verbosity = {"info", "trace"}

            # Enable `verbose` logging for just the frontend module:
            logger.verbosity = {"frontend": "verbose"}

            # Enable `verbose` and `timing` logging for just the frontend module:
            logger.verbosity = {"frontend": {"verbose", "timing"}}
        """

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, new_verbosity):
        if isinstance(new_verbosity, VerbosityTrie):
            self._verbosity = new_verbosity
            return

        verbosity_map = {}
        if not isinstance(new_verbosity, dict):
            new_verbosity = {"": new_verbosity}

        if "" not in new_verbosity:
            new_verbosity[""] = "info"

        for key, values in new_verbosity.items():
            if isinstance(values, str):
                values = [values]

            # Update values based on `enables`.
            updated_values = set(values)
            for value in values:
                config = self.VERBOSITY_CONFIGS[value]
                updated_values.update(config.enables)

            verbosity_map[key] = updated_values

        self._verbosity = VerbosityTrie(verbosity_map)

    @contextlib.contextmanager
    def use_verbosity(self, verbosity: str):
        """
        Returns a context manager that temporarily adjusts the verbosity of the logger.
        """
        old_verbosity = copy.copy(self.verbosity)

        try:
            self.verbosity = verbosity
            yield
        finally:
            self.verbosity = old_verbosity

    @contextlib.contextmanager
    def indent(self, level: int = 4):
        """
        Returns a context manager that indents all contained logging messages.
        This can be nested.

        Args:
            level: The number of spaces by which to indent.
        """
        old_indentation = self._indentation

        try:
            self._indentation += level
            yield
        finally:
            self._indentation = old_indentation

    def log(self, message: Union[str, Callable[[], str]], verbosity: str, stack_depth: int = 2) -> None:
        """
        Logs a message to standard output.

        Generally, you should use the methods that are tied to a severity, such as `info` or `timing`.
        `VERBOSITY_CONFIGS` includes a complete list of available methods.

        Args:
            message: The message to log. This can be provided as a callable in which case it will not
                be called unless the message actually needs to be logged.
            verbosity: The verbosity at which to log this message.
            stack_depth: The stack depth to use when determining which file the message is being logged from.
        """
        assert (
            verbosity in self.VERBOSITY_CONFIGS
        ), f"Unknown verbosity setting: {verbosity}. Available options were: {list(self.VERBOSITY_CONFIGS.keys())}"

        # Get the file path relative to the module root
        def module_path(path):
            import tripy

            module_root = tripy.__path__[0]
            return os.path.realpath(os.path.relpath(path, module_root))

        def get_rel_file_path():
            file_path = sys._getframe(stack_depth).f_code.co_filename
            # If we can't get a valid path, keep walking the stack until we can.
            index = stack_depth
            while index > 0 and not os.path.exists(file_path):
                index -= 1
                file_path = sys._getframe(index).f_code.co_filename

            return module_path(file_path)

        def should_log():
            path = None
            # Don't actually need to get the path if there are no non-default entries in the trie.
            if self.verbosity.has_non_default_entries:
                path = get_rel_file_path()
            return verbosity in self.verbosity.get_verbosity_set(path)

        if not should_log():
            return

        if callable(message):
            message = message()

        message = indent(message, prefix=" " * self._indentation)

        config = self.VERBOSITY_CONFIGS[verbosity]
        print(f"{config.color}{config.prefix}{message}{Style.reset}")

    def __getattr__(self, name) -> Any:
        if name in self.VERBOSITY_CONFIGS:
            return partial(self.log, verbosity=name, stack_depth=3)

        return super().__getattr__(name)


logger = _Logger()
