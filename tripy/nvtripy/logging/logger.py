#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import contextlib
import copy
import os
import sys
from functools import partial
from textwrap import indent
from typing import Any, Callable, Dict, Union, Set

from colored import Style

from nvtripy.logging.verbosity import VerbosityConfig, make_verbosity_map
from nvtripy.logging.verbosity_trie import VerbosityTrie
from nvtripy import export


class _Logger:
    """
    The global logger used across Tripy.

    The verbosity can be controlled using the ``verbosity`` property and may be either a single string
    or set of strings:

    .. code-block:: python
        :linenos:
        :caption: Setting Logging Verbosity

        # Need to reset verbosity when we finish the example to not affect other tests # doc: omit
        # doc: no-print-locals
        old_verbosity = tp.logger.verbosity # doc: omit
        old_color = tp.logger.enable_color # doc: omit
        tp.logger.enable_color = False # doc: omit

        tp.logger.verbose("This will NOT be logged")
        tp.logger.verbosity = "verbose"
        tp.logger.verbose("This will be logged")

        tp.logger.verbosity = old_verbosity # doc: omit
        tp.logger.enable_color = old_color # doc: omit

    Possible values for this come from the keys of ``logger.VERBOSITY_CONFIGS``.

    It may alternatively be provided in a per-path dictionary to set per-module/submodule
    verbosities:

    .. code-block:: python
        :linenos:
        :caption: Per-Submodule Logging Verbosities

        # doc: no-print-locals
        old_verbosity = tp.logger.verbosity # doc: omit
        old_color = tp.logger.enable_color # doc: omit
        tp.logger.enable_color = False # doc: omit


        # Enable `verbose` logging for just the frontend module:
        tp.logger.verbosity = {"frontend": "verbose"}

        # Enable `verbose` and `timing` logging for just the frontend module:
        tp.logger.verbosity = {"frontend": {"verbose", "timing"}}

        tp.logger.verbosity = old_verbosity # doc: omit
        tp.logger.enable_color = old_color # doc: omit


    Verbosity can also be changed temporarily by using the ``use_verbosity`` context manager:

    .. code-block:: python
        :linenos:
        :caption: Setting Logging Verbosity Temporarily

        old_color = tp.logger.enable_color # doc: omit
        tp.logger.enable_color = False # doc: omit


        tp.logger.verbose("This will NOT be logged")

        with tp.logger.use_verbosity("verbose"):
            tp.logger.verbose("This will be logged")

        tp.logger.verbose("This will NOT be logged")

        tp.logger.enable_color = old_color # doc: omit
    """

    VERBOSITY_CONFIGS: Dict[str, VerbosityConfig] = make_verbosity_map()

    def __init__(self) -> None:
        self._indentation = 0
        self.verbosity: Union[str, Set[str], Dict[str, str], Dict[str, Set[str]]] = "info"
        self.enable_color = True
        self._already_logged_hashes = set()

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

    def log(
        self, message: Union[str, Callable[[], str]], verbosity: str, mode: str = "each", stack_depth: int = 2
    ) -> None:
        """
        Logs a message to standard output.

        Generally, you should use the methods that are tied to a severity, such as `info` or `timing`.
        `VERBOSITY_CONFIGS` includes a complete list of available methods.

        Args:
            message: The message to log. This can be provided as a callable in which case it will not
                be called unless the message actually needs to be logged.
            verbosity: The verbosity at which to log this message.
            mode: Indicates when or how to log the message. Available modes are:
                - "each": Log the message each time.
                - "once": Only log a message the first time it is seen.
            stack_depth: The stack depth to use when determining which file the message is being logged from.
        """
        assert (
            verbosity in self.VERBOSITY_CONFIGS
        ), f"Unknown verbosity setting: {verbosity}. Available options were: {list(self.VERBOSITY_CONFIGS.keys())}"

        # Get the file path relative to the module root
        def module_path(path):
            import nvtripy

            module_root = nvtripy.__path__[0]
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

        if mode == "once":
            message_hash = hash(message)
            if message_hash in self._already_logged_hashes:
                return
            self._already_logged_hashes.add(message_hash)

        if callable(message):
            message = message()

        message = indent(message, prefix=" " * self._indentation)

        config = self.VERBOSITY_CONFIGS[verbosity]
        message = f"{config.prefix}{message}"
        if self.enable_color:
            message = f"{config.color}{message}{Style.reset}"

        print(message)

    def __getattr__(self, name) -> Any:
        if name in self.VERBOSITY_CONFIGS:
            return partial(self.log, verbosity=name, stack_depth=3)

        raise AttributeError(f"Logger has no attribute: '{name}'.")


logger = export.public_api(autodoc_options=[":annotation:"], symbol="logger")(_Logger())
