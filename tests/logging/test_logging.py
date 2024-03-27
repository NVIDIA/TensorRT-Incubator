import copy
import os

import pytest

import tripy as tp


class TestLogging:
    def test_basic(self, capsys):
        message = "Test Message"
        tp.logger.info(message)

        captured = capsys.readouterr()

        assert message in captured.out

    @pytest.mark.parametrize(
        "verbosity, expected",
        [
            # Enable `info` logging in all modules
            (
                "info",
                {"": {"info", "warning", "error"}},
            ),
            # Enable `flat_ir` logging in all modules
            (
                "flat_ir",
                {"": {"flat_ir"}},
            ),
            # Enable `info` and `trace` logging in all modules
            (
                {"info", "trace"},
                {"": {"info", "warning", "error", "trace"}},
            ),
            # Enable `error` logging for just the frontend module:
            (
                {"frontend": "error"},
                {"frontend": {"": {"error"}}, "": {"error", "warning", "info"}},
            ),
            # Enable `error` and `timing` logging for just the frontend module:
            (
                {"frontend": {"error", "timing"}},
                {"frontend": {"": {"timing", "error"}}, "": {"error", "warning", "info"}},
            ),
        ],
    )
    def test_verbosity_trie(self, verbosity, expected):
        old_verbosity = copy.copy(tp.logger.verbosity)
        try:
            tp.logger.verbosity = verbosity
            assert tp.logger.verbosity == expected
        finally:
            # Reset verbosity so we don't corrupt other tests
            tp.logger.verbosity = old_verbosity

    def test_use_verbosity(self):
        default_verbosity = {"": {"info", "warning", "error"}}
        assert tp.logger.verbosity == default_verbosity
        with tp.logger.use_verbosity("error"):
            assert tp.logger.verbosity == {"": {"error"}}
        assert tp.logger.verbosity == default_verbosity

    def test_disabled_verbosity_does_not_log(self, capsys):
        with tp.logger.use_verbosity("warning"):
            tp.logger.info("This message should not be logged!")

        assert not capsys.readouterr().out

    def test_can_disable_using_filename(self, capsys):
        # Paths are expected to be part of the tripy module, so we do something slightly
        # hacky to express this file relative to the tripy module (even though it's not part of it).
        path = os.path.realpath(os.path.relpath(__file__, tp.__path__[0]))
        with tp.logger.use_verbosity({path: "warning"}):
            tp.logger.info("This message should not be logged!")

        assert not capsys.readouterr().out

    def test_indent(self, capsys):
        message = "This message should be indented"
        with tp.logger.indent():
            tp.logger.info(message)

        out = capsys.readouterr().out
        print(out)
        assert (" " * 4 + message) in out

        # After we exit the context manager, messages should no longer be indented.
        message = "This message should NOT be indented"
        tp.logger.info(message)

        out = capsys.readouterr().out
        print(out)
        assert message in out
        assert (" " * 4 + message) not in out

    def test_log_callable(self, capsys):
        message = "Message to log"
        tp.logger.info(lambda: message)

        out = capsys.readouterr().out
        print(out)
        assert message in out
