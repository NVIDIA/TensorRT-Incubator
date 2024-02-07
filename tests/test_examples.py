import copy
import glob
import os
import shutil
from textwrap import dedent
from typing import Callable, List, Set, Sequence

import pytest

from tests.helper import ROOT_DIR

EXAMPLES_ROOT = os.path.join(ROOT_DIR, "examples")


class Marker:
    """
    Represents special markers in example READMEs used to convey information to
    the testing infrastructure.

    Special markers follow the format:

    <!-- Tripy Test: <NAME> Start -->

    and:

    <!-- Tripy Test: <NAME> End -->

    marking the start and end of the block respectively.
    """

    def __init__(
        self, matches_start_func: Callable[[str], bool] = None, matches_end_func: Callable[[str], bool] = None
    ):
        self.matches_start = matches_start_func

        self.matches_end = matches_end_func

    @staticmethod
    def from_name(name: str) -> "Marker":
        return Marker(
            matches_start_func=lambda line: line == f"<!-- Tripy Test: {name} Start -->",
            matches_end_func=lambda line: line == f"<!-- Tripy Test: {name} End -->",
        )


AVAILABLE_MARKERS = {
    # For command markers, the start marker may be annotated with a language tag, e.g. ```py, so an exact match is too strict.
    "command": Marker(
        matches_start_func=lambda line: line.startswith("```"),
        matches_end_func=lambda line: line == "```",
    ),
    # Marks an entire block to be ignored by the tests.
    "ignore": Marker.from_name("IGNORE"),
    # Marks an entire block as being expected to fail.
    "xfail": Marker.from_name("XFAIL"),
    # Marks that a block contains the expected output from the immediate previous block.
    "expected_stdout": Marker.from_name("EXPECTED_STDOUT"),
}


class MarkerTracker:
    """
    Keeps track of active markers in the current README on a line-by-line basis.
    """

    def __init__(self, readme_path: str):
        self.readme_path: str = readme_path
        self.active_markers: Set[Marker] = set()
        self.entering_markers: Set[Marker] = set()  # The markers that we are currently entering
        self.exiting_markers: Set[Marker] = set()  # The markers that we are currently exiting

    def __enter__(self) -> "MarkerTracker":
        self.file = open(self.readme_path, "r")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.file.close()

    def __iter__(self) -> str:
        for line in self.file.readlines():
            stripped_line = line.strip()
            self.entering_markers.clear()
            self.exiting_markers.clear()

            for marker in AVAILABLE_MARKERS.values():
                if not self.is_in(marker) and marker.matches_start(stripped_line):
                    self.active_markers.add(marker)
                    self.entering_markers.add(marker)
                elif marker.matches_end(stripped_line):
                    self.active_markers.remove(marker)
                    self.exiting_markers.add(marker)

            yield line.rstrip()

    def is_in(self, marker: Marker) -> bool:
        """
        Whether we are currently on a line between the specified start and end marker.
        This will always return False for a line containing the marker itself.
        """
        return marker in self.active_markers and not (self.entering(marker) or self.exiting(marker))

    def entering(self, marker):
        return marker in self.entering_markers

    def exiting(self, marker):
        return marker in self.exiting_markers


class CommandBlock:
    def __init__(self, markers: Set[Marker]):
        self.content: str = None
        self.markers = markers

    def add(self, line: str):
        if self.content is None:
            self.content = line
        else:
            self.content += f"\n{line}"

    def has_marker(self, name: str):
        return AVAILABLE_MARKERS[name] in self.markers

    def __str__(self):
        return dedent(self.content)


# Extract any ``` blocks from the README
# NOTE: This parsing logic is not smart enough to handle multiple separate commands in a single block.
def load_command_blocks_from_readme(readme) -> List[CommandBlock]:
    with open(readme, "r") as f:
        contents = f.read()
        # Check that the README has all the expected sections.
        assert "## Introduction" in contents, "All example READMEs should have an 'Introduction' section!"
        assert "## Running The Example" in contents, "All example READMEs should have a 'Running The Example' section!"

    cmd_blocks = []
    with MarkerTracker(readme) as tracker:
        for line in tracker:
            # We use copy here so we don't accidentally alias.
            if tracker.entering(AVAILABLE_MARKERS["command"]):
                current_block = CommandBlock(markers=copy.copy(tracker.active_markers))
            elif tracker.exiting(AVAILABLE_MARKERS["command"]):
                cmd_blocks.append(copy.copy(current_block))
            elif tracker.is_in(AVAILABLE_MARKERS["command"]):
                current_block.add(line)

    return cmd_blocks


class Example:
    def __init__(self, path_components: Sequence[str], artifact_names: Sequence[str] = []):
        self.path = os.path.join(EXAMPLES_ROOT, *path_components)
        self.artifacts = [os.path.join(self.path, name) for name in artifact_names]
        # Ensures no files in addition to the specified artifacts were created.
        self.original_files = []

    def _get_file_list(self):
        return [path for path in glob.iglob(os.path.join(self.path, "*")) if "__pycache__" not in path]

    def _remove_artifacts(self, must_exist=True):
        for artifact in self.artifacts:
            if must_exist:
                print(f"Checking for the existence of artifact: {artifact}")
                assert os.path.exists(artifact), f"{artifact} does not exist!"
            elif not os.path.exists(artifact):
                continue

            if os.path.isdir(artifact):
                shutil.rmtree(artifact)
            else:
                os.remove(artifact)

    def __enter__(self):
        self._remove_artifacts(must_exist=False)

        self.original_files = self._get_file_list()
        readme = os.path.join(self.path, "README.md")
        return load_command_blocks_from_readme(readme)

    def run(self, block, sandboxed_install_run):
        return sandboxed_install_run(block, cwd=self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Checks for and removes artifacts expected by this example
        """
        self._remove_artifacts()
        assert (
            self._get_file_list() == self.original_files
        ), "Unexpected files were created. If this is the desired behavior, add the file paths to `artifact_names`"

    def __str__(self):
        return os.path.relpath(self.path, EXAMPLES_ROOT)


EXAMPLES = [Example(["nanogpt"])]


@pytest.mark.l1
@pytest.mark.parametrize("example", EXAMPLES, ids=lambda case: str(case))
def test_examples(example, sandboxed_install_run):
    statuses = []
    with example as command_blocks:
        for block in command_blocks:
            if block.has_marker("ignore"):
                continue

            block_text = str(block)
            if block.has_marker("expected_stdout"):
                print("Checking command output against expected output:")
                assert statuses[-1].stdout.strip() == block_text.strip()
            else:
                status = example.run(block_text, sandboxed_install_run)

                details = f"Note: Command was: {block_text}.\n==== STDOUT ====\n{status.stdout}\n==== STDERR ====\n{status.stderr}"
                if block.has_marker("xfail"):
                    assert not status.success, f"Command that was expected to fail did not fail. {details}"
                else:
                    assert status.success, f"Command that was expected to succeed did not succeed. {details}"
                statuses.append(status)
