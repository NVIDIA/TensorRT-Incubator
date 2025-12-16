"""
This module is a pytest plugin intended to be used for running upstream JAX unit tests
with the MLIR-TensorRT PJRT backend.
It provides three important features:
1. It captures the artifacts (error log and MLIR crash repro) for failed tests.
Given `top_level_dir` for artifacts via `--mtrt-pjrt-artifact-dir`, it will create a
directory structure like:
```
top_level_dir/
    <test_module>/
        <test_name>/
            error.txt
            repro.mlir
```
`test_module` is the directory name of the JAX unit test file (without the .py extension).
`test_name` is the name of the test function inside the JAX unit test file.

2. It also supports skipping tests that are not supported by MLIR-TensorRT, for any reason.
The config file for skipping tests is a JSON file specified via `--mtrt-jax-unittest-config-json`.
The config file has the following format:
```
{
    "test_module": {
        "test_name": "skip_reason"
    }
}
```
Similar to artifacts saving, `test_module` is the directory name of the JAX unit test
file (without the .py extension) and `test_name` is the name of the test function inside
the JAX unit test file. `skip_reason` is a string that describes why the test is not
supported by MLIR-TensorRT, which is added to the test report.

3. It generates a test summary report which is saved in the directory of `test_module`.

To use this plugin, you need to pass the following options to pytest:

--mtrt-pjrt-artifact-dir: The directory to save the test artifacts.
--mtrt-jax-unittest-config-json: The JSON file containing the unsupported tests.

Example usage:
```
pytest -p mtrt_jax_pytest_plugin --mtrt-pjrt-artifact-dir=./mtrt_test_logs --mtrt-jax-unittest-config-json=./mtrt_jax_unittest_config.json
```
NOTE: This plugin is also compatible with xdist.
"""

from pathlib import Path
import pytest
import shutil
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class TestConfig:
    """Configuration for test execution."""

    artifact_dir: str
    unsupported_tests: Dict[str, Dict[str, str]]


@dataclass
class TestStats:
    """Statistics for test execution."""

    passed: int
    failed: int
    skipped: int
    total: int = 0
    executed: int = 0

    @property
    def coverage(self) -> float:
        return (self.executed / self.total) * 100 if self.total > 0 else 0

    @property
    def success_rate(self) -> float:
        return (self.passed / self.total) * 100 if self.total > 0 else 0

    @property
    def failure_rate(self) -> float:
        return (self.failed / self.executed) * 100 if self.executed > 0 else 0


def pytest_addoption(parser, pluginmanager) -> None:
    """Add custom pytest command line options."""
    parser.addoption(
        "--mtrt-pjrt-artifact-dir",
        dest="MTRT_PJRT_ARTIFACT_DIR",
        help="Directory to save MTRT JAX unit test artifacts",
    )
    parser.addoption(
        "--mtrt-jax-unittest-config-json",
        dest="MTRT_JAX_UNITTEST_CONFIG_JSON",
        default="",
        help="JSON file containing MLIR-TensorRT unsupported tests",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Initialize session-wide test configuration."""
    artifact_dir = Path(session.config.getoption("MTRT_PJRT_ARTIFACT_DIR")).absolute()
    config_path = session.config.getoption("MTRT_JAX_UNITTEST_CONFIG_JSON")
    test_config_dict = (
        {} if not config_path else json.loads(Path(config_path).read_text())
    )
    session.test_config = TestConfig(
        artifact_dir=str(artifact_dir),
        unsupported_tests=test_config_dict,
    )


def sanitize_name(name):
    """Sanitize test name to be used as a directory name."""
    return re.sub(r"[\[\],\(\)=_  \n]", "", name)


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Set up test environment and handle test skipping."""
    config = getattr(item.session, "test_config", None)
    if not config:
        return

    test_module = Path(item.nodeid.split("::")[0].replace(".py", ""))

    # Check for unsupported tests
    if test_module.name in config.unsupported_tests:
        test_config = config.unsupported_tests[test_module.name]
        if item.name in test_config:
            pytest.skip(
                f"Test {item.name} is not supported by MLIR-TensorRT. "
                f"Reason: {test_config[item.name]}"
            )

    # Set up test directories
    test_module_dir = Path(config.artifact_dir) / test_module
    test_item_dir = test_module_dir / sanitize_name(item.name)

    # Clean and create directories
    shutil.rmtree(test_item_dir, ignore_errors=True)
    test_item_dir.mkdir(parents=True, exist_ok=True)

    # Store metadata in user_properties
    item.user_properties.append(("test_module_dir", str(test_module_dir)))
    item.user_properties.append(("test_item_dir", str(test_item_dir)))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call) -> None:
    """Generate detailed error report when a test fails."""
    outcome = yield
    result = outcome.get_result()

    if not (call.when == "call" and result.failed):
        return

    # Get test directories from user_properties
    test_item_dir = next(
        (v for k, v in item.user_properties if k == "test_item_dir"), None
    )
    if not test_item_dir:
        return

    error_content = [
        result.longreprtext,
        "\nSTDERR:\n-------",
        result.capstderr,
        "\nLOG:\n----",
        result.caplog,
        "\nSTDOUT:\n-------",
        result.capstdout,
    ]
    (Path(test_item_dir) / "error.txt").write_text("\n".join(error_content))


def pytest_runtest_teardown(item: pytest.Item) -> None:
    """Clean up test artifacts and save crash repro if available."""
    # Get test directories from user_properties
    test_item_dir = next(
        (v for k, v in item.user_properties if k == "test_item_dir"), None
    )
    if not test_item_dir:
        return

    error_file = Path(test_item_dir) / "error.txt"
    if not error_file.exists():
        shutil.rmtree(Path(test_item_dir), ignore_errors=True)
        return

    # Copy MLIR produced crash reproducer if exists
    # Check runner bash script for more details
    crash_repro = Path.cwd() / "crash.mlir"
    if crash_repro.exists():
        shutil.move(crash_repro, Path(test_item_dir) / "repro.mlir")


@pytest.hookimpl(hookwrapper=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Generate test summary report for a test module."""
    yield

    def get_test_module_dir(stats: Dict[str, List[Any]]) -> Optional[Path]:
        """Extract test module directory from the first available report."""
        for status in ["passed", "failed", "skipped"]:
            if reports := stats.get(status, []):
                for prop in reports[0].user_properties:
                    if prop[0] == "test_module_dir":
                        return prop[1]
        return None

    stats = terminalreporter.stats
    test_module_dir = get_test_module_dir(stats)
    if not test_module_dir:
        return

    # Calculate statistics
    test_stats = TestStats(
        passed=len(stats.get("passed", [])),
        failed=len(stats.get("failed", [])),
        skipped=len(stats.get("skipped", [])),
    )
    test_stats.total = test_stats.passed + test_stats.failed + test_stats.skipped
    test_stats.executed = test_stats.passed + test_stats.failed

    # Generate report content
    report_content = [
        "########### Test summary ###########",
        f"Passed: {test_stats.passed}",
        f"Failed: {test_stats.failed}",
        f"Skipped: {test_stats.skipped}",
        f"Total: {test_stats.total}",
        f"Coverage: {test_stats.coverage:.2f}% (executed/total * 100)",
        f"Success rate: {test_stats.success_rate:.2f}% (passed/total * 100)",
        f"Failure rate: {test_stats.failure_rate:.2f}% (failed/executed * 100)",
        "*at this point, all skips are treated as failure",
        "",
        "----------- Detailed Report ----------",
        "########### PASSED ###########",
    ]

    # Add test results
    report_content.extend(e.nodeid for e in stats.get("passed", []))
    report_content.extend(["########### FAILED ###########"])
    report_content.extend(
        f"{e.nodeid}\nSTDERR:\n{e.capstderr}\n*****" for e in stats.get("failed", [])
    )
    report_content.extend(["########### SKIPPED ###########"])
    report_content.extend(
        f"{e.nodeid} REASON: {e.longrepr[-1]}" for e in stats.get("skipped", [])
    )

    # Write report
    report_file = Path(test_module_dir) / "report.txt"
    report_file.write_text("\n".join(report_content))
    terminalreporter.write_sep("-", f"Report written to {report_file}")
