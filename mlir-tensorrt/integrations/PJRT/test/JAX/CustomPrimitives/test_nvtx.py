# REQUIRES: host-has-at-least-1-gpus
# REQUIRES: host-has-nsight-systems
# RUN: %pick-one-gpu %mlir-trt-jax-py %s
"""E2E test for NVTX range annotations.

Tests single push/pop, variadic push/pop, decorator, and nested ranges.
"""

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

USER_DOMAIN = "MLIR TensorRT User Annotations"

HARNESS = """\
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.xla_bridge import _discover_and_register_pjrt_plugins
_discover_and_register_pjrt_plugins()
from mlir_tensorrt_jax.mtrt_ops.nvtx_range import (
    mtrt_nvtx_push, mtrt_nvtx_pop, mtrt_nvtx_annotate,
    NVTX_COLOR, register_nvtx_range_lowering
)
register_nvtx_range_lowering()
x = np.ones((2, 4), dtype=np.float32)
"""


@pytest.fixture(scope="module")
def tmpdir():
    d = tempfile.mkdtemp(prefix="nvtx_e2e_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _run_and_check(tmpdir, name, expected_ranges, code):
    script = os.path.join(tmpdir, f"{name}.py")
    Path(script).write_text(code)
    report_base = os.path.join(tmpdir, name)

    result = subprocess.run(
        [
            "nsys",
            "profile",
            "--trace=nvtx,cuda",
            "--force-overwrite=true",
            "-o",
            report_base,
            sys.executable,
            script,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    report = report_base + ".nsys-rep"
    assert result.returncode == 0 and os.path.exists(
        report
    ), f"nsys profiling failed: {result.stderr[:500]}"

    sqlite_path = report_base + ".sqlite"
    subprocess.run(
        [
            "nsys",
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            "-o",
            sqlite_path,
            report,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert os.path.exists(sqlite_path), "nsys export to sqlite failed"

    conn = sqlite3.connect(sqlite_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT domainId FROM NVTX_EVENTS WHERE text = ?", (USER_DOMAIN,)
        ).fetchall()
        assert rows, f"Domain '{USER_DOMAIN}' not found in NVTX_EVENTS"
        domain_id = rows[0][0]

        rows = conn.execute(
            "SELECT DISTINCT text FROM NVTX_EVENTS "
            "WHERE domainId = ? AND text IS NOT NULL AND text != ?",
            (domain_id, USER_DOMAIN),
        ).fetchall()
        found = {r[0] for r in rows}
        missing = [n for n in expected_ranges if n not in found]
        assert not missing, f"Missing NVTX ranges: {missing}, found: {found}"
    finally:
        conn.close()
        os.unlink(sqlite_path)


@pytest.mark.requires_at_least_n_gpus(n=1)
@pytest.mark.requires_nsight_systems
def test_single_push_pop(tmpdir):
    _run_and_check(
        tmpdir,
        "test_single",
        ["single_range"],
        HARNESS
        + """
@jax.jit
def test_fn(x):
    x, rid = mtrt_nvtx_push(x, name="single_range", color=NVTX_COLOR.GREEN)
    y = jnp.exp(x)
    y = mtrt_nvtx_pop(y, range_id=rid)
    return y
result = test_fn(x)
np.testing.assert_allclose(result, np.exp(x), rtol=1e-5)
""",
    )


@pytest.mark.requires_at_least_n_gpus(n=1)
@pytest.mark.requires_nsight_systems
def test_variadic_push_pop(tmpdir):
    _run_and_check(
        tmpdir,
        "test_variadic",
        ["variadic_range"],
        HARNESS
        + """
@jax.jit
def test_fn(a, b):
    (a, b), rid = mtrt_nvtx_push(a, b, name="variadic_range", color=NVTX_COLOR.BLUE)
    c = a + b
    c = mtrt_nvtx_pop(c, range_id=rid)
    return c
result = test_fn(x, x)
np.testing.assert_allclose(result, x + x, rtol=1e-5)
""",
    )


@pytest.mark.requires_at_least_n_gpus(n=1)
@pytest.mark.requires_nsight_systems
def test_decorator(tmpdir):
    _run_and_check(
        tmpdir,
        "test_decorator",
        ["decorated_softmax"],
        HARNESS
        + """
@mtrt_nvtx_annotate(name="decorated_softmax", color=NVTX_COLOR.MAGENTA)
def my_softmax(x):
    e = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    return e / jnp.sum(e, axis=-1, keepdims=True)
@jax.jit
def test_fn(x):
    return my_softmax(x)
result = test_fn(x)
np.testing.assert_allclose(result, jax.nn.softmax(x, axis=-1), rtol=1e-5)
""",
    )


@pytest.mark.requires_at_least_n_gpus(n=1)
@pytest.mark.requires_nsight_systems
def test_nested_ranges(tmpdir):
    _run_and_check(
        tmpdir,
        "test_nested",
        ["outer_range", "inner_range"],
        HARNESS
        + """
@jax.jit
def test_fn(x):
    x, rid_outer = mtrt_nvtx_push(x, name="outer_range", color=NVTX_COLOR.GREEN)
    x, rid_inner = mtrt_nvtx_push(x, name="inner_range", color=NVTX_COLOR.RED)
    y = jnp.exp(x)
    y = mtrt_nvtx_pop(y, range_id=rid_inner)
    y = y + x
    y = mtrt_nvtx_pop(y, range_id=rid_outer)
    return y
result = test_fn(x)
np.testing.assert_allclose(result, np.exp(x) + x, rtol=1e-5)
""",
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
