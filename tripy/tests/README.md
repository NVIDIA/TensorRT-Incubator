# Tests

## Test Types

- **Unit tests** mirror the structure of the code.

    - **Example:** Tests for [`nvtripy/frontend/tensor.py`](../nvtripy/frontend/tensor.py)
        are located in [`tests/frontend/test_tensor.py`](../tests/frontend/test_tensor.py).

- **Integration tests** are under [`tests/integration`](../tests/integration/).

- **Performance tests** are under [`tests/performance`](../tests/performance/). There are 3 kinds:

    - Regression tests, using
        [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest/)
        and the
        [Continuous Benchmark GitHub Action](https://github.com/marketplace/actions/continuous-benchmark).

        - **Tip:** View historical perf charts by opening
        [`index.html` from the `benchmarks` branch](https://github.com/NVIDIA/TensorRT-Incubator/blob/benchmarks/dev/bench/index.html)
        in a browser.

    - Comparative tests between Tripy and `torch.compile`.

    - Overhead tests to measure runtime overheads vs. running the MLIR executable directly.


## Running Tests

To run all tests:

```bash
pytest tests/ -v
```

To run L0 tests:

```bash
pytest tests/ -v -m "not l1 and not l1_release_package" -n 4 --dist worksteal --ignore tests/performance
pytest tests/performance -v -m "not l1 and not l1_release_package"
```

- **Note:** Performance tests are run separately because they must run serially.


## Test Cadence

Use pytest markers to indicate which job a test should run in.

- The default is **L0**, i.e. the pull request pipeline.

- Markers will overwrite the implicit L0 marker but are otherwise additive.

<!-- Tripy: TEST: IGNORE Start -->
- **Example:** Marking tests to run in **L1** (nightly):

    ```py
    @pytest.mark.l1
    def test_really_slow_things():
        ...
    ```
<!-- Tripy: TEST: IGNORE End -->

- **See also:** [pyproject.toml](../pyproject.toml) for all supported markers.


## Profiling

The `--profile` option will generate profiling data under `prof/`:

- `pstats` file for each test
- `combined.prof` for all tests together

**Example:** To profile L0 functional tests:

```bash
pytest tests/ -v -m "not l1 and not l1_release_package" --ignore tests/performance --profile
```

Visualize the results using `snakeviz`.

- **Note:** Ensure you launched the container with port forwarding (`-p 8080:8080`).

**Example:**

```bash
snakeviz prof/combined.prof -s --hostname 0.0.0.0
```

Navigate to:
http://localhost:8080/snakeviz/%2Ftripy%2Fprof%2Fcombined.prof
in a browser.


## Coverage Reports

To generate code coverage reports:

```bash
pytest --cov=nvtripy/ --cov-report=html --cov-config=.coveragerc tests/ -v
```

You can view the report at `htmlcov/index.html` in a browser.


## Dead Code Detection

Use `vulture` to detect dead code.

- **Note:** This **will** include false positives!

```bash
vulture . --sort-by-size
```

Exclude false positives (but also some true positives) with:

```bash
vulture . --sort-by-size --min-confidence=100
```
