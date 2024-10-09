# Tests

The tests directory includes both unit and integration tests. For the former, the file
structure is meant to exactly mirror the structure of the code. That means, for example
that `tripy/path/to/<file>.py` will have all of its unit tests in `tests/path/to/test_<file>.py`.
The `tests/integration` directory captures the latter group of tests.


## Running Tests

You can run all tests locally in the development container by running:
```bash
pytest tests/ -v -n 4 --dist worksteal --ignore tests/performance
pytest tests/performance -v
```

Performance tests are run separately because they must run serially to ensure
accurate measurements.

You can also provide marker arguments to only run specific test cadences
(see [the test cadence section](#test-cadence) below). For example, to run only
L0 tests, use:

```bash
pytest tests/ -v -m "not l1 and not manual" -n 4 --dist worksteal --ignore tests/performance
pytest tests/performance -v -m "not l1 and not manual"
```


## Profiling

You can profile test runtimes in the development container using the
`--profile` option, which will generate `pstats` files for each test
in a `prof/` directory, along with a `combined.prof` file for all the
tests together.

For example, to profile L0 tests, run:

```bash
pytest tests/ -v -m "not l1 and not manual" --profile
```

You can visualize the results using `snakeviz`.

*NOTE: Ensure that you launched the development container with port forwarding,*
*i.e. the `-p 8080:8080` option.*

For example:

```bash
snakeviz prof/combined.prof -s --hostname 0.0.0.0
```

Then, in a browser, navigate to:
http://localhost:8080/snakeviz/%2Ftripy%2Fprof%2Fcombined.prof



## Coverage Reports

You can generate code coverage reports locally by running:

```bash
pytest --cov=tripy/ --cov-report=html --cov-config=.coveragerc tests/ -v
```

To view the report, open the `htmlcov/index.html` file from the root directory in a browser.


## Dead Code Detection

Our development container includes a static analysis tool called `vulture` that can
detect dead code. *This **will** include false positives for our code, so be careful!*

You can run it with:

```bash
vulture tripy tests --sort-by-size
```

To exclude false positives, use:

```bash
vulture tripy tests --sort-by-size --min-confidence=100
```


## Adding Tests

When modifying or adding new files in `tripy`, make sure that you also modify or add the corresponding
unit test files under `tests`. For integration tests, you can find an appropriate file in
`tests/integration` or create a new one if none of the existing files fit your needs.

### Test Cadence

We don't necessarily want to run every test in every single pipeline. You can use special
pytest markers to indicate the cadence for a test. For example:

<!-- Tripy: TEST: IGNORE Start -->

```py
@pytest.mark.l1
def test_really_slow_things():
    ...
```

<!-- Tripy: TEST: IGNORE End -->

Supported markers are documented in [pyproject.toml](../pyproject.toml).

### Docstring Tests

For public-facing interfaces, you should add examples in the docstrings.
Avoid doing this for internal interfaces since we do not build documentation for
those anyway.

Any code blocks in docstrings are automatically tested by `tests/test_ux.py`.
The code block format is:
```py
"""
.. code-block:: python
    :linenos:
    :caption: Descriptive Title

    <example code>
"""
```

Any caption other than `Example` will have a prefix of `Example: ` prepended to it.

**NOTE: The docstrings must *not* import `tripy`, `numpy`, or `torch`. They will be imported**
    **automatically as `tp`, `np`, and `torch` respectively. Any other modules will need to be imported.**


### Performance Tests

In addition to functional tests, we also run performance tests of three kinds:

1. Regression tests, which compare current Tripy performance to historical data
    to ensure we don't regress. We use the
    [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest/)
    plugin to gather data and the
    [Continuous Benchmark GitHub Action](https://github.com/marketplace/actions/continuous-benchmark)
    for regression testing.

    You can view graphs and charts of the historical data by opening the
    [`index.html` file from the `benchmarks` branch](https://github.com/NVIDIA/TensorRT-Incubator/blob/benchmarks/dev/bench/index.html)
    in a browser.

2. Comparative tests, which compare Tripy and `torch.compile`.

3. Overhead tests, which check the overhead introduced by Tripy as compared
    to running the underlying MLIR executable by itself. This is done by measuring
    how long it takes to run an empty executable since in that case, all the time
    is taken by the Tripy wrapper code.
