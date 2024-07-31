# Tests

The tests directory includes both unit and integration tests. For the former, the file
structure is meant to exactly mirror the structure of the code. That means, for example
that `tripy/path/to/<file>.py` will have all of its unit tests in `tests/path/to/test_<file>.py`.
The `tests/integration` directory captures the latter group of tests.


## Running Tests

You can run all tests locally in the development container by running:
```bash
pytest tests/ -v
```

You can also provide marker arguments to only run specific test cadences
(see [the test cadence section](#test-cadence) below). For example, to run only
L0 tests, use:

```bash
pytest tests/ -v -m "l0 or not l1"
```

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

The markers we currently support are:

- `l0`: Indicates that the test should be run in each merge request.
        This marker is applied by default if no other markers are present.

- `l1`: Indicates that the test should be run at a nightly cadence.


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
