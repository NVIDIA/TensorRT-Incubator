# Tests

The tests directory includes both unit and integration tests. For the former, the file
structure is meant to exactly mirror the structure of the code. That means, for example
that `tripy/path/to/<file>.py` will have all of its unit tests in `tests/path/to/test_<file>.py`.
The `tests/integration` directory captures the latter group of tests.

## Running Tests

You can run tests locally in the development container by running:
```bash
pytest tests/ -v
```

## Adding Tests

When modifying or adding new files in `tripy`, make sure that you also modify or add the corresponding
unit test files under `tests`. For integration tests, you can find an appropriate file in
`tests/integration` or create a new one if none of the existing files fit your needs.

### Docstring Tests

For public-facing interfaces, you should add examples in the docstrings.
Avoid doing this for internal interfaces since we do not build documentation for
those anyway.

Any code blocks (i.e. `.. code:: python` followed by a newline and then an indented block)
in docstrings are automatically tested by `tests/test_ux.py`.

For example:
```py
"""
.. code:: python

    <example code>
"""
```

**NOTE: The docstrings must *not* import `tripy` or `numpy`. They will be imported **
    **automatically as `tp` and `np` respectively. Any other modules will need to be imported*.**
