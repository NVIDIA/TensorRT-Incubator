# Documentation

This directory includes all the source files for the public API documentation.
The structure is such that similar or related concepts are grouped under the same
file. For example, `Module` and `Parameter` are both documented under `nn.rst`.
Note that each file renders as a separate page.

## Building Documentation Locally

You can build the documentation locally in the development container by running:
```bash
sphinx-build docs build/docs -j 6 -W
```
To view the documentation, you can open `build/docs/index.html` in a browser.

## Adding Documentation

To add documentation for a new class or function, find the appropriate `.rst` file in
`docs/` or add a new one if none of the existing files are appropriate.
If you added a new file, update `docs/index.rst` to include it.

### Docstrings

The public facing docstrings are preprocessed before documentation is generated.
Specifically, for any code examples in the docstrings, `assert`s are stripped out and
the code is executed so that the output can be displayed as a code block immediately
following the one containing the example. Thus, make sure to include `print`s in your
example code so that the output is helpful to look at.
