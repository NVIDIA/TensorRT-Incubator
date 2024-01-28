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

Code examples in public facing docstrings are preprocessed before
documentation is generated. Specifically:

1. Any code examples in the docstrings are executed so that their output can be
    displayed after the code block. Several modules, including `tripy` (as `tp`),
    `numpy` (as `np`) and `torch` are automatically imported and can be used in
    code examples.

2. The values of any `tripy` type local variables are appended to the output.
    You can customize this behavior:

    - To disable it completely, add `# doc: no-print-locals` as a separate line
        at the top of the code block.

    - To only display certain variables, add `# doc: print-locals` followed by a space
        separate list of variable names. For example: `# doc: print-locals inp out`.

3. Any `assert` statements are stripped out.

4. Any lines that end with `# doc: omit` are stripped out.
