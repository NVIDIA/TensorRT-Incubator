# Documentation

This directory includes all the source files for the public API documentation.
The structure is such that similar or related concepts are grouped under the same
file. For example, `Module` and `Parameter` are both documented under `nn.rst`.

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

Each `.rst` file will render as a separate page.

### Docstrings

For all public facing docstrings, we have several requirements:

- The function signature must have type annotations for all parameters and return type.

- The docstring must explain what the operation is doing.

- The parameters must be documented in the `Args:` section of the docstring.

- The return value must be documented in the `Returns:` section of the docstring.

- The docstring must include a code example (denoted by `.. code-block:: python`).


#### Docstring Code Examples

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
        separated list of variable names. For example: `# doc: print-locals inp out`.

3. Any `assert` statements are stripped out.

4. Any lines that end with `# doc: omit` are stripped out.


### Developer Documentation

In addition to the API reference, we also include documentation for developers of Tripy
in [docs/development](/docs/development/). The markdown files there are included in `.rst`
files and parsed by the Myst parser. This means we need to make some special considerations:

1. We cannot use the `[[TOC]]` directive to automatically generate tables of contents.
    Instead, use:

        ```{contents} Table of Contents
        :depth: 3
        ```

2. All links to files in the repository must be absolute and start with `source:` so that
    Myst can replace them with URLs to our remote repository. Otherwise, the links will
    cause the relevant file to be downloaded.

    Links to markdown files are an exception; if a markdown file is linked without `source:`,
    it will point to the corresponding page in the rendered documentation (if present).
