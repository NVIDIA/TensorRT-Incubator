# Documentation

This directory includes all the source files for the public API documentation.

## Building Documentation Locally

You can build the documentation locally in the development container by running:
```bash
python3 docs/generate_rsts.py
sphinx-build build/doc_sources build/docs -c docs/ -j 4 -W -n
```
To view the documentation, you can open `build/docs/index.html` in a browser.

## Adding Documentation

### How It Works

The `export.public_api()` decorator allows you to specify metadata for documentation
generation, such as where in the documentation hierarchy the API should be documented.

The `generate_rsts.py` script uses this information to automatically generate a directory
structure and populate it with `.rst` files.

For more information, see the docstring for [`export.public_api()`](../tripy/export.py).

### Docstrings

For all public facing docstrings, we have several requirements:

- The function signature must have type annotations for all parameters and return type.

- The docstring must explain what the operation is doing.

- The parameters must be documented in the `Args:` section of the docstring.

- The return value must be documented in the `Returns:` section of the docstring.

- The docstring must include a code example (denoted by `.. code-block:: python`).


### Guides

In addition to the API reference, we also include various guides in subdirectories
of [docs](.).

Each such subdirectory must start with a `pre<N>_` or `post<N>_` prefix, which indicates
the ordering of each set of guides in the index/side bar. Specifically, `pre` indicates
that the guides in that directory should precede the API reference documentation, while
`post` indicates that they should follow it. The number indicates the relative ordering
with respect to other sets of guides. For example, if we have the following directories:

- `pre0_user_guides`
- `pre1_examples`
- `post0_developer_guides`

then the documentation will have the following ordering:

- User Guides
- Examples
- API Reference
- Developer Guides

The markdown files there are included in `.rst` files and parsed by the Myst parser.
This means we need to make some special considerations:

1. We cannot use the `[[TOC]]` directive to automatically generate tables of contents.
    Our Sphinx theme will automatically generate tables of contents, so you can omit these entirely.

2. All links to files in the repository must be absolute and start with `source:` so that
    Myst can replace them with URLs to our remote repository. Otherwise, the links will
    cause the relevant file to be downloaded. For example:
    ```
    [Fill operation](source:/tripy/frontend/trace/ops/fill.py)
    ```

    Links to markdown files are an exception; if a markdown file is part of the *rendered*
    documentation, it should be linked to using the `project:` tag instead.

3. For links to documentation for APIs, you can use the following syntax:

    ```md
    {<api_kind>}`<api_name>`
    ```

    For example:

    ```md
    {class}`tripy.Tensor`
    ```

    `<api_kind>` can take on any value that is a valid role provided by
    [Sphinx's Python domain](https://www.sphinx-doc.org/en/master/usage/domains/python.html).

Guides may use the markers specified in [tests/helper.py](../tests/helper.py) to customize
how the documentation is interpreted (see `AVAILABLE_MARKERS` in that file).


### Code Examples

Code examples in public facing docstrings and guides are preprocessed before
documentation is generated. Specifically:

- Any code examples are executed so that their output can be
    displayed after the code block. Several modules, including `tripy` (as `tp`),
    `numpy` (as `np`), `cupy` (as `cp`), and `torch` are automatically imported
    and can be used in code examples.

- The values of any `tripy` type local variables are appended to the output.
    You can customize this behavior:

    - To only display certain variables, add `# doc: print-locals` followed by a space
        separated list of variable names. For example: `# doc: print-locals inp out`.

    - To only disable certain variables, add `# doc: no-print-locals` followed by a space
        separated list of variable names. For example: `# doc: no-print-locals inp out`.

    - To disable it completely, add just `# doc: no-print-locals` without specifying any variables.

- In docstrings, but not guides, any `assert` statements are stripped out.

- Any lines that end with `# doc: omit` are stripped out.

- By default, documentation generation will fail if an exception is thrown by a code snippet.
    In order to allow exceptions, add `# doc: allow-exception`.

To avoid running code entirely, you can add `# doc: no-eval` in the docstring. Note that this will
not prevent the code block from being executed in the tests.


### Dynamically Generating Content In Guides

In some cases, it's useful to run Python code and include the output in a guide *without* including
the Python code itself. To do so, you can use a trick like this:

```md
    <!-- Tripy: DOC: OMIT Start -->
    ```py
    # doc: no-print-locals
    print("This line should be rendered into the docs")
    ```
    <!-- Tripy: DOC: OMIT End -->
```

This works because `DOC: OMIT` removes the encapsulated text from the post-processed markdown file
but does *not* prevent the block from being evaluated (to do that, use `# doc: no-eval` as normal).

We include special logic that will omit the `Output:` heading when the output
is generated in this way.
