# Documentation

This directory contains **source** files for public guides and **configuration**
files for the public documentation.


## Building Documentation Locally

In the development container, run:

```bash
python3 docs/generate_rsts.py
sphinx-build build/doc_sources build/docs -c docs/ -j 4 -W -n
```

- To **view** the docs, launch an HTTP server *outside* the container.
    From the [tripy root directory](../), run:

    ```bash
    python3 -m http.server 8001 --directory build/docs
    ```

    Then navigate to http://localhost:8001 in a browser.


## How It Works

1. [`generate_rsts.py`](./generate_rsts.py) creates `.rst` files based on public APIs
    and guides (details below).

2. Sphinx generates HTML documentation based on the `.rst`s.


### API Documentation

Public APIs are indicated by the [`export.public_api()`](../nvtripy/export.py) decorator,
which specifies doc metadata for each API (e.g. location).

**Requirements:**

- Signature must have **type annotations**.

- All parameters and return values must be documented.

- Docstring must include *at least* **one [code example](#code-examples)**.

- If the function accepts `tp.Tensor`s, must indicate **data type constraints**
    with the [`wrappers.interface`](../nvtripy/frontend/wrappers.py) decorator.

**Example:**

```py
@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "int4", "int32", "int64", "bool", "int8"],
    },
)
def relu(input: "nvtripy.Tensor") -> "nvtripy.Tensor":
    r"""
    Applies Rectified Linear Unit (RELU) function
    to each element of the input tensor:

    :math:`\text{relu}(x) = \max(0,x)`

    Args:
        input: The input tensor.

    Returns:
        A tensor of the same shape as the input.

    .. code-block:: python
        :linenos:

        input = tp.Tensor([1., 2., 3., 4.])
        output = tp.relu(input)

        t = torch.tensor([1, 2, 3, 4], dtype=torch.float32) # doc: omit
        assert tp.allclose(output, tp.Tensor(torch.nn.functional.relu(t)))
    """
```


### Guides

**Guides** are included in various subdirectories of [docs](.) for longer-form
content, e.g. **workflows** or **concepts**.

- The `pre<N>_` or `post<N>_` directory prefix indicates ordering of guide sets;
    `pre` guides **precede** the API reference, `post` guides **follow** it.

Guides are **markdown** files parsed by the Myst parser, therefore:

- Do **not** use `[[TOC]]` or include a table of contents manually; one will be generated automatically.

- Links to files must be **absolute** paths and use a `source:` tag.

    - **Exception:** links to **rendered** markdown files (e.g. other guides) should use a `project:` tag.

    - **Example:**

        ```md
        [Broadcast operation](source:/nvtripy/trace/ops/broadcast.py)
        ```

    - **Why:** Other links will cause the file to be downloaded instead of linking to the repository.

- Links to API documentation should use the syntax:

    ```md
    {<api_kind>}`<fully_qualified_api_name>`
    ```

    `<api_kind>` should be a role from
    [Sphinx's Python domain](https://www.sphinx-doc.org/en/master/usage/domains/python.html).

    - **Example:**

    ```md
    {class}`nvtripy.Tensor`
    ```

#### Tip: Dynamically Generating Content

You can embed code whose **output** is included but whose **content** is excluded from the guide
using the `DOC: OMIT` marker:

```md
    <!-- Tripy: DOC: OMIT Start -->
    ```py
    # doc: no-print-locals
    print("This line should be rendered into the docs")
    ```
    <!-- Tripy: DOC: OMIT End -->
```

- **See also:** `AVAILABLE_MARKERS` in [tests/helper.py](../tests/helper.py) for a complete list of markers.


### Code Examples

Code blocks in docstrings/guides are **preprocessed**:

- Some modules are **automatically imported** - do **not** import these manually:
    - `nvtripy` (as `tp`)
    - `numpy` (as `np`)
    - `cupy` (as `cp`)
    - `torch`

- In docstrings, but **not** guides, `assert` statements are removed.

- Lines ending with `# doc: omit` are stripped out.

- Code is **executed** and any output is displayed in the docs.

    - `# doc: allow-exception` allows exceptions to be thrown. By default, they are treated as failures.

    - `# doc: no-output` omits output from the docs (but still executes the code).

    - `# doc: no-eval` disables execution but this means the code will be **untested**!

    - `# doc: ignore-line` disables execution of the indicated line but still includes it in the rendered code.

- Local variables are also displayed. You can customize this:

    - **Include** only specific variables: `# doc: print-locals <var1> <var2> ...`
    - **Exclude** *specific* variables: `# doc: no-print-locals <var1> <var2> ...`
    - **Exclude** *all* variables: `# doc: no-print-locals` (with no arguments).
