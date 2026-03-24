---
name: tripy-documentation
description: 'Write API documentation for nvtripy following project conventions. Use when: writing docstrings for ops or modules, adding code examples, using @export.public_api document_under paths, creating Sphinx RST cross-references, understanding the docs build pipeline.'
---

# API Documentation for nvtripy

## When to Use

- Writing or updating docstrings for public API functions, classes, or modules
- Adding working code examples to documentation
- Choosing the correct `document_under` path for `@export.public_api`
- Understanding how documentation is generated and built

## Documentation Pipeline

1. **`@export.public_api(document_under="...")`** registers APIs in `PUBLIC_APIS` list
2. **`docs/generate_rsts.py`** reads `PUBLIC_APIS` and generates `.rst` files in the docs hierarchy
3. **Sphinx** builds the final HTML docs from those `.rst` files
4. Docstring code examples are extracted and validated during testing

## `@export.public_api` Parameters

```python
@export.public_api(
    document_under="operations/functions",   # Doc hierarchy path
    autodoc_options=[":special-members:"],    # Sphinx autodoc options
    bypass_dispatch=True,                     # Skip function registry overhead
)
```

### `document_under` Path Conventions

| Path | Use For |
|------|---------|
| `"operations/functions"` | General tensor operations (softmax, reshape, etc.) |
| `"operations/initializers"` | Tensor creation (ones, zeros, full, arange) |
| `"operations/modules"` | Neural network modules (Linear, LayerNorm, etc.) |
| `"compiling_code/compile.rst"` | Compilation-related APIs |
| `"compiling_code/input_info/index.rst"` | InputInfo and related classes |
| `"config.rst"` | Configuration variables |

The path creates a directory structure: `"operations/functions"` → `operations/functions/<name>.rst`.

APIs targeting the same `.rst` file render on the same page.

### `autodoc_options`

- `[":special-members:"]` — Include `__init__`, `__call__`, etc.
- `[":no-members:", ":no-special-members:"]` — Show only the class/module itself
- `[":no-value:"]` — Hide the default value of a variable

### `bypass_dispatch`

- `True` on a function: Disables the function registry's overload dispatch and type-checking (performance optimization)
- `True` on a class: Bypass dispatch for ALL methods
- `["__init__", "__call__"]`: Bypass only for listed methods

## Docstring Format

### Functions

```python
def my_op(input: "nvtripy.Tensor", dim: int = 0) -> "nvtripy.Tensor":
    r"""
    Brief one-line description of what the function does.

    Longer description with math if applicable:

    :math:`\text{my_op}(x) = f(x)`

    Args:
        input: The input tensor.
        dim: The dimension to operate along.

    Returns:
        A tensor of the same shape as the input.

    .. code-block:: python
        :linenos:

        input = tp.iota([2, 3], dtype=tp.float32)
        output = tp.my_op(input, dim=0)

        assert tp.allclose(output, expected)

    .. seealso:: :func:`related_func`, :class:`RelatedClass`
    """
```

### Classes (Modules)

```python
class MyModule(Module):
    r"""
    Brief description with math notation.

    :math:`\text{MyModule}(x) = xW^T + b`
    """

    dtype: datatype.dtype
    r"""The data type used to perform the operation."""

    weight: Tensor
    r"""The :math:`W` parameter of shape :math:`[\text{out}, \text{in}]`."""

    def __init__(self, features: int, dtype: datatype.dtype = datatype.float32) -> None:
        r"""
        Args:
            features: Size of the feature dimension.
            dtype: The data type for parameters.

        .. code-block:: python
            :linenos:

            module = tp.MyModule(3)
            module.weight = tp.iota(module.weight.shape)

            input = tp.iota((2, 3))
            output = module(input)

            assert cp.from_dlpack(output).get().shape == (2, 3)

            torch_out = torch.nn.functional.my_op(torch.from_dlpack(input))  # doc: omit
            assert np.allclose(cp.from_dlpack(output).get(), cp.from_dlpack(torch_out).get())
        """
```

## Code Example Conventions

### Required elements

1. **Setup**: Create inputs using `tp.iota()`, `tp.ones()`, `tp.zeros()`, or `tp.Tensor()`
2. **Operation**: Call the function/module under test
3. **Assertion**: Verify the result with `assert` (using `tp.allclose`, `np.array_equal`, or shape checks)

### Available imports in code blocks

Code examples automatically have access to:
- `tp` (nvtripy)
- `np` (numpy)
- `cp` (cupy)
- `torch`

### Special directives

- `# doc: omit` — Line is excluded from rendered documentation but still executes
- `# doc: no-print-locals <var>` — Suppresses automatic printing of the variable
- `:linenos:` — Always include for numbered lines

### Cross-references

- `:func:`function_name`` — Link to a function
- `:class:`ClassName`` — Link to a class
- `:math:`expression`` — Inline LaTeX math
- `.. seealso:: :func:`related`" — "See also" section at the end

### Math notation

Use `r"""` raw strings for docstrings containing `:math:` to avoid backslash issues.

LaTeX examples:
- Inline: `:math:`\text{softmax}(x_{i})``
- Block: Use `\Large` / `\normalsize` for fraction sizing
- Common: `:math:`\bar{x}`` (mean), `:math:`\sigma^2`` (variance), `:math:`\epsilon`` (epsilon)

## Checklist

- [ ] `@export.public_api(document_under="...")` with correct hierarchy path
- [ ] Docstring uses `r"""` if it contains `:math:` directives
- [ ] Args section documents all parameters with types
- [ ] Returns section describes output shape/type
- [ ] `.. code-block:: python` with `:linenos:` and working assertions
- [ ] `.. seealso::` links to related functions/classes
- [ ] Field docstrings for all dataclass fields on modules
- [ ] `# doc: omit` for verification-only lines that shouldn't appear in docs
