## Plan: Migrate Operators to New Constraint System

This plan will systematically port operators to use the `input_requirements`/`output_guarantees` constraint system. The migration follows proven patterns from [`cast.py`](nvtripy/frontend/ops/cast.py), [`concatenate.py`](nvtripy/frontend/ops/concatenate.py), and [`ones.py`](nvtripy/frontend/ops/ones.py).

### Steps

1. **Migrate simple type-preserving unary operators** (~16 ops in [`nvtripy/frontend/ops/unary/`](nvtripy/frontend/ops/unary/)): Convert operators like `exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `sigmoid`, `relu`, `gelu`, `silu`, `abs`, `neg`, `invert` using pattern: `input_requirements=OneOf(GetInput("input").dtype, [dtypes])` and `output_guarantees=GetReturn(0).dtype == GetInput("input").dtype`. Run tests with `pytest tests/integration/test_operator_constraints.py -k "exp|log|sqrt" -s` and corresponding sanity tests. **Commit changes** with message like "Migrate unary operators to new constraint system".

2. **Migrate shape manipulation operators** (~12 ops in [`nvtripy/frontend/ops/`](nvtripy/frontend/ops/)): Convert `reshape`, `transpose`, `permute`, `flatten`, `squeeze`, `unsqueeze`, `expand`, `repeat`, `stack`, `slice`, `flip` using the same type-preserving pattern. Test with `pytest tests/integration/test_operator_constraints.py -k "reshape|transpose|permute" -s`. **Commit changes** with message like "Migrate shape manipulation operators to new constraint system".

3. **Migrate binary arithmetic and comparison operators** (~16 ops in [`nvtripy/frontend/ops/binary/`](nvtripy/frontend/ops/binary/)): Convert `add`, `sub`, `mul`, `div`, `floor_div`, `mod`, `pow`, `maximum`, `minimum`, `equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal`, `logical_or` using pattern: `OneOf(GetInput("self").dtype, [...]) & (GetInput("other").dtype == GetInput("self").dtype)`. Comparison ops return `dt.bool`. Test with `pytest tests/integration/test_operator_constraints.py -k "add|mul|equal" -s`. **Commit changes** with message like "Migrate binary operators to new constraint system".

4. **Migrate reduction operators** (~11 ops in [`nvtripy/frontend/ops/reduce/`](nvtripy/frontend/ops/reduce/)): Convert `sum`, `prod`, `mean`, `max`, `min`, `var`, `all`, `any`, plus multi-type ops `argmax`, `argmin`, `topk` (note: `topk` has multiple returns). Use `If` constraints for multi-type variables. Test with `pytest tests/integration/test_operator_constraints.py -k "sum|mean|argmax" -s`. **Commit changes** with message like "Migrate reduction operators to new constraint system".

5. **Migrate initializers with optional dtype parameters** (~8 ops in [`nvtripy/frontend/ops/`](nvtripy/frontend/ops/)): Convert `zeros`, `zeros_like`, `full`, `full_like`, `iota`, `arange` using `If(GetInput("dtype") != None, ...)` pattern for conditional output guarantees. Test with `pytest tests/integration/test_operator_constraints.py -k "zeros|full|iota|arange" -s`. **Commit changes** with message like "Migrate initializer operators to new constraint system".

6. **Migrate advanced operations and special cases** (~15 ops): Convert `matmul`, `gather`, `where`, `masked_fill`, `outer`, `pad`, `softmax`, `cumsum`, `copy`, `resize`, `triu`, `tril`, `avgpool`, `maxpool` in their respective files. Handle multi-type variables and complex logic. Migrate `quantize`/`dequantize` in [`plugin_qdp.py`](nvtripy/frontend/ops/plugin_qdp.py) and [`quantize.py`](nvtripy/frontend/ops/quantize.py)/[`dequantize.py`](nvtripy/frontend/ops/dequantize.py) with coordinated type pairs. Test each with `-k` filtering. **Commit changes** with message like "Migrate advanced operators to new constraint system".

### Implementation Notes

- Translate any existing exception logic directly to boolean logic using `&`, `|`, `~` operators without creating helper functions
