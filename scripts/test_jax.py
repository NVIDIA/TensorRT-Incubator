"""
This script dumps out JAX MHLO IR for debugging"
"""
import jax
import jax.numpy as jnp


def func(x):
    return x.at[0].get()


x = jnp.zeros((2,))
# Get JAXExpr IR
jaxpr = jax.make_jaxpr(func)(x)
ir = jax.jit(func).lower(x)

# Get MHLO IR
mhlo = ir.compiler_ir("mhlo")
print(f"MHLO IR: \n{mhlo}")
# # Get optimized_mhlo
compiled_mhlo = ir.compile()
print(f"Compiled MHLO: \n{compiled_mhlo.as_text()}")
