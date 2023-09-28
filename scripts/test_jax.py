"""
This script dumps out JAX MHLO IR for debugging" 
"""
import jax
import jax.numpy as jnp
import jaxlib

x = jnp.zeros((2,))
f = lambda x: x.at[0].get()

# Get JAXExpr IR
jaxpr = jax.make_jaxpr(f)(x)
ir = jax.jit(f).lower(x)

# Get MHLO IR
mhlo = ir.compiler_ir('mhlo')
print(f"MHLO IR: \n{mhlo}")
# # Get optimized_mhlo
compiled_mhlo = ir.compile()
print(f"Compiled MHLO: \n{compiled_mhlo.as_text()}")
