from __future__ import annotations
import jax
import jax.numpy as jnp

from jaxga.signatures import positive_signature
from ..jaxga import reduce_bases
from functools import cache

# https://stackoverflow.com/questions/72265516/permutation-sign-for-batched-vectors-python
@jax.jit
def compute_parity(p):
  return jnp.linalg.det(jax.jacobian(jnp.sort)(p.astype(float))).astype(int)



# TODO this is what needs to be improved
@cache
def get_mv_reduce_same(a_blade_indices):
    blade_to_index = {}
    blade_to_blade_index = {}

    indices = []
    unique_count = len(set(a_blade_indices))
    out_indices = [[] for _ in range(unique_count)]
    out_signs = [[] for _ in range(unique_count)]

    for i, blade_index in enumerate(a_blade_indices):
        blade_index_set = frozenset(blade_index)
        if blade_index_set in blade_to_index:
            index = blade_to_index[blade_index_set]
            sign, _ = reduce_bases(
                blade_index, blade_to_blade_index[blade_index_set], positive_signature
            )
        else:
            index = len(blade_to_index)
            sign = 1
            blade_to_index[blade_index_set] = index
            blade_to_blade_index[blade_index_set] = blade_index
            indices.append(blade_index)
        out_indices[index].append(i)
        out_signs[index].append(sign)

    def _values_mv_reduce_same(a_values):
        out_batch_shape = a_values.shape[1:]
        result = jnp.empty([len(out_indices), *out_batch_shape], dtype=jnp.float32)
        for i, (mm, signs) in enumerate(zip(out_indices, out_signs)):
            for j, (m, sign) in enumerate(zip(mm, signs)):
                if j == 0:
                    result = result.at[i].set(sign * a_values[m])
                else:
                    result = result.at[i].add(sign * a_values[m])
        return result

    _values_mv_reduce_same_jit = jax.jit(_values_mv_reduce_same)

    return _values_mv_reduce_same_jit, tuple(indices)

# def compute_parity(p):
#     return jnp.linalg.det(jax.jacobian(jnp.sort)(p.astype(float))).astype(int)

# def compute_parity(p):
#   return jnp.linalg.det(jax.jacobian(jnp.sort)(p.astype(float))).astype(int)



# In [13]: p=jax.jit(compute_parity)
# 51.3 µs ± 57.6 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
