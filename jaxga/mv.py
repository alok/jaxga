from __future__ import annotations

from typing import Any, Callable, Literal, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

from .jaxga import mv_repr, reverse_indices
from .ops.add import get_mv_add
from .ops.dual import get_mv_dual
from .ops.inverse import get_mv_inverse
from .ops.keepnonzero import get_mv_keep_nonzero
from .ops.multiply import get_mv_multiply
from .ops.reduce_same import get_mv_reduce_same
from .ops.sandwich import get_mv_sandwich
from .ops.select import get_mv_select
from .ops.simple_exp import get_mv_simple_exp
from .signatures import positive_signature

# TODO register mv as pytree

# TODO context manager to handle quadratic form?
# TODO preserve even/odd split
Signature = str
@jax.jit
def parity(permutation: jnp.ndarray) -> tuple[jnp.ndarray, int]:
    """A linear algebra approach to the parity of a permutation."""
    # TODO sort only once
    sorted_perm = jnp.sort(permutation)
    sign = jnp.linalg.det(jax.jacobian(jnp.sort)(permutation.astype(float))).astype(int)
    return sorted_perm, sign
@jdc.pytree_dataclass
class Blade:
    indices: jnp.ndarray[int]
    value: jnp.floating


# TODO look into dropping quadratic form (sig) or attaching it somehow with a decorator
@jdc.pytree_dataclass
class MV:
    values: jnp.ndarray
    indices: tuple[int]
    signature: Signature

class MultiVector:
    def __init__(
        self,
        values: jnp.array,
        indices: Sequence[int],
        signature: Callable[[Union[Tuple[()], int]], int] = positive_signature,
    ) -> None:
        self.values = values
        self.indices = tuple(indices)
        self.signature = signature

    def e(*indices: Sequence[int], **kwargs: dict[str, Any]):
        signature = kwargs["signature"] if "signature" in kwargs else positive_signature
        batch_shape = (
            ((1,) + tuple(kwargs["batch_shape"])) if "batch_shape" in kwargs else (1,)
        )
        return MultiVector(
            values=jnp.ones(batch_shape, dtype=jnp.float32),
            indices=(tuple(indices),),
            signature=signature,
        )

    def __add__(self, other):
        if not isinstance(other, MultiVector):
            other = MultiVector.e() * other

        mv_add, out_indices = get_mv_add(self.indices, other.indices)
        out_values = mv_add(self.values, other.values)
        return MultiVector(
            values=out_values, indices=out_indices, signature=self.signature
        )

    def __radd__(self, other):
        if not isinstance(other, MultiVector):
            other = MultiVector.e() * other

        mv_add, out_indices = get_mv_add(other.indices, self.indices)
        out_values = mv_add(other.values, self.values)
        return MultiVector(
            values=out_values, indices=out_indices, signature=self.signature
        )

    def __sub__(self, other):
        if not isinstance(other, MultiVector):
            other = MultiVector.e() * other

        mv_add, out_indices = get_mv_add(self.indices, other.indices)
        out_values = mv_add(self.values, -other.values)
        return MultiVector(
            values=out_values, indices=out_indices, signature=self.signature
        )

    def __rsub__(self, other):
        if not isinstance(other, MultiVector):
            other = MultiVector.e() * other

        mv_add, out_indices = get_mv_add(other.indices, self.indices)
        out_values = mv_add(other.values, -self.values)
        return MultiVector(
            values=out_values, indices=out_indices, signature=self.signature
        )

    def __mul__(self, other):
        if isinstance(other, MultiVector):
            mv_multiply, out_indices = get_mv_multiply(
                self.indices, other.indices, self.signature
            )
            out_values = mv_multiply(self.values, other.values)
            return MultiVector(
                values=out_values, indices=out_indices, signature=self.signature
            )
        return MultiVector(
            values=self.values * other, indices=self.indices, signature=self.signature
        )

    def __rmul__(self, other):
        if isinstance(other, MultiVector):
            mv_multiply, out_indices = get_mv_multiply(
                other.indices, self.indices, self.signature
            )
            out_values = mv_multiply(other.values, self.values)
            return MultiVector(values=out_values, indices=out_indices)
        return MultiVector(
            values=self.values * other, indices=self.indices, signature=self.signature
        )

    def __xor__(self, other):
        mv_multiply, out_indices = get_mv_multiply(
            self.indices, other.indices, self.signature, "op"
        )
        out_values = mv_multiply(self.values, other.values)
        return MultiVector(
            values=out_values, indices=out_indices, signature=self.signature
        )

    def __or__(self, other):
        mv_multiply, out_indices = get_mv_multiply(
            self.indices, other.indices, self.signature, "ip"
        )
        out_values = mv_multiply(self.values, other.values)
        return MultiVector(
            values=out_values, indices=out_indices, signature=self.signature
        )

    def __invert__(self):
        return MultiVector(
            values=self.values,
            indices=reverse_indices(self.indices),
            signature=self.signature,
        )

    def __neg__(self):
        return MultiVector(
            values=-self.values, indices=self.indices, signature=self.signature
        )

    def sandwich(self, other):
        mv_sandwich, out_indices = get_mv_sandwich(
            self.indices, other.indices, self.signature
        )
        out_values = mv_sandwich(self.values, other.values)
        return MultiVector(
            values=out_values, indices=out_indices, signature=self.signature
        )

    def inverse(self):
        mv_inv, inv_indices = get_mv_inverse(self.indices, self.signature)
        inv_values = mv_inv(self.values)
        return MultiVector(
            values=inv_values, indices=inv_indices, signature=self.signature
        )

    def __truediv__(self, other):
        if isinstance(other, MultiVector):
            return self * other.inverse()
        return self * (1 / other)

    def __rtruediv__(self, other):
        return other * self.inverse()

    def __repr__(self):
        return mv_repr(self.indices, self.values)

    def __getitem__(self, select_indices):
        mv_select, out_indices = get_mv_select(self.indices, select_indices)
        out_values = mv_select(self.values)
        return MultiVector(out_values, out_indices, signature=self.signature)

    def simple_exp(self):
        mv_simple_exp, out_indices = get_mv_simple_exp(self.indices, self.signature)
        out_values = mv_simple_exp(self.values)
        return MultiVector(out_values, out_indices, signature=self.signature)

    def keep_nonzero(self):
        mv_keep_nonzero, out_indices = get_mv_keep_nonzero(self.indices, self.values)
        out_values = mv_keep_nonzero(self.values)
        return MultiVector(out_values, out_indices, signature=self.signature)

    def reduce_same(self):
        mv_reduce_same, out_indices = get_mv_reduce_same(self.indices)
        out_values = mv_reduce_same(self.values)
        return MultiVector(out_values, out_indices, signature=self.signature)

    def dual(self, dims):
        mv_dual, out_indices = get_mv_dual(self.indices, dims)
        out_values = mv_dual(self.values)
        return MultiVector(out_values, out_indices, signature=self.signature)

    def unitize(self):
        # TODO
        return self / self.weight()

    def weight(self):
        return sqrt(self.antidot(self.antireverse()))

    def bulk(self):
        return sqrt(self.dot(self.reverse()))

    def geometric_norm(self):
        return self.bulk() + self.weight()

    def has_geom_property(self):
        return self.dot(self.reverse()) == self * self.reverse() and self.antidot(
            self.antireverse()
        ) == self.antiprod(self, self.antireverse())

    def adjoint():
        return self.reverse().involute()

    def left_complement(self):
        return self.reverse().antiprod(antiscalar(self.signature)) + One().antiprod(
            self.reverse()
        )

    # TODO define sin/cos/tan/pow using https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4361175/


# The spin representation of the twofold cover of an odd orthogonal group, the
# odd spin group, and the two half-spin representations of the twofold cover of
# an even orthogonal group, the even spinor group, are fundamental
# representations that *cannot* be realized in the space of tensors.
