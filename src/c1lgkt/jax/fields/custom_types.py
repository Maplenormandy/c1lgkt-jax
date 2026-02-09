"""
Module defining custom types used in the field providers and particle tools modules.
"""
from jaxtyping import Real, Complex, Array, ArrayLike, PyTree

# %% Custom types for fields

type ScalarArray = Real[Array, "Nq"]
type ScalarArrayLike = Real[ArrayLike, "Nq"]
type VectorArray = Real[Array, "3 Nq"]
type VectorTuple = tuple[Real[Array, "Nq"], Real[Array, "Nq"], Real[Array, "Nq"]]
