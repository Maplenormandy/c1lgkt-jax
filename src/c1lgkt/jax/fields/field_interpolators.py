# %% -*- coding: utf-8 -*-
"""
@author: maple

This file contains type definitions for field interpolators
"""

import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import interpax

import abc
from typing import NamedTuple

from jaxtyping import ArrayLike, Real

# %% Allowable field signatures

class RotatingFrame(NamedTuple):
    omega: Real  # rotation frequency
    t0: Real     # reference time

class ZonalField(NamedTuple):
    interp: interpax.Interpolator1D    # Interpolator1D over psi

class BallooningField(NamedTuple):
    n: int                             # toroidal mode number
    frame: RotatingFrame               # rotating frame parameters
    interp: interpax.Interpolator2D    # Interpolator2D over (psi, theta)

class HelicalField(NamedTuple):
    n: int                             # toroidal mode number
    m: int                             # poloidal mode number
    frame: RotatingFrame               # rotating frame parameters
    interp: interpax.Interpolator1D    # Interpolator1D over psi

class FieldProvider(eqx.Module):
    @abc.abstractmethod
    def compute_fields(
            self,
            t, state,
            psi_and_ff,
            geom_terms,
            ):
        pass  # to be implemented by subclasses