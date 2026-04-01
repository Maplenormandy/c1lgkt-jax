"""
This module contains tools for specifying initial conditions in a reproducible manner
"""

from typing import NamedTuple


import jax.numpy as jnp
import jax

import equinox as eqx

import abc

from ..particles.particle_motion import PusherState, PusherArgs

# %% Abstract base class for initial condition provider

class AbstractStateProvider(eqx.Module):
    """
    Abstract base class for initial condition providers.
    """
    
    @abc.abstractmethod
    def as_state(self) -> tuple[float, PusherState]:
        """
        Return the initial time and initial state (t0, y0).
        """
        raise NotImplementedError


# %% 
