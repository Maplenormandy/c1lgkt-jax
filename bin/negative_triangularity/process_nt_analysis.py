# %% Import JAX and enable 64-bit precision
import jax
jax.config.update("jax_enable_x64", True)


# %% Import other libraries
import equinox as eqx
import interpax
import diffrax
import optimistix as optx

import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
from dataclasses import dataclass
from functools import partial
import numpy as np
from collections import namedtuple
from typing import NamedTuple
import matplotlib.pyplot as plt
import os

from netCDF4 import Dataset

from jaxtyping import ArrayLike, Real
from tqdm import tqdm

import pickle

# %% Load my own libraries
from c1lgkt.jax.fields.equilibrium import Equilibrium
from c1lgkt.jax.fields.clebsch import ClebschMappingBuilder
from c1lgkt.jax.fields.field_providers import EikonalFieldProvider, ZonalFieldProvider
import c1lgkt.jax.particles.particle_motion as particle_motion
import c1lgkt.jax.particles.particle_tools as particle_tools


# %% Load equilibrium and the punctures file
eq = Equilibrium.from_gfile('./scratch/data/g193802.04490')

with open('./scratch/outputs/puncs.pkl', 'rb') as f:
    ppuncs, npuncs = pickle.load(f)

# %% Process the punctures

omega_frame = 2*jnp.pi*20.0 / 7

pp = particle_motion.elec

def process_puncture(punc: particle_tools.PunctureData):
    """
    Takes a PunctureData and returns pairs ((Lphi0, varphi0, Lphi1, varphi1)) that give the
    action of the Poincare map
    """
    
    t = punc.tp
    r, varphi, z, upar, mu = punc.yp

    num_punc = len(t) # pyright: ignore

    # Varphi in the moving frame
    varphi_frame = varphi - omega_frame * t

    # Pad r and z with zeros so JAX doesn't recompile
    r_pad = jnp.pad(r, (0, 64 - num_punc))
    z_pad = jnp.pad(z, (0, 64 - num_punc))
    psi_pad = eq.interp_psi(r_pad, z_pad)
    psi = psi_pad[:num_punc]
    bv_pad = eq.compute_bv(r_pad, z_pad)
    modb_pad = jnp.linalg.norm(bv_pad, axis=0)


    # Compute p_parallel; assume apar = 0
    ppar = pp.m * upar
    lphi = pp.z * psi + ppar * r * bv_pad[1,:num_punc] / modb_pad[:num_punc]

    # Apply a canonical transform that makes varphi 2pi periodic, so the neural network doesn't need to do the lifting
    varphi_frame = 7*varphi_frame
    lphi = lphi / 7

    # Reshape into pairs
    return np.array([lphi[:-1], varphi_frame[:-1], lphi[1:], varphi_frame[1:]]).T

ppairs_list = list(map(process_puncture, ppuncs))
ppairs = np.concatenate(ppairs_list)
print(ppairs.shape)


np.savez('./data/training_data.npz', data=ppairs[:,:2], labels=ppairs[:,2:])