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

from pathlib import Path

# %% Load my own libraries
from c1lgkt.jax.analysis.configuration import load_yaml_config, realize_initial_conditions
import c1lgkt.jax.particles.particle_motion as particle_motion
import c1lgkt.jax.particles.particle_tools as particle_tools

# %% Load configuration


args, ic_gen, analysis = load_yaml_config('./scratch/data/nt_resonance_analysis.yaml')
y0, mask, ic = realize_initial_conditions(ic_gen, args)

eq = args.eq

t0 = 0.0

# Set up ODE
term = diffrax.ODETerm(particle_motion.f_driftkinetic)
solver = diffrax.Dopri5()
saveat = diffrax.SaveAt(t0=True, t1=True, steps=True)
stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

# %% Integrate particle trajectories in blocks and compute punctures

import pickle

# Get parameters from config
max_steps = int(analysis['max_steps'])
dt0 = float(analysis['dt0'])
batch_size = int(analysis['batch_size'])
output_dir = Path(analysis['output_dir'])

# number of particles
nump = y0.r.shape[0]

# Make the output directory if it doesn't exist yet
os.makedirs(output_dir, exist_ok=True)

# Convert y0 to numpy arrays for easier slicing
y0 = jtu.tree_map(lambda x: np.array(x), y0)

@jax.jit
def integrate_single(y0_single):
    sol = diffrax.diffeqsolve(
        term, solver, t0=t0, t1=10000.0, dt0=dt0, y0=y0_single,
        args=args, stepsize_controller=stepsize_controller,
        saveat=saveat,
        max_steps=max_steps,
        throw=False
    )

    return sol.ts, sol.ys

def compute_punctures(ts, ys):
    ys = jtu.tree_map(lambda x: np.array(x)[:,np.isfinite(ts)].T, ys)
    ts = ts[np.isfinite(ts)]

    return particle_tools.compute_punctures(ts, ys, ys.z - eq.zaxis, condpunc=(ys.r - eq.raxis) > 0.0)[0]

for batch_i in tqdm(range(0, nump, batch_size), dynamic_ncols=True):
    # Get current batch of initial conditions
    y0_batch = jtu.tree_map(lambda x: jnp.asarray(x[batch_i:batch_i+batch_size]), y0)

    ts, ys = jax.vmap(integrate_single)(y0_batch)


    def compute_puncture(k):
        return compute_punctures(ts[k,:], jtu.tree_map(lambda x: x[k:k+1,:], ys))

    # Compute punctures in outer midplane
    puncs_batch = [compute_puncture(k) for k in range(batch_size)]
    
    # Save the punctures
    with open(output_dir / f'puncs_resonance_{batch_i//batch_size:03d}.pkl', 'wb') as f:
        pickle.dump(puncs_batch, f)


