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

# %% Load my own libraries
from c1lgkt.jax.analysis.configuration import load_yaml_config, realize_initial_conditions
import c1lgkt.jax.particles.particle_motion as particle_motion
import c1lgkt.jax.particles.particle_tools as particle_tools

# %% Load configuration

args, ic_gen = load_yaml_config('./scratch/data/nt_analysis.yaml')
y0, mask, ic = realize_initial_conditions(ic_gen, args)

eq = args.eq

t0 = 0.0

# Set up ODE
term = diffrax.ODETerm(particle_motion.f_driftkinetic)
solver = diffrax.Dopri5()
saveat = diffrax.SaveAt(t0=True, t1=True, steps=True)
stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

# Test f_driftkinetic
dy0 = particle_motion.f_driftkinetic(t0, y0, args)

# %% Integrate particle trajectories in blocks and compute punctures

import pickle

num_blocks = 512
#num_blocks = 64
save_interval = 64
ppuncs = particle_tools.PunctureData.empty_list_like(y0)
npuncs = particle_tools.PunctureData.empty_list_like(y0)
ppuncs_poloidal = particle_tools.PunctureData.empty_list_like(y0)
npuncs_poloidal = particle_tools.PunctureData.empty_list_like(y0)

pbar = tqdm(range(num_blocks), dynamic_ncols=True)
for i in pbar:
    pbar.set_description('t = {:.6f}'.format(t0))

    # Integrate particle trajectories
    sol = diffrax.diffeqsolve(
        term, solver, t0=t0, t1=10000.0, dt0=2e-6, y0=y0,
        args=args, stepsize_controller=diffrax.ConstantStepSize(), saveat=saveat,
        max_steps=512,
        throw=False
    )

    # Extract solution and compute punctures
    r_sol, varphi_sol, z_sol, vpar_sol, mu_sol = sol.ys # pyright: ignore
    ppuncs_i, npuncs_i = particle_tools.compute_punctures(sol.ts, sol.ys, z_sol - eq.zaxis) # pyright: ignore

    # Accumulate punctures
    ppuncs = jax.tree.map(lambda *xs: np.concatenate(xs), ppuncs, ppuncs_i)
    npuncs = jax.tree.map(lambda *xs: np.concatenate(xs), npuncs, npuncs_i)

    # Add poloidal punctures
    #phase = eikonal_field.n[0]*varphi_sol - (sol.ts*eikonal_field.omega[0])[:,None] # pyright: ignore
    #ppuncs_i, npuncs_i = particle_tools.compute_punctures(sol.ts, sol.ys, phase, period=2*jnp.pi) # pyright: ignore
    #ppuncs_poloidal = jax.tree.map(lambda *xs: np.concatenate(xs), ppuncs_poloidal, ppuncs_i)
    #npuncs_poloidal = jax.tree.map(lambda *xs: np.concatenate(xs), npuncs_poloidal, npuncs_i)

    # Update initial conditions for next block
    t0 = sol.ts[-1] # pyright: ignore
    y0 = particle_motion.PusherState(
        r=r_sol[-1],
        varphi=varphi_sol[-1],
        z=z_sol[-1],
        upar=vpar_sol[-1],
        mu=mu_sol[-1]
    )

    

    if (i+1)%save_interval == 0:
        # Save punctures
        with open(f'./scratch/outputs/nt/puncs_long_pair_{i//save_interval}.pkl', 'wb') as f:
            pickle.dump((ppuncs, npuncs), f)
            #pickle.dump((ppuncs, npuncs, ppuncs_poloidal, npuncs_poloidal), f)

        # Clear out old punctures to save memory
        ppuncs = particle_tools.PunctureData.empty_list_like(y0)
        npuncs = particle_tools.PunctureData.empty_list_like(y0)

        #ppuncs_poloidal = particle_tools.PunctureData.empty_list_like(y0)
        #npuncs_poloidal = particle_tools.PunctureData.empty_list_like(y0)