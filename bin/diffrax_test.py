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
from c1lgkt.jax.fields.equilibrium import Equilibrium
from c1lgkt.jax.fields.clebsch import ClebschMappingBuilder
from c1lgkt.jax.fields.field_providers import EikonalFieldProvider
import c1lgkt.jax.particles.particle_motion as particle_motion
import c1lgkt.jax.particles.particle_tools as particle_tools

# %% Load equilibrium and set up clebsch mapping
print('Loading equilibrium...')
eq = Equilibrium.from_eqdfile('./tests/D3D141451.eqd')

clebsch_builder = ClebschMappingBuilder()
# Check if the mapping file exists
if os.path.isfile('./outputs/theta_map_D3D141451.eqx'):
    print('Loading existing theta map...')
    theta_map = clebsch_builder.load_theta_map('./outputs/theta_map_D3D141451.eqx')
else:
    print('Building new theta map...')
    theta_map = clebsch_builder.build_theta_map(eq)
    eqx.tree_serialise_leaves('./outputs/theta_map_D3D141451.eqx', theta_map)
# Build Clebsch representation
if os.path.isfile('./outputs/clebsch_D3D141451.eqx'):
    print('Loading existing Clebsch representation...')
    clebsch = clebsch_builder.load_clebsch('./outputs/clebsch_D3D141451.eqx')
else:
    print('Building new Clebsch representation...')
    clebsch = clebsch_builder.build_clebsch(theta_map, eq)
    eqx.tree_serialise_leaves('./outputs/clebsch_D3D141451.eqx', clebsch)

# %% Set up eikonal field provider
print('Setting up eikonal field provider...')
coefs = np.zeros((2, 4, 6))
coefs[0, 2, 0] = 3e-5
coefs[1, 2, 0] = 2e-5

theta_x1 = theta_map(jnp.array([theta_map.nulls.x1.rz[0]]), jnp.array([theta_map.nulls.x1.rz[1]]))[0]
theta_x2 = theta_map(jnp.array([theta_map.nulls.x2.rz[0]]), jnp.array([theta_map.nulls.x2.rz[1]]))[0]

field_provider = EikonalFieldProvider(
    clebsch = clebsch,
    n = jnp.array([3, 3], dtype=int),
    omega = jnp.array([0.0, 0.0]),
    psi0 = jnp.array([0.95*eq.psix, 0.95*eq.psix]),
    psi_scale = jnp.array([0.02*eq.psix, 0.02*eq.psix]),
    theta0 = jnp.array([theta_x1 + 0.01, theta_x2 - 0.01]),
    alpha_scale = jnp.array([5.0, 5.0]),
    gh_coefs = jnp.array(coefs)
)

# %% Pusher for drift-kinetic particles
pp = particle_motion.elec

# Set up ODE
term = diffrax.ODETerm(particle_motion.f_driftkinetic)
solver = diffrax.Dopri5()
saveat = diffrax.SaveAt(t0=True, t1=True, steps=True)
args = particle_motion.PusherArgs(eq=eq, pp=pp, theta_map=theta_map, fields=[field_provider])
stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

num_particles = 4096
# Set up initial conditions
y0 = (
    jnp.linspace(jnp.max(eq.lcfsrz[0,:])-0.08, jnp.max(eq.lcfsrz[0,:])-0.005, num_particles),
    jnp.zeros(num_particles),
    jnp.zeros(num_particles),
    jnp.ones(num_particles) * pp.vt,
    jnp.ones(num_particles) * pp.m * pp.vt**2 * 0.1
)
t0 = 0.0

# Test f_driftkinetic
dy0 = particle_motion.f_driftkinetic(0.0, y0, args)

# %% Integrate particle trajectories in blocks and compute punctures
print('Integrating particle trajectories and computing punctures...')

num_blocks = 1
ppuncs = [particle_tools.PunctureData(
    tp = jnp.zeros((0,)),
    yp = jax.tree.map(lambda x: jnp.zeros((0,)), y0)
)] * y0[0].shape[0]
npuncs = [particle_tools.PunctureData(
    tp = jnp.zeros((0,)),
    yp = jax.tree.map(lambda x: jnp.zeros((0,)), y0)
)] * y0[0].shape[0]

# JIT compile the solver
# Integrate particle trajectories
@jax.jit
def compute_sol(y0):
    # Integrate particle trajectories
    sol = diffrax.diffeqsolve(
        term, solver, t0=t0, t1=1000.0, dt0=2e-6, y0=y0,
        args=args, stepsize_controller=diffrax.ConstantStepSize(), saveat=saveat,
        max_steps=256,
        throw=False
    )
    return sol

sol = compute_sol(y0)


# Extract solution and compute punctures
r_sol, varphi_sol, z_sol, vpar_sol, mu_sol = sol.ys

# Update initial conditions for next block
t0 = sol.ts[-1]
y0 = (
    r_sol[-1],
    varphi_sol[-1],
    z_sol[-1],
    vpar_sol[-1],
    mu_sol[-1]
)

with jax.profiler.trace("./outputs/jax-trace", create_perfetto_link=True):
    for i in tqdm(range(num_blocks)):
        # Integrate particle trajectories
        sol = diffrax.diffeqsolve(
            term, solver, t0=t0, t1=1000.0, dt0=2e-6, y0=y0,
            args=args, stepsize_controller=diffrax.ConstantStepSize(), saveat=saveat,
            max_steps=256,
            throw=False
        )

        # Extract solution and compute punctures
        r_sol, varphi_sol, z_sol, vpar_sol, mu_sol = sol.ys
        #ppuncs_i, npuncs_i = particle_tools.compute_punctures(sol.ts, sol.ys, varphi_sol, period=2*jnp.pi/3)

        # Accumulate punctures
        #ppuncs = jax.tree.map(lambda *xs: jnp.concatenate(xs), ppuncs, ppuncs_i)
        #npuncs = jax.tree.map(lambda *xs: jnp.concatenate(xs), npuncs, npuncs_i)

        # Update initial conditions for next block
        t0 = sol.ts[-1]
        y0 = (
            r_sol[-1],
            varphi_sol[-1],
            z_sol[-1],
            vpar_sol[-1],
            mu_sol[-1]
        )



