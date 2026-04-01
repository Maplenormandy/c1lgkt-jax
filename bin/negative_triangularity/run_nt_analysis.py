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
from c1lgkt.jax.fields.field_providers import EikonalFieldProvider, ZonalFieldProvider
import c1lgkt.jax.particles.particle_motion as particle_motion
import c1lgkt.jax.particles.particle_tools as particle_tools

# %% Load equilibrium and set up clebsch mapping
eq = Equilibrium.from_gfile('./scratch/data/g193802.04490')

clebsch_builder = ClebschMappingBuilder()
# Check if the mapping file exists
if os.path.isfile('./scratch/outputs/theta_map_193802.eqx'):
    print('Loading existing theta map...')
    theta_map = clebsch_builder.load_theta_map('./scratch/outputs/theta_map_193802.eqx')
else:
    print('Building new theta map...')
    theta_map = clebsch_builder.build_theta_map(eq)
    eqx.tree_serialise_leaves('./scratch/outputs/theta_map_193802.eqx', theta_map)
# Build Clebsch representation
if os.path.isfile('./scratch/outputs/clebsch_193802.eqx'):
    print('Loading existing Clebsch representation...')
    clebsch = clebsch_builder.load_clebsch('./scratch/outputs/clebsch_193802.eqx')
else:
    print('Building new Clebsch representation...')
    clebsch = clebsch_builder.build_clebsch(theta_map, eq)
    eqx.tree_serialise_leaves('./scratch/outputs/clebsch_193802.eqx', clebsch)

# %% Set up fields

# Load Er from pfile
zonal_fields = ZonalFieldProvider.from_pfile('./scratch/data/p193802.04490', eq)
interp_phi_scaled = interpax.Interpolator1D(zonal_fields.interp_phi.x, zonal_fields.interp_phi.f * 1.7, method='cubic2')
zonal_fields = ZonalFieldProvider(interp_phi=interp_phi_scaled, interp_apar=zonal_fields.interp_apar)

coefs = jnp.zeros((1, 4, 6))
# Te ~ 0.25 keV at rho ~ 0.94. We want e phi / Te ~ ntilde / n0 ~ 3e-2; phi is in kV
# This ends up being around 7.5 V
coefs = coefs.at[0, 0, 0].set(0.25*3e-2)

eikonal_field = EikonalFieldProvider(
    theta_map,
    clebsch,
    # n = 6 leads to k_\theta of about 0.143 rad/cm at the OMP
    jnp.array([6], dtype=int),
    # taking f about 28 kHz
    jnp.array([2*jnp.pi*27.0]),
    # Perturbation is centered around rho = 0.94
    jnp.array([0.94*eq.psix]),
    # Width is about 0.045 in normalized psi
    jnp.array([0.045*eq.psix]),
    # Mode is peaked at the outer midplane
    jnp.array([0.0]),
    # Poloidal extent of the mode; somewhat arbitrary for now
    jnp.array([jnp.pi]),
    # Amplitude of the mode; see earlier
    coefs
    )

# Set up interpolator for r(psi) so we can initialize particles in psi space
r_outer = jnp.linspace(eq.raxis, eq.rmax, 128)
psi_outer = eq.interp_psi(r_outer, jnp.full_like(r_outer, eq.zaxis))
interp_router = interpax.Interpolator1D(psi_outer, r_outer, method='cubic2')

# reference position
r_ref = interp_router(0.945*eq.psix)
z_ref = eq.zaxis
bv = eq.compute_bv(jnp.array([r_ref]), jnp.array([z_ref]))
modb = jnp.linalg.norm(bv)

# Electrons 
pp = particle_motion.elec
# Reference kinetic energy of particles ~ Te ~ 0.27 keV
ev_ref = 0.27*1.5
# cos(pitch angle) ~ sqrt(0.33)
xi_ref = jnp.sqrt(0.33)
# Initial parallel velocity
upar_ref = pp.vt * xi_ref * np.sqrt(ev_ref)
# Initial magnetic moment
mu_ref = pp.m * (1-xi_ref**2) * (pp.vt * np.sqrt(ev_ref))**2 / 2 / modb

# Set up the reference state
t0 = 0.0
state_ref = (
    jnp.array([r_ref]),
    jnp.array([0.0]),
    jnp.array([z_ref]),
    jnp.array([upar_ref]),
    jnp.array([mu_ref]),
)

# Set up the PusherArgs
args = particle_motion.PusherArgs(
    eq=eq,
    pp=pp,
    #fields=[zonal_fields, eikonal_field]
    fields=[zonal_fields]
)

# Compute the integrals of motion for the reference state
ham_ref, lphi_ref = particle_tools.compute_integrals(t0, state_ref, args)
# Unpack the reference integrals
ham_ref = ham_ref[0]
lphi_ref = lphi_ref[0]

# %% Set up initial conditions

# Number of particles
nump = 96

# Initial positions: low-discrepancy sequence in varphi and psi
g = 1.32471795724474602596
a1 = 1.0/g
a2 = 1.0/(g*g)
samp1 = (0.5 + jnp.arange(nump) * a1) % 1.0
samp2 = (0.5 + jnp.arange(nump) * a2) % 1.0

varphi0 = samp1*(2*jnp.pi/7)
r0 = interp_router(0.92*eq.psix + samp2*0.05*eq.psix)
#r0 = jnp.full_like(varphi0, r_ref)
z0 = jnp.full_like(r0, eq.zaxis)
mu0 = jnp.full_like(r0, mu_ref)

# Set up the initial partial state tuple
y0 = particle_motion.PusherState(
    r=r0,
    varphi=varphi0,
    z=z0,
    upar=jnp.full_like(r0, upar_ref),
    mu=mu0
)

# Compute the parallel kinetic energies
omega_frame = eikonal_field.omega[0] / eikonal_field.n[0]

# Get the parallel kinetic energy
kpar, upar_omega = particle_tools.compute_parallel_energy(t0, y0, (ham_ref, lphi_ref), omega_frame, args)

upar0 = jnp.sqrt(2 * kpar / pp.m) + upar_omega

y0 = particle_motion.PusherState(
    r=r0,
    varphi=varphi0,
    z=z0,
    upar=upar0,
    mu=mu0
)

# Set up ODE
term = diffrax.ODETerm(particle_motion.f_driftkinetic)
solver = diffrax.Dopri5()
saveat = diffrax.SaveAt(t0=True, t1=True, steps=True)
stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

# Test f_driftkinetic
dy0 = particle_motion.f_driftkinetic(t0, y0, args)

# %% Integrate particle trajectories in blocks and compute punctures

import pickle

num_blocks = 64
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