#!/usr/bin/env python
# coding: utf-8

# # Tests with Vlasov

# In[1]:


# %% Import JAX and enable 64-bit precision
import jax
jax.config.update("jax_enable_x64", True)


# In[2]:


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
import matplotlib as mpl

import lineax

from jaxtyping import ArrayLike, Real, Complex
from tqdm.notebook import tqdm

import sympy as sp
from IPython.display import display_latex

from c1lgkt.jax.analysis.initial_conditions import quasirandom

import operator
import optax

from c1lgkt.jax.analysis.wba import wba_weights

from jax._src.core import ShapeDtypeStruct as ShapeDtypeStruct
from jaxtyping import Array, Inexact, PyTree


# In[16]:


# %% Vlasov dynamics

max_steps = 256
decimate = 64
batch_size = 4096

@jax.jit
def drift_vlasov(t, y, args):
    x, p = y
    vlasov_params, wave_params = args
    dxdt = p / vlasov_params.m
    dpdt = vlasov_params.q * vlasov_params.e_field(t, x, wave_params) - vlasov_params.gamma * p
    return dxdt, dpdt

def diffusion_vlasov(t, y, args):
    x, p = y
    vlasov_params, wave_params = args
    dW_x = 0.0
    dW_p = jnp.sqrt(2 * vlasov_params.gamma * vlasov_params.m)
    return lineax.FunctionLinearOperator(lambda v: (jnp.zeros_like(v), dW_p * v), input_structure=ShapeDtypeStruct((batch_size,), jnp.float64))

class WaveParams(NamedTuple):
    # Wave number
    k: Real
    # Frequency
    omega: Real

class HarmonicElectricField(eqx.Module):
    # Amplitude of the electric field
    e0: Real

    def __call__(self, t, x, wave_params: WaveParams):
        # phi = phi0 * cos(omega t - k x), so E = -dphi/dx = -phi0 * k * sin(omega t - k x), div(E) = phi0 * k^2 * cos(omega t - k x) = e0 * k * cos(omega t - k x)
        return -self.e0 * jnp.sin(wave_params.omega * t - wave_params.k * x)

class VlasovParams(NamedTuple):
    # Particle mass
    m: Real
    # Particle charge
    q: Real
    # Damping rate
    gamma: Real
    # Density
    n0: Real
    # Electric field function
    e_field: HarmonicElectricField


# In[17]:


# %% Terms related to fluid moments and Poisson equation
@jax.jit
def re_exp(a: Complex, phase: Real[ArrayLike, "..."]) -> Real[ArrayLike, "..."]:
    return jnp.real(a) * jnp.cos(phase) - jnp.imag(a) * jnp.sin(phase)

class HarmonicFluidWeights(eqx.Module):
    n1: Complex
    u1: Complex
    t1: Complex
    q1: Complex

    @jax.jit
    def __call__(self, ts: PyTree, ys: PyTree, wave_params: WaveParams):
        x, p = ys
        x0 = x[0,:]
        p0 = p[0,:]
        phase = wave_params.omega * ts[0] - wave_params.k * x0
        p_weights = jnp.exp(re_exp(self.n1, phase) + p0 * re_exp(self.u1, phase) + 0.5 * (p0**2 - 1.0) * re_exp(self.t1, phase) + (p0**3 - 3.0 * p0) * re_exp(self.q1, phase))
        return p_weights / jnp.sum(p_weights)

# Set up weights for WBA and compute the difference from the initial weights
wba_w = wba_weights(max_steps//decimate+1)

tweight0 = jnp.zeros(max_steps//decimate+1)
tweight0 = tweight0.at[0].set(1.0)
tweight1 = wba_w

diff_weights = tweight1 - tweight0

@jax.jit
def compute_fluid_moments(ts: PyTree, ys: PyTree, p_weights: PyTree, t_weights: PyTree, wave_params: WaveParams):
    xs, ps = ys
    phase = wave_params.k * xs - wave_params.omega * ts[:, None]
    weights = p_weights[None,:] * t_weights[:, None]
    cos_weight = weights * jnp.cos(phase) / jnp.pi
    sin_weight = -weights * jnp.sin(phase) / jnp.pi

    n1 = jnp.sum(cos_weight) + 1j* jnp.sum(sin_weight)
    u1 = jnp.sum(cos_weight * ps) + 1j * jnp.sum(sin_weight * ps)
    t1 = jnp.sum(cos_weight * (ps**2)) + 1j * jnp.sum(sin_weight * (ps**2))
    q1 = jnp.sum(cos_weight * (ps**3)) + 1j * jnp.sum(sin_weight * (ps**3))

    return n1, u1, t1, q1

#response1 = compute_fluid_moments(sol.ts, sol.ys, HarmonicFluidWeights(k=k_wave, omega=omega, n1=0.0j, u1=0.00j, t1=0.0j), tweight1)
#norm1 = jax.tree_util.tree_map(lambda a: jnp.abs(a)**2, response1)


# In[35]:


# %% Objective function

from collections.abc import Sequence


class VlasovProblemInput(eqx.Module):
    """
    Class holding input parameters to the vlasov problem
    """
    m: Real
    q: Real
    gamma: Real
    k_wave: Real
    e0: Real
    n0: Real


class VlasovProblemInnerUnknowns(eqx.Module):
    """
    Class holding unknown parameters to be solved for in the inner problem of the vlasov problem
    """
    n1: Complex
    u1: Complex
    t1: Complex
    q1: Complex

    def to_array(self) -> Array:
        return jnp.array([jnp.real(self.n1), jnp.imag(self.n1), jnp.real(self.u1), jnp.imag(self.u1), jnp.real(self.t1), jnp.imag(self.t1), jnp.real(self.q1), jnp.imag(self.q1)])

    @staticmethod
    def from_array(arr: Array) -> VlasovProblemInnerUnknowns:
        return VlasovProblemInnerUnknowns(n1=arr[0] + 1j * arr[1], u1=arr[2] + 1j * arr[3], t1=arr[4] + 1j * arr[5], q1=arr[6] + 1j * arr[7])

class VlasovProblemOuterUnknowns(eqx.Module):
    """
    Class holding unknown parameters to be solved for in the outer problem of the vlasov problem
    """
    omega: Real

    def to_array(self) -> Array:
        return jnp.array([self.omega])

    @staticmethod
    def from_array(arr: Array) -> VlasovProblemOuterUnknowns:
        return VlasovProblemOuterUnknowns(omega=arr[0])


@jax.jit
def fn_inner(x, args: tuple[VlasovProblemInput, VlasovProblemOuterUnknowns, diffrax.Solution]):
    vlasov_input, outer_unknowns, sol = args
    inner_unknowns = VlasovProblemInnerUnknowns.from_array(x)

    wave_params = WaveParams(k=vlasov_input.k_wave, omega=outer_unknowns.omega)
    p_weights = HarmonicFluidWeights(n1=inner_unknowns.n1, u1=inner_unknowns.u1, t1=inner_unknowns.t1, q1=inner_unknowns.q1)(sol.ts, sol.ys, wave_params)
    dn, du, dt, dq = compute_fluid_moments(sol.ts, sol.ys, p_weights, diff_weights, wave_params)

    return jnp.array([jnp.real(dn), jnp.imag(dn), jnp.real(du), jnp.imag(du), jnp.real(dt), jnp.imag(dt), jnp.real(dq), jnp.imag(dq)])

@jax.jit
def fn_outer(x, args: tuple[VlasovProblemInput, Array]):
    # Unpack unknowns
    outer_unknowns = VlasovProblemOuterUnknowns.from_array(x)

    # Set up parameters for the Vlasov simulation
    vlasov_input, key = args

    field = HarmonicElectricField(e0=vlasov_input.e0)
    vlasov_params = VlasovParams(m=vlasov_input.m, q=vlasov_input.q, gamma=vlasov_input.gamma, n0=vlasov_input.n0, e_field=field)
    wave_params = WaveParams(k=vlasov_input.k_wave, omega=outer_unknowns.omega)

    diffrax_args = (vlasov_params, wave_params)

    # Set up initial conditions for the Vlasov simulation
    z1, z2 = quasirandom([[0.0, 2*jnp.pi], [-1.0, 1.0]], batch_size)
    # Random offset
    key, subkey = jax.random.split(key)
    offset = jax.random.uniform(subkey, shape=(2,))
    z1 = (z1 + offset[0]) % (2*jnp.pi)
    z2 = (z2 + offset[1] + 1.0) % 2.0 - 1.0
    p0 = jax.lax.erf_inv(z2)

    t0, t1 = 0.0, 2*jnp.pi/4.0
    dt = (t1 - t0) / max_steps

    y0 = (z1, p0)

    # Set up terms
    bm = diffrax.VirtualBrownianTree(t0, t1, tol=dt/2.0, shape=(batch_size,), key=key)
    drift_term = diffrax.ODETerm(drift_vlasov)
    diffusion_term = diffrax.ControlTerm(diffusion_vlasov, bm)
    terms = diffrax.MultiTerm(drift_term, diffusion_term)

    # Set up solver and saveat
    solver = diffrax.Heun()
    saveat = diffrax.SaveAt(t0=True, t1=True, steps=decimate)

    # Compute samples of Vlasov equation
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=y0,
        args=diffrax_args,
        saveat=saveat,
        max_steps=max_steps,
    )

    # Compute solution to the inner problem
    solver = optx.Newton(rtol=1e-6, atol=1e-6)
    theta0 = jnp.zeros(8)
    theta_solve = optx.root_find(fn_inner, solver, theta0, args=(vlasov_input, outer_unknowns, sol), adjoint=optx.ImplicitAdjoint())
    inner_sol = VlasovProblemInnerUnknowns.from_array(theta_solve.value)

    # Compute residuals for the outer problem, which in this case is just the real part of the dispersion relation
    p_weights = HarmonicFluidWeights(n1=inner_sol.n1, u1=inner_sol.u1, t1=inner_sol.t1, q1=inner_sol.q1)(sol.ts, sol.ys, wave_params)
    nresp, uresp, tresp, qresp = compute_fluid_moments(sol.ts, sol.ys, p_weights, wba_w, wave_params)
    return jnp.array([vlasov_input.e0 * vlasov_input.k_wave - jnp.real(nresp) * vlasov_input.q * vlasov_input.n0])


def jacrev_and_value(func, argnums: int | Sequence[int] =0):
    def wrapper(*args, **kwargs):
        value = func(*args, **kwargs)
        return value, value
    return jax.jacrev(wrapper, argnums=argnums, has_aux=True)


# In[36]:

# Initial guess for the outer unknowns (omega), and set up the Vlasov problem input and random key
theta0 = jnp.array([4.0])
vlasov_input = VlasovProblemInput(m=1.0, q=-1.0, gamma=0.1, k_wave=1.0, e0=1e-3, n0=13.0)
key = jax.random.PRNGKey(0)

my_func = jacrev_and_value(fn_outer, argnums=0)
outer_step_fn = jax.jit(my_func)


# In[37]:


theta_step = theta0


# In[38]:

# Manual Newton's method for the outer problem, using the Jacobian and value from the jacrev_and_value wrapper
for k in range(10):
    outer_step = outer_step_fn(theta_step, (vlasov_input, key))
    theta_step = theta_step - outer_step[1] / outer_step[0][0,0]
    print(outer_step)
    print(theta_step)


# In[25]:


theta_step
