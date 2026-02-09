# %% -*- coding: utf-8 -*-
"""
@author: maple

File containing classes which handle magnetic geometry computations. Mainly, these are responsible for generating
the mapping between geometric poloidal angle theta_g and straight-field-line poloidal angle theta, as well as the safety
factor q(psi) and toroidal flux function phi(psi).

The basic idea is that these quantities involve non-local integrations over the equilibrium, so we have a different
class for handling them.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import interpax
import diffrax
import optimistix as optx

from typing import NamedTuple
from functools import partial

from jaxtyping import ArrayLike, Real

from .equilibrium import Equilibrium

# %% Pushers for field lines

class FieldlineArgs(NamedTuple):
    eq: Equilibrium  # Equilibrium object

@jax.jit
def f_fieldline(t, state, args: FieldlineArgs):
    """
    Push a magnetic field line in (R, varphi, Z) coordinates.
    """
    eq = args.eq

    r, varphi, z = state

    br, bt, bz = eq.compute_bv(r, z)

    drdt = br
    dvarphidt = bt / r  # Convert Bphi to angular velocity
    dzdt = bz

    return (drdt, dvarphidt, dzdt)

@jax.jit
def f_fieldline_axial(t, state, args: FieldlineArgs):
    """
    Push a magnetic field line in (rhog, varphi, thetag) coordinates, where rhog and thetag are geometric cylindrical coordinates in (R,Z)
    """
    eq = args.eq

    rhog, varphi, thetag = state

    r = rhog * jnp.cos(thetag) + eq.raxis
    z = rhog * jnp.sin(thetag) + eq.zaxis

    drdt, dvarphidt, dzdt = f_fieldline(t, (r, varphi, z), args)

    drhogdt = drdt * jnp.cos(thetag) + dzdt * jnp.sin(thetag)
    dthetagdt = (-drdt * jnp.sin(thetag) + dzdt * jnp.cos(thetag)) / rhog

    return (drhogdt, dvarphidt, dthetagdt)

def cond_axial_crossing(n: int):
    """
    Returns a function which is zero when the field line winds around n times in the geometric poloidal angle thetag
    """
    def cond_fn(t, y, args: FieldlineArgs, **kwargs):
        
        eq = args.eq

        rhog, varphi, thetag = y

        return jnp.ravel(thetag)[0] - n*2*jnp.pi
    
    return cond_fn

# %% Performing field line pushing

class FieldlinePusher(eqx.Module):
    """
    This is a helper class which sets up diffrax components for pushing field lines.

    TODO: I was hoping this would reduce JIT times by reusing compiled functions, but it seems like the functions
    get re-jitted sometimes, but not always??? Need to investigate further.
    """
    term: diffrax.ODETerm = eqx.field(static=True)
    solver: diffrax.AbstractSolver = eqx.field(static=True)
    stepsize_controller: diffrax.AbstractStepSizeController = eqx.field(static=True)
    root_finder: optx.AbstractRootFinder = eqx.field(static=True)
    event: diffrax.Event = eqx.field(static=True)
    saveat: diffrax.SaveAt = eqx.field(static=True)

    def __init__(self):
        """
        Set up diffrax components for field line pushing. No arguments needed.
        """
        self.term = diffrax.ODETerm(f_fieldline_axial)
        self.solver = diffrax.Dopri5()
        self.stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
        self.root_finder = optx.Newton(rtol=1e-7, atol=1e-7)
        self.event = diffrax.Event((cond_axial_crossing(1), cond_axial_crossing(-1)), self.root_finder)
        self.saveat = diffrax.SaveAt(t0=True, t1=True, dense=True)

    @jax.jit
    @partial(jax.vmap, in_axes=(None, None, 0))
    def compute_q_and_dtheta(self, eq: Equilibrium, rhog0: Real):
        # How long should we push the field lines for
        max_t_final = 10.0 * 2 * jnp.pi * eq.raxis / jnp.max(eq.ff)
        
        # Arguments for field line pushing
        args = FieldlineArgs(eq=eq)

        # Initial state
        y0 = (rhog0, 0, 0)

        # Integrate forward to find q and dtheta
        sol = diffrax.diffeqsolve(
            self.term,
            self.solver,
            t0=jnp.array(0.0),
            t1=jnp.array(max_t_final),
            dt0=jnp.array(2e-3*max_t_final),
            y0=y0,
            args=args,
            saveat=self.saveat,
            stepsize_controller=self.stepsize_controller,
            event=self.event
        )

        # Extract final state
        rhog_ys, varphi_ys, thetag_ys = sol.ys # pyright: ignore

        # Compute q value as number of toroidal turns per poloidal turn. We've completed one poloidal turn, so we divide the toroidal angle by 2pi
        q = varphi_ys[-1]/(2*jnp.pi)

        # Get the densely evaluated field line points to compute dtheta
        rhog, varphi, thetag = jax.lax.map(sol.evaluate, jnp.linspace(0, sol.ts[-1], 256, endpoint=False)) # pyright: ignore
        # Difference between geometric and straight-field-line poloidal angle
        dtheta = thetag - (varphi / varphi_ys[-1]) * 2 * jnp.pi
        # Resample dtheta onto a uniform grid in thetag
        dtheta_resamp = interpax.interp1d(jnp.linspace(0, 2*jnp.pi, 128, endpoint=False), thetag, dtheta, method='cubic2', period=2*jnp.pi)

        return q, dtheta_resamp


# %%

class GeometryHandler(eqx.Module):
    # Interpolator over psi for the outer midplane minor radius squared. Note that psi ~ a^2 near the axis, so this interpolator is well-behaved.
    interp_amid2: interpax.Interpolator1D

    # Interpolators for q(psi) and dtheta(psi, theta_g)
    Nsurf: int                              # number of psi surfaces used in constructing q and dtheta profiles      
    psi_surf: Real[ArrayLike, "{Nsurf}"]    # psi values at which q and dtheta are evaluated
    q_surf: Real[ArrayLike, "{Nsurf}"]      # corresponding q values
    interp_q: interpax.Interpolator1D       # Interpolator over psi for the safety factor q(psi)
    interp_dtheta: interpax.Interpolator2D  # Interpolator over (psi, theta_g) such that theta = theta_g + interp_dtheta(psi, theta_g)

    # Toroidal flux
    interp_torflux: interpax.PPoly

    def __init__(self, eq: Equilibrium, pusher: FieldlinePusher, nsurf: int = 192):
        ## First, compute interpolation functions for mapping psi to midplane radius
        # Grid of midplane r points
        rmid = eq.rgrid[eq.rgrid > eq.raxis]
        # Compute psi using the equilibrium's psi interpolator
        psi_midplane = eq.interp_psi(rmid, jnp.full_like(rmid, eq.zaxis))
        # Add in the magnetic axis point
        rmid = jnp.concatenate([jnp.array([eq.raxis]), rmid])
        psi_midplane = jnp.concatenate([jnp.array([0.0]), psi_midplane])
        # Set up the interpolator
        self.interp_amid2 = interpax.Interpolator1D(psi_midplane, (rmid-eq.raxis)**2, method='cubic2')

        # Grid of psi values to compute profiles on
        self.Nsurf = nsurf
        self.psi_surf = jnp.linspace(0, eq.psix*0.99, nsurf+1)[1:]  # avoid axis and separatrix
        rhog_surf = jnp.sqrt(self.interp_amid2(self.psi_surf))
        # Compute q and dtheta profiles
        self.q_surf, dtheta_profile = pusher.compute_q_and_dtheta(eq, rhog_surf)
        # Set up interpolators
        self.interp_q = interpax.Interpolator1D(self.psi_surf, self.q_surf, method='cubic2')
        self.interp_dtheta = interpax.Interpolator2D(self.psi_surf, jnp.linspace(0, 2*jnp.pi, 128, endpoint=False), dtheta_profile, period=(None, 2*jnp.pi))
        # Set up toroidal flux interpolator. Note that interp_q_ppoly is the same as self.interp_q but in PPoly form.
        interp_q_ppoly = interpax.CubicSpline(self.psi_surf, self.q_surf)
        self.interp_torflux = interp_q_ppoly.antiderivative()
