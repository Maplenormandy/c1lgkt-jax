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

from jaxtyping import ArrayLike, Real

from .equilibrium import Equilibrium

# %% Pushers for field lines

class FieldlineArgs(NamedTuple):
    eq: Equilibrium  # Equilibrium object

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

# %%

class GeometryHandler(eqx.Module):
    # Interpolator over psi for the outer midplane minor radius squared. Note that psi ~ a^2 near the axis, so this interpolator is well-behaved.
    interp_amid2: interpax.Interpolator1D

    # Interpolators for q(psi) and dtheta(psi, theta_g)
    psi_surf: Real[ArrayLike, "Nsurf"]      # psi values at which q and dtheta are evaluated
    q_surf: Real[ArrayLike, "Nsurf"]        # corresponding q values
    interp_q: interpax.Interpolator1D       # Interpolator over psi for the safety factor q(psi)
    interp_dtheta: interpax.Interpolator2D  # Interpolator over (psi, theta_g) such that theta = theta_g + interp_dtheta(psi, theta_g)

    # Toroidal flux
    interp_torflux: interpax.PPoly

    def __init__(self, eq: Equilibrium):
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

        ## Next, compute the q profiles and dtheta profiles using diffrax
        # Set up terms for field line tracing
        term = diffrax.ODETerm(f_fieldline_axial)
        solver = diffrax.Dopri5()
        saveat = diffrax.SaveAt(t0=True, t1=True, dense=True)
        args = FieldlineArgs(eq=eq)
        stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
        root_finder = optx.Newton(rtol=1e-7, atol=1e-7)
        event = diffrax.Event((cond_axial_crossing(1), cond_axial_crossing(-1)), root_finder)
        # This gives an estimate of how long we need to follow field lines for; approximately 10 toroidal windings
        max_t_final = 10.0 * 2 * jnp.pi * eq.raxis / jnp.max(eq.ff)

        # Set up function to compute q and dtheta profiles
        def compute_q_and_dtheta(psi0):
            # Initial rhog value from psi
            rhog0 = jnp.sqrt(self.interp_amid2(psi0))
            # Initial state
            y0 = (rhog0, 0, 0)

            # Integrate forward to find q and dtheta
            sol = diffrax.diffeqsolve(
                term,
                solver,
                t0=0.0,
                t1=max_t_final,
                dt0=2e-3*max_t_final,
                y0=y0,
                args=args,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                event=event
            )

            # Extract final state
            rhog_ys, varphi_ys, thetag_ys = sol.ys

            # Compute q value as number of toroidal turns per poloidal turn. We've completed one poloidal turn, so we divide the toroidal angle by 2pi
            q = varphi_ys[-1]/(2*jnp.pi)

            # Get the densely evaluated field line points to compute dtheta
            rhog, varphi, thetag = jax.lax.map(sol.evaluate, jnp.linspace(0, sol.ts[-1], 256, endpoint=False))
            # Difference between geometric and straight-field-line poloidal angle
            dtheta = thetag - (varphi / varphi_ys[-1]) * 2 * jnp.pi
            # Resample dtheta onto a uniform grid in thetag
            dtheta_resamp = interpax.interp1d(jnp.linspace(0, 2*jnp.pi, 128, endpoint=False), thetag, dtheta, method='cubic2', period=2*jnp.pi)

            return q, dtheta_resamp
        
        # Vectorize over psi0
        compute_q_and_dtheta_vmap = jax.vmap(compute_q_and_dtheta)
        # Grid of psi values to compute profiles on
        self.psi_surf = jnp.linspace(0, eq.psix*0.99, 193)[1:]  # avoid axis and separatrix
        # Compute q and dtheta profiles
        self.q_surf, dtheta_profile = compute_q_and_dtheta_vmap(self.psi_surf)
        # Set up interpolators
        self.interp_q = interpax.Interpolator1D(self.psi_surf, self.q_surf, method='cubic2')
        self.interp_dtheta = interpax.Interpolator2D(self.psi_surf, jnp.linspace(0, 2*jnp.pi, 128, endpoint=False), dtheta_profile, period=(None, 2*jnp.pi))
        # Set up toroidal flux interpolator. Note that interp_q_ppoly is the same as self.interp_q but in PPoly form.
        interp_q_ppoly = interpax.CubicSpline(self.psi_surf, self.q_surf)
        self.interp_torflux = interp_q_ppoly.antiderivative()

