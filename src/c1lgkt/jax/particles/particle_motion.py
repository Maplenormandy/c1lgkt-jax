"""
Contains codes for the vector fields associated with particle motion.

Typically I work in cylindrical coordinates (R, phi, Z).
"""

from typing import NamedTuple

from ..fields.equilibrium import Equilibrium
import jax.numpy as jnp
import jax
import interpax

from jaxtyping import ArrayLike, Real

# %% Arguments for the various pushers

class ParticleParams(NamedTuple):
    """
    NamedTuple holding particle properties.

    Some unit normalizations:
    - Length: 1m
    - Magnetic field: 1T
    - Energy: 1keV
    - Time: 1ms
    - Charge: e
    - Mass: (1e * 1T * 1m)^2 / (1 keV) = 1.60217663e-22 kg
      => google search calculator version: (1 electron charge * 1 T * 1 m)^2 / 1 keV
      
    z: float
        Particle charge state (in units of e)
    m: float
        Particle mass (in units of proton mass)
    vt: float
        Reference particle velocity at 1 keV in reference units; technically a derived quantity but very useful to have around
    """
    z: float
    m: float
    vt: float

# Some default particle parameters
deut = ParticleParams(z=1, m=2.08793698e-5, vt=1/jnp.sqrt(2.08793698e-5))
elec = ParticleParams(z=-1, m=5.6856301e-9, vt=1/jnp.sqrt(5.6856301e-9))

class PusherArgs(NamedTuple):
    """
    NamedTuple holding arguments for all particle pushers. Note most things here are optional.
    """
    eq: Equilibrium | None = None
    pp: ParticleParams | None = None
    zonal_fields: interpax.Interpolator1D | None = None # Temporary; storage for zonally symmetric fields


# %% Functions for gyrokinetic particle pushing with zonally symmetric equilibria

def f_driftkinetic(t, state, args: PusherArgs):
    """
    Push a (single) drift-kinetic tracer in (R, varphi, Z) coordinates.
    Note y = (R, varphi, Z, v_||, mu). (TODO: Is this the best set of coordinates for EM?)
    """
    r, varphi, z, vll, mu = state
    eq = args.eq
    pp = args.pp
    
    ## Magnetic terms
    psi_ev, ff_ev = eq.compute_psi_and_ff(r, z)
    bv, bu, modb, gradmodb, curlbu = eq.compute_geom_terms(r, psi_ev, ff_ev)
    ## Bstar and Bstar parallel
    bstar = bv + (pp.m / pp.z) * curlbu * vll[None, ...]
    bstarll = jnp.sum(bu*bstar, axis=0)

    ## Electric potential gradient
    # TODO: think about a way to implement electromagnetic fields
    #dphi = compute_fields(t, r, z, varphi, psi_ev, eq, fields, frame)
    dphi = jnp.zeros_like(gradmodb)
    if args.zonal_fields is not None:
        (psi, psidr, psidz, psidrr, psidrz, psidzz) = psi_ev
        dzpot = args.zonal_fields(psi, dx=1)
        dphi = jnp.array([psidr * dzpot, jnp.zeros_like(psidr), psidz * dzpot])
    

    ## Finally compute the (spatial) gradient of the Hamiltonian
    gradh = mu * gradmodb + pp.z * dphi

    rdot = (jnp.cross(bu, gradh, axis=0) / pp.z + vll[None, ...] * bstar) / bstarll[None, ...]

    drdt = rdot[0, ...]
    dvarphidt = rdot[1, ...] / r
    dzdt = rdot[2, ...]
    dvlldt = -(jnp.sum(bstar*gradh, axis=0) / bstarll) / pp.m
    dmudt = jnp.zeros_like(mu)

    return (drdt, dvarphidt, dzdt, dvlldt, dmudt)

def f_driftkinetic_midplane(t, state, args: PusherArgs):
    """
    Push a drift-kinetic tracer, but also follow winding angles around the magnetic axis and around (v_||, Z)
    """
    # Unpack the state
    r, varphi, z, vll, mu, theta_pol, theta_vll = state
    # We need these arguments
    eq = args.eq
    pp = args.pp
    
    # Pass through to the basic drift-kinetic pusher
    drdt, dvarphidt, dzdt, dvlldt, dmudt = f_driftkinetic(t, (r, varphi, z, vll, mu), args)

    # Compute the winding rates
    dthetapoldt = ((r - eq.raxis) * dzdt - (z - eq.zaxis) * drdt) / ((r - eq.raxis)**2 + (z - eq.zaxis)**2)
    dthetavlldt = ((vll / pp.vt) * dzdt - (z - eq.zaxis) * dvlldt / pp.vt) / ((z - eq.zaxis)**2 + (vll/pp.vt)**2)

    # Return everything
    return (drdt, dvarphidt, dzdt, dvlldt, dmudt, dthetapoldt, dthetavlldt)