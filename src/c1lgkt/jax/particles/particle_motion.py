"""
Contains codes for the vector fields associated with particle motion.

Typically everything is done in cylindrical coordinates (R, phi, Z).
"""

from typing import NamedTuple, TypeVar, Generic

from ..fields.equilibrium import Equilibrium, PsiTuple
from ..fields.clebsch import ThetaMapping
from ..fields.field_providers import AbstractFieldProvider, sum_field_grads, sum_fields
import jax.numpy as jnp
import jax
import interpax
import jax.tree_util as jtu

from jaxtyping import ArrayLike, Real, PyTree, Array
from diffrax._custom_types import RealScalarLike, VF, Y, Args, Control
from diffrax._path import AbstractPath

from functools import reduce

from diffrax import AbstractTerm

from ..custom_types import ScalarArray, ScalarArrayLike, ScalarFields, VectorFields

# %% Arguments for the various pushers

# Reference mass in kg
m_kev = 1.60217663e-22
# 1 amu in reference mass units
amu = 1.03642697e-5

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
    z: Real
    m: Real
    vt: Real

    @classmethod
    def build_from_config(cls, obj: dict) -> ParticleParams:
        """
        Builds ParticleParams from a dictionary
        """
        if 'name' in obj:
            return cls.species(obj['name'])
        elif 'm' in obj and 'z' in obj:
            m = obj['m'] * amu
            z = obj['z']
            vt = 1 / jnp.sqrt(m * amu)
            return cls(z=z, m=m, vt=vt)
        else:
            # TODO: Method for specifying custom particle parameters?
            raise ValueError('Invalid particle species configuration')

    @classmethod
    def species(cls, name: str):
        match name:
            case 'e':
                return cls(z=-1, m=5.6856301e-9, vt=1/jnp.sqrt(5.6856301e-9))
            case 'D':
                return cls(z=1, m=2.08793698e-5, vt=1/jnp.sqrt(2.08793698e-5))
            case _:
                raise ValueError(f'Unknown species {name}')

# %% Definition of gyrokinetic Hamiltonian


class PusherArgs(NamedTuple):
    """
    NamedTuple holding arguments for all particle pushers. Note most things here are optional.
    """
    eq: Equilibrium
    pp: ParticleParams
    fields: list[AbstractFieldProvider]

    def compute_fields(self, t: Real, r: ScalarArrayLike, varphi: ScalarArrayLike, z: ScalarArrayLike, psi_ev: PsiTuple) -> ScalarFields:
        """
        Compute the fields at the given position and time.
        """
        fields = reduce(sum_fields, (f(t, r, varphi, z, psi_ev) for f in self.fields))
        return fields
    
    def compute_grad_fields(self, t: Real, r: ScalarArrayLike, varphi: ScalarArrayLike, z: ScalarArrayLike, psi_ev: PsiTuple) -> tuple[VectorFields, ScalarFields]:
        """
        Compute the field gradients at the given position and time.
        """
        grad_fields, fields = reduce(sum_field_grads, (f.grad_and_value(t, r, varphi, z, psi_ev) for f in self.fields))
        return grad_fields, fields

class PusherState(NamedTuple):
    """
    NamedTuple holding the state of a particle pusher
    """
    r: ScalarArray
    varphi: ScalarArray
    z: ScalarArray
    upar: ScalarArray
    mu: ScalarArray

    @classmethod
    def empty(cls, nq: int) -> PusherState:
        """
        Return an empty PusherState, of the appropriate shape for nq particles.
        """
        return cls(
            r=jnp.empty(nq),
            varphi=jnp.empty(nq),
            z=jnp.empty(nq),
            upar=jnp.empty(nq),
            mu=jnp.empty(nq)
        )

# %% Functions for gyrokinetic particle pushing with zonally symmetric equilibria

@jax.jit
def f_driftkinetic(t: Real, y: PusherState, args: PusherArgs):
    """
    Push a (single) drift-kinetic tracer in (R, varphi, Z) coordinates.
    Note y = (R, varphi, Z, u_||, mu). (TODO: Is this the best set of coordinates for EM?)
    """
    # Unpack the state
    r, varphi, z, upar, mu = y

    # Unpack the arguments
    eq = args.eq
    pp = args.pp
    #theta_map = args.theta_map
    
    ## Magnetic terms
    psi_ev, ff_ev = eq.compute_psi_and_ff(r, z)
    bv, bu, modb, gradmodb, curlbu = eq.compute_geom_terms(r, psi_ev, ff_ev)
    # Bstar and Bstar parallel
    bstar = bv + (pp.m / pp.z) * curlbu * upar[None, ...]
    bstarpar = jnp.sum(bu*bstar, axis=0)

    ## Compute the fields

    # Evaluate the fields and their gradients over the list of field providers
    grad_fields, fields = args.compute_grad_fields(t, r, varphi, z, psi_ev)
    # Unpack
    apar = fields.get('apar', jnp.zeros_like(r))
    (dapar_dr, dapar_dvarphi, dapar_dz) = grad_fields.get('apar', (jnp.zeros_like(r), jnp.zeros_like(r), jnp.zeros_like(r)))
    (dphi_dr, dphi_dvarphi, dphi_dz) = grad_fields.get('phi', (jnp.zeros_like(r), jnp.zeros_like(r), jnp.zeros_like(r)))

    ## Compute gradients of the Hamiltonian

    # p_|| = dH/du_||
    ppar = pp.m * upar - pp.z * apar

    # Electric potential gradient
    gradphi = jnp.array([dphi_dr, dphi_dvarphi / r, dphi_dz])
    # Magnetic potential gradient
    gradapar = jnp.array([dapar_dr, dapar_dvarphi / r, dapar_dz])
    
    ## Finally compute the (spatial) gradient of the Hamiltonian
    gradh = mu * gradmodb + pp.z * gradphi - (pp.z / pp.m) * ppar[None, ...] * gradapar

    rdot = (jnp.cross(bu, gradh, axis=0) / pp.z + ppar[None, ...] * bstar / pp.m) / bstarpar[None, ...]

    drdt = rdot[0, ...]
    dvarphidt = rdot[1, ...] / r
    dzdt = rdot[2, ...]
    dupardt = -(jnp.sum(bstar*gradh, axis=0) / bstarpar) / pp.m
    dmudt = jnp.zeros_like(mu)

    return PusherState(drdt, dvarphidt, dzdt, dupardt, dmudt)



# %% Term for implementing drift-kinetic pushing simultaneously with collisions

_Control = TypeVar("_Control", bound=Control)

def _prod(vf, control):
    return jnp.tensordot(jnp.conj(vf), control, axes=jnp.ndim(control))

def _sum(*x):
    return sum(x[1:], x[0])

class GyroSDETerm(AbstractTerm, Generic[_Control]):
    control: AbstractPath[_Control]

    def vf(self, t: RealScalarLike, y: PusherState, args: PusherArgs) -> tuple[PyTree[ArrayLike], ...]:
        # Unpack the state
        r, varphi, z, upar, mu = y

        # Unpack the arguments
        eq = args.eq
        pp = args.pp
        #theta_map = args.theta_map
        
        ## Magnetic terms
        psi_ev, ff_ev = eq.compute_psi_and_ff(r, z)
        bv, bu, modb, gradmodb, curlbu = eq.compute_geom_terms(r, psi_ev, ff_ev)
        # Bstar and Bstar parallel
        bstar = bv + (pp.m / pp.z) * curlbu * upar[None, ...]
        bstarpar = jnp.sum(bu*bstar, axis=0)

        ## Compute the fields

        # Evaluate the fields and their gradients over the list of field providers
        grad_fields, fields = args.compute_grad_fields(t, r, varphi, z, psi_ev)
        # Unpack
        apar = fields.get('apar', 0.0)
        (dapar_dr, dapar_dvarphi, dapar_dz) = grad_fields.get('apar', (0.0, 0.0, 0.0))
        (dphi_dr, dphi_dvarphi, dphi_dz) = grad_fields.get('phi', (0.0, 0.0, 0.0))

        ## Compute gradients of the Hamiltonian

        # p_|| = dH/du_||
        ppar = pp.m * upar - pp.z * apar

        # Electric potential gradient
        gradphi = jnp.array([dphi_dr, dphi_dvarphi / r, dphi_dz])
        # Magnetic potential gradient
        gradapar = jnp.array([dapar_dr, dapar_dvarphi / r, dapar_dz])
        
        ## Finally compute the (spatial) gradient of the Hamiltonian
        gradh = mu * gradmodb + pp.z * gradphi - (pp.z / pp.m) * ppar[None, ...] * gradapar

        rdot = (jnp.cross(bu, gradh, axis=0) / pp.z + ppar[None, ...] * bstar / pp.m) / bstarpar[None, ...]

        ## Drift part of the SDE
        drdt = rdot[0, ...]
        dvarphidt = rdot[1, ...] / r
        dzdt = rdot[2, ...]
        dupardt = -(jnp.sum(bstar*gradh, axis=0) / bstarpar) / pp.m
        dmudt = jnp.zeros_like(mu)

        vf_drift = PusherState(drdt, dvarphidt, dzdt, dupardt, dmudt)

        ## Diffusion part of the SDE

        ## Compute coordinate transform
        # kinetic momentum, I think?
        modp = jnp.sqrt(ppar**2 + 2 * pp.m * mu * modb)
        # pitch angle
        xi = ppar / modp

        kappax = 1.0
        kappapar = 1.0
        kappaperp = 1.0

        
        drdwt = (jnp.array([1.0 - bu[0, ...]**2, -bu[0, ...] * bu[1, ...], -bu[0, ...] * bu[2, ...]]) * kappax)
        dvarphidwt = (jnp.array([-bu[1, ...] * bu[0, ...], 1.0 - bu[1, ...]**2, -bu[1, ...] * bu[2, ...]]) * kappax / r[None, ...])
        dzdwt = (jnp.array([-bu[2, ...] * bu[0, ...], -bu[2, ...] * bu[1, ...], 1.0 - bu[2, ...]**2]) * kappax)
        dupardwt = (jnp.array([xi, modp]) * (kappapar / pp.m))
        dmudwt = (jnp.array([modp * (1 - xi**2), -modp**2 * xi]) * (jnp.sqrt((1-xi**2) / modp**2) / (pp.m * modb)) * kappaperp)

        vf_diff = PusherState(drdwt.T, dvarphidwt.T, dzdwt.T, dupardwt.T, dmudwt.T)

        return (vf_drift, vf_diff)
    
    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> tuple[PyTree[ArrayLike], ...]:
        return (t1-t0, self.control.evaluate(t0, t1, **kwargs))

    def prod(
        self, vf: tuple[PyTree[ArrayLike], ...], control: tuple[PyTree[ArrayLike], ...]
    ) -> Y:
        vf_drift, vf_diff = vf
        contr_drift, contr_diff = control
        out = (
            jtu.tree_map(lambda v: contr_drift * v, vf_drift),
            jtu.tree_map(lambda v: _prod(v, contr_diff), vf_diff),
        )
        return jtu.tree_map(_sum, *out)
    
    def vf_prod(
        self,
        t: RealScalarLike,
        y: Y,
        args: Args,
        control: tuple[PyTree[ArrayLike], ...],
    ) -> Y:
        vf = self.vf(t, y, args)
        return self.prod(vf, control)


# %% Legacy code

# def f_driftkinetic_midplane(t, state, args: PusherArgs):
#     """
#     Push a drift-kinetic tracer, but also follow winding angles around the magnetic axis and around (v_||, Z), as well as the poloidal action
#     """
#     # Unpack the state
#     r, varphi, z, vll, mu, thetag, theta_vll, action = state
#     # We need these arguments
#     eq = args.eq
#     pp = args.pp
    
#     ## Magnetic terms
#     psi_ev, ff_ev = eq.compute_psi_and_ff(r, z)
#     (psi, psidr, psidz, psidrr, psidrz, psidzz) = psi_ev
#     bv, bu, modb, gradmodb, curlbu = eq.compute_geom_terms(r, psi_ev, ff_ev)
#     ## Bstar and Bstar parallel
#     bstar = bv + (pp.m / pp.z) * curlbu * vll[None, ...]
#     bstarll = jnp.sum(bu*bstar, axis=0)

#     ## Electric potential gradient
    
#     #dphi = compute_fields(t, r, z, varphi, psi_ev, eq, fields, frame)
#     dphi = jnp.zeros_like(gradmodb)
#     if args.zonal_fields is not None:
#         dzpot = args.zonal_fields(psi, dx=1)
#         dphi = jnp.array([psidr * dzpot, jnp.zeros_like(psidr), psidz * dzpot])
    

#     ## Finally compute the (spatial) gradient of the Hamiltonian
#     gradh = mu * gradmodb + pp.z * dphi

#     rdot = (jnp.cross(bu, gradh, axis=0) / pp.z + vll[None, ...] * bstar) / bstarll[None, ...]

#     drdt = rdot[0, ...]
#     dvarphidt = rdot[1, ...] / r
#     dzdt = rdot[2, ...]
#     dvlldt = -(jnp.sum(bstar*gradh, axis=0) / bstarll) / pp.m
#     dmudt = jnp.zeros_like(mu)

#     # Compute the winding rates
#     dthetagdt = ((r - eq.raxis) * dzdt - (z - eq.zaxis) * drdt) / ((r - eq.raxis)**2 + (z - eq.zaxis)**2)
#     dthetavlldt = ((vll / pp.vt) * dzdt - (z - eq.zaxis) * dvlldt / pp.vt) / ((z - eq.zaxis)**2 + (vll/pp.vt)**2)

#     # The one form is A \cdot dx + m v_|| b \cdot dx
#     # TODO: There's some sort of sign issue here that I need to think about more carefully?
#     oneformr = pp.z * args.vector_potential_r(r, z) + pp.m * vll * bu[0, ...]
#     oneformz = pp.m * vll * bu[2, ...]
#     dactiondt = -(oneformr * drdt + oneformz * dzdt) / (2 * jnp.pi)

#     # Return everything
#     return (drdt, dvarphidt, dzdt, dvlldt, dmudt, dthetagdt, dthetavlldt, dactiondt)


# def compute_driftkinetic_integrals(t, state, args: PusherArgs):
#     """
#     Compute the toroidal angular momentum and the Hamiltonian
#     """
#     # Unpack the state
#     r, varphi, z, vll, mu = state
#     eq = args.eq
#     pp = args.pp

#     psi = eq.interp_psi(r, z)
#     bv = eq.compute_bv(r, z)
#     modb = jnp.linalg.norm(bv, axis=0)

#     pot = args.zonal_fields(psi)

#     ham = 0.5 * pp.m * vll**2 + mu * modb + pp.z * pot
#     lphi = pp.z * psi + pp.m * vll * r * bv[1,...] / modb

#     return lphi, ham