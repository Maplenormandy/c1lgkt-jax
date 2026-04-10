"""
Contains codes for the vector fields associated with particle motion.

Typically everything is done in cylindrical coordinates (R, phi, Z).
"""

from typing import NamedTuple, TypeVar, Generic, Literal

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

from functools import partial, reduce

from diffrax import AbstractTerm

from ..custom_types import ScalarArray, ScalarArrayLike, ScalarFields, VectorFields

from jax.scipy.special import erf

# %% Arguments for the various pushers

# Reference charge in units of C
q_ref = 1.60217663e-19
# Reference energy in units of J
e_ref = 1.60217663e-16
# Reference mass in kg
m_kev = 1.60217663e-22

# 1 amu in reference mass units
amu = 1.03642697e-5
# collisional prefactor (e^4 / (4 pi epsilon0^2)) in reference units
# => Google search calculator version: (electron charge) ^ 4 / (permittivity of free space) ^ 2 / (1.60217663e-22 kg)^2 * (1ms)^4 / ( 4 * pi)
e0_prefactor = 2.60563432e-23

class ParticleParams(NamedTuple):
    """
    NamedTuple holding particle properties.

    Some unit normalizations:
    - Length: 1m
    - Magnetic field: 1T
    - Electrostatic potential: 1kV
    - Energy: 1keV
    - Time: 1ms
    - Charge: e
    - Mass: (1e * 1T * 1m)^2 / (1 keV) = 1.60217663e-22 kg
      => google search calculator version: (1 electron charge * 1 T * 1 m)^2 / 1 keV
      
    z: float
        Particle charge state (in units of e)
    m: float
        Particle mass (in reference mass units, i.e. 1.03642697e-5 for 1 amu)
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

def _inc_gamma_psi(x: Real) -> Real:
    R"""
    The regularized incomplete lower gamma function, which appears in collisional expressions.
    $\psi(x) = 2 / \sqrt{\pi} \int_0^x \sqrt{t} e^{-t} dt$
    """
    return erf(jnp.sqrt(x)) - (2 / jnp.sqrt(jnp.pi)) * jnp.sqrt(x) * jnp.exp(-x)

_gv_inc_gamma_psi = jax.numpy.vectorize(jax.value_and_grad(_inc_gamma_psi))

@partial(jax.jit, static_argnames=['mode'])
def _gyro_vf(t: Real, y: PusherState, args: PusherArgs, *, mode: Literal["ode", "sde"] = 'ode') -> PusherState | tuple[PusherState, PusherState]:
    """
    Push gyrokinetic tracers in (R, varphi, Z, upar, mu) coordinates. If mode="ode", returns the
    drift-kinetic vector field. If mode="sde", returns a tuple of (drift vector field, diffusion
    vector field).
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

    ## Drift part of the ODE
    drdt = rdot[0, ...]
    dvarphidt = rdot[1, ...] / r
    dzdt = rdot[2, ...]
    dupardt = -(jnp.sum(bstar*gradh, axis=0) / bstarpar) / pp.m
    dmudt = jnp.zeros_like(mu)
    
    if mode == "ode":
        vf_drift = PusherState(drdt, dvarphidt, dzdt, dupardt, dmudt)
        return vf_drift
    
    elif mode == "sde":
        ## Diffusion part of the SDE

        ## Perpendicular coordinate vectors
        e2 = jnp.array([-bu[1, ...], bu[0, ...], jnp.zeros_like(bu[0, ...])])
        e3 = jnp.cross(bu, e2, axis=0)
        e2 = e2 / jnp.linalg.norm(e2, axis=0)
        e3 = e3 / jnp.linalg.norm(e3, axis=0)

        ## Compute coordinate transform
        # kinetic momentum, I think?
        modp2 = ppar**2 + 2 * pp.m * mu * modb
        modp = jnp.sqrt(modp2)
        # pitch angle
        xi = ppar / modp
        # perpendicular energy fraction
        e_perp = (1 - xi**2)


        ## Computing the diffusion coefficients; from NRL plasma formulary, converted to SI
        # A simple estimate for the coulomb logarithm for now
        coulomb_log = 15.0
        # Density and temperature of the colliding species
        n_beta = 1e19 # for now, constant density in m^-3
        t_beta = 0.25 # for now, constant temperature in keV
        m_beta = 2.08793698e-5 # for now, deuterium mass in reference units
        x_beta = m_beta * (modp2 / pp.m**2) / (2 * t_beta)
        # Collision frequency prefactor
        nu0 = e0_prefactor * n_beta * coulomb_log / (modp**3 / pp.m)
        # Terms which appear in the diffusion coefficients
        ginc, dginc = _gv_inc_gamma_psi(x_beta)

        # Diffusion and drag coefficients
        nu_s = ginc * nu0 * (pp.m/m_beta)
        diff_par = 0.5 * (ginc/x_beta) * nu0 * (modp2)
        diff_perp = ((1 - 0.5 / x_beta) * ginc + dginc) * nu0 * (modp2)
        diff_x = ((diff_par - diff_perp) * e_perp / 2.0 + diff_perp) / bstarpar**2

        # Compute the matrix elements; via Hirvijoki 2013
        kp_par = jnp.sqrt(2 * diff_par)
        kp_perp = jnp.sqrt(2 * diff_perp * e_perp / modp2)
        kp_x = jnp.sqrt(2 * diff_x)

        # Diffusion vector fields; spatial part
        drdwt = jnp.array([e2[0, ...], e3[0, ...]]) * kp_x
        dvarphidwt = jnp.array([e2[1, ...], e3[1, ...]]) * kp_x / r
        dzdwt = jnp.array([e2[2, ...], e3[2, ...]]) * kp_x
        # Diffusion vector fields; velocity part
        dupardwt = jnp.array([
            kp_par * xi / pp.m,
            kp_perp * modp / pp.m
            ])
        dmudwt = jnp.array([
            kp_par * 2 * mu / modp,
            -kp_perp * xi * modp2 / (pp.m * modb)
            ])

        # Compute drift terms plus drag. n.b. since the Heun method converges to the stratonovich
        # solution, we only include the drag term in the drift vector field
        vf_drift = PusherState(drdt, dvarphidt, dzdt, dupardt - nu_s*(ppar/pp.m) , dmudt - 2*nu_s*mu)
        vf_diff = PusherState(drdwt, dvarphidwt, dzdwt, dupardwt, dmudwt)

        return (vf_drift, vf_diff)

def f_driftkinetic(t: Real, y: PusherState, args: PusherArgs):
    """
    Push gyrokinetic tracers in (R, varphi, Z, upar, mu) coordinates.
    """
    return _gyro_vf(t, y, args, mode="ode")



# %% Term for implementing drift-kinetic pushing simultaneously with collisions

_Control = TypeVar("_Control", bound=Control)

def _prod(vf, control):
    return jnp.tensordot(jnp.conj(vf), control, axes=jnp.ndim(control))

def _sum(*x):
    return sum(x[1:], x[0])

class GyroSDETerm(AbstractTerm, Generic[_Control]):
    control1: AbstractPath[_Control]
    control2: AbstractPath[_Control]

    def vf(self, t: RealScalarLike, y: PusherState, args: PusherArgs) -> tuple[PusherState, PusherState]:
        return _gyro_vf(t, y, args, mode="sde")
    
    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> tuple[PyTree[ArrayLike], ...]:
        return (t1-t0, self.control1.evaluate(t0, t1, **kwargs), self.control2.evaluate(t0, t1, **kwargs))

    def prod(
        self, vf: tuple[PusherState, PusherState], control: tuple[PyTree[ArrayLike], ...]
    ) -> Y:
        vf_drift, vf_diff = vf
        contr_drift, contr_diff1, contr_diff2 = control
        
        # Note that we need a special product function here, since the diffusion vector field has
        # a block diagonal form that needs to be taken into account

        out_drift = jtu.tree_map(lambda v: contr_drift * v, vf_drift)
        out_diff = PusherState(
            jnp.sum(vf_diff.r * contr_diff1, axis=0),
            jnp.sum(vf_diff.varphi * contr_diff1, axis=0),
            jnp.sum(vf_diff.z * contr_diff1, axis=0),
            jnp.sum(vf_diff.upar * contr_diff2, axis=0),
            jnp.sum(vf_diff.mu * contr_diff2, axis=0)
        )
        return jtu.tree_map(_sum, out_drift, out_diff)
    
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