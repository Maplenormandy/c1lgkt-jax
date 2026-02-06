# %% -*- coding: utf-8 -*-
"""
@author: maple

Class which handles the Clebsch representation of magnetic fields
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import interpax
import optimistix as optx
import diffrax

import matplotlib.pyplot as plt
import matplotlib as mpl

from typing import Type, TypeVar, NamedTuple
from jaxtyping import ArrayLike, Real

from .equilibrium import Equilibrium

import re

from functools import partial

# %% Functions associated with magnetic nulls

class MagneticNull(NamedTuple):
    """
    NamedTuple which collects useful data for nulls in the poloidal magnetic field
    """
    # R, Z coordinates
    rz: Real[ArrayLike, "2"]
    # psi value of the null
    psi: Real

    # Hessian of psi at the null
    hess: Real[ArrayLike, "2 2"]

    # Eigendecomposition of the Hessian, H = Q Lambda Q^T
    lam: Real[ArrayLike, "2"]
    q: Real[ArrayLike, "2 2"]

    # S matrix, which gives Sylvester's inertia for the Hessian H = S D S^T where D = diag(\pm 1, \pm 1).
    s: Real[ArrayLike, "2 2"]
    sinv: Real[ArrayLike, "2 2"]

class MagneticNullInfo(NamedTuple):
    """
    NamedTuple which collects the main magnetic nulls in the equilibrium, as well as some auxiliary shaping parameters.
    x1 is the 'main' null, x2 is the secondary null, and axis is the magnetic axis
    """
    axis: MagneticNull
    x1: MagneticNull
    x2: MagneticNull

    # Minor radius of the outboard midplane, (rmax - rmin)/2
    amid: Real

def _objective_magnetic_null(y: Real[ArrayLike, "2"], args: Equilibrium) -> Real[ArrayLike, "2"]:
    """
    Objective function for finding magnetic nulls, i.e. points where grad(psi) = 0
    """
    eq = args
    dr = eq.interp_psi(y[0], y[1], dx=1)
    dz = eq.interp_psi(y[0], y[1], dy=1)
    return jnp.array([dr, dz])


# %% Functions associated with computing theta via (u,v) fields

#type UvParams = Real[ArrayLike, "8"]

class UvParams(NamedTuple):
    """
    NamedTuple which holds parameters for computing the (u,v) fields used to define theta = arctan2(v,u)
    """
    # Fictional o-point locations
    o1: Real[ArrayLike, "2"]
    o2: Real[ArrayLike, "2"]
    # Inertia-related parameters
    p1: Real[ArrayLike, "2"]
    p2: Real[ArrayLike, "2"]

    def from_array(arr: Real[ArrayLike, "8"]) -> "UvParams":
        return UvParams(
            o1=arr[0:2],
            o2=arr[2:4],
            p1=arr[4:6],
            p2=arr[6:8],
        )

@jax.jit
def _compute_uv(r: Real, z: Real, nulls: MagneticNullInfo, params: UvParams) -> tuple[Real, Real]:
    """
    Computes the auxiliary (u,v) fields used to compute theta = arctan2(v,u). The strategy is to place zeros of a
    complex(-ish) function f(z) = (z-z0) (z-z1) (z-z2), then compute u = Re(f(z)) and v = Im(f(z)).
    
    z0 corresponds to the magnetic axis, while z1 and z2 correspond to the fictional o-points.
    This is the internal, unbatched function with parameters that specify the locations of the fictional o-points.
    """
    # Pack coordinate array
    rz = jnp.array([r, z])

    # Local coordinates near the axis
    eta0 = (rz - nulls.axis.rz) @ nulls.axis.s

    # Local coordinates for fictional o-points. Note unit normalization does not matter.
    eta1 = rz - params.o1
    eta2 = rz - params.o2

    # Local coordinates near x-point
    xi1 = (rz - nulls.x1.rz) @ nulls.x1.s
    xi2 = (rz - nulls.x2.rz) @ nulls.x2.s

    a_norm = nulls.amid * jnp.sqrt(jnp.abs(nulls.axis.lam[0]))

    # Complex coordinates for the u, v field
    z0 = jnp.tanh(eta0[0]/a_norm) + 1j*(eta0[1]/a_norm)
    z1 = eta1[0] + 1j*eta1[1]
    z2 = eta2[0] + 1j*eta2[1]

    xi1_sq = xi1**2
    xi2_sq = xi2**2

    uv = z0 * z1 * z2
    uv = uv * (1 + 1j * (jnp.dot(params.p1, xi1_sq - 1.0) * jnp.exp(-0.5 * jnp.sum(xi1_sq) / (nulls.x1.psi)**2) + jnp.dot(params.p2, xi2_sq - 1.0) * jnp.exp(-0.5 * jnp.sum(xi2_sq) / (nulls.x2.psi)**2)))

    return jnp.real(uv), jnp.imag(uv)


@jax.jit
def _grad_theta(r: Real, z: Real, nulls: MagneticNullInfo, params: UvParams) -> tuple[Real, Real]:
    """
    Computes the smoothly-varying gradient of theta = arctan2(v,u) (i.e. across the branch cut) with respect to (r,z)
    using the auxiliary (u,v) fields.

    This is the internal, unbatched function with parameters that specify the locations of the fictional o-points.
    """
    (u, v), vjp_fun = jax.vjp(lambda r_in, z_in: _compute_uv(r_in, z_in, nulls, params), r, z)
    uv2 = u**2 + v**2

    return vjp_fun((-v / uv2, u / uv2))

# Hessian of theta
_hess_theta = jax.jacfwd(_grad_theta, argnums=(0, 1))


## These functions help compute the parameters needed for the (u,v) field

@jax.jit
def _objective_uv_params(y: Real[ArrayLike, "4"], args: MagneticNullInfo) -> Real[ArrayLike, "4"]:
    """
    Objective function which ensures that grad(theta) = 0 at the x-points. This is the coarse version without inertia constraints.
    """
    nulls = args

    # Augment y to full 8 parameters with zeros for inertia-related params
    y_aug = jnp.array([y[0], y[1], y[2], y[3], 0.0, 0.0, 0.0, 0.0])
    # convert to params
    uv_params = UvParams.from_array(y_aug)

    # Compute grad(theta) at the x-points
    dth1 = jnp.array(_grad_theta(nulls.x1.rz[0], nulls.x1.rz[1], nulls, uv_params))
    dth2 = jnp.array(_grad_theta(nulls.x2.rz[0], nulls.x2.rz[1], nulls, uv_params))

    # Return grad(theta) components at both x-points
    return jnp.array([dth1[0], dth1[1], dth2[0], dth2[1]])

@jax.jit
def _objective_uv_params_refine(y: Real[ArrayLike, "8"], args: MagneticNullInfo) -> Real[ArrayLike, "8"]:
    """
    Objective function which ensures that grad(theta) = 0 at the x-points, as well as enforcing inertia constraints.
    """
    nulls = args

    # Convert to params
    uv_params = UvParams.from_array(y)

    # Compute grad(theta) at the x-points
    dth1 = jnp.array(_grad_theta(nulls.x1.rz[0], nulls.x1.rz[1], nulls, uv_params))
    dth2 = jnp.array(_grad_theta(nulls.x2.rz[0], nulls.x2.rz[1], nulls, uv_params))

    # Compute Hessian of theta at the x-points
    hth1 = jnp.array(_hess_theta(nulls.x1.rz[0], nulls.x1.rz[1], nulls, uv_params))
    hth2 = jnp.array(_hess_theta(nulls.x2.rz[0], nulls.x2.rz[1], nulls, uv_params))

    # Compute the inertia at the x-points, which we want to take the form [[0, a], [a, 0]]
    inertia1 = nulls.x1.sinv @ hth1 @ nulls.x1.sinv.T
    inertia2 = nulls.x2.sinv @ hth2 @ nulls.x2.sinv.T

    # Return grad(theta) components and inertia components at both x-points
    return jnp.array([dth1[0], dth1[1], dth2[0], dth2[1], inertia1[0,0], inertia1[1,1], inertia2[0,0], inertia2[1,1]])



# %% Functions related to solving for alpha

type ClebschFieldlineArgs = tuple[Equilibrium, MagneticNullInfo, UvParams]

def cond_bounding_box(t, y, args: ClebschFieldlineArgs, **kwargs):
    """
    Condition function to check if the field line is within the bounding box of the EFIT domain
    """
    # Unpack args
    eq, _, _ = args

    # Unpack state
    r, z, alpha = y

    return (r - eq.rmin) * (eq.rmax - r) * (z - eq.zmin) * (eq.zmax - z)

@jax.jit
def fn_fieldline(t, y, args: ClebschFieldlineArgs):
    """
    """
    # Unpack args
    eq, nulls, uv_params = args

    # Unpack state
    r, z, alpha = y

    br, bt, bz = eq.compute_bv(r, z)

    gradtheta = _grad_theta(r, z, nulls, uv_params)
    psidr = eq.interp_psi(r, z, dx=1)
    psidz = eq.interp_psi(r, z, dy=1)

    dtheta = br * gradtheta[0] + bz * gradtheta[1]

    # Note: jacobian is actually r * dtheta; noting that dvarphi = bt / r, dalpha = dvarphi / dtheta
    jacobian = (psidr * gradtheta[1] - psidz * gradtheta[0])
    dalpha = bt / jacobian

    return (br/dtheta, bz/dtheta, dalpha)

# %% Clebsch field class

class ThetaMapping(eqx.Module):
    """
    Class which is responsible for computing the theta mapping and its derivatives
    """
    ## Info for main magnetic null points.
    nulls: MagneticNullInfo

    ## Parameters involved in (u,v) field computation
    uv_params: UvParams
    
    ## Inertia of theta near the x-points, which potentially could be used to compute the singular component of alpha separately
    #inertia1: Real
    #inertia2: Real
    # Here's the old code that computed inertia values
    #self.inertia1 = (self.x1.sinv @ jnp.array(self._hess_theta(self.x1.rz[0], self.x1.rz[1], sol_refine.value)) @ self.x1.sinv.T)[0,1]
    #self.inertia2 = (self.x2.sinv @ jnp.array(self._hess_theta(self.x2.rz[0], self.x2.rz[1], sol_refine.value)) @ self.x2.sinv.T)[0,1]

    ### Methods

    ## Public methods which use internally-stored parameters

    def _compute_uv(self, r: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"]) -> tuple[Real[ArrayLike, "N"], Real[ArrayLike, "N"]]:
        """
        Computes the auxiliary (u,v) fields used to compute theta = arctan2(v,u) using the internally-stored parameters.
        """
        return jax.vmap(_compute_uv, in_axes=(0,0,None,None))(r, z, self.nulls, self.uv_params)
    
    def __call__(self, r: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"]) ->  Real[ArrayLike, "N"]:
        """
        Computes theta = arctan2(v,u) using the auxiliary (u,v) fields and the internally-stored parameters.
        """
        u, v = self._compute_uv(r, z)
        return jnp.arctan2(v, u)

    def grad(self, r: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"]) -> tuple[Real[ArrayLike, "N"], Real[ArrayLike, "N"]]:
        """
        Computes the smoothly-varying gradient of theta = arctan2(v,u) (i.e. across the branch cut) with respect to (r,z)
        using the auxiliary (u,v) fields and the internally-stored parameters.
        """
        return jax.vmap(_grad_theta, in_axes=(0,0,None,None))(r, z, self.nulls, self.uv_params)

    def hessian(self, r: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"]) -> tuple[tuple[Real[ArrayLike, "N"], Real[ArrayLike, "N"]], tuple[Real[ArrayLike, "N"], Real[ArrayLike, "N"]]]:
        """
        Computes the Hessian of theta = arctan2(v,u) with respect to (r,z) using the auxiliary (u,v) fields and the internally-stored parameters.
        """
        return jax.vmap(_hess_theta, in_axes=(0,0,None,None))(r, z, self.nulls, self.uv_params)
        
# %% Class for Clebsch field?

class ClebschMapping(eqx.Module):
    """
    Class which is responsible for computing the Clebsch field (i.e. alpha mapping) and its derivatives as a function
    of (psi, theta), as well as providing weight functions to control branch cut behavior
    """
    interp_alpha: interpax.Interpolator2D

    def __call__(self, psi: Real[ArrayLike, "N"], theta: Real[ArrayLike, "N"]) -> tuple[Real[ArrayLike, "Nbranch N"], Real[ArrayLike, "Nbranch N"]]:
        """
        Computes alpha(psi, theta), sampled on the primary as well the +/-2pi branch cuts (in that order),
        along with weight functions that indicate the desired branch cut behavior. Nbranch is the number of branches (3), and N is the number of points.

        TODO: Experiment with computing the branches through a vector-valued interpolation
        """
        ## First, sample alpha
        alpha = self.interp_alpha(psi, theta)
        alpha_p = self.interp_alpha(psi, theta + 2*jnp.pi)
        alpha_n = self.interp_alpha(psi, theta - 2*jnp.pi)

        ## For now, let's just return uniform weights;
        return jnp.stack([alpha, alpha_p, alpha_n], axis=0), jnp.ones((3, ) + psi.shape)
        

# %% Clebsch field builder class

class ClebschMappingBuilder(eqx.Module):
    """
    Class which holds various solvers used in building the Clebsch field.
    """

    ## Newton solvers
    newton_magnetic_null: optx.Newton
    newton_uv_params: optx.Newton
    newton_uv_params_refine: optx.Newton

    ## ODE solver info
    term: diffrax.ODETerm
    solver: diffrax.AbstractSolver
    stepsize_controller: diffrax.AbstractStepSizeController
    event: diffrax.Event
    saveats: tuple[diffrax.SaveAt, diffrax.SaveAt]

    ## Values of psi and theta to evaluate at
    theta_eval: Real[ArrayLike, "1024"]

    def _find_magnetic_null(self, rz_guess: Real[ArrayLike, "2"], eq: Equilibrium) -> MagneticNull:
        """
        Takes a guess for the R,Z coordinates of a magnetic null and refines it using Newton's method. Then, computes
        auxiliary information such as the Hessian, eigenvalues/eigenvectors, and Sylvester's inertia matrix Q.
        """

        # Refine the magnetic null using Newton's method
        sol = optx.root_find(_objective_magnetic_null, self.newton_magnetic_null, rz_guess, args=eq, throw=False)
        rz_null = sol.value

        # Compute the Hessian of psi at the null
        psidrr = eq.interp_psi(rz_null[0], rz_null[1], dx=2)
        psidrz = eq.interp_psi(rz_null[0], rz_null[1], dx=1, dy=1)
        psidzz = eq.interp_psi(rz_null[0], rz_null[1], dy=2)
        hess = jnp.array([[psidrr, psidrz], [psidrz, psidzz]])

        # Compute eigenvalue information
        w, v = jnp.linalg.eigh(hess)

        ## Reorder eigenvalues/vectors to ensure (R,Z)-like coordinates
        # Sort by R component magnitude descending
        w_permute = jnp.argsort(-jnp.abs(v[0, :]))
        w = w[w_permute]
        v = v[:, w_permute]
        # Ensure that first eigenvector points roughly in +R direction
        v = v.at[:, 0].set(jnp.sign(v[0, 0]) * v[:, 0])
        # Ensure right-handedness
        v = v.at[:, 1].set(jnp.sign(jnp.linalg.det(v)) * v[:, 1])

        # Compute Q for Sylvester's inertia; i.e. H = Q D Q^T where D = diag(\pm 1, \pm 1)
        q = v @ jnp.diag(jnp.sqrt(jnp.abs(w)))
        qinv = jnp.linalg.inv(q)

        return MagneticNull(
            rz=rz_null,
            psi=eq.interp_psi(rz_null[0], rz_null[1]),
            hess=hess,
            lam=w, q=v, s=q, sinv=qinv
        )
    
    def _init_empty_magnetic_null(self) -> MagneticNull:
        """
        Returns an empty MagneticNull object with all zeros.
        """
        return MagneticNull(
            rz=jnp.zeros((2,)),
            psi=0.0,
            hess=jnp.zeros((2,2)),
            lam=jnp.zeros((2,)),
            q=jnp.zeros((2,2)),
            s=jnp.zeros((2,2)),
            sinv=jnp.zeros((2,2))
        )
    
    def _find_uv_params(self, nulls: MagneticNullInfo) -> UvParams:
        """
        Computes the parameters for the (u,v) field such that grad(theta) = 0 at the x-points, with inertia constraints.
        """
        # Initial guess for (u,v) parameters based on null locations
        y0 = jnp.array([2*nulls.x1.rz[0] - nulls.axis.rz[0], 2*nulls.x1.rz[1] - nulls.axis.rz[1], 2*nulls.x2.rz[0] - nulls.axis.rz[0], 2*nulls.x2.rz[1] - nulls.axis.rz[1]])
        sol = optx.root_find(_objective_uv_params, self.newton_uv_params, y0, args=nulls, throw=False)
        # Refine the solution to enforce inertia constraints
        y0_refine = jnp.array([sol.value[0], sol.value[1], sol.value[2], sol.value[3], 0.0, 0.0, 0.0, 0.0])
        sol_refine = optx.root_find(_objective_uv_params_refine, self.newton_uv_params_refine, y0_refine, args=nulls, throw=False)
        
        return UvParams.from_array(sol_refine.value)
    
    @partial(jax.vmap, in_axes=(None, 0, None))
    def _compute_alpha(self, r0, args: ClebschFieldlineArgs):
        """
        Starting from initial major radius r0 on the magnetic axis midplane, computes alpha by integrating along the field line
        """
        # Unpack args
        eq, nulls, uv_params = args

        ## Compute remainder of initial state
        # Initial Z is at the magnetic axis plane
        z0 = jnp.full_like(r0, nulls.axis.rz[1])
        # We don't necessarily start at theta = 0, so compute initial theta from (u,v)
        u0, v0 = _compute_uv(r0, z0, nulls, uv_params)
        theta0 = jnp.arctan2(v0, u0)

        # Initial state (r, z, alpha=0)
        y0 = (r0, z0, jnp.zeros_like(r0))

        # First, we want to integrate to theta = 0. Note that time is parameterized by theta. We manually perform an RK4 step
        k1 = fn_fieldline(theta0, y0, args)
        y1 = jax.tree.map(lambda a, b: a - 0.5 * b * (theta0/2), y0, k1)
        k2 = fn_fieldline(theta0/2, y1, args)
        y2 = jax.tree.map(lambda a, b: a - 0.5 * b * (theta0/2), y0, k2)
        k3 = fn_fieldline(theta0/2, y2, args)
        y3 = jax.tree.map(lambda a, b: a - b * (theta0), y0, k3)
        k4 = fn_fieldline(0.0, y3, args)
        y0 = jax.tree.map(lambda a, b, c, d, e: a - (b + 2*c + 2*d + e) * (theta0/6), y0, k1, k2, k3, k4)

        # Next, we want to integrate to \pm 4 pi in theta to get a full field line
        sol_fwd = diffrax.diffeqsolve(
            self.term, self.solver,
            t0=0.0, t1=4*jnp.pi+1e-3, dt0=5e-2, y0=y0,
            args=args,
            saveat=self.saveats[0],
            stepsize_controller=self.stepsize_controller,
            event=self.event,
            max_steps=512,
            throw=False
        )
        sol_bak = diffrax.diffeqsolve(
            self.term, self.solver,
            t0=0.0, t1=-4*jnp.pi-1e-3, dt0=-5e-2, y0=y0,
            args=args,
            saveat=self.saveats[1],
            stepsize_controller=self.stepsize_controller,
            event=self.event,
            max_steps=512,
            throw=False
        )

        r_fwd, z_fwd, alpha_fwd = sol_fwd.ys
        r_bak, z_bak, alpha_bak = sol_bak.ys

        # Find the first infinite value index in the forward solution
        fwd_ind = jnp.argmax(jnp.isinf(alpha_fwd))
        # Replace inf values with the last finite value
        alpha_fwd = jnp.where(jnp.isinf(alpha_fwd), alpha_fwd[fwd_ind-1], alpha_fwd)

        # Find the first infinite value index in the backward solution
        bak_ind = jnp.argmax(jnp.isinf(alpha_bak))
        # Replace inf values with the last finite value
        alpha_bak = jnp.where(jnp.isinf(alpha_bak), alpha_bak[bak_ind-1], alpha_bak)

        # Join together the solution and return it
        return jnp.concatenate([alpha_bak[::-1], alpha_fwd[:]], axis=0)
    
    def build_theta_map(self, eq: Equilibrium) -> ThetaMapping:
        """
        Computes the theta mapping from the given equilibrium
        """
        ## First, compute the locations of the main magnetic nulls.
        axis = self._find_magnetic_null(jnp.array([eq.raxis, eq.zaxis]), eq)
        x1 = self._find_magnetic_null(jnp.array([eq.rx, eq.zx]), eq)
        # Guess the location of the second null by flipping the primary null across the magnetic axis midplane
        x2 = self._find_magnetic_null(jnp.array([eq.rx, 2*eq.zaxis - eq.zx]), eq)

        ## Compute minor radius
        a = 0.5 * (jnp.max(eq.lcfsrz[0,:]) - jnp.min(eq.lcfsrz[0,:]))

        ## Assemble null info
        nulls = MagneticNullInfo(axis=axis, x1=x1, x2=x2, amid=a)

        ## Next, compute the parameters for the (u,v) field such that grad(theta) = 0 at the x-points
        uv_params = self._find_uv_params(nulls)

        ## Construct the theta_map object
        theta_map = ThetaMapping(
            nulls=nulls,
            uv_params=uv_params,
        )

        return theta_map
    
    def load_theta_map(self, eqx_filename: str) -> ThetaMapping:
        """
        Loads a theta mapping from a given eqx file
        """
        # Open up the file
        with open(eqx_filename, 'rb') as f:
            # Create an empty theta map to use as a template
            nulls_empty = MagneticNullInfo(
                axis=self._init_empty_magnetic_null(),
                x1=self._init_empty_magnetic_null(),
                x2=self._init_empty_magnetic_null(),
                amid=0.0)
            theta_map_empty = ThetaMapping(
                nulls=nulls_empty,
                uv_params=UvParams.from_array(jnp.zeros((8,))),
            )
            # Deserialize the theta map
            theta_map = eqx.tree_deserialise_leaves(f, theta_map_empty)

        return theta_map

    def build_clebsch(self, theta_map: ThetaMapping, eq: Equilibrium) -> ClebschMapping:
        """
        Computes the Clebsch mapping from the given equilibrium
        """
        nulls = theta_map.nulls
        uv_params = theta_map.uv_params

        ## Finally, we want to compute alpha on a grid of psi and theta values for interpolation later
        r0 = jnp.sqrt(jnp.linspace(0.01, 0.99, 512)) * (eq.rmax - nulls.axis.rz[0]) + (nulls.axis.rz[0])
        z0 = jnp.full_like(r0, nulls.axis.rz[1])
        psi_eval = eq.interp_psi(r0, z0)

        alpha_eval = self._compute_alpha(r0, (eq, nulls, uv_params))

        interp_alpha = interpax.Interpolator2D(psi_eval, self.theta_eval, alpha_eval, method='monotonic', extrap=True)

        clebsch = ClebschMapping(
            interp_alpha=interp_alpha
        )

        return clebsch
    
    def load_clebsch(self, eqx_filename: str) -> ClebschMapping:
        """
        Loads a Clebsch mapping from a given eqx file
        """
        # Open up the file
        with open(eqx_filename, 'rb') as f:
            # Create an empty clebsch mapping to use as a template
            psi_eval = jnp.linspace(0.0, 1.0, 512)  # Placeholder; actual values will be loaded
            theta_eval = jnp.linspace(-4*jnp.pi, 4*jnp.pi, 1024, endpoint=False)
            alpha_eval = jnp.zeros((512, 1024))  # Placeholder; actual values will be loaded
            interp_alpha_empty = interpax.Interpolator2D(psi_eval, theta_eval, alpha_eval, method='monotonic', extrap=True)
            clebsch_empty = ClebschMapping(
                interp_alpha=interp_alpha_empty
            )
            # Deserialize the clebsch mapping
            clebsch = eqx.tree_deserialise_leaves(f, clebsch_empty)

        return clebsch

    def __init__(self):
        """
        Sets up solvers for various steps in building the Clebsch field.
        """

        ## Newton solvers
        self.newton_magnetic_null = optx.Newton(rtol=1e-12, atol=1e-12)
        self.newton_uv_params = optx.Newton(rtol=1e-8, atol=1e-8)
        self.newton_uv_params_refine = optx.Newton(rtol=1e-8, atol=1e-8)

        ## ODE solver info
        self.term = diffrax.ODETerm(fn_fieldline)
        self.solver = diffrax.Dopri5()
        self.stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
        self.event = diffrax.Event(cond_bounding_box)

        # Values of theta to evaluate at for the alpha integration
        self.theta_eval = jnp.linspace(-4*jnp.pi, 4*jnp.pi, 1024, endpoint=False)

        self.saveats = (
            diffrax.SaveAt(ts=self.theta_eval[512:]),
            diffrax.SaveAt(ts=self.theta_eval[511::-1])
        )