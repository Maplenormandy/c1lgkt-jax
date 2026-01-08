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

    # Eigendecomposition of the Hessian, H = V W V^T
    w: Real[ArrayLike, "2"]
    v: Real[ArrayLike, "2 2"]

    # Q matrix, which gives Sylvester's inertia for the Hessian H = Q D Q^T where D = diag(\pm 1, \pm 1).
    q: Real[ArrayLike, "2 2"]
    qinv: Real[ArrayLike, "2 2"]

class MagneticNullInfo(NamedTuple):
    """
    NamedTuple which collects the main magnetic nulls in the equilibrium, as well as some auxiliary shaping parameters.
    x1 is the 'main' null, x2 is the secondary null, and axis is the magnetic axis
    """
    axis: MagneticNull
    x1: MagneticNull
    x2: MagneticNull

    # Minor radius of the outboard midplane, (rmax - rmin)/2
    a: Real

def _objective_magnetic_null(y: Real[ArrayLike, "2"], args: Equilibrium) -> Real[ArrayLike, "2"]:
    """
    Objective function for finding magnetic nulls, i.e. points where grad(psi) = 0
    """
    eq = args
    dr = eq.interp_psi(y[0], y[1], dx=1)
    dz = eq.interp_psi(y[0], y[1], dy=1)
    return jnp.array([dr, dz])


# %% Functions associated with computing theta via (u,v) fields

type UvParams = Real[ArrayLike, "8"]

def _compute_uv(r: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"], nulls: MagneticNullInfo, params: UvParams) -> tuple[Real[ArrayLike, "N"], Real[ArrayLike, "N"]]:
    """
    Computes the auxiliary (u,v) fields used to compute theta = arctan2(v,u). The strategy is to place zeros of a
    complex(-ish) function f(z) = (z-z0) (z-z1) (z-z2), then compute u = Re(f(z)) and v = Im(f(z)).
    
    z0 corresponds to the magnetic axis, while z1 and z2 correspond to the fictional o-points.
    This is the internal function with parameters that specify the locations of the fictional o-points.
    """
    # Local coordinates near the axis
    eta0 = jnp.tensordot(nulls.axis.q.T, jnp.array([r - nulls.axis.rz[0], z - nulls.axis.rz[1]]), axes=1)

    # Local coordinates for fictional o-points. Note unit normalization does not matter.
    eta1 = jnp.array([r - params[0], z - params[1]])
    eta2 = jnp.array([r - params[2], z - params[3]])

    # Local coordinates near x-point
    xi1 = jnp.tensordot(nulls.x1.q.T, jnp.array([r - nulls.x1.rz[0], z - nulls.x1.rz[1]]), axes=1)
    xi2 = jnp.tensordot(nulls.x2.q.T, jnp.array([r - nulls.x2.rz[0], z - nulls.x2.rz[1]]), axes=1)

    a_norm = nulls.a * jnp.sqrt(jnp.abs(nulls.axis.w[0]))

    # Complex coordinates for the u, v field
    z0 = jnp.tanh(eta0[0,...]/a_norm) + 1j*(eta0[1,...]/a_norm)
    z1 = eta1[0,...] + 1j*eta1[1,...] + (params[4] + 1j*params[5]) * jnp.exp(-0.5 * (xi1[0,...]**2 + xi1[1,...]**2))
    z2 = eta2[0,...] + 1j*eta2[1,...] + (params[6] + 1j*params[7]) * jnp.exp(-0.5 * (xi2[0,...]**2 + xi2[1,...]**2))

    uv = z0 * z1 * z2

    return jnp.real(uv), jnp.imag(uv)

# Jacobian of (u,v) with respect to (r,z)
_jac_uv = jax.jacfwd(_compute_uv, argnums=(0,1))

def _grad_theta(r: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"], nulls: MagneticNullInfo, params: UvParams) -> tuple[Real[ArrayLike, "N"], Real[ArrayLike, "N"]]:
    """
    Computes the smoothly-varying gradient of theta = arctan2(v,u) (i.e. across the branch cut) with respect to (r,z)
    using the auxiliary (u,v) fields
    """
    u, v = _compute_uv(r, z, nulls, params)
    duv_drz = _jac_uv(r, z, nulls, params)

    dudr, dudz = duv_drz[0]
    dvdr, dvdz = duv_drz[1]

    uv2 = u**2 + v**2

    return ((u * dvdr - v * dudr) / uv2, (u * dvdz - v * dudz) / uv2)

# Hessian of theta
_hess_theta = jax.jacfwd(_grad_theta, argnums=(0, 1))


## These functions help compute the parameters needed for the (u,v) field

def _objective_uv_params(y: Real[ArrayLike, "4"], args: MagneticNullInfo) -> Real[ArrayLike, "4"]:
    """
    Objective function which ensures that grad(theta) = 0 at the x-points. This is the coarse version without inertia constraints.
    """
    nulls = args

    # Augment y to full 8 parameters with zeros for inertia-related params
    y_aug = jnp.array([y[0], y[1], y[2], y[3], 0.0, 0.0, 0.0, 0.0])

    # Compute grad(theta) at the x-points
    dth1 = jnp.array(_grad_theta(nulls.x1.rz[0], nulls.x1.rz[1], nulls, y_aug))
    dth2 = jnp.array(_grad_theta(nulls.x2.rz[0], nulls.x2.rz[1], nulls, y_aug))

    # Return grad(theta) components at both x-points
    return jnp.array([dth1[0], dth1[1], dth2[0], dth2[1]])

def _objective_uv_params_refine(y: Real[ArrayLike, "8"], args: MagneticNullInfo) -> Real[ArrayLike, "8"]:
    """
    Objective function which ensures that grad(theta) = 0 at the x-points, as well as enforcing inertia constraints.
    """
    nulls = args

    # Compute grad(theta) at the x-points
    dth1 = jnp.array(_grad_theta(nulls.x1.rz[0], nulls.x1.rz[1], nulls, y))
    dth2 = jnp.array(_grad_theta(nulls.x2.rz[0], nulls.x2.rz[1], nulls, y))

    # Compute Hessian of theta at the x-points
    hth1 = jnp.array(_hess_theta(nulls.x1.rz[0], nulls.x1.rz[1], nulls, y))
    hth2 = jnp.array(_hess_theta(nulls.x2.rz[0], nulls.x2.rz[1], nulls, y))

    # Compute the inertia at the x-points, which we want to take the form [[0, a], [a, 0]]
    inertia1 = nulls.x1.qinv @ hth1 @ nulls.x1.qinv.T
    inertia2 = nulls.x2.qinv @ hth2 @ nulls.x2.qinv.T

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

    jacobian = (psidr * gradtheta[1] - psidz * gradtheta[0])
    dalpha = bt / jacobian

    return (br/dtheta, bz/dtheta, dalpha/dtheta)

# %% Clebsch field class

class ClebschMapping(eqx.Module):
    """
    Class which is responsible for computing and handling Clebsch representations of magnetic fields, B = nabla psi x nabla alpha.
    This alpha is primarily used for the eikonal representation of microinstabilities in the presence of magnetic shear.
    """
    ## Info for main magnetic null points.
    nulls: MagneticNullInfo

    ## Parameters involved in (u,v) field computation
    uv_params: UvParams
    
    ## Inertia of theta near the x-points, which potentially could be used to compute the singular component of alpha separately
    #inertia1: Real
    #inertia2: Real
    # Here's the old code that computed inertia values
    #self.inertia1 = (self.x1.qinv @ jnp.array(self._hess_theta(self.x1.rz[0], self.x1.rz[1], sol_refine.value)) @ self.x1.qinv.T)[0,1]
    #self.inertia2 = (self.x2.qinv @ jnp.array(self._hess_theta(self.x2.rz[0], self.x2.rz[1], sol_refine.value)) @ self.x2.qinv.T)[0,1]

    ## Interpolator for alpha field
    interp_alpha: interpax.Interpolator2D

    ### Methods

    ## Public methods which use internally-stored parameters

    def compute_uv(self, r: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"]) -> tuple[Real[ArrayLike, "N"], Real[ArrayLike, "N"]]:
        """
        Computes the auxiliary (u,v) fields used to compute theta = arctan2(v,u) using the internally-stored parameters.
        """
        return _compute_uv(r, z, self.nulls, self.uv_params)
    
    def jac_uv(self, r: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"]) -> tuple[tuple[Real[ArrayLike, "N"], Real[ArrayLike, "N"]], tuple[Real[ArrayLike, "N"], Real[ArrayLike, "N"]]]:
        """
        Computes the Jacobian of the (u,v) fields with respect to (r,z) using the internally-stored parameters.
        """
        return _jac_uv(r, z, self.nulls, self.uv_params)

    def grad_theta(self, r: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"]) -> tuple[Real[ArrayLike, "N"], Real[ArrayLike, "N"]]:
        """
        Computes the smoothly-varying gradient of theta = arctan2(v,u) (i.e. across the branch cut) with respect to (r,z)
        using the auxiliary (u,v) fields and the internally-stored parameters.
        """
        return _grad_theta(r, z, self.nulls, self.uv_params)

    def hess_theta(self, r: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"]) -> tuple[tuple[Real[ArrayLike, "N"], Real[ArrayLike, "N"]], tuple[Real[ArrayLike, "N"], Real[ArrayLike, "N"]]]:
        """
        Computes the Hessian of theta = arctan2(v,u) with respect to (r,z) using the auxiliary (u,v) fields and the internally-stored parameters.
        """
        return _hess_theta(r, z, self.nulls, self.uv_params)
        

        

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

        # Compute Q for Sylvester's inertia
        q = v @ jnp.diag(jnp.sqrt(jnp.abs(w)))
        qinv = jnp.linalg.inv(q)

        return MagneticNull(
            rz=rz_null,
            psi=eq.interp_psi(rz_null[0], rz_null[1]),
            hess=hess,
            w=w, v=v, q=q, qinv=qinv
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
        
        return sol_refine.value
    
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

        # First, we want to integrate to theta = 0. Note that time is parameterized by theta
        sol_theta0 = diffrax.diffeqsolve(
            self.term, self.solver,
            t0=theta0, t1=0.0, dt0=5e-4, y0=y0,
            args=args,
            stepsize_controller=self.stepsize_controller,
            event=self.event,
            max_steps=16,
            throw=False
        )

        # Now, we get a new initial state at theta = 0
        y0 = y0 = (sol_theta0.ys[0][0], sol_theta0.ys[1][0], sol_theta0.ys[2][0])

        # Next, we want to integrate to \pm 4 pi in theta to get a full field line
        sol_fwd = diffrax.diffeqsolve(
            self.term, self.solver,
            t0=0.0, t1=4*jnp.pi+1e-3, dt0=5e-4, y0=y0,
            args=args,
            saveat=self.saveats[0],
            stepsize_controller=self.stepsize_controller,
            event=self.event,
            max_steps=512,
            throw=False
        )
        sol_bak = diffrax.diffeqsolve(
            self.term, self.solver,
            t0=0.0, t1=-4*jnp.pi-1e-3, dt0=-5e-4, y0=y0,
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

    def clebsch_from_equilibrium(self, eq: Equilibrium) -> ClebschMapping:
        ## First, compute the locations of the main magnetic nulls.
        axis = self._find_magnetic_null(jnp.array([eq.raxis, eq.zaxis]), eq)
        x1 = self._find_magnetic_null(jnp.array([eq.rx, eq.zx]), eq)
        # TODO: Figure out a better way to guess the location of this second null
        x2 = self._find_magnetic_null(jnp.array([1.2321, 1.1871]), eq)

        ## Compute minor radius
        a = 0.5 * (jnp.max(eq.lcfsrz[0,:]) - jnp.min(eq.lcfsrz[0,:]))

        ## Assemble null info
        nulls = MagneticNullInfo(axis=axis, x1=x1, x2=x2, a=a)

        ## Next, compute the parameters for the (u,v) field such that grad(theta) = 0 at the x-points
        uv_params = self._find_uv_params(nulls)

        ## Finally, we want to compute alpha on a grid of psi and theta values for interpolation later
        r0 = jnp.sqrt(jnp.linspace(0.01, 0.99, 512)) * (eq.rmax - nulls.axis.rz[0]) + (nulls.axis.rz[0])
        z0 = jnp.full_like(r0, nulls.axis.rz[1])
        psi_eval = eq.interp_psi(r0, z0)

        alpha_eval = self._compute_alpha(r0, (eq, nulls, uv_params))

        interp_alpha = interpax.Interpolator2D(psi_eval, self.theta_eval, alpha_eval, method='monotonic', extrap=True)

        return ClebschMapping(
            nulls=nulls,
            uv_params=uv_params,
            interp_alpha=interp_alpha
        )

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
            diffrax.SaveAt(ts=self.theta_eval[512:0:-1])
        )