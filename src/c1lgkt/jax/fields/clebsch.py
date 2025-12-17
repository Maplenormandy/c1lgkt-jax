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

import matplotlib.pyplot as plt
import matplotlib as mpl

from typing import Type, TypeVar, NamedTuple
from jaxtyping import ArrayLike, Real

from .equilibrium import Equilibrium

import re

# %%

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

class ClebschField(eqx.Module):
    """
    Class which is responsible for computing and handling Clebsch representations of magnetic fields.
    """
    
    ## Magnetic equilibrium
    eq: Equilibrium

    ## Info for main magnetic null points. x1 is the 'main' null, x2 is the secondary null, and axis is the magnetic axis
    axis: MagneticNull
    x1: MagneticNull
    x2: MagneticNull

    # Normalized minor radius
    amin_norm: Real
    # Parameters for the (u,v) field computation
    uv_params: Real[ArrayLike, "8"]
    # Inertia of theta near the x-points
    inertia1: Real
    inertia2: Real

    def refine_magnetic_null(self, rz_guess: Real[ArrayLike, "2"]) -> MagneticNull:
        """
        Takes a guess for the R,Z coordinates of a magnetic null and refines it using Newton's method. Then, computes
        auxiliary information such as the Hessian, eigenvalues/eigenvectors, and Sylvester's inertia matrix Q.
        """

        ## Function to optimize via Newton's method
        def fn(y, args):
            dr = self.eq.interp_psi(y[0], y[1], dx=1)
            dz = self.eq.interp_psi(y[0], y[1], dy=1)
            return jnp.array([dr, dz])
        
        ## Set up the solver. Can use pretty fine tolerances since the problem is fairly easy.
        solver = optx.Newton(rtol=1e-12, atol=1e-12)

        # Refine the magnetic null using Newton's method
        sol = optx.root_find(fn, solver, rz_guess)
        rz_null = sol.value

        # Compute the Hessian of psi at the null
        psidrr = self.eq.interp_psi(rz_null[0], rz_null[1], dx=2)
        psidrz = self.eq.interp_psi(rz_null[0], rz_null[1], dx=1, dy=1)
        psidzz = self.eq.interp_psi(rz_null[0], rz_null[1], dy=2)
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
            psi=self.eq.interp_psi(rz_null[0], rz_null[1]),
            hess=hess,
            w=w, v=v, q=q, qinv=qinv
        )
    
    def compute_uv_(self, r: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"], params: Real[ArrayLike, "8"]) -> tuple[Real[ArrayLike, "N"], Real[ArrayLike, "N"]]:
        """
        Computes the auxiliary (u,v) fields used to compute theta = arctan2(v,u). The strategy is to place zeros of a
        complex(-ish) function f(z) = (z-z0) (z-z1) (z-z2), then compute u = Re(f(z)) and v = Im(f(z)).
        
        z0 corresponds to the magnetic axis, while z1 and z2 correspond to the fictional o-points.
        This is the internal function with parameters that specify the locations of the fictional o-points.
        """
        # Local coordinates near the axis
        eta0 = jnp.tensordot(self.axis.q.T, jnp.array([r - self.axis.rz[0], z - self.axis.rz[1]]), axes=1)

        # Local coordinates for fictional o-points. Note unit normalization does not matter.
        eta1 = jnp.array([r - params[0], z - params[1]])
        eta2 = jnp.array([r - params[2], z - params[3]])

        # Local coordinates near x-point
        xi1 = jnp.tensordot(self.x1.q.T, jnp.array([r - self.x1.rz[0], z - self.x1.rz[1]]), axes=1)
        xi2 = jnp.tensordot(self.x2.q.T, jnp.array([r - self.x2.rz[0], z - self.x2.rz[1]]), axes=1)

        # Complex coordinates for the u, v field
        z0 = jnp.tanh(eta0[0,...]/self.amin_norm) + 1j*(eta0[1,...]/self.amin_norm)
        z1 = eta1[0,...] + 1j*eta1[1,...] + (params[4] + 1j*params[5]) * jnp.exp(-0.5 * (xi1[0,...]**2 + xi1[1,...]**2))
        z2 = eta2[0,...] + 1j*eta2[1,...] + (params[6] + 1j*params[7]) * jnp.exp(-0.5 * (xi2[0,...]**2 + xi2[1,...]**2))

        uv = z0 * z1 * z2

        return jnp.real(uv), jnp.imag(uv)
    
    jac_uv_ = jax.jacfwd(compute_uv_, argnums=(1,2))

    def grad_theta_(self, r: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"], params: Real[ArrayLike, "8"]) -> tuple[Real[ArrayLike, "N"], Real[ArrayLike, "N"]]:
        """
        Computes the smoothly-varying gradient of theta = arctan2(v,u) (i.e. across the branch cut) with respect to (r,z)
        using the auxiliary (u,v) fields
        """
        u, v = self.compute_uv_(r, z, params)
        duv_drz = self.jac_uv_(r, z, params)

        dudr, dudz = duv_drz[0]
        dvdr, dvdz = duv_drz[1]

        uv2 = u**2 + v**2

        return ((u * dvdr - v * dudr) / uv2, (u * dvdz - v * dudz) / uv2)
        
    # Hessian of theta
    hess_theta_ = jax.jacfwd(grad_theta_, argnums=(1, 2))

    def __init__(self, eq: Equilibrium):
        self.eq = eq

        ## First, compute the locations of the main magnetic nulls.
        self.axis = self.refine_magnetic_null(jnp.array([self.eq.raxis, self.eq.zaxis]))
        self.x1 = self.refine_magnetic_null(jnp.array([self.eq.rx, self.eq.zx]))
        # TODO: Figure out a better way to guess the location of this second null
        self.x2 = self.refine_magnetic_null(jnp.array([1.2321, 1.1871]))

        self.amin_norm = 0.5 * (jnp.max(eq.lcfsrz[0,:]) - jnp.min(eq.lcfsrz[0,:])) * jnp.sqrt(jnp.abs(self.axis.w[0]))

        ## Objective function for the theta matching at x-points
        def fn_theta_objective(y, args):
            y_aug = jnp.array([y[0], y[1], y[2], y[3], 0.0, 0.0, 0.0, 0.0])
            dth1 = jnp.array(self.grad_theta_(self.x1.rz[0], self.x1.rz[1], y_aug))
            dth2 = jnp.array(self.grad_theta_(self.x2.rz[0], self.x2.rz[1], y_aug))

            return jnp.array([dth1[0], dth1[1], dth2[0], dth2[1]])

        ## Objective function for the theta matching at x-points
        def fn_theta_objective_refine(y, args):
            dth1 = jnp.array(self.grad_theta_(self.x1.rz[0], self.x1.rz[1], y))
            dth2 = jnp.array(self.grad_theta_(self.x2.rz[0], self.x2.rz[1], y))

            hth1 = jnp.array(self.hess_theta_(self.x1.rz[0], self.x1.rz[1], y))
            hth2 = jnp.array(self.hess_theta_(self.x2.rz[0], self.x2.rz[1], y))

            # We want the inertia = [[0, a], [a, 0]]
            inertia1 = self.x1.qinv @ hth1 @ self.x1.qinv.T
            inertia2 = self.x2.qinv @ hth2 @ self.x2.qinv.T

            return jnp.array([dth1[0], dth1[1], dth2[0], dth2[1], inertia1[0,0], inertia1[1,1], inertia2[0,0], inertia2[1,1]])
        
        solver = optx.Newton(rtol=1e-8, atol=1e-8)
        y0 = jnp.array([2*self.x1.rz[0] - self.axis.rz[0], 2*self.x1.rz[1] - self.axis.rz[1], 2*self.x2.rz[0] - self.axis.rz[0], 2*self.x2.rz[1] - self.axis.rz[1]])
        sol = optx.root_find(fn_theta_objective, solver, y0)

        solver_refine = optx.Newton(rtol=1e-8, atol=1e-8)
        y0_refine = jnp.array([sol.value[0], sol.value[1], sol.value[2], sol.value[3], 0.0, 0.0, 0.0, 0.0])
        sol_refine = optx.root_find(fn_theta_objective_refine, solver_refine, y0_refine, throw=False)

        self.uv_params = sol_refine.value
        self.inertia1 = (self.x1.qinv @ jnp.array(self.hess_theta_(self.x1.rz[0], self.x1.rz[1], sol_refine.value)) @ self.x1.qinv.T)[0,1]
        self.inertia2 = (self.x2.qinv @ jnp.array(self.hess_theta_(self.x2.rz[0], self.x2.rz[1], sol_refine.value)) @ self.x2.qinv.T)[0,1]