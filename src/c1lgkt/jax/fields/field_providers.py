# %% -*- coding: utf-8 -*-
"""
@author: maple

This file contains type definitions for field interpolators
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import interpax

import abc
from typing import NamedTuple

from jaxtyping import ArrayLike, Real, Complex, Integer

import abc

from .clebsch import ClebschMapping

# %% Allowable field signatures


type EmFields = tuple[Real[ArrayLike, "N"], Real[ArrayLike, "N"]]
type GradSingleField = tuple[Real[ArrayLike, "N"], Real[ArrayLike, "N"], Real[ArrayLike, "N"]]
type GradEmFields = tuple[GradSingleField, GradSingleField]


class AbstractFieldProvider(eqx.Module):
    @abc.abstractmethod
    def __call__(self,
        t: Real,
        psi: Real[ArrayLike, "N"], theta: Real[ArrayLike, "N"], varphi: Real[ArrayLike, "N"]
        ) -> EmFields:
        """
        Compute field components (phi, A_parallel) at given (t, psi, theta, varphi).
        """
        raise NotImplementedError
    
    def grad(self,
        t: Real,
        psi: Real[ArrayLike, "N"], theta: Real[ArrayLike, "N"], varphi: Real[ArrayLike, "N"]
        ) -> GradEmFields:
        """
        Computes the gradients of the fields with respect to (psi, theta, varphi).
        """
        raise NotImplementedError

class ZonalFieldProvider(AbstractFieldProvider):
    """
    Field provider for (time-indepenent) zonally symmetric fields. Might do time-dependent later
    """

    # Zonal electrostatic potential
    interp_phi: interpax.Interpolator1D
    # Zonal A_parallel
    interp_apar: interpax.Interpolator1D

    def __call__(self,
        t: Real,
        psi: Real[ArrayLike, "N"], theta: Real[ArrayLike, "N"], varphi: Real[ArrayLike, "N"]
        ) -> EmFields:
        return self.interp_phi(psi), self.interp_apar(psi)
    
    def grad(self,
        t: Real,
        psi: Real[ArrayLike, "N"], theta: Real[ArrayLike, "N"], varphi: Real[ArrayLike, "N"]
        ) -> GradEmFields:
        return (
            (self.interp_phi(psi), jnp.zeros_like(psi), jnp.zeros_like(psi)),
            (self.interp_apar(psi), jnp.zeros_like(psi), jnp.zeros_like(psi))
        )

class EikonalFieldProvider(AbstractFieldProvider):
    """
    Field provider for fields with eikonal structure:

    phi = Re{ A(psi, alpha_RZ) exp(i (omega t - n alpha)) }
    """

    # Interpolator for the clebsch function alpha_RZ(psi, theta), so B = \nabla psi x \nabla (alpha_RZ - \varphi)
    clebsch: ClebschMapping

    # List of eikonal modes; we should be thinking of this like a sum over saddle points in the stationary phase approximation
    # for the eikonal integral, maybe?
    n: Integer[ArrayLike, "Nmode"]
    omega: Real[ArrayLike, "Nmode"]
    psi0: Real[ArrayLike, "Nmode"]
    psi_scale: Real[ArrayLike, "Nmode"]
    interp_alpha0: interpax.Interpolator1D
    alpha_scale: Real[ArrayLike, "Nmode"]

    # Gauss-Hermite data for each mode
    gh_coefs: Real[ArrayLike, "Nmode 4 6"] # Nmode, real/imag parts of phi/apar, then 6 GH coefficients

    def __init__(self,
        clebsch: ClebschMapping,
        n: Integer[ArrayLike, "Nmode"],
        omega: Real[ArrayLike, "Nmode"],
        psi0: Real[ArrayLike, "Nmode"],
        psi_scale: Real[ArrayLike, "Nmode"],
        theta0: Real[ArrayLike, "Nmode"],
        alpha_scale: Real[ArrayLike, "Nmode"],
        gh_coefs: Real[ArrayLike, "Nmode 4 6"]
        ):
        """
        Initializes an eikonal field provider. The only nontrivial part is constructing the interpolator for alpha0(psi), which is done here.
        """
        self.clebsch = clebsch
        self.n = n
        self.omega = omega
        self.psi0 = psi0
        self.psi_scale = psi_scale
        
        self.alpha_scale = alpha_scale
        self.gh_coefs = gh_coefs

        # Build interpolator for alpha0(psi)
        alpha0 = jax.lax.map(lambda t: clebsch.interp_alpha(clebsch.interp_alpha.x, jnp.full_like(clebsch.interp_alpha.x, t)), theta0).transpose()
        self.interp_alpha0 = interpax.Interpolator1D(clebsch.interp_alpha.x, alpha0, kind='cubic2', extrap=True)

    def _eval(self, t: Real,
        psi: Real, theta: Real, varphi: Real
        ) -> tuple[Real, Real]:
        """
        Un-vectored evaluation of the fields at a single point; used to do autodiff on the gradients
        """

        # Compute alpha; each should be shape (Nbranch)
        # TODO: Weights are not used currently...
        alpha, weights = self.clebsch(psi, theta)
        # Compute alpha0; will be shape (Nmode)
        alpha0 = self.interp_alpha0(psi)

        # Compute the normalized radial coordinate; will be shape (Nmode)
        x = (psi - self.psi0) / self.psi_scale
        # Compute the normalized binormal coordinate; will be shape (Nmode, Nbranch)
        y = (alpha[None, :] - alpha0[:, None]) / self.alpha_scale[:, None]

        # Gaussian weight; will be shape (Nmode, Nbranch)
        g = jnp.exp(-0.5 * ((x**2)[:, None] + y**2))

        # Shorthand for gauss-hermite coefficients
        c = self.gh_coefs

        # Compute the Hermite polynomials
        h10 = x # (Nmode)
        h01 = y # (Nmode, Nbranch)
        h20 = (4 * x**2 - 2) # (Nmode)
        h11 = (x[:, None] * y) # (Nmode, Nbranch)
        h02 = (4 * y**2 - 2) # (Nmode, Nbranch)

        # Compute the field amplitudes p = (phi_real, phi_imag, apar_real, apar_imag) for each mode and branch, shape (Nmode, Nbranch, 4)
        p = c[:, :, 0][:, None, :] + \
            c[:, :, 1][:, None, :] * h10[:, None, None] + \
            c[:, :, 2][:, None, :] * h01[:, :   , None] + \
            c[:, :, 3][:, None, :] * h20[:, None, None] + \
            c[:, :, 4][:, None, :] * h11[:, :   , None] + \
            c[:, :, 5][:, None, :] * h02[:, :   , None]
        # Multiply by Gaussian envelope
        p *= g[:, :, None]

        # Next, compute the phases; shape (Nmode, Nbranch)
        phase = self.omega[:, None] * t - self.n[:, None] * (varphi - (alpha[None, :] - alpha0[:, None]))

        phi = jnp.sum(p[:, :, 0] * jnp.cos(phase) - p[:, :, 1] * jnp.sin(phase), axis=(0,1))
        apar = jnp.sum(p[:, :, 2] * jnp.cos(phase) - p[:, :, 3] * jnp.sin(phase), axis=(0,1))

        return phi, apar

    def __call__(self,
        t: Real,
        psi: Real[ArrayLike, "N"], theta: Real[ArrayLike, "N"], varphi: Real[ArrayLike, "N"]
        ) -> EmFields:

        return jax.vmap(self._eval, in_axes=(None, 0, 0, 0))(t, psi, theta, varphi)
    
    _grad = jax.vmap(jax.jacfwd(_eval, argnums=(2,3,4)), in_axes=(None, None, 0, 0, 0))

    def grad(self,
        t: Real,
        psi: Real[ArrayLike, "N"], theta: Real[ArrayLike, "N"], varphi: Real[ArrayLike, "N"]
        ) -> GradEmFields:

        return self._grad(t, psi, theta, varphi)