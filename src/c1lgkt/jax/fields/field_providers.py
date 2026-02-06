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

from .clebsch import ClebschMapping, ThetaMapping

# %% Allowable field signatures


type EmFields = tuple[Real[ArrayLike, "N"], Real[ArrayLike, "N"]]
type GradSingleField = tuple[Real[ArrayLike, "N"], Real[ArrayLike, "N"], Real[ArrayLike, "N"]]
type GradEmFields = tuple[GradSingleField, GradSingleField]


class AbstractFieldProvider(eqx.Module):
    @abc.abstractmethod
    def __call__(self,
        t: Real,
        r: Real[ArrayLike, "N"], varphi: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"],
        psi_ev: tuple
        ) -> EmFields:
        """
        Compute field components (phi, A_parallel) at given (t, R, varphi, Z). Also takes in psi as auxiliary information, as many field providers will want to use it as the radial coordinate.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def value_and_grad(self,
        t: Real,
        r: Real[ArrayLike, "N"], varphi: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"],
        psi_ev: tuple
        ) -> tuple[EmFields, GradEmFields]:
        """
        Computes the values and gradients of the fields with respect to (R, varphi, Z).
        """
        raise NotImplementedError


# %% Some field providers

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
        r: Real[ArrayLike, "N"], varphi: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"],
        psi_ev: tuple
        ) -> EmFields:
        # Unpack psi and its gradients
        (psi, psidr, psidz, psidrr, psidrz, psidzz) = psi_ev
        return self.interp_phi(psi), self.interp_apar(psi)
    
    def value_and_grad(self,
        t: Real,
        r: Real[ArrayLike, "N"], varphi: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"],
        psi_ev: tuple
        ) -> tuple[EmFields, GradEmFields]:
        # Unpack psi and its gradients
        (psi, psidr, psidz, psidrr, psidrz, psidzz) = psi_ev

        dphi = self.interp_phi(psi, dx=1)
        dapar = self.interp_apar(psi, dx=1)

        return (
            (self.interp_phi(psi), self.interp_apar(psi)),
            (
                (dphi * psidr, jnp.zeros_like(psi), dphi * psidz),
                (dapar * psidr, jnp.zeros_like(psi), dapar * psidz)
            )
        )
    
class RZFourierFieldProvider(AbstractFieldProvider):
    """
    Field provider for fields which are real in (R,Z) and (sparse) Fourier in varphi. Phase = omega t - n varphi
    """

    # List of Fourier modes
    n: Integer[ArrayLike, "Nmode"]
    omega: Real[ArrayLike, "Nmode"]

    # Complex Fourier coefficients for phi and A_parallel, as functions of (R,Z)
    interp_phi_coefs: interpax.Interpolator2D
    interp_apar_coefs: interpax.Interpolator2D

    def sum_fourier(coefs: Complex[ArrayLike, "Nq Nmode"], phases: Real[ArrayLike, "Nq Nmode"]) -> Real[ArrayLike, "Nq"]:
        return jnp.sum(jnp.real(coefs) * jnp.cos(phases) - jnp.imag(coefs) * jnp.sin(phases), axis=-1)

    def __call__(self,
        t: Real,
        r: Real[ArrayLike, "N"], varphi: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"],
        psi_ev: tuple
        ) -> EmFields:
        # Evaluate the Fourier coefficients at the given (R,Z)
        phi_coefs = self.interp_phi_coefs(r, z)
        apar_coefs = self.interp_apar_coefs(r, z)
        phases = (self.omega * t)[None,:] - self.n[None, :] * varphi[:, None]

        # Now we can compute the fields by summing over modes
        phi = RZFourierFieldProvider.sum_fourier(phi_coefs, phases)
        apar = RZFourierFieldProvider.sum_fourier(apar_coefs, phases)

        return phi, apar
    
    def value_and_grad(self,
        t: Real,
        r: Real[ArrayLike, "N"], varphi: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"],
        psi_ev: tuple
        ) -> tuple[EmFields, GradEmFields]:
        # Evaluate the Fourier coefficients at the given (R,Z) and their gradients
        phi_coefs = self.interp_phi_coefs(r, z)
        apar_coefs = self.interp_apar_coefs(r, z)
        dphi_coefs_dr = self.interp_phi_coefs(r, z, dx=1, dy=0)
        dphi_coefs_dz = self.interp_phi_coefs(r, z, dx=0, dy=1)
        dapar_coefs_dr = self.interp_apar_coefs(r, z, dx=1, dy=0)
        dapar_coefs_dz = self.interp_apar_coefs(r, z, dx=0, dy=1)
        phases = (self.omega * t)[None,:] - self.n[None, :] * varphi[:, None]

        # Now we can compute the fields and their gradients by summing over modes
        phi = RZFourierFieldProvider.sum_fourier(phi_coefs, phases)
        dphi_dr = RZFourierFieldProvider.sum_fourier(dphi_coefs_dr, phases)
        dphi_dz = RZFourierFieldProvider.sum_fourier(dphi_coefs_dz, phases)
        dphi_dvarphi = RZFourierFieldProvider.sum_fourier(-1j * self.n[None, :] * phi_coefs, phases)
        apar = RZFourierFieldProvider.sum_fourier(apar_coefs, phases)
        dapar_dr = RZFourierFieldProvider.sum_fourier(dapar_coefs_dr, phases)
        dapar_dz = RZFourierFieldProvider.sum_fourier(dapar_coefs_dz, phases)
        dapar_dvarphi = RZFourierFieldProvider.sum_fourier(-1j * self.n[None, :] * apar_coefs, phases)

        return (
            (phi, apar),
            (
                (dphi_dr, dphi_dvarphi, dphi_dz),
                (dapar_dr, dapar_dvarphi, dapar_dz)
            )
        )

class EikonalFieldProvider(AbstractFieldProvider):
    """
    Field provider for fields with eikonal structure:

    phi = Re{ A(psi, alpha_RZ) exp(i (omega t - n alpha)) }
    """

    # Interpolator for the clebsch function alpha_RZ(psi, theta), so B = \nabla psi x \nabla (alpha_RZ - \varphi)
    theta_map: ThetaMapping
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
        theta_map: ThetaMapping,
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
        self.theta_map = theta_map
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

    @jax.jit
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

    _grad_eval = jax.vmap(jax.jacrev(_eval, argnums=(2,3,4)), in_axes=(None, None, 0, 0, 0))

    @jax.jit
    def __call__(self,
        t: Real,
        r: Real[ArrayLike, "N"], varphi: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"],
        psi_ev: tuple
        ) -> EmFields:
        # Unpack psi and its gradients
        (psi, psidr, psidz, psidrr, psidrz, psidzz) = psi_ev

        # Compute theta
        theta = self.theta_map(r,z)

        return jax.vmap(self._eval, in_axes=(None, 0, 0, 0))(t, psi, theta, varphi)

    @jax.jit
    def value_and_grad(self,
        t: Real,
        r: Real[ArrayLike, "N"], varphi: Real[ArrayLike, "N"], z: Real[ArrayLike, "N"],
        psi_ev: tuple
        ) -> tuple[EmFields, GradEmFields]:

        # Unpack psi and its gradients
        (psi, psidr, psidz, psidrr, psidrz, psidzz) = psi_ev

        # Compute theta and its gradients
        theta = self.theta_map(r,z)
        thetadr, thetadz = self.theta_map.grad(r,z)

        # Get fields
        phi, apar = jax.vmap(self._eval, in_axes=(None, 0, 0, 0))(t, psi, theta, varphi)

        # Compute the jacobian of the fields
        (dphi_dpsi, dphi_dtheta, dphi_dvarphi), (dapar_dpsi, dapar_dtheta, dapar_dvarphi) = self._grad_eval(t, psi, theta, varphi)

        # Return the gradients
        return (
            (phi, apar),
            (
                (dphi_dpsi * psidr + dphi_dtheta * thetadr, dphi_dvarphi, dphi_dpsi * psidz + dphi_dtheta * thetadz),
                (dapar_dpsi * psidr + dapar_dtheta * thetadr, dapar_dvarphi, dapar_dpsi * psidz + dapar_dtheta * thetadz)
            )
        )