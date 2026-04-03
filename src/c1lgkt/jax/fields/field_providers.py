# %% -*- coding: utf-8 -*-
"""
@author: maple

This file contains type definitions for field interpolators
"""

from operator import eq

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import interpax

import numpy as np

import abc
from typing import NamedTuple, Type

from jaxtyping import ArrayLike, Real, Complex, Integer, Array

import abc

from .clebsch import ClebschMapping, ThetaMapping, ClebschMappingBuilder
from .equilibrium import Equilibrium, PsiTuple

from ..custom_types import ScalarArray, ScalarArrayLike, VectorTuple

from functools import partial

# %% Allowable field signatures

type EmFieldTuple = tuple[ScalarArray, ScalarArray]
type GradEmFieldTuple = tuple[VectorTuple, VectorTuple]


type_registry: dict[str, Type['AbstractFieldProvider']] = {}

class AbstractFieldProvider(eqx.Module):
    @abc.abstractmethod
    def __call__(self,
        t: Real,
        r: ScalarArrayLike, varphi: ScalarArrayLike, z: ScalarArrayLike,
        psi_ev: PsiTuple
        ) -> EmFieldTuple:
        """
        Compute field components (phi, A_parallel) at given (t, R, varphi, Z). Also takes in psi as auxiliary information,
        as many field providers will want to use it as the radial coordinate.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def value_and_grad(self,
        t: Real,
        r: ScalarArrayLike, varphi: ScalarArrayLike, z: ScalarArrayLike,
        psi_ev: PsiTuple
        ) -> tuple[EmFieldTuple, GradEmFieldTuple]:
        """
        Computes the values and gradients of the fields with respect to (R, varphi, Z).
        """
        raise NotImplementedError
    
    @classmethod
    @abc.abstractmethod
    def _build_from_config(cls, config: dict, eq: Equilibrium) -> AbstractFieldProvider:
        """
        Builds a field provider from a configuration dictionary. The exact format of the dictionary
        will depend on the specific field provider, but it should contain all necessary information
        to construct the field provider.
        """
        raise NotImplementedError
    
    @staticmethod
    def build_from_config(config: dict, eq: Equilibrium) -> AbstractFieldProvider:
        """
        This method dispatches to the appropriate _build_from_config method based on the 'type' field
        in the configuration dictionary. The 'type' field should specify the type of field provider to construct (e.g. 'eikonal', 'zonal', etc.).
        """
        if 'type' not in config:
            raise ValueError('Field provider configuration must contain type')
        
        type_name = config.pop('type')
        if type_name not in type_registry:
            raise ValueError(f'Unknown field provider type {type_name}')
        
        cls = type_registry[type_name]
        return cls._build_from_config(config, eq)




def register_field_provider(cls, type_name: str):
    type_registry[type_name] = cls
    return cls


# %% Some field providers

@partial(register_field_provider, type_name='zonal')
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
        r: ScalarArrayLike, varphi: ScalarArrayLike, z: ScalarArrayLike,
        psi_ev: PsiTuple
        ) -> EmFieldTuple:
        # Unpack psi and its gradients
        (psi, psidr, psidz, psidrr, psidrz, psidzz) = psi_ev
        return self.interp_phi(psi), self.interp_apar(psi)
    
    def value_and_grad(self,
        t: Real,
        r: ScalarArrayLike, varphi: ScalarArrayLike, z: ScalarArrayLike,
        psi_ev: PsiTuple
        ) -> tuple[EmFieldTuple, GradEmFieldTuple]:
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

    @classmethod
    def from_pfile(cls: Type[ZonalFieldProvider], filename: str, eq: Equilibrium) -> ZonalFieldProvider:
        with open(filename, 'r') as f:
            data = f.readlines()

            # Find the line where the data starts
            for line in range(len(data)):
                if 'er(kV/m)' in data[line]:
                    break

            # Determine the number of psi points from the header line
            tokens = data[line].split()
            n_psi = int(tokens[0])

            # Read in the grid values
            psi_grid = np.zeros(n_psi)
            er_grid = np.zeros(n_psi)
            erprime_grid = np.zeros(n_psi)

            # Now read in the data
            for k in range(line+1, line+1+n_psi):
                values = list(map(lambda t: float(t.strip()), data[k].split()))
                psi_grid[k - (line+1)] = values[0] * eq.psix
                er_grid[k - (line+1)] = values[1]
                erprime_grid[k - (line+1)] = values[2] / eq.psix

            # We add a small scrape-off-layer region to ensure the fields go to zero
            scale = (1.0 / eq.psix)/0.005
            c1 = er_grid[-1]
            c2 = erprime_grid[-1] + scale * c1

            # Create extra grid points beyond the LCS
            psi_extra = np.arange(1.005, 1.2, 0.005) * eq.psix

            # Compute the fields at the extra grid points using an exponential decay model
            er_extra = (c1 + c2 * (psi_extra - eq.psix)) * np.exp(-scale * (psi_extra - eq.psix))
            erprime_extra = (-scale * er_extra + c2 * np.exp(-scale * (psi_extra - eq.psix)))

            # Extend the grids with the extra points
            psi_grid = np.concatenate((psi_grid, psi_extra))
            er_grid = np.concatenate((er_grid, er_extra))
            erprime_grid = np.concatenate((erprime_grid, erprime_extra))

            # In order to fit the electric field and its derivative, we upsample the data
            er_spline = interpax.Interpolator1D(psi_grid, er_grid, kind='cubic', fx=erprime_grid)
            # Hardcoded upsampling to 512 points; should be good enough for now, but could make this more flexible later if needed
            psi_dense = jnp.linspace(0.0, eq.psix*1.2, 512, endpoint=False)
            er_dense = er_spline(psi_dense)
            
            # We need to compute R on the outboard midplane in order to integrate the electric field
            r_outer = jnp.linspace(eq.raxis, eq.rmax, 128)
            psi_outer = eq.interp_psi(r_outer, jnp.full_like(r_outer, eq.zaxis))
            interp_router = interpax.Interpolator1D(psi_outer, r_outer, method='cubic2')
            r_dense = interp_router(psi_dense)

            # Integrate the electric field to get the potential
            phi = -jnp.concatenate((jnp.array([0]), jnp.cumsum((er_dense[:-1] + er_dense[1:]) * (jnp.diff(r_dense)) * 0.5)))

            # Set up interpolators
            interp_phi = interpax.Interpolator1D(psi_dense, phi - phi[-1], method='cubic2')
            interp_apar = interpax.Interpolator1D(psi_dense, jnp.zeros_like(psi_dense), method='cubic2')

            return cls(interp_phi, interp_apar)
        
    @classmethod
    def _build_from_config(cls, config: dict, eq: Equilibrium) -> ZonalFieldProvider:
        """
        Builds a zonal field provider from a configuration dictionary. The dictionary should contain the filename of the pfile to load the fields from, as well as an equilibrium object to get the necessary normalization factors.
        """
        if 'pfile' in config:
            return cls.from_pfile(config['pfile'], eq)
        else:
            raise ValueError('Zonal field provider configuration must contain pfile')

def sum_fourier(coefs: Complex[Array, "Nq Nmode"], phases: Real[Array, "Nq Nmode"]) -> Real[Array, "Nq"]:
    return jnp.sum(jnp.real(coefs) * jnp.cos(phases) - jnp.imag(coefs) * jnp.sin(phases), axis=-1)

@partial(register_field_provider, type_name='rzfourier')
class RZFourierFieldProvider(AbstractFieldProvider):
    """
    Field provider for fields which are real in (R,Z) and (sparse) Fourier in varphi. Phase = omega t - n varphi
    """

    # List of Fourier modes
    n: Integer[Array, "Nmode"]
    omega: Real[Array, "Nmode"]

    # Complex Fourier coefficients for phi and A_parallel, as functions of (R,Z)
    interp_phi_coefs: interpax.Interpolator2D
    interp_apar_coefs: interpax.Interpolator2D
    
    @jax.jit
    def _eval(self,
        t: Real,
        r: ScalarArray, varphi: ScalarArray, z: ScalarArray,
        psi_ev: PsiTuple
        ) -> EmFieldTuple:
        """
        Internal jitted evaluation function that assumes all inputs are already arrays of the correct shape
        """
        # Evaluate the Fourier coefficients at the given (R,Z)
        phi_coefs = self.interp_phi_coefs(r, z)
        apar_coefs = self.interp_apar_coefs(r, z)
        phases = (self.omega * t)[None,:] - self.n[None, :] * varphi[:, None]

        # Now we can compute the fields by summing over modes
        phi = sum_fourier(phi_coefs, phases)
        apar = sum_fourier(apar_coefs, phases)

        return phi, apar
    
    @jax.jit
    def _value_and_grad(self,
        t: Real,
        r: ScalarArray, varphi: ScalarArray, z: ScalarArray,
        psi_ev: PsiTuple
        ) -> tuple[EmFieldTuple, GradEmFieldTuple]:
        # Evaluate the Fourier coefficients at the given (R,Z) and their gradients
        phi_coefs = self.interp_phi_coefs(r, z)
        apar_coefs = self.interp_apar_coefs(r, z)
        dphi_coefs_dr = self.interp_phi_coefs(r, z, dx=1, dy=0)
        dphi_coefs_dz = self.interp_phi_coefs(r, z, dx=0, dy=1)
        dapar_coefs_dr = self.interp_apar_coefs(r, z, dx=1, dy=0)
        dapar_coefs_dz = self.interp_apar_coefs(r, z, dx=0, dy=1)
        phases = (self.omega * t)[None,:] - self.n[None, :] * varphi[:, None]

        # Now we can compute the fields and their gradients by summing over modes
        phi = sum_fourier(phi_coefs, phases)
        dphi_dr = sum_fourier(dphi_coefs_dr, phases)
        dphi_dz = sum_fourier(dphi_coefs_dz, phases)
        dphi_dvarphi = sum_fourier(-1j * self.n[None, :] * phi_coefs, phases)
        apar = sum_fourier(apar_coefs, phases)
        dapar_dr = sum_fourier(dapar_coefs_dr, phases)
        dapar_dz = sum_fourier(dapar_coefs_dz, phases)
        dapar_dvarphi = sum_fourier(-1j * self.n[None, :] * apar_coefs, phases)

        return (
            (phi, apar),
            (
                (dphi_dr, dphi_dvarphi, dphi_dz),
                (dapar_dr, dapar_dvarphi, dapar_dz)
            )
        )

    def __call__(self,
        t: Real,
        r: ScalarArrayLike, varphi: ScalarArrayLike, z: ScalarArrayLike,
        psi_ev: PsiTuple
        ) -> EmFieldTuple:
        r, varphi, z = jnp.asarray(r), jnp.asarray(varphi), jnp.asarray(z)

        return self._eval(t, r, varphi, z, psi_ev)
    
    def value_and_grad(self,
        t: Real,
        r: ScalarArrayLike, varphi: ScalarArrayLike, z: ScalarArrayLike,
        psi_ev: PsiTuple
        ) -> tuple[EmFieldTuple, GradEmFieldTuple]:
        r, varphi, z = jnp.asarray(r), jnp.asarray(varphi), jnp.asarray(z)

        return self._value_and_grad(t, r, varphi, z, psi_ev)
    
    @classmethod
    def _build_from_config(cls, config: dict, eq: Equilibrium) -> RZFourierFieldProvider:
        """
        Builds an RZFourierFieldProvider from a configuration dictionary. The dictionary should contain the necessary information to construct the interpolators for the Fourier coefficients, as well as the mode numbers and frequencies.
        """
        raise NotImplementedError('RZFourierFieldProvider does not yet have a build_from_config method implemented')

@partial(register_field_provider, type_name='eikonal')
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
    # n is the toroidal mode number
    n: Integer[Array, "Nmode"]
    # omega is the frequency, phase ~ omega t - n (varphi - alpha_RZ). Note the toroidal phase
    # velocity is omega/n
    omega: Real[Array, "Nmode"]
    psi0: Real[Array, "Nmode"]
    psi_scale: Real[Array, "Nmode"]
    interp_alpha0: interpax.Interpolator1D
    alpha_scale: Real[Array, "Nmode"]

    # Gauss-Hermite data for each mode
    gh_coefs: Real[Array, "Nmode 4 6"] # Nmode, real/imag parts of phi/apar, then 6 GH coefficients

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
        self.n = jnp.asarray(n, dtype=int)
        self.omega = jnp.asarray(omega)
        self.psi0 = jnp.asarray(psi0)
        self.psi_scale = jnp.asarray(psi_scale)
        
        self.alpha_scale = jnp.asarray(alpha_scale)
        self.gh_coefs = jnp.asarray(gh_coefs)

        # Build interpolator for alpha0(psi)
        alpha0 = jax.lax.map(lambda t: clebsch.interp_alpha(clebsch.interp_alpha.x, jnp.full_like(clebsch.interp_alpha.x, t)), jnp.asarray(theta0)).transpose()
        self.interp_alpha0 = interpax.Interpolator1D(clebsch.interp_alpha.x, alpha0, kind='cubic2', extrap=True)

    @jax.jit
    def _scalar_eval(self, t: Real,
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
    
    _eval = jnp.vectorize(_scalar_eval, excluded=(0,1))

    _grad_eval = jnp.vectorize(jax.jacrev(_scalar_eval, argnums=(2,3,4)), excluded=(0,1))

    def __call__(self,
        t: Real,
        r: ScalarArrayLike, varphi: ScalarArrayLike, z: ScalarArrayLike,
        psi_ev: PsiTuple
        ) -> EmFieldTuple:
        r, varphi, z = jnp.asarray(r), jnp.asarray(varphi), jnp.asarray(z)

        # Unpack psi and its gradients
        (psi, psidr, psidz, psidrr, psidrz, psidzz) = psi_ev

        # Compute theta
        theta = self.theta_map(r,z)

        return self._eval(t, psi, theta, varphi)

    def value_and_grad(self,
        t: Real,
        r: ScalarArrayLike, varphi: ScalarArrayLike, z: ScalarArrayLike,
        psi_ev: PsiTuple
        ) -> tuple[EmFieldTuple, GradEmFieldTuple]:
        r, varphi, z = jnp.asarray(r), jnp.asarray(varphi), jnp.asarray(z)

        # Unpack psi and its gradients
        (psi, psidr, psidz, psidrr, psidrz, psidzz) = psi_ev

        # Compute theta and its gradients
        theta = self.theta_map(r,z)
        thetadr, thetadz = self.theta_map.grad(r,z)

        # Get fields
        phi, apar = self._eval(t, psi, theta, varphi)

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
    
    @classmethod
    def _build_from_config(cls, config: dict, eq: Equilibrium) -> EikonalFieldProvider:
        clebsch_builder = ClebschMappingBuilder()
        clebsch_config = config.pop('clebsch')
        
        theta_map, clebsch = clebsch_builder.build_from_config(clebsch_config, eq)

        # Special treatment for gh_coefs
        n_mode = len(np.asarray(config['n']))

        gh_coefs_config = np.asarray(config.pop('gh_coefs'))
        gh_coefs = np.zeros(n_mode * 4 * 6)
        gh_coefs[:len(gh_coefs_config)] = gh_coefs_config
        config['gh_coefs'] = gh_coefs.reshape((n_mode, 4, 6))

        return cls(theta_map, clebsch, **config)



    
# %%
