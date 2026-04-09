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
from typing import Mapping, NamedTuple, Sequence, Type

from jaxtyping import ArrayLike, Real, Complex, Integer, Array

import abc

from .clebsch import ClebschMapping, ThetaMapping, ClebschMappingBuilder
from .equilibrium import Equilibrium, PsiTuple

from ..custom_types import ScalarArray, ScalarArrayLike, VectorTuple, ScalarFields, VectorFields

from functools import partial, reduce

# %% Allowable field signatures

type_registry: dict[str, Type['AbstractFieldProvider']] = {}

class AbstractFieldProvider(eqx.Module):
    @abc.abstractmethod
    def __call__(self,
        t: Real,
        r: ScalarArrayLike, varphi: ScalarArrayLike, z: ScalarArrayLike,
        psi_ev: PsiTuple
        ) -> ScalarFields:
        """
        Compute field components (phi, A_parallel) at given (t, R, varphi, Z). Also takes in psi as auxiliary information,
        as many field providers will want to use it as the radial coordinate.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def grad_and_value(self,
        t: Real,
        r: ScalarArrayLike, varphi: ScalarArrayLike, z: ScalarArrayLike,
        psi_ev: PsiTuple
        ) -> tuple[VectorFields, ScalarFields]:
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

def jacrev_and_value(func, argnums: int | Sequence[int] =0):
    def wrapper(*args, **kwargs):
        value = func(*args, **kwargs)
        return value, value
    return jax.jacrev(wrapper, argnums=argnums, has_aux=True)

@partial(register_field_provider, type_name='zonal')
class ZonalFieldProvider(AbstractFieldProvider):
    """
    Field provider for (time-indepenent) zonally symmetric fields. Might do time-dependent later
    """

    # Interpolator function for the fields, as function(s) of psi
    interp: interpax.Interpolator1D

    # List of field components provided by this provider
    fields: str | list[str] = eqx.field(static=True)

    def __call__(self,
        t: Real,
        r: ScalarArrayLike, varphi: ScalarArrayLike, z: ScalarArrayLike,
        psi_ev: PsiTuple
        ) -> ScalarFields:

        # Unpack psi and its gradients
        (psi, psidr, psidz, psidrr, psidrz, psidzz) = psi_ev

        # Evaluate the fields at the given psi values
        ev = self.interp(psi)

        # Return the fields in a dictionary
        if isinstance(self.fields, str):
            return {self.fields: ev}
        else:
            return {field: ev[...,i] for i, field in enumerate(self.fields)}

    def grad_and_value(self,
        t: Real,
        r: ScalarArrayLike, varphi: ScalarArrayLike, z: ScalarArrayLike,
        psi_ev: PsiTuple
        ) -> tuple[VectorFields, ScalarFields]:
        # Unpack psi and its gradients
        (psi, psidr, psidz, psidrr, psidrz, psidzz) = psi_ev

        # Evaluate the fields and their gradients at the given psi values
        ev = self.interp(psi)
        dev = self.interp(psi, dx=1)

        # Return the fields in a dictionary
        if isinstance(self.fields, str):
            return (
                {self.fields: (dev * psidr, jnp.zeros_like(psi), dev * psidz)},
                {self.fields: ev}
            )
        else:
            return (
                {field: (dev[...,i] * psidr, jnp.zeros_like(psi), dev[...,i] * psidz) for i, field in enumerate(self.fields)},
                {field: ev[...,i] for i, field in enumerate(self.fields)},
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

            return cls(interp_phi, fields='phi')
        
    @classmethod
    def _build_from_config(cls, config: dict, eq: Equilibrium) -> ZonalFieldProvider:
        """
        Builds a zonal field provider from a configuration dictionary. The dictionary should contain the filename of the pfile to load the fields from, as well as an equilibrium object to get the necessary normalization factors.
        """
        if 'pfile' in config:
            return cls.from_pfile(config['pfile'], eq)
        else:
            raise ValueError('Zonal field provider configuration must contain pfile')

def sum_fourier(coefs: Complex[Array, "Nq Nmode Nfield"], phases: Real[Array, "Nq Nmode"]) -> Real[Array, "Nq Nfield"]:
    z1 = jnp.einsum('ijk...,ijk->i...', jnp.real(coefs), jnp.cos(phases))
    z2 = jnp.einsum('ijk...,ijk->i...', jnp.imag(coefs), jnp.sin(phases))
    return z1 - z2

@partial(register_field_provider, type_name='rzfourier')
class RZFourierFieldProvider(AbstractFieldProvider):
    """
    Field provider for fields which are real in (R,Z) and (sparse) Fourier in varphi. Phase = omega t - n varphi
    """

    # List of Fourier modes
    n: Integer[Array, "Nmode"]
    omega: Real[Array, "Nmode"]

    # Complex Fourier coefficients for phi and A_parallel, as functions of (R,Z)
    interp_coefs: interpax.Interpolator2D

    # List of field components provided by this provider
    fields: str | list[str] = eqx.field(static=True)
    
    @jax.jit
    def _eval(self,
        t: Real,
        r: ScalarArray, varphi: ScalarArray, z: ScalarArray,
        psi_ev: PsiTuple
        ) -> ScalarFields:
        """
        Internal jitted evaluation function that assumes all inputs are already arrays of the correct shape
        """
        # Evaluate the Fourier coefficients at the given (R,Z)
        coefs = self.interp_coefs(r, z)
        phases = (self.omega * t)[None,:] - self.n[None, :] * varphi[:, None]

        # Now, we can compute the fields by summing over modes
        ev = sum_fourier(coefs, phases)

        # Return the fields in a dictionary
        if isinstance(self.fields, str):
            return {self.fields: ev}
        else:
            return {field: ev[...,i] for i, field in enumerate(self.fields)}
    
    @jax.jit
    def _grad_and_value(self,
        t: Real,
        r: ScalarArray, varphi: ScalarArray, z: ScalarArray,
        psi_ev: PsiTuple
        ) -> tuple[VectorFields, ScalarFields]:
        # Evaluate the Fourier coefficients at the given (R,Z) and their gradients
        coefs = self.interp_coefs(r, z)
        dcoefs_dr = self.interp_coefs(r, z, dx=1, dy=0)
        dcoefs_dz = self.interp_coefs(r, z, dx=0, dy=1)
        phases = (self.omega * t)[None,:] - self.n[None, :] * varphi[:, None]

        # Now we can compute the fields and their gradients by summing over modes
        ev = sum_fourier(coefs, phases)
        dev_dr = sum_fourier(dcoefs_dr, phases)
        dev_dz = sum_fourier(dcoefs_dz, phases)
        dev_dvarphi = sum_fourier(-1j * self.n[None, :, None] * coefs, phases)

        if isinstance(self.fields, str):
            return (
                {self.fields: (dev_dr, dev_dvarphi, dev_dz) },
                {self.fields: ev }
            )
        else:
            return (
                {field: (dev_dr[...,i], dev_dvarphi[...,i], dev_dz[...,i]) for i, field in enumerate(self.fields)},
                {field: ev[...,i] for i, field in enumerate(self.fields)}
            )

    def __call__(self,
        t: Real,
        r: ScalarArrayLike, varphi: ScalarArrayLike, z: ScalarArrayLike,
        psi_ev: PsiTuple
        ) -> ScalarFields:
        r, varphi, z = jnp.asarray(r), jnp.asarray(varphi), jnp.asarray(z)

        return self._eval(t, r, varphi, z, psi_ev)
    
    def grad_and_value(self,
        t: Real,
        r: ScalarArrayLike, varphi: ScalarArrayLike, z: ScalarArrayLike,
        psi_ev: PsiTuple
        ) -> tuple[VectorFields, ScalarFields]:
        r, varphi, z = jnp.asarray(r), jnp.asarray(varphi), jnp.asarray(z)

        return self._grad_and_value(t, r, varphi, z, psi_ev)
    
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

    
    # n is the toroidal mode number
    n: Integer[Array, "Nmode"]
    # omega is the frequency, phase ~ omega t - n (varphi - alpha_RZ). Note the toroidal phase
    # velocity is omega/n
    omega: Real[Array, "Nmode"]

    # List of eikonal modes; we should be thinking of this like a sum over saddle points in the stationary phase approximation
    # for the eikonal integral, maybe?
    psi0: Real[Array, "Nmode"]
    psi_scale: Real[Array, "Nmode"]
    interp_alpha0: interpax.Interpolator1D
    alpha_scale: Real[Array, "Nmode"]

    # Gauss-Hermite data for each mode
    gh_coefs: Complex[Array, "Nmode Ncoef Nfield"] # Nmode, then GH coefficients

    # Fields provided
    fields: str | list[str] = eqx.field(static=True)

    def __init__(self,
        theta_map: ThetaMapping,
        clebsch: ClebschMapping,
        n: Integer[ArrayLike, "Nmode"],
        omega: Real[ArrayLike, "Nmode"],
        psi0: Real[ArrayLike, "Nmode"],
        psi_scale: Real[ArrayLike, "Nmode"],
        theta0: Real[ArrayLike, "Nmode"],
        alpha_scale: Real[ArrayLike, "Nmode"],
        gh_coefs: Complex[ArrayLike, "Nmode Ncoef Nfield"],
        fields: str | list[str],
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

        self.fields = fields

    @jax.jit
    def _scalar_eval(self, t: Real,
        psi: Real, theta: Real, varphi: Real
        ) -> dict[str, Real]:
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

        # Shorthand for gauss-hermite coefficients, shape (Nmode, Ncoef, Nfield)
        c = self.gh_coefs

        # Start accumulating the field; p is shape (Nmode, Nbranch, Nfield)
        p = c[:, 0, ...][:, None, ...]

        # Compute the Hermite polynomials
        if c.shape[1] > 1:
            h10 = x # (Nmode)
            h01 = y # (Nmode, Nbranch)
            p = p + jnp.einsum('i...,i->i...', c[:, 1, ...], h10)[:, None, ...]
            p = p + jnp.einsum('i...,ij->ij...', c[:, 2, ...], h01)
        if c.shape[1] > 3:
            h20 = (4 * x**2 - 2) # (Nmode)
            h11 = (x[:, None] * y) # (Nmode, Nbranch)
            h02 = (4 * y**2 - 2) # (Nmode, Nbranch)
            p = p + jnp.einsum('i...,i->i...', c[:, 3, ...], h20)[:, None, ...]
            p = p + jnp.einsum('i...,ij->ij...', c[:, 4, ...], h11)
            p = p + jnp.einsum('i...,ij->ij...', c[:, 5, ...], h02)

        # Multiply by Gaussian envelope
        p *= g[:, :, ...]

        # Next, compute the phases; shape (Nmode, Nbranch)
        phase = self.omega[:, None] * t - self.n[:, None] * (varphi - (alpha[None, :] - alpha0[:, None]))

        z1 = jnp.einsum('jk...,jk->...', jnp.real(p), jnp.cos(phase))
        z2 = jnp.einsum('jk...,jk->...', jnp.imag(p), jnp.sin(phase))
        ev = z1 - z2

        # Return the fields in a dictionary
        if isinstance(self.fields, str):
            return {self.fields: ev}
        else:
            return {field: ev[...,i] for i, field in enumerate(self.fields)}
    
    _eval = jnp.vectorize(_scalar_eval, excluded=(0,1))

    _grad_and_value = jnp.vectorize(jacrev_and_value(_scalar_eval, argnums=(2,3,4)), excluded=(0,1))

    def __call__(self,
        t: Real,
        r: ScalarArrayLike, varphi: ScalarArrayLike, z: ScalarArrayLike,
        psi_ev: PsiTuple
        ) -> ScalarFields:
        r, varphi, z = jnp.asarray(r), jnp.asarray(varphi), jnp.asarray(z)

        # Unpack psi and its gradients
        (psi, psidr, psidz, psidrr, psidrz, psidzz) = psi_ev

        # Compute theta
        theta = self.theta_map(r,z)

        return self._eval(t, psi, theta, varphi)

    def grad_and_value(self,
        t: Real,
        r: ScalarArrayLike, varphi: ScalarArrayLike, z: ScalarArrayLike,
        psi_ev: PsiTuple
        ) -> tuple[VectorFields, ScalarFields]:
        r, varphi, z = jnp.asarray(r), jnp.asarray(varphi), jnp.asarray(z)

        # Unpack psi and its gradients
        (psi, psidr, psidz, psidrr, psidrz, psidzz) = psi_ev

        # Compute theta and its gradients
        theta = self.theta_map(r,z)
        thetadr, thetadz = self.theta_map.grad(r,z)

        # Get fields
        dev, ev = self._grad_and_value(t, psi, theta, varphi)

        # Convert the gradients from (psi, theta, varphi) space to (R, varphi, Z) space using the chain rule
        for key in dev.keys():
            dev_dpsi, dev_dtheta, dev_dvarphi = dev[key]
            dev_dr = dev_dpsi * psidr + dev_dtheta * thetadr
            dev_dz = dev_dpsi * psidz + dev_dtheta * thetadz
            dev[key] = (dev_dr, dev_dvarphi, dev_dz)

        # Return the gradients
        return dev, ev
    
    @classmethod
    def _build_from_config(cls, config: dict, eq: Equilibrium) -> EikonalFieldProvider:
        clebsch_builder = ClebschMappingBuilder()
        clebsch_config = config.pop('clebsch')
        
        theta_map, clebsch = clebsch_builder.build_from_config(clebsch_config, eq)

        # Special treatment for gh_coefs
        n_mode = len(np.asarray(config['n']))

        gh_coefs_config = np.asarray(config.pop('gh_coefs'), dtype='complex')
        if isinstance(config['fields'], str):
            config['gh_coefs'] = gh_coefs_config.reshape((n_mode, -1))
        else:
            config['gh_coefs'] = gh_coefs_config.reshape((n_mode, -1, len(config['fields'])))

        return cls(theta_map, clebsch, **config)

def sum_fields(x: ScalarFields, y: ScalarFields) -> ScalarFields:
    keys = x.keys() | y.keys()
    return {
        k: reduce(jnp.add, (d[k] for d in (x, y) if k in d))
        for k in keys
    }

def sum_field_grads(x: tuple[VectorFields, ScalarFields], y: tuple[VectorFields, ScalarFields]) -> tuple[VectorFields, ScalarFields]:
    keys_d = x[0].keys() | y[0].keys()
    keys_s = x[1].keys() | y[1].keys()
    
    sum_s = {
        k: reduce(jnp.add, (d[1][k] for d in (x, y) if k in d[1]))
        for k in keys_s
    }
    sum_d = {
        k: tuple(reduce(jnp.add, (d[0][k][i] for d in (x, y) if k in d[0])) for i in range(3))
        for k in keys_d
    }
    
    return sum_d, sum_s # pyright: ignore

    
# %%
