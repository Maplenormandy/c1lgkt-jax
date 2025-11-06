# %% -*- coding: utf-8 -*-
"""
@author: maple

Class for magnetic and profile equilibria. Based off of equilibrium.m provided by Hongxuan Zhu
"""

import jax.numpy as jnp
import equinox as eqx
import interpax

import matplotlib.pyplot as plt
import matplotlib as mpl

from typing import Type, TypeVar, NamedTuple


# %%

# List of variables expected in the magnetic geometry. Used to generate some code
eqd_vars = 'mr mz mpsi rmin rmax zmin zmax rgrid zgrid raxis zaxis psix rx zx psi ff psirz wallrz lcfsrz'.split()
setter_code = '\n'.join('self.' + s + ' = kwargs["' + s + '"]' for s in eqd_vars)

# Some static typing stuff to help with type hinting
T = TypeVar('T', bound='Equilibrium')


class Equilibrium(eqx.Module):
    """
    Class which is essentially responsible for holding EFIT and profile-level data

    Attributes
    ----------
    mr, mz, mpsi: int
        Number of grid points in R, Z, and psi respectively
    rmin, rmax, zmin, zmax: float
        Minimum and maximum R and Z values in the grid
    rgrid, zgrid: jnp.ndarray
        1D arrays of R and Z grid points
    raxis, zaxis: float
        R and Z coordinates of the magnetic axis
    psix: float
        Value of poloidal flux on the LCFS
    rx, zx: float
        R and Z coordinates of the X-point
    psi: jnp.ndarray
        1D array of poloidal flux values on which F(psi) is defined
    ff: jnp.ndarray
        1D array of toroidal field function values F(psi)
    psirz: jnp.ndarray
        2D array of poloidal flux values on the (R,Z) grid
    wallrz: jnp.ndarray
        2D array of (R,Z) coordinates defining the wall shape
    lcfsrz: jnp.ndarray
        2D array of (R,Z) coordinates defining the LCFS shape
    interp_ff: interpax.Interpolator1D
        Interpolator for F(psi)
    interp_psi: interpax.Interpolator2D
        Interpolator for psi(R,Z)
    """
    mr: int
    mz: int
    mpsi: int

    rmin: float
    rmax: float
    zmin: float
    zmax: float

    rgrid: jnp.ndarray
    zgrid: jnp.ndarray

    raxis: float
    zaxis: float
    psix: float
    rx: float
    zx: float

    psi: jnp.ndarray
    ff: jnp.ndarray

    psirz: jnp.ndarray

    wallrz: jnp.ndarray
    lcfsrz: jnp.ndarray

    interp_ff: interpax.Interpolator1D
    interp_psi: interpax.Interpolator2D

    def __init__(self, **kwargs):
        """
        Instantiates an equilibrium with magnetic data initialized from keyword arguments
        """
        # Use the generated setter_code above to get the benefit of code completion
        self.mr = kwargs["mr"]
        self.mz = kwargs["mz"]
        self.mpsi = kwargs["mpsi"]
        self.rmin = kwargs["rmin"]
        self.rmax = kwargs["rmax"]
        self.zmin = kwargs["zmin"]
        self.zmax = kwargs["zmax"]
        self.rgrid = kwargs["rgrid"]
        self.zgrid = kwargs["zgrid"]
        self.raxis = kwargs["raxis"]
        self.zaxis = kwargs["zaxis"]
        self.psix = kwargs["psix"]
        self.rx = kwargs["rx"]
        self.zx = kwargs["zx"]
        self.psi = kwargs["psi"]
        self.ff = kwargs["ff"]
        self.psirz = kwargs["psirz"]
        self.wallrz = kwargs["wallrz"]
        self.lcfsrz = kwargs["lcfsrz"]

        # Set up the interpolators
        self.interp_ff = interpax.Interpolator1D(self.psi, self.ff, method='cubic2', extrapolate=True)
        self.interp_psi = interpax.Interpolator2D(self.rgrid, self.zgrid, self.psirz.T, method='cubic2')

    def compute_bv(self, r, z):
        """
        Compute the magnetic field vector;
        B = (Br, Bphi, Bz) = (-1/R dpsi/dz, F(psi)/R, 1/R dpsi/dR)
        """
        eq = self

        # Evaluate psi and its derivatives
        psi = eq.interp_psi(r, z)
        drpsi = eq.interp_psi(r, z, dx=1)
        dzpsi = eq.interp_psi(r, z, dy=1)

        # Detect if a particle is outside the LCFS; if so, use different interpolation
        outside_lcfs = jnp.logical_or(psi > eq.psix, z < eq.zx)
        ff = jnp.where(outside_lcfs, eq.ff[-1], eq.interp_ff(psi))

        return (-dzpsi / r, -ff / r, drpsi / r)
    
    def compute_psi_and_ff(self, r, z):
        """
        Shorthand function for computing psi, ff, and its derivatives
        """
        # Evaluate psi and its derivatives
        psi = self.interp_psi(r, z)
        psidr = self.interp_psi(r, z, dx=1)
        psidz = self.interp_psi(r, z, dy=1)
        psidrr = self.interp_psi(r,z,dx=2)
        psidrz = self.interp_psi(r,z,dx=1,dy=1)
        psidzz = self.interp_psi(r,z,dy=2)

        # Detect if a particle is outside the LCFS; if so, use different interpolation
        outside_lcfs = jnp.logical_or(psi > self.psix, z < self.zx)
        ff = jnp.where(outside_lcfs, self.ff[-1], self.interp_ff(psi))
        dff = jnp.where(outside_lcfs, 0.0, self.interp_ff(psi, dx=1))

        # Store the evaluations of the interpolations, to pass to other functions that will use it
        psi_ev = (psi, psidr, psidz, psidrr, psidrz, psidzz)
        ff_ev = (ff, dff)

        return psi_ev, ff_ev
    
    def compute_geom_terms(self, r, psi_ev, ff_ev):
        """
        Computes unit vector b, |B|, grad|B|, and curl(b) given psi and ff evaluations
        """
        (psi, psidr, psidz, psidrr, psidrz, psidzz) = psi_ev
        (ff, dff) = ff_ev

        # B vector
        bv = jnp.array([-psidz / r, -ff / r, psidr / r])
        # |B|
        modb = jnp.linalg.norm(bv, axis=0)
        # B unit vector
        bu = bv / modb[None, ...]
        
        ## Evaluate grad|B| in the following manner:
        # grad|B| = (grad(R|B|) - B grad(R)) / R
        # grad(R|B|) = grad(sqrt(R**2 B**2)) = grad((RB)**2) / 2 / R|B|
        # grad((RB)**2) = grad(F(psi)**2 + |grad(psi)|**2)
        # grad(F(psi)**2) = 2 F'(psi) grad(psi)
        # grad(|grad(psi)|**2) = 2 Hess(psi) grad(psi)
        rmodb = r * modb
        gradpsi = jnp.array([psidr, jnp.zeros_like(r), psidz])
        gradf2_half = (ff * dff)[None, ...] * gradpsi

        # Note that Hess(psi) is symmetric in Cylindrical coordinates.
        # Due to axisymmetry, we only need to evaluate the Hess(psi) in the R,Z plane
        gradgradpsi2_half = jnp.array([psidr * psidrr + psidz * psidrz, jnp.zeros_like(r), psidr * psidrz + psidz * psidzz])
        gradrmodb = (gradf2_half + gradgradpsi2_half) / rmodb[None, ...]

        gradmodb = (gradrmodb - jnp.array([modb, jnp.zeros_like(r), jnp.zeros_like(r)])) / r[None, ...]

        ## Evalute curl(bhat) = curl(B/|B|) = (|B| curl(B) - grad|B| x B) / B**2
        curlb = jnp.array([dff * psidz / r, -(psidzz + psidrr) / r - 2*psidr / r**2, -dff * psidr / r])
        curlbu = (curlb - jnp.cross(gradmodb, bu, axis=0)) / modb[None, ...]
        return bv, bu, modb, gradmodb, curlbu
    
    def plot_magnetic_geometry(self, ax: mpl.axes.Axes, monochrome=True, alpha=1.0):
        """
        Function which plots the magnetic geometry (flux surfaces, wall, LCFS) on a given axis
        """
        eq = self
        
        if monochrome:
            ax.contour(eq.rgrid, eq.zgrid, eq.psirz, levels=64, colors=['tab:gray'], linewidths=mpl.rcParams['lines.linewidth']*0.5, alpha=alpha)
            ax.plot(eq.wallrz[:,0], eq.wallrz[:,1], c='k')
            ax.plot(eq.lcfsrz[:,0], eq.lcfsrz[:,1], c='k', alpha=alpha)
            ax.set_aspect('equal')
        else:
            ax.contour(eq.rgrid, eq.zgrid, eq.psirz, levels=64, linewidths=mpl.rcParams['lines.linewidth']*0.5, alpha=alpha)
            ax.plot(eq.wallrz[:,0], eq.wallrz[:,1])
            ax.plot(eq.lcfsrz[:,0], eq.lcfsrz[:,1], alpha=alpha)
            ax.set_aspect('equal')

    @classmethod
    def from_eqdfile(cls: Type[T], filename: str) -> T:
        """
        Loads a *.eqd file and returns an instance of an equilibrium class
        """
        
        with open(filename, 'r') as f:
            data = f.readlines()
            
            # Read and convert the line-by-line data at the top of the file
            mr, mz, mpsi = map(int,data[1].split())
            rmin, rmax, zmin, zmax = map(float,data[2].split())
            raxis, zaxis, baxis = map(float,data[3].split()) # Note: baxis is not used
            psix, rx, zx = map(float, data[4].split())
            
            rgrid = jnp.linspace(rmin, rmax, mr)
            zgrid = jnp.linspace(zmin, zmax, mz)

            # Take the lines corresponding to psi, join them into a single array, then parse them
            begin_read = 5
            end_read = begin_read+1+(mpsi-1)//4
            psi = jnp.array((' '.join(data[begin_read:end_read])).split(), dtype=float)
            # Update the read line, then proceed to the next block of data
            begin_read = end_read
            end_read = begin_read+1+(mpsi-1)//4
            ff = jnp.array((' '.join(data[begin_read:end_read])).split(), dtype=float) # Toroidal magnetic field?
            
            # Get 2d psi grid as a function of (R,z)
            begin_read = end_read
            end_read = begin_read+1+(mr*mz-1)//4
            psirz = jnp.array((' '.join(data[begin_read:end_read])).split(), dtype=float).reshape((mz,mr))
            
            # Get the shape of the wall
            begin_read = end_read
            mw = int(data[begin_read+1].strip())
            begin_read = begin_read + 2
            end_read = begin_read + 1 + (2*mw-1)//2
            wallrz = jnp.array((' '.join(data[begin_read:end_read])).split(), dtype=float).reshape((mw,2))
            
            # Get the shape of the LCFS
            begin_read = end_read
            ml = int(data[begin_read+1].strip())
            begin_read = begin_read + 2
            end_read = begin_read + 1 + (2*ml-1)//2
            lcfsrz = jnp.array((' '.join(data[begin_read:end_read])).split(), dtype=float).reshape((ml,2))
        
        # Use reflection to set these variables out of laziness
        local_vars = locals()
        eq_kwargs = {key: local_vars[key] for key in eqd_vars}
        return cls(**eq_kwargs)
