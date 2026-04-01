"""
Contains tools for working with particles.
"""

import numpy as np

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Real, Bool, PyTree, Array

from typing import NamedTuple

from functools import reduce

from .particle_motion import PusherArgs, PusherState

from ..custom_types import ScalarArray, ScalarArrayLike

class PunctureData(NamedTuple):
    """
    NamedTuple holding puncture data.

    t_punc: Real[ArrayLike, "Npunc"]
        Times of punctures
    y_punc: PyTree of Real[ArrayLike, "Npunc"]
        States at punctures
    """
    tp: Real[ArrayLike, "Npunc"]
    yp: PyTree[Real[ArrayLike, "Npunc"]]

    @classmethod
    def empty_list_like(cls, state: PusherState) -> list[PunctureData]:
        """
        Creates a list of empty PunctureData objects with the same structure as the given state.
        """
        return list(cls(
            tp = np.zeros((0,)),
            yp = jax.tree.map(lambda x: np.zeros((0,)), state)
        ) for _ in range(state.r.shape[0]))


def compute_punctures(ts: Real[ArrayLike, "Nt"], ys: PyTree[Real[ArrayLike, "Nt *shape"]], fpunc: Real[ArrayLike, "Nt N"], condpunc: Bool[ArrayLike, "Nt"] | None = None, period: float =-1):
    """
    Compute punctures. Returns a tuples consisting of positive punctures and
    one consisting of negative punctures.

    We also provide a condition array condpunc that indicates whether
    or not to consider a puncture at a given time step; both endpoints of the time interval
    need to have condpunc = True to consider a puncture there.

    Period can be specified; if period > 0, then we consider punctures
    whenever fpunc crosses an integer multiple of period. If period <= 0,
    then we consider punctures whenever fpunc crosses zero.

    If the first point is a puncture, it is counted; if the last point is a puncture, it is ignored

    Note that any inf values should be filtered out ahead of time
    """
    ts = np.asarray(ts)
    fpunc = np.asarray(fpunc)
    if condpunc is not None:
        condpunc = np.asarray(condpunc)

    # Get the number of particles
    nump = fpunc.shape[1]

    # Prepare output arrays
    ppuncs = [None] * nump
    npuncs = [None] * nump

    for k in range(nump):
        # Select the relevant arrays
        f = fpunc[:,k]
        
        if condpunc is None:
            # If no condition is given, always consider punctures
            c = True
        else:
            # Accept only intervals where condpunc is True at both ends
            c = condpunc[:-1,k] & condpunc[1:,k]

        # First, let's detect candidate punctures
        if period < 0:
            # When period < 0, we look for zero crossings
            pjumps = ( f[:-1] <= 0.0) & ( f[1:] > 0.0) & c
            njumps = (-f[:-1] <= 0.0) & (-f[1:] > 0.0) & c
        else:
            # When period > 0, we look for crossings of integer multiples of period
            pjumps = (( f[:-1] // period) < ( f[1:] // period)) & c
            njumps = ((-f[:-1] // period) < (-f[1:] // period)) & c

        # Get the indices of the punctures
        ind_ppuncs = np.argwhere(pjumps)
        ind_npuncs = np.argwhere(njumps)

        ## Now, for each puncture, compute the time and state at the puncture by linear interpolation

        # Prepare output arrays
        tp_ppuncs = np.empty(ind_ppuncs.shape[0])
        tp_npuncs = np.empty(ind_npuncs.shape[0])
        # ys will be stored as list of PyTrees
        yp_ppuncs = [None] * ind_ppuncs.shape[0]
        yp_npuncs = [None] * ind_npuncs.shape[0]

        # Compute a linear interpolation to the puncture point.
        # TODO: Do some sort of fancier interpolation to compute the punctures?
        for j in range(ind_ppuncs.shape[0]):
            tind = ind_ppuncs[j,0]

            # Compute the fraction along the interval where the puncture occurs
            if period < 0:
                ffrac = -f[tind] / (f[tind+1] - f[tind])
            else:
                ffrac = ((-f[tind] + period/2) % period - period/2) / (f[tind+1] - f[tind])

            # Assign the output data
            tp_ppuncs[j] = (1-ffrac)*ts[tind] + ffrac*ts[tind+1]
            yp_ppuncs[j] = jax.tree.map(lambda z: (1-ffrac)*z[tind,k] + ffrac*z[tind+1,k], ys)

        for j in range(ind_npuncs.shape[0]):
            tind = ind_npuncs[j,0]

            if period < 0:
                ffrac = -f[tind] / (f[tind+1] - f[tind])
            else:
                ffrac = ((-f[tind] + period/2) % period - period/2) / (f[tind+1] - f[tind])

            tp_npuncs[j] = (1-ffrac)*ts[tind] + ffrac*ts[tind+1]
            yp_npuncs[j] = jax.tree.map(lambda z: (1-ffrac)*z[tind,k] + ffrac*z[tind+1,k], ys)

        # Compute tree transposes
        if len(yp_ppuncs) > 0:
            yp_ppuncs = jax.tree.map(lambda *xs: np.stack(xs), *yp_ppuncs)
        else:
            yp_ppuncs = jax.tree.map(lambda x: np.zeros((0,)), ys)
        ppuncs[k] = PunctureData(tp_ppuncs, yp_ppuncs)
        if len(yp_npuncs) > 0:
            yp_npuncs = jax.tree.map(lambda *xs: np.stack(xs), *yp_npuncs)
        else:
            yp_npuncs = jax.tree.map(lambda x: np.zeros((0,)), ys)
        npuncs[k] = PunctureData(tp_npuncs, yp_npuncs)

    return ppuncs, npuncs

@jax.jit
def compute_integrals(t: Real, y: PusherState, args: PusherArgs) -> tuple[ScalarArray, ScalarArray]:
    """
    From the time, state, and PusherArgs, compute the Hamiltonian and the canonical toroidal angular momentum.
    """
    # Unpack the arguments
    r, varphi, z, upar, mu = y
    eq = args.eq
    pp = args.pp
    fields = args.fields

    psi_ev, ff_ev = eq.compute_psi_and_ff(r, z)
    bv = eq.compute_bv(r, z)
    modb = jnp.linalg.norm(bv, axis=0)

    # Compute the fields
    fields_eval = [f(t, r, varphi, z, psi_ev) for f in fields]
    # Sum up the values
    fields_eval_sum = reduce(lambda a, b: jax.tree.map(lambda x, y: x + y, a, b), fields_eval)
    phi, apar = fields_eval_sum

    # Compute the integrals
    ppar = pp.m * upar - pp.z * apar
    ham = 0.5 * ppar**2 / pp.m + mu * modb + pp.z * phi
    lphi = pp.z * psi_ev[0] + ppar * r * bv[1,:] / modb

    return ham, lphi

def compute_parallel_energy(t: Real, y: PusherState, integrals: tuple[Real, Real], omega: Real, args: PusherArgs) -> tuple[ScalarArray, ScalarArray]:
    """
    From the time, state, integrals, and PusherArgs, compute the parallel energy.
    Note that the upar term in the state is ignored; the parallel energy is the
    Kpar = ppar**2 / 2 m term in the Hamiltonian, where ppar = m upar - z apar

    For convenience, it returns a tuple of (Kpar, upar0) which is useful for computing the initial
    parallel velocity. Kpar = (upar - upar0)**2 / 2 m
    """

    # Unpack the arguments
    r, varphi, z, upar, mu = y
    eq = args.eq
    pp = args.pp
    fields = args.fields
    ham, lphi = integrals

    # Compute the adiabatic invariant
    kam = ham - omega * lphi

    # Compute magnetic stuff
    psi_ev, ff_ev = eq.compute_psi_and_ff(r, z)
    bv = eq.compute_bv(r, z)
    modb = jnp.linalg.norm(bv, axis=0)

    # Compute the fields
    fields_eval = [f(t, r, varphi, z, psi_ev) for f in fields]
    # Sum up the values
    fields_eval_sum = reduce(lambda a, b: jax.tree.map(lambda x, y: x + y, a, b), fields_eval)
    phi, apar = fields_eval_sum

    # Compute the parallel energy
    ppar0 = pp.m * omega * r * bv[1,:] / modb
    kpar = kam + ppar0**2 / (2 * pp.m) - mu * modb - pp.z * (phi - omega * psi_ev[0])

    upar0 = (ppar0 + pp.z * apar) / pp.m

    return kpar, upar0