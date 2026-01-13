"""
Contains tools for working with particles.
"""

import numpy as np

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Real, Bool, PyTree

from typing import NamedTuple

class PunctureData(NamedTuple):
    """
    NamedTuple holding puncture data.

    t_punc: Real[ArrayLike, "Np"]
        Times of punctures
    y_punc: PyTree of Real[ArrayLike, "Np"]
        States at punctures
    """
    tp: Real[ArrayLike, "Np"]
    yp: PyTree | None

def compute_punctures(ts: Real[ArrayLike, "Nt"], ys, fpunc: Real[ArrayLike, "Nt N"], condpunc: Bool[ArrayLike, "Nt"] | None = None, period=-1):
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
            yp_ppuncs = jax.tree.map(lambda *xs: jnp.stack(xs), *yp_ppuncs)
        else:
            yp_ppuncs = jax.tree.map(lambda x: jnp.empty((0,)), ys)
        ppuncs[k] = PunctureData(tp_ppuncs, yp_ppuncs)
        if len(yp_npuncs) > 0:
            yp_npuncs = jax.tree.map(lambda *xs: jnp.stack(xs), *yp_npuncs)
        else:
            yp_npuncs = jax.tree.map(lambda x: jnp.empty((0,)), ys)
        npuncs[k] = PunctureData(tp_npuncs, yp_npuncs)

    return ppuncs, npuncs