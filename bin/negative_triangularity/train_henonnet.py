#!/usr/bin/env python
"""
Train a HenonNet symplectic neural network surrogate for a Poincaré map.

Two layer architectures are available, both symplectic:

Standard (Burby et al. 2021):
    Each Hénon layer applies H[V,η](x,y) = (y+η, -x+∇V(y)) four times,
    where V(y) = Wout · tanh(Win*(y-ymean)/ydiam + bin).

Periodic (Drimalas et al. 2025, Sec. IV.C):
    Each Hénon layer alternates two base maps — HL = H2 ∘ H1 ∘ H2 ∘ H1 —
    where
        H1[V1,η1](x,y) = (y+η1, -x+∇_y V1(cos y, sin y))
        H2[V2,η2](x,y) = (y+η2, -x+∇V2(y))
    V1 is periodic in y with period 2π; V2 is the standard normalised MLP.
    This satisfies HL(x, y+2nπ) = HL(x,y) + (0, 2nπ), i.e. the map
    commutes with 2π shifts in the varphi coordinate.

Usage:
    python train_henonnet.py [options]
    python train_henonnet.py --epochs 500 --layers 64 64 64 64 --save data/henonnet.eqx
    python train_henonnet.py --no-periodic   # use standard (non-periodic) layers
"""

import argparse
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, PyTree
from typing import Iterator, Tuple


# ---------------------------------------------------------------------------
# HenonNet architecture
# ---------------------------------------------------------------------------

class HenonLayer(eqx.Module):
    """
    A single Hénon layer: 4 applications of H[V,η](x,y) = (y+η, -x+∇V(y)).

    Trainable parameters: Win [ni], Wout [ni], bin [ni], eta [].
    Fixed (static) fields: ymean, ydiam — normalization constants for y.
    """
    Win:  Array  # [ni]
    Wout: Array  # [ni]
    bin:  Array  # [ni]
    eta:  Array  # []
    ymean: float = eqx.field(static=True)
    ydiam: float = eqx.field(static=True)

    def __init__(self, ni: int, key: Array, ymean: float, ydiam: float):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        scale = 0.01
        self.Win  = jax.random.normal(k1, (ni,)) * scale
        self.Wout = jax.random.normal(k2, (ni,)) * scale
        self.bin  = jax.random.normal(k3, (ni,)) * scale
        self.eta  = jax.random.normal(k4, ())    * scale
        self.ymean = float(ymean)
        self.ydiam = float(ydiam)

    def _V(self, y: Array) -> Array:
        """Scalar potential V(y) ∈ R."""
        y_norm = (y - self.ymean) / self.ydiam
        hidden = jnp.tanh(self.Win * y_norm + self.bin)
        return jnp.dot(self.Wout, hidden)

    def _henon_step(self, x: Array, y: Array) -> Tuple[Array, Array]:
        grad_V = jax.grad(self._V)(y)
        return y + self.eta, -x + grad_V

    def __call__(self, z: Float[Array, "2"]) -> Float[Array, "2"]:
        x, y = z[0], z[1]
        for _ in range(4):
            x, y = self._henon_step(x, y)
        return jnp.stack([x, y])


class PeriodicHenonLayer(eqx.Module):
    """
    Hénon layer with exact 2π-periodicity in the y (varphi) coordinate.

    Implements HL = H2 ∘ H1 ∘ H2 ∘ H1, where
        H1[V1,η1](x,y) = (y+η1,  -x + ∇_y V1(cos y, sin y))
        H2[V2,η2](x,y) = (y+η2,  -x + ∇V2(y))

    V1 takes (cos y, sin y) as input, so it is automatically 2π-periodic.
    V2 takes normalised y, identical to the standard HenonLayer.

    The layer satisfies HL(x, y+2nπ) = HL(x,y) + (0, 2nπ) for all n ∈ Z,
    so any composition of PeriodicHenonLayers commutes with 2π shifts in y.
    (Drimalas et al. 2025, Phys. Plasmas 32, 103901, Sec. IV.C)
    """
    # H1 parameters — periodic potential V1(cos y, sin y)
    Win1_c: Array  # [ni]  cos(y) input weights
    Win1_s: Array  # [ni]  sin(y) input weights
    Wout1:  Array  # [ni]
    bin1:   Array  # [ni]
    eta1:   Array  # []
    # H2 parameters — standard normalised potential V2(y)
    Win2:  Array   # [ni]
    Wout2: Array   # [ni]
    bin2:  Array   # [ni]
    eta2:  Array   # []
    ymean: float = eqx.field(static=True)
    ydiam: float = eqx.field(static=True)

    def __init__(self, ni: int, key: Array, ymean: float, ydiam: float):
        keys = jax.random.split(key, 9)
        scale = 0.01
        self.Win1_c = jax.random.normal(keys[0], (ni,)) * scale
        self.Win1_s = jax.random.normal(keys[1], (ni,)) * scale
        self.Wout1  = jax.random.normal(keys[2], (ni,)) * scale
        self.bin1   = jax.random.normal(keys[3], (ni,)) * scale
        self.eta1   = jax.random.normal(keys[4], ())    * scale
        self.Win2   = jax.random.normal(keys[5], (ni,)) * scale
        self.Wout2  = jax.random.normal(keys[6], (ni,)) * scale
        self.bin2   = jax.random.normal(keys[7], (ni,)) * scale
        self.eta2   = jax.random.normal(keys[8], ())    * scale
        self.ymean  = float(ymean)
        self.ydiam  = float(ydiam)

    def _V1(self, y: Array) -> Array:
        """Periodic potential: inputs are (cos y, sin y) — period 2π in y."""
        hidden = jnp.tanh(self.Win1_c * jnp.cos(y) + self.Win1_s * jnp.sin(y) + self.bin1)
        return jnp.dot(self.Wout1, hidden)

    def _V2(self, y: Array) -> Array:
        """Standard normalised potential."""
        y_norm = (y - self.ymean) / self.ydiam
        hidden = jnp.tanh(self.Win2 * y_norm + self.bin2)
        return jnp.dot(self.Wout2, hidden)

    def _H1(self, x: Array, y: Array) -> Tuple[Array, Array]:
        return y + self.eta1, -x + jax.grad(self._V1)(y)

    def _H2(self, x: Array, y: Array) -> Tuple[Array, Array]:
        return y + self.eta2, -x + jax.grad(self._V2)(y)

    def __call__(self, z: Float[Array, "2"]) -> Float[Array, "2"]:
        x, y = z[0], z[1]
        x, y = self._H1(x, y)
        x, y = self._H2(x, y)
        x, y = self._H1(x, y)
        x, y = self._H2(x, y)
        return jnp.stack([x, y])


class HenonNet(eqx.Module):
    """HenonNet: composition of N Hénon layers (standard or periodic)."""
    layers: list

    def __init__(
            self,
            unit_list: list,
            key: Array,
            ymean: float,
            ydiam: float,
            *,
            periodic: bool = True,
    ):
        """
        Args:
            unit_list: hidden-layer widths, one entry per Hénon layer
            key: PRNG key
            ymean, ydiam: normalization constants for the y coordinate (varphi)
            periodic: if True (default), use PeriodicHenonLayer (2π symmetry in y);
                      if False, use the standard HenonLayer
        """
        keys = jax.random.split(key, len(unit_list))
        layer_cls = PeriodicHenonLayer if periodic else HenonLayer
        self.layers = [layer_cls(ni, k, ymean, ydiam) for ni, k in zip(unit_list, keys)]

    def __call__(self, z: Float[Array, "2"]) -> Float[Array, "2"]:
        for layer in self.layers:
            z = layer(z)
        return z


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def data_loader(
        X: Float[Array, "N 2"],
        Y: Float[Array, "N 2"],
        batch_size: int,
        *,
        key: Array,
) -> Iterator[Tuple[Float[Array, "B 2"], Float[Array, "B 2"]]]:
    N = X.shape[0]
    perm = jax.random.permutation(key, N)
    for i in range(N // batch_size):
        idx = perm[i * batch_size : (i + 1) * batch_size]
        yield X[idx], Y[idx]


@eqx.filter_jit
def loss_fn(
        model: HenonNet,
        x: Float[Array, "B 2"],
        y: Float[Array, "B 2"],
) -> Array:
    y_pred = jax.vmap(model)(x)
    return jnp.mean((y_pred - y) ** 2)


def make_step_fn(optim: optax.GradientTransformation):
    """Return a JIT-compiled training step that closes over `optim`."""
    @eqx.filter_jit
    def make_step(
            model: HenonNet,
            opt_state: PyTree,
            x: Float[Array, "B 2"],
            y: Float[Array, "B 2"],
    ):
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, new_opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, new_opt_state, loss_val
    return make_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train HenonNet Poincaré map surrogate")
    parser.add_argument("--data",       default="data/training_data.npz")
    parser.add_argument("--n-train",    type=int,       default=None,
                        help="Training samples to use (default: all)")
    parser.add_argument("--layers",     type=int, nargs="+", default=[32, 32, 32, 32],
                        help="Hidden units per Hénon layer (default: 32 32 32 32)")
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--save",       default="data/henonnet.eqx",
                        help="Path to save trained model weights")
    parser.add_argument("--periodic", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use periodic Hénon layers (default: True). "
                             "Pass --no-periodic for the standard architecture.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    d = np.load(args.data)
    X_all = d["data"].astype(np.float32)    # (N, 2): [Lphi,  varphi ]
    Y_all = d["labels"].astype(np.float32)  # (N, 2): [Lphi', varphi']
    N_total = X_all.shape[0]

    n_train = args.n_train if args.n_train is not None else N_total
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(N_total)[:n_train]

    X_train = jnp.array(X_all[idx])
    Y_train = jnp.array(Y_all[idx])
    print(f"Loaded {N_total} samples; using {n_train} for training")

    # ------------------------------------------------------------------
    # Normalization: use y-coordinate (varphi, col 1) statistics
    # computed over the combined input/output set so the model generalises
    # to the full range of y values seen during evaluation.
    # ------------------------------------------------------------------
    y_vals = np.concatenate([X_all[:, 1], Y_all[:, 1]])
    ymean = float(y_vals.mean())
    ydiam = float(y_vals.std())
    print(f"y normalization: ymean={ymean:.4f}, ydiam={ydiam:.4f}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    key = jax.random.PRNGKey(args.seed)
    model = HenonNet(args.layers, key, ymean, ydiam, periodic=args.periodic)
    layer_type = "PeriodicHenonLayer" if args.periodic else "HenonLayer"
    print(f"HenonNet architecture: {args.layers} × {layer_type}")

    # Cosine decay LR schedule
    steps_per_epoch = n_train // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    schedule = optax.cosine_decay_schedule(args.lr, total_steps, alpha=0.01)
    optim = optax.adam(schedule)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    make_step = make_step_fn(optim)

    print(f"Training for {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    losses = []
    for epoch in range(args.epochs):
        epoch_key = jax.random.PRNGKey(args.seed + epoch + 1)
        loss_val = jnp.nan
        for x_batch, y_batch in data_loader(X_train, Y_train, args.batch_size, key=epoch_key):
            model, opt_state, loss_val = make_step(model, opt_state, x_batch, y_batch)
        losses.append(float(loss_val))

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:4d}  loss = {float(loss_val):.4e}")

    print(f"Done. Final loss: {losses[-1]:.4e}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_dir = os.path.dirname(args.save)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    eqx.tree_serialise_leaves(args.save, model)
    losses_path = args.save.replace(".eqx", "_losses.npy")
    np.save(losses_path, np.array(losses))
    print(f"Saved model  → {args.save}")
    print(f"Saved losses → {losses_path}")

    return model, losses


if __name__ == "__main__":
    main()
