# Initial Conditions TOML Format

This document describes the initial-condition specification for particle simulations.

The specification lives under the `[initial_conditions]` table and is evaluated as a tagged expression tree. Leaf nodes create tagged sample arrays. Branch nodes combine those arrays using tag-aware operations.

## Top-Level Table

`[initial_conditions]` keys:

- `root` (required): Name of the node that produces the final tagged set.
- `coords` (optional): Coordinate order used for validation and assembly. Default: `"R", "varphi", "Z", "upar", "mu"`.
- `seed` (optional): Integer seed used by random samplers.

Nodes are defined under `[initial_conditions.nodes.<name>]`.

## Node Types

Leaf node:

- `kind = "leaf"`
- `tag` (required): Coordinate tag the sampler produces.
- `sampler` (required): Sampler name.
- Sampler-specific parameters.
- `seed` (optional): Overrides the top-level seed for this node.

Op node:

- `kind = "op"`
- `op` (required): Operation name.
- `inputs` (required): List of node names.

The `kind` key may be omitted if `sampler` or `op` is present.

## Samplers

Built-in samplers:

- `linspace`: `start`, `stop`, `num`, optional `endpoint` (default `true`).
- `arange`: `start` (default `0`), `stop`, `step` (default `1`).
- `values`: `values` (array literal).
- `constant`: `value`, optional `num` (default `1`).
- `uniform`: `low` (default `0`), `high` (default `1`), `num` (requires seed).
- `normal`: `mean` (default `0`), `std` (default `1`), `num` (requires seed).

## Operations

Built-in ops:

- `stack`: Merge blocks with disjoint tags. All inputs must have the same length.
- `meshgrid`: Cartesian product across input blocks. Tags within a block are treated as correlated.
- `concat`: Concatenate multiple blocks with identical tag sets.

`meshgrid` ordering uses `indexing="ij"` semantics. The last input varies fastest after flattening.

## Example 1: Meshgrid + Stack

```toml
[initial_conditions]
root = "stack_all"
coords = ["R", "varphi", "Z", "upar", "mu"]

[initial_conditions.nodes.r0]
kind = "leaf"
tag = "R"
sampler = "linspace"
start = 1.3
stop = 1.7
num = 16

[initial_conditions.nodes.phi0]
kind = "leaf"
tag = "varphi"
sampler = "linspace"
start = 0.0
stop = 6.283185307179586
num = 32
endpoint = false

[initial_conditions.nodes.z0]
kind = "leaf"
tag = "Z"
sampler = "constant"
value = 0.0

[initial_conditions.nodes.upar0]
kind = "leaf"
tag = "upar"
sampler = "constant"
value = 0.0

[initial_conditions.nodes.mu0]
kind = "leaf"
tag = "mu"
sampler = "constant"
value = 0.0

[initial_conditions.nodes.mesh_rphi]
kind = "op"
op = "meshgrid"
inputs = ["r0", "phi0", "z0"]

[initial_conditions.nodes.stack_all]
kind = "op"
op = "stack"
inputs = ["mesh_rphi", "upar0", "mu0"]
```

## Example 2: Union of Two Clouds

```toml
[initial_conditions]
root = "union"
seed = 123

[initial_conditions.nodes.r_a]
tag = "R"
sampler = "normal"
mean = 1.5
std = 0.05
num = 64

[initial_conditions.nodes.r_b]
tag = "R"
sampler = "normal"
mean = 1.7
std = 0.02
num = 64

[initial_conditions.nodes.phi_a]
tag = "varphi"
sampler = "uniform"
low = 0.0
high = 6.283185307179586
num = 64

[initial_conditions.nodes.phi_b]
tag = "varphi"
sampler = "uniform"
low = 0.0
high = 6.283185307179586
num = 64

[initial_conditions.nodes.z_a]
tag = "Z"
sampler = "constant"
value = 0.0
num = 64

[initial_conditions.nodes.z_b]
tag = "Z"
sampler = "constant"
value = 0.0
num = 64

[initial_conditions.nodes.upar_a]
tag = "upar"
sampler = "constant"
value = 0.0
num = 64

[initial_conditions.nodes.upar_b]
tag = "upar"
sampler = "constant"
value = 0.0
num = 64

[initial_conditions.nodes.mu_a]
tag = "mu"
sampler = "constant"
value = 0.0
num = 64

[initial_conditions.nodes.mu_b]
tag = "mu"
sampler = "constant"
value = 0.0
num = 64

[initial_conditions.nodes.stack_a]
op = "stack"
inputs = ["r_a", "phi_a", "z_a", "upar_a", "mu_a"]

[initial_conditions.nodes.stack_b]
op = "stack"
inputs = ["r_b", "phi_b", "z_b", "upar_b", "mu_b"]

[initial_conditions.nodes.union]
op = "concat"
inputs = ["stack_a", "stack_b"]
```

## Python Usage

```python
from c1lgkt.jax.analysis import build_initial_conditions_from_toml, assemble_state

samples = build_initial_conditions_from_toml("initial_conditions.toml")
state = assemble_state(samples, axis=0)
```

## Extending

Custom samplers and ops can be registered at runtime using `register_sampler` and `register_op`.
