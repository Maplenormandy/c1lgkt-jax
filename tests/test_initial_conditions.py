import math
import unittest
from pathlib import Path

import jax.numpy as jnp

from c1lgkt.jax.analysis import build_initial_conditions_from_toml, build_initial_conditions, assemble_state


class InitialConditionsTomlTests(unittest.TestCase):
    def test_meshgrid_stack_example(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        toml_path = f"{repo_root}/docs/initial_conditions_example.toml"
        samples = build_initial_conditions_from_toml(str(toml_path))

        self.assertEqual(set(samples.keys()), {"R", "varphi", "Z", "upar", "mu"})
        for tag, arr in samples.items():
            self.assertEqual(arr.shape, (6,), msg=f"{tag} has wrong shape")

        expected_r = jnp.asarray([1.0, 1.0, 1.1, 1.1, 1.2, 1.2])
        expected_phi = jnp.asarray([0.0, math.pi, 0.0, math.pi, 0.0, math.pi])
        expected_z = jnp.zeros((6,))

        self.assertTrue(jnp.allclose(samples["R"], expected_r))
        self.assertTrue(jnp.allclose(samples["varphi"], expected_phi))
        self.assertTrue(jnp.allclose(samples["Z"], expected_z))

        state = assemble_state(samples, axis=0)
        self.assertEqual(state.shape, (5, 6))

    def test_concat_union(self) -> None:
        spec = {
            "initial_conditions": {
                "root": "union",
                "nodes": {
                    "r_a": {"tag": "R", "sampler": "values", "values": [1.0, 1.1]},
                    "r_b": {"tag": "R", "sampler": "values", "values": [2.0]},
                    "phi_a": {"tag": "varphi", "sampler": "values", "values": [0.0, 0.5]},
                    "phi_b": {"tag": "varphi", "sampler": "values", "values": [1.5]},
                    "z_a": {"tag": "Z", "sampler": "constant", "value": 0.0, "num": 2},
                    "z_b": {"tag": "Z", "sampler": "constant", "value": 0.0, "num": 1},
                    "upar_a": {"tag": "upar", "sampler": "constant", "value": 0.0, "num": 2},
                    "upar_b": {"tag": "upar", "sampler": "constant", "value": 0.0, "num": 1},
                    "mu_a": {"tag": "mu", "sampler": "constant", "value": 0.0, "num": 2},
                    "mu_b": {"tag": "mu", "sampler": "constant", "value": 0.0, "num": 1},
                    "stack_a": {"op": "stack", "inputs": ["r_a", "phi_a", "z_a", "upar_a", "mu_a"]},
                    "stack_b": {"op": "stack", "inputs": ["r_b", "phi_b", "z_b", "upar_b", "mu_b"]},
                    "union": {"op": "concat", "inputs": ["stack_a", "stack_b"]},
                },
            }
        }

        samples = build_initial_conditions(spec)
        self.assertEqual(samples["R"].shape, (3,))
        self.assertTrue(jnp.allclose(samples["R"], jnp.asarray([1.0, 1.1, 2.0])))
        self.assertTrue(jnp.allclose(samples["varphi"], jnp.asarray([0.0, 0.5, 1.5])))

    def test_stack_scalar_broadcast(self) -> None:
        spec = {
            "initial_conditions": {
                "root": "stack_all",
                "nodes": {
                    "r0": {"tag": "R", "sampler": "values", "values": [1.0, 1.1, 1.2]},
                    "phi0": {"tag": "varphi", "sampler": "values", "values": [0.0, 0.5, 1.0]},
                    "z0": {"tag": "Z", "sampler": "constant", "value": 0.0},
                    "upar0": {"tag": "upar", "sampler": "constant", "value": 2.5},
                    "mu0": {"tag": "mu", "sampler": "constant", "value": 0.0},
                    "stack_all": {"op": "stack", "inputs": ["r0", "phi0", "z0", "upar0", "mu0"]},
                },
            }
        }

        samples = build_initial_conditions(spec)
        self.assertEqual(samples["R"].shape, (3,))
        self.assertTrue(jnp.allclose(samples["Z"], jnp.asarray([0.0, 0.0, 0.0])))
        self.assertTrue(jnp.allclose(samples["upar"], jnp.asarray([2.5, 2.5, 2.5])))


if __name__ == "__main__":
    unittest.main()
