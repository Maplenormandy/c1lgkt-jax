"""
This module contains code to load and set up PusherArgs
"""

from typing import Any
from ..particles.particle_motion import PusherArgs, ParticleParams, PusherState
from ..fields.equilibrium import Equilibrium

from ..fields.field_providers import AbstractFieldProvider

from .initial_conditions import build_initial_conditions, InitialConditionGenerator

import yaml
import jax.numpy as jnp

from jaxtyping import Array, Bool

# %%

default_context = {
    'pi': jnp.pi,
    'e': jnp.e,
    'sqrt': jnp.sqrt,
}

# Check for handlebars in the args and kwargs, and replace them with the evaluated expressions
def replace_handlebars(x: Any, context: dict) -> Any:
    """
    Handlebar replacement allows us to write expressions in the JSON that will be evaluated at runtime.
    """
    if isinstance(x, list):
        return [replace_handlebars(xi, context) for xi in x]
    elif isinstance(x, dict):
        return {k: replace_handlebars(v, context) for k, v in x.items()}
    elif isinstance(x, str) and x.startswith('{{') and x.endswith('}}'):
        expr = x[2:-2]
        return eval(expr, globals=None, locals=context)
    else:
        return x

def build_pusher_args(config: dict) -> PusherArgs:
    """
    Builds pusher args from a dictionary
    """
    
    ## Equilibrium
    eq_config = config.get('equilibrium', {})
    eq = Equilibrium.build_from_config(eq_config)

    # Default context
    context = default_context | {'eq': eq}
    
    ## Species parameters
    species_config = replace_handlebars(config.get('species', {}), context)
    pp = ParticleParams.build_from_config(species_config)

    context = context | {'pp': pp}
    
    ## Fields
    fields_config = replace_handlebars(config.get('fields', []), context)
    fields = list(map(lambda cfg: AbstractFieldProvider.build_from_config(cfg, eq), fields_config))
    
    return PusherArgs(eq=eq, pp=pp, fields=fields)


def load_yaml_config(path: str) -> tuple[PusherArgs, InitialConditionGenerator]:
    """
    Loads a YAML configuration file and returns it as a dictionary
    """
    
    with open(path, 'r') as f:
        configs = list(yaml.safe_load_all(f))

    if len(configs) != 2:
        raise ValueError(f'Expected 2 documents in the YAML file, but got {len(configs)}')

    pusher_args = build_pusher_args(configs[0])
    ic_generator = build_initial_conditions(configs[1])
    return pusher_args, ic_generator


def realize_initial_conditions(ic_generator: InitialConditionGenerator, args: PusherArgs) -> tuple[PusherState, Bool[Array, "..."], dict[str, Any]]:
    """
    Realizes the initial conditions by evaluating the generators and transforms specified in the InitialConditionGenerator.
    """
    # Default context
    context = default_context | {
        'eq': args.eq,
        'pp': args.pp,
        'fields': args.fields,
        'args': args
    }

    # Realize the initial conditions
    ic = ic_generator.realize(context)

    # Get the filter mask for the particles
    mask = jnp.isfinite(ic['upar'])

    # Set up the initial conditions as a PusherState
    y0 = PusherState(
        r=ic['R'][mask],
        varphi=ic['varphi'][mask],
        z=ic['Z'][mask],
        upar=ic['upar'][mask],
        jperp1=jnp.sqrt(2*ic['mu'][mask]),
        jperp2=jnp.zeros_like(ic['mu'][mask])
    )

        
    return y0, mask.reshape(ic['shape']), ic