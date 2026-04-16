"""
This module contains tools for specifying initial conditions in a reproducible manner
"""
import jax.numpy as jnp

import equinox as eqx
import interpax

import abc

from ..particles import particle_motion, particle_tools
from dataclasses import dataclass, field, asdict
from typing import Literal, Any, Callable
import json, yaml
import abc

from ..custom_types import ScalarArray

# %% Custom samplers

def quasirandom(bounds: list[list[float]], num: int) -> ScalarArray | tuple[ScalarArray, ...]:
    """
    Generates num quasirandom samples in the given bounds. The dimension is inferred from the length of the bounds list
    """
    if len(bounds) == 1:
        g = 1.6180339887498948482
        a1 = 1.0/g
        z1 = (0.5 + a1*jnp.arange(num)) % 1
        y1 = bounds[0][0] + z1 * (bounds[0][1] - bounds[0][0])
        return y1
    elif len(bounds) == 2:
        g = 1.32471795724474602596
        a1 = 1.0/g
        a2 = 1.0/(g*g)
        z1, z2 = map(lambda a: (0.5 + a*jnp.arange(num)) % 1, (a1, a2))
        y1, y2 = map(lambda z, b: b[0] + z * (b[1] - b[0]), (z1, z2), bounds)
        return y1, y2
    elif len(bounds) == 3:
        g = 1.22074408460575947536
        a1 = 1.0/g
        a2 = 1.0/(g*g)
        a3 = 1.0/(g*g*g)
        z1, z2, z3 = map(lambda a: (0.5 + a*jnp.arange(num)) % 1, (a1, a2, a3))
        y1, y2, y3 = map(lambda z, b: b[0] + z * (b[1] - b[0]), (z1, z2, z3), bounds)
        return y1, y2, y3
    else:
        raise NotImplementedError("Quasirandom sampling only implemented for 1D, 2D, and 3D bounds")


# %% Ops for joining outputs from multiple inputs


def op_broadcast(inputs: list[dict[str, Any]]) -> dict[str, Any]:
    """
    This op takes a dictionary of outputs from multiple particles and combines them into a single dictionary,
    broadcasting them to the same shape. The shape is determined by the common broadcast shape of all the values.
    This is a disjoint union, so the keys of the dictionary must be unique
    """
    joined = {}
    shapes = []

    for child in inputs:
        child_shape = child.pop('shape', ())
        shapes.append(child_shape)

        for key, value in child.items():
            if key in joined:
                raise ValueError(f"Duplicate key {key} found in children dictionaries. Cannot join.")
            joined[key] = jnp.broadcast_to(value, child_shape) # broadcast the value to the child shape and store it in the joined dictionary

    # Check if we can broadcast all the values to the same shape
    shape = jnp.broadcast_shapes(*shapes)
    
    # We need to broadcast all the values to the same shape, which should be the shape of the first non-scalar value we encounter
    for key, value in joined.items():
        joined[key] = jnp.broadcast_to(value, shape).ravel()

    joined['shape'] = shape

    return joined

def op_meshgrid(inputs: list[dict[str, Any]]) -> dict[str, Any]:
    """
    This op takes a list of dicts and combines them into a single dict via a
    cartesian-product-like broadcast. Given inputs with shapes [s1, s2, ..., sn],
    each value in input i is reshaped to (*1s_before, *si, *1s_after) so that
    the final broadcast shape is (*s1, *s2, ..., *sn).
    Scalars (shape ()) contribute no axes to the output shape.
    """
    joined = {}
    shapes = []

    # --- Pass 1: validate and collect per-input shapes ---
    for i, child in enumerate(inputs):
        child_shape = child.pop('shape', ())
        shapes.append(child_shape)

        for key, value in child.items():
            if key in joined:
                raise ValueError(f"Duplicate key {key} in child {i}. Cannot meshgrid.")
            s = jnp.shape(value)
            joined[key] = jnp.broadcast_to(value, child_shape) # broadcast the value to the child shape and store it in the joined dictionary

    # --- Compute output shape and per-input reshape targets ---
    # Each input i gets reshaped to: (*1s, *shapes[i], *1s)
    # where the leading 1s cover all axes from inputs 0..i-1
    # and the trailing 1s cover all axes from inputs i+1..n-1.
    total_shape = sum(shapes, start=())
    n_total = len(total_shape)

    # Build reshape for input i: prefix of 1s, then own shape, then suffix of 1s
    def reshape_for(value, i):
        prefix_len = sum(len(s) for s in shapes[:i])
        suffix_len = sum(len(s) for s in shapes[i+1:])
        new_shape = (1,) * prefix_len + shapes[i] + (1,) * suffix_len
        return jnp.broadcast_to(jnp.reshape(value, new_shape), total_shape)

    # --- Pass 2: reshape and broadcast each value ---
    result = {}
    for i, child in enumerate(inputs):
        for key in child.keys():
            result[key] = reshape_for(joined[key], i).ravel()

    result['shape'] = total_shape

    return result


op_registry = {
    'meshgrid': op_meshgrid,
    'broadcast': op_broadcast,
}


def reduce_inputs(inputs: dict | list[dict[str, Any]]) -> dict[str, Any]:
    
    if isinstance(inputs, dict):
        # If it's already a dict, just return it
        return inputs
    
    elif isinstance(inputs, list):
        # If it's a list, we need to join the dicts together
        # We look for any dict in the list that has a special key "__op" which specifies the op to use for joining
        # If no such dict is found, we default to broadcasting all inputs together

        to_join = list()
        op = 'broadcast'  # default op

        for child in inputs:
            if '__op' in child:
                op = child['__op']
            else:
                to_join.append(child)

        if op not in op_registry:
            raise ValueError(f"Op '{op}' not found in registry.")
        
        func = op_registry[op]

        return func(to_join)
    
    else:
        raise ValueError(f"Invalid input type: {type(inputs)}. Must be dict or list of dict.")

# %% Custom transformations

def outboard_midplane(input: dict[str, Any], context: dict) -> dict[str, Any]:
    """
    This transform takes psi and and outputs R and Z on the outboard midplane.
    """
    eq = context['eq']
    r_outer = jnp.linspace(eq.raxis, eq.rmax, 128)
    psi_outer = eq.interp_psi(r_outer, jnp.full_like(r_outer, eq.zaxis))
    interp_router = interpax.Interpolator1D(psi_outer, r_outer, method='cubic2')

    psi = input.pop('psi')
    r = interp_router(psi)
    z = jnp.full_like(r, eq.zaxis)

    input['R'] = r
    input['Z'] = z
    
    return input

def upar_mu(input: dict[str, Any], context: dict) -> dict[str, Any]:
    """
    This transform takes the kinetic energy and pitch angle and outputs mu and upar.
    """
    eq = context['eq']

    xi = input.pop('xi')
    ev = input.pop('ev')
    r = input['R']
    z = input['Z']

    bv = eq.compute_bv(r, z)
    modb = jnp.linalg.norm(bv, axis=0)

    pp = context['pp']

    upar = pp.vt * xi * jnp.sqrt(ev)
    mu = pp.m * (1-xi**2) * (pp.vt * jnp.sqrt(ev))**2 / (2*modb)
    
    input['mu'] = mu
    input['upar'] = upar

    return input

def integrals(input: dict[str, Any], context: dict) -> dict[str, Any]:
    """
    This transform computes the integrals of motion
    """
    if 'psi' in input:
        input = outboard_midplane(input, context)
    if 'xi' in input:
        input = upar_mu(input, context)
        
    t0 = input.get('t0', context.get('t0', 0.0))
    r = input.pop('R')
    varphi = input.pop('varphi', jnp.zeros_like(r))
    z = input.pop('Z')
    upar = input.pop('upar')
    mu = input['mu']

    state = particle_motion.PusherState(
        r=r,
        varphi=varphi,
        z=z,
        upar=upar,
        jperp1=jnp.sqrt(2*mu),
        jperp2=jnp.zeros_like(mu)
    )

    args = context['args']

    ham, lphi = particle_tools.compute_integrals(t0, state, args)

    input['H'] = ham
    input['Lphi'] = lphi

    return input

def from_integrals(input: dict[str, Any], context: dict) -> dict[str, Any]:
    omega_frame = input.get('omega_frame', context.get('omega_frame', 0.0))
    ham = input['H']
    lphi = input['Lphi']

    upar = jnp.zeros_like(ham)

    state = particle_motion.PusherState(
        r=input['R'],
        varphi=input['varphi'],
        z=input['Z'],
        upar=upar,
        jperp1=jnp.sqrt(2*input['mu']),
        jperp2=jnp.zeros_like(input['mu'])
    )

    pp = context['pp']

    kpar, upar_omega = particle_tools.compute_parallel_energy(0.0, state, (ham, lphi), omega_frame, context['args'])
    upar0 = jnp.sqrt(2 * kpar / pp.m) + upar_omega

    input['upar'] = upar0

    return input



# %% Registries of allowed samplers, transforms, and ops

sampler_registry = {
    'linspace': jnp.linspace,
    'quasirandom': quasirandom,
}

transform_registry = {
    'outboard_midplane': outboard_midplane,
    'upar_mu': upar_mu,
    'integrals': integrals,
    'from_integrals': from_integrals,
}


# %%

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


class AbstractInputNode(eqx.Module):
    @abc.abstractmethod
    def realize(self, context: dict) -> dict[str, Any]:
        raise NotImplementedError

class ConstantLeaf(AbstractInputNode):
    constant: str | float | list[str | float]
    axes: str | list[str]

    def realize(self, context: dict) -> dict[str, Any]:
        value = replace_handlebars(self.constant, context)
        
        match self.axes:
            case str(axis):
                output = jnp.asarray(value)
                shape = jnp.shape(output)
                return {axis: output, "shape": shape}
            case list(axes):
                output = [jnp.asarray(val) for val in value]
                shape = jnp.broadcast_shapes(*[jnp.shape(out) for out in output])
                return {axis: out for axis, out in zip(axes, output)} | {"shape": shape}
            case _:
                raise ValueError(f"Invalid type for axes: {type(self.axes)}. Must be str or list[str].")

class SamplerLeaf(AbstractInputNode):
    sampler: str
    axes: str | list[str]
    args: list[Any] = eqx.field(default_factory=list)
    kwargs: dict[str, Any] = eqx.field(default_factory=dict)

    def realize(self, context: dict) -> dict[str, Any]:
        # Check that the specified sampler is in the registry
        if self.sampler not in sampler_registry:
            raise ValueError(f"Sampler '{self.sampler}' not found in registry.")

        args = replace_handlebars(self.args, context)
        kwargs = replace_handlebars(self.kwargs, context)

        func = sampler_registry[self.sampler]
        output = func(*args, **kwargs) # pyright: ignore

        match self.axes:
            case str(axis):
                output = jnp.asarray(output)
                shape = jnp.shape(output)
                return {axis: jnp.asarray(output), "shape": shape}
            case list(axes):
                output = [jnp.asarray(out) for out in output]
                shape = jnp.broadcast_shapes(*[jnp.shape(out) for out in output])
                return {axis: out for axis, out in zip(axes, output)} | {"shape": shape}
            case _:
                raise ValueError(f"Invalid type for axes: {type(self.axes)}. Must be str or list[str].")

class TransformNode(AbstractInputNode):
    transform: str
    input: AbstractInputNode | list[AbstractInputNode]

    def realize(self, context: dict) -> dict[str, Any]:
        if self.transform not in transform_registry:
            raise ValueError(f"Transform '{self.transform}' not found in registry.")
        
        func = transform_registry[self.transform]

        if isinstance(self.input, list):
            child_output = reduce_inputs([child.realize(context) for child in self.input])
        else:
            child_output = self.input.realize(context)

        return func(child_output, context)
    
class OpLeaf(AbstractInputNode):
    """
    This is a special leaf which changes the nature of the joining of inputs
    """
    op: str

    def realize(self, context: dict) -> dict[str, Any]:
        """
        This is a special leaf which tells the reduction operation to change how it joins the inputs together.
        """
        return {"__op": self.op}
        

class InitialConditionGenerator(AbstractInputNode):
    inputs: AbstractInputNode | list[AbstractInputNode]

    def realize(self, context: dict) -> dict[str, Any]:
        def realize_child(child: AbstractInputNode | list[AbstractInputNode]):
            match child:
                case list(children):
                    return reduce_inputs([realize_child(c) for c in children])
                case AbstractInputNode():
                    return child.realize(context)
                case _:
                    raise ValueError(f"Invalid child type: {type(child)}. Must be AbstractInputNode or list of AbstractInputNode.")
        
        # Realize all of the children and reduce them together
        output = realize_child(self.inputs)

        # Now, check whether or not we need to apply any final transformations to the output
        if 'omega_frame' in output:
            output = from_integrals(output, context)

        return output

def object_hook(obj: dict) -> Any:
    if 'sampler' in obj:
        return SamplerLeaf(**obj)
    elif 'constant' in obj:
        return ConstantLeaf(**obj)
    elif 'transform' in obj:
        return TransformNode(**obj)
    elif 'op' in obj:
        return OpLeaf(**obj)
    elif 'inputs' in obj:
        return InitialConditionGenerator(**obj)
    return obj


def build_initial_conditions(obj: Any) -> Any:
    """
    Builds initial conditions from a JSON-like dictionary. The dictionary can specify samplers,
    constants, and transforms, which will be evaluated at runtime to produce the final initial conditions.
    """
    if isinstance(obj, dict):
        processed = {k: build_initial_conditions(v) for k, v in obj.items()}
        return object_hook(processed)
    elif isinstance(obj, list):
        return [build_initial_conditions(item) for item in obj]
    else:
        return obj  # scalar — leave as-is
