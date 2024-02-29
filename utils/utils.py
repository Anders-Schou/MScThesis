from collections.abc import Sequence, Callable

import jax
import jax.numpy as jnp


def cyclic_diff(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.subtract(x, jnp.roll(x, -1))


def remove_points(arr: jnp.ndarray, fun: Callable) -> jnp.ndarray:
    return arr[jnp.invert(fun(arr))].copy()


def keep_points(arr: jnp.ndarray, fun: Callable) -> jnp.ndarray:
    return arr[fun(arr)].copy()


def out_shape(fun, *args):
    print(f"{fun.__name__:<15}    {jax.eval_shape(fun, *args).shape}")

