from collections.abc import Sequence, Callable
from functools import wraps
import os
import json
from time import perf_counter

import jax
import jax.numpy as jnp
from torch.utils.tensorboard import SummaryWriter


def cyclic_diff(x: jax.Array) -> jax.Array:
    return jnp.subtract(x, jnp.roll(x, -1))


def remove_points(arr: jax.Array, fun: Callable) -> jax.Array:
    return arr[jnp.invert(fun(arr))].copy()


def keep_points(arr: jax.Array, fun: Callable) -> jax.Array:
    return arr[fun(arr)].copy()


def out_shape(fun, *args):
    print(f"{fun.__name__:<15}    {jax.eval_shape(fun, *args).shape}")


def limits2vertices(xlim: Sequence, ylim: Sequence) -> list[tuple[list]]:
    """
    Works for rectangles only. Returns pairs of end points for
    each of the four sides in counterclockwise order.
    """
    
    v = [
         ([xlim[0], ylim[0]], [xlim[1], ylim[0]]), # Lower horizontal
         ([xlim[1], ylim[0]], [xlim[1], ylim[1]]), # Right vertical
         ([xlim[1], ylim[1]], [xlim[0], ylim[1]]), # Upper horizontal
         ([xlim[0], ylim[1]], [xlim[0], ylim[0]])  # Left vertical
    ]
    return v


def normal_eq(x: jax.Array, y: jax.Array, base: list[Callable], ridge: float | None = None):
    X = jnp.concatenate([b(x).reshape(-1, 1) for b in base], axis=-1)
    XT_X = X.T @ X
    XT_y = X.T @ y
    if ridge is None:
        return jnp.linalg.solve(XT_X, XT_y).ravel()
    return jnp.linalg.solve(XT_X+ridge*jnp.identity(len(base)), XT_y).ravel()


def pretty_json(hp):
  json_hp = json.dumps(hp, indent=2)
  return "".join("\t" + line for line in json_hp.splitlines(True))


def log_settings(settings):
    os.system("rm -rf " + settings["io"]["log_dir"]+"/"+settings["id"]+"/*")
    writer = SummaryWriter(log_dir=settings["io"]["log_dir"]+"/"+settings["id"])
    writer.add_text("settings.json", pretty_json(settings))
    writer.close()
    return


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t1 = perf_counter()
        v = func(*args, **kwargs)
        t2 = perf_counter()
        print(f"Time for function '{func.__name__}': {t2-t1:.6f} seconds")
        return v
    return wrapper
