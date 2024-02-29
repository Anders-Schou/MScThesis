import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

def save_fig(dir: str, file_name: str, format: str = "png", clear = False):
    if not file_name.endswith("." + format):
        file_name += ("." + format)
    plt.savefig(os.path.join(dir, file_name), format=format)
    if clear:
        plt.clf()


def plot_polygon(ax, vertices: jnp.ndarray, *args, **kwargs):
    v = jnp.squeeze(vertices)
    if v.ndim != 2:
        raise ValueError(f"Input must only have 2 dimensions with length > 1, but there were {v.ndim}.")
    x = jnp.append(v[:, 0].ravel(), v[0, 0])
    y = jnp.append(v[:, 1].ravel(), v[0, 1])
    ax.plot(x, y, *args, **kwargs)


def plot_circle(ax, radius: float, resolution: int, *args, **kwargs) -> None:
    theta = jnp.linspace(0, 2*jnp.pi, resolution+1)
    x = radius * jnp.cos(theta)
    y = radius * jnp.sin(theta)
    ax.plot(x, y, *args, **kwargs)
    return