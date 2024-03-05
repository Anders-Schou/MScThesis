import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

def save_fig(dir: str, file_name: str, format: str = "png", clear = False):
    if not file_name.endswith("." + format):
        file_name += ("." + format)
    plt.savefig(os.path.join(dir, file_name), format=format, bbox_inches="tight")
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


def get_plot_variables(xlim, ylim, grid = 201):
    x = jnp.linspace(xlim[0], xlim[1], grid)
    y = jnp.linspace(ylim[0], ylim[1], grid)
    X, Y = jnp.meshgrid(x, y)
    plotpoints = jnp.concatenate((X.reshape((-1, 1)), Y.reshape((-1, 1))), axis=1)
    return X, Y, plotpoints


# Plot phi function
def plot_potential(X, Y, Z, Z_true, fig_dir, name, extension="png", radius = 2, circle_res = 100):
        CLEVELS = 101
        
        fig, ax = plt.subplots(1, 3, figsize=(22, 5))
        ax[0].set_aspect('equal', adjustable='box')
        ax[0].set_title("True solution", fontsize=20)
        p1 = ax[0].contourf(X, Y, Z_true, levels=CLEVELS)
        plt.colorbar(p1, ax=ax[0])

        ax[1].set_aspect('equal', adjustable='box')
        ax[1].set_title("Prediction", fontsize=20)
        p2 = ax[1].contourf(X, Y, Z, levels=CLEVELS)
        plt.colorbar(p2, ax=ax[1])
        
        ax[2].set_aspect('equal', adjustable='box')
        ax[2].set_title("Abs. error", fontsize=20)
        p3 = ax[2].contourf(X, Y, jnp.abs(Z - Z_true), levels=CLEVELS)
        plt.colorbar(p3, ax=ax[2])
        
        [plot_circle(ax[i], radius, circle_res, color="red") for i in range(3)]
        save_fig(fig_dir, name, extension)
        plt.clf()


# Plot stress tensor
def plot_stress(X, Y, Z, Z_true, fig_dir, name, extension="png", radius = 2, circle_res = 100):
    CLEVELS = 101
    FONTSIZE = 40

    fig, ax = plt.subplots(3, 4, figsize=(55, 30))
    ax[0, 0].set_aspect('equal', adjustable='box')
    ax[0, 0].set_title("XX stress", fontsize=FONTSIZE)
    p1 = ax[0, 0].contourf(X, Y, Z[0], levels=CLEVELS)
    plt.colorbar(p1, ax=ax[0, 0])

    ax[0, 1].set_aspect('equal', adjustable='box')
    ax[0, 1].set_title("XY stress", fontsize=FONTSIZE)
    p2 = ax[0, 1].contourf(X, Y, Z[1], levels=CLEVELS)
    plt.colorbar(p2, ax=ax[0, 1])
    
    ax[0, 2].set_aspect('equal', adjustable='box')
    ax[0, 2].set_title("YX stress", fontsize=FONTSIZE)
    p3 = ax[0, 2].contourf(X, Y, Z[2], levels=CLEVELS)
    plt.colorbar(p3, ax=ax[0, 2])

    ax[0, 3].set_aspect('equal', adjustable='box')
    ax[0, 3].set_title("YY stress", fontsize=FONTSIZE)
    p4 = ax[0, 3].contourf(X, Y, Z[3], levels=CLEVELS)
    plt.colorbar(p4, ax=ax[0, 3])



    ax[1, 0].set_aspect('equal', adjustable='box')
    ax[1, 0].set_title("True XX stress", fontsize=FONTSIZE)
    p1 = ax[1, 0].contourf(X, Y, Z_true[0], levels=CLEVELS)
    plt.colorbar(p1, ax=ax[1, 0])

    ax[1, 1].set_aspect('equal', adjustable='box')
    ax[1, 1].set_title("True XY stress", fontsize=FONTSIZE)
    p2 = ax[1, 1].contourf(X, Y, Z_true[1], levels=CLEVELS)
    plt.colorbar(p2, ax=ax[1, 1])
    
    ax[1, 2].set_aspect('equal', adjustable='box')
    ax[1, 2].set_title("True YX stress", fontsize=FONTSIZE)
    p3 = ax[1, 2].contourf(X, Y, Z_true[2], levels=CLEVELS)
    plt.colorbar(p3, ax=ax[1, 2])

    ax[1, 3].set_aspect('equal', adjustable='box')
    ax[1, 3].set_title("True YY stress", fontsize=FONTSIZE)
    p4 = ax[1, 3].contourf(X, Y, Z_true[3], levels=CLEVELS)
    plt.colorbar(p4, ax=ax[1, 3])



    ax[2, 0].set_aspect('equal', adjustable='box')
    ax[2, 0].set_title("Abs. error of XX stress", fontsize=FONTSIZE)
    p1 = ax[2, 0].contourf(X, Y, jnp.abs(Z[0]-Z_true[0]), levels=CLEVELS)
    plt.colorbar(p1, ax=ax[2, 0])

    ax[2, 1].set_aspect('equal', adjustable='box')
    ax[2, 1].set_title("Abs. error of XY stress", fontsize=FONTSIZE)
    p2 = ax[2, 1].contourf(X, Y, jnp.abs(Z[1]-Z_true[1]), levels=CLEVELS)
    plt.colorbar(p2, ax=ax[2, 1])
    
    ax[2, 2].set_aspect('equal', adjustable='box')
    ax[2, 2].set_title("Abs. error of YX stress", fontsize=FONTSIZE)
    p3 = ax[2, 2].contourf(X, Y, jnp.abs(Z[2]-Z_true[2]), levels=CLEVELS)
    plt.colorbar(p3, ax=ax[2, 2])

    ax[2, 3].set_aspect('equal', adjustable='box')
    ax[2, 3].set_title("Abs. error of YY stress", fontsize=FONTSIZE)
    p4 = ax[2, 3].contourf(X, Y, jnp.abs(Z[3]-Z_true[3]), levels=CLEVELS)
    plt.colorbar(p4, ax=ax[2, 3])


    
    [plot_circle(ax[i, j], radius, circle_res, color="red") for i in range(3) for j in range(4)]
    save_fig(fig_dir, name, extension)
    plt.clf()