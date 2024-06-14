import math

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

import numpy as np
from scipy.stats import qmc

import jax
import jax.numpy as jnp

from datahandlers.generators import generate_rectangle
from utils.plotting import save_fig


fig_format = "pdf"

lims = [0, jnp.pi]
endpoints = [
    [lims[0], lims[0]],
    [lims[1], lims[0]],
    [lims[1], lims[1]],
    [lims[0], lims[1]]
]

def line(i, unit_points):
    # print(unit_points.shape)
    # print(unit_points.ravel()[:5])

    return jnp.array(endpoints[i%4])*unit_points + jnp.array(endpoints[(i+1)%4])*(1-unit_points)



N = 1000
R = 100
keys = jax.random.split(jax.random.PRNGKey(1234), 3)

# Uniform sampling
p_unif = jax.random.uniform(keys[0], (N, 2))*jnp.pi
ru_keys = jax.random.split(keys[0], 4)
r_unif = [line(i, jax.random.uniform(ru_keys[i], (R, 1))) for i in range(4)]

# Sobol sampling
p_sobol = qmc.Sobol(2, seed=int(jax.random.randint(keys[1], (), 0, jnp.iinfo(jnp.int32).max))
                    ).random_base2(math.ceil(jnp.log2(N)))[:N]*jnp.pi
rs_keys = jax.random.split(keys[1], 4)
r_sobol = [line(i, qmc.Sobol(1, seed=int(jax.random.randint(rs_keys[i], (), 0, jnp.iinfo(jnp.int32).max))
                             ).random_base2(math.ceil(jnp.log2(R)))[:R]
                ) for i in range(4)]

# Halton sampling
p_halton = qmc.Halton(2, seed=int(jax.random.randint(keys[2], (), 0, jnp.iinfo(jnp.int32).max))).random(N)*jnp.pi
rh_keys = jax.random.split(keys[2], 4)
r_halton = [line(i, qmc.Halton(1, seed=int(jax.random.randint(rh_keys[i], (), 0, jnp.iinfo(jnp.int32).max))
                               ).random(R)
                 ) for i in range(4)]


fig, ax = plt.subplots(1, 3, figsize=(25, 10))
titles = ["Uniform", "Sobol", "Halton"]

p = [p_unif, p_sobol, p_halton]
r = [r_unif, r_sobol, r_halton]

for c in range(3):
    ax[c].scatter(*[p[c][:, i] for i in [0, 1]], s=25)
    [ax[c].scatter(*[r[c][t][:, i] for i in [0, 1]], c="red", s=25) for t in range(4)]
    ax[c].set_title(titles[c], fontsize=50)
    ax[c].set_aspect('equal', adjustable='box')
    ax[c].set_xticks(np.linspace(0, np.pi, 5))
    ax[c].set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"], fontsize=25)
    ax[c].set_xlabel(r"$x_1$", fontsize=40)
    ax[c].set_yticks(np.linspace(0, np.pi, 5))
    ax[c].set_yticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"], fontsize=25)
    if c == 0:
        ax[c].set_ylabel(r"$x_2$", fontsize=40, rotation=0, ha="right", labelpad=4)
fig.savefig("figures/qmc." + fig_format, bbox_inches="tight")

