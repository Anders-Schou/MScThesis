
import matplotlib.pyplot as plt
from matplotlib import rc

import jax
import jax.numpy as jnp

from utils.plotting import save_fig


n = 15
nn = 1000

key = jax.random.PRNGKey(0)


lims = [-jnp.pi, jnp.pi]

s = jax.random.uniform(key, (n,), minval=lims[0], maxval=lims[-1])

# s = jnp.linspace(*lims, 2*n+1)[1::2]


lf = lambda x: jnp.sin(x)
hf = lambda x: jnp.sin(10*x)

xx = jnp.linspace(*lims, nn+1)

f = [lf, hf]
fignames = ["lowfreq", "highfreq"]

xlim = lims
ylim = [-1.25, 1.25]
rc('text', usetex=True)
for i in range(2):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.plot(xx, f[i](xx), color="k")
    ax.scatter(s, f[i](s), color="k")
    ax.vlines(s, *ylim, color="k")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("auto", adjustable="box")
    ax.set_xticks([lims[0], 0, lims[-1]])
    ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=25)
    ax.set_yticks([])
    ax.set_yticklabels([])
    save_fig("figures", fignames[i], "pdf", fig=fig)