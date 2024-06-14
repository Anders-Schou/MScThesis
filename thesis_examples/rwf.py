from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, cm, ticker, colors as pcol
rc("text", usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')

import jax
import jax.numpy as jnp


N = 1000
_INIT = -0.3
_MINS = jnp.array([-1., 0.25*(jnp.sqrt(0.6)-1)]) # [global, local] mins
_SHIFT = jnp.mean(_MINS)
_LEVELS = jnp.linspace(0.01, 24.2, 801)**3
# _LEVELS = jnp.concatenate([jnp.arange(0., 0.5, 401), jnp.linspace(0.5**(1/3), 24.2, 101)**3])
opt_fun = lambda x: (x+1.+_SHIFT)**2*(5.*(x+_SHIFT)**2+0.25)+0.01
init = _INIT - _SHIFT
mins = _MINS - _SHIFT


ww = jnp.linspace(mins[0]-0.2, mins[1]+0.2, N+1)
m = jnp.max(jnp.abs(ww))
wf = jnp.linspace(-m, m, N+1)*4
we = wf#jnp.exp(wf**2)*jnp.sign(wf)
vv, ss = jnp.meshgrid(we, we)

wf1 = wf[wf<0]
wf2 = wf[wf>0]
hyperbolas = [(wf1, mins[0]/wf1), (wf2, mins[0]/wf2),
              (wf1, mins[1]/wf1), (wf2, mins[1]/wf2),
              (wf1, init   /wf1), (wf2, init   /wf2)]
colors = ["red", "red", "tab:orange", "tab:orange", "blue", "blue"]


fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].plot(ww, opt_fun(ww), linewidth=4, color="black")
ax[0].scatter(mins[0], opt_fun(mins[0]), s=100, color="red", alpha=1., zorder=2)
ax[0].scatter(mins[1], opt_fun(mins[1]), s=100, color="tab:orange", alpha=1., zorder=3)
ax[0].scatter(init, opt_fun(init), s=100, color="blue", alpha=1., zorder=4)
ax[0].set_xlabel(r"$w$", fontsize=40)
ax[0].set_ylabel(r"$\mathcal{L}(w)$", rotation=0, fontsize=40, ha="right", labelpad=16)
ax[0].legend(["_hidden", r"\textbf{Global minimum}", r"\textbf{Local mininum}", r"\textbf{Initialization}"],
             fontsize=30, loc="lower right")

ymin = jnp.min(opt_fun(ww))
ymax = jnp.max(opt_fun(ww))
ydiff = 0.1*(ymax - ymin)
ax[0].set_ylim([ymin-ydiff, ymax+ydiff])


# ax[1].set_aspect("equal", adjustable="datalim")
ax[1].contourf(ss, vv, (opt_fun(ss*vv)), levels=_LEVELS, norm=pcol.LogNorm(), cmap="Greys").set_edgecolor("face")
for h, c in zip(hyperbolas, colors):
    ax[1].plot(*h, linewidth=2, linestyle="--", color=c, alpha=1.)
ax[1].set_xlim([wf[0], wf[-1]])
ax[1].set_ylim([wf[0], wf[-1]])
ax[1].set_xlabel(r"$s$", fontsize=40)
ax[1].set_ylabel(r"$v$", rotation=0, fontsize=40, ha="right", labelpad=16)



for i in range(2):
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_xticklabels([])
    ax[i].set_yticklabels([])



fig.savefig("figures/rwf.pdf", bbox_inches="tight")