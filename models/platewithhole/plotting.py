import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np

from utils.plotting import log_img, save_fig


_DEFAULT_RADIUS = 2
_DEFAULT_CIRCLE_RES = 100
_CLEVELS = 101
_FONTSIZE = 40




def plot_loss(loss_arr: jax.Array, loss_map: dict, *,
              fig_dir,
              name,
              epoch_step = None,
              extension="png",
              figsize = (35, 30)):

    num_plots = len(loss_map.keys())
    fig, ax = plt.subplots(num_plots, 1, figsize=figsize)
    plot_split = list(loss_map.keys())

    if epoch_step is not None:
        epochs = epoch_step*np.arange(loss_arr.shape[0])
        for i in range(num_plots):
            ax[i].semilogy(epochs, loss_arr[:, loss_map[plot_split[i]]], linewidth=5)
            ax[i].tick_params(axis='x', labelsize=_FONTSIZE)
            ax[i].tick_params(axis='y', labelsize=_FONTSIZE)
            # ax[i].fill_between(epochs[epochs % 10000 >= 5000], 0, facecolor='gray', alpha=.5)
    else:
        for i in range(num_plots):
            ax[i].semilogy(loss_arr[:, loss_map[plot_split[i]]], linewidth=5)
            ax[i].tick_params(axis='x', labelsize=_FONTSIZE)
            ax[i].tick_params(axis='y', labelsize=_FONTSIZE)
        
    save_fig(fig_dir, name, extension)
    plt.clf()
    return



# def plot_potential(X, Y, Z, *, fig_dir, name,
#                    extension="png",
#                    radius = _DEFAULT_RADIUS,
#                    circle_res = _DEFAULT_CIRCLE_RES,
#                    log_dir="", 
#                    save=True, 
#                    log=False,
#                    step=None):
#     """
#     Function for plotting potential function.
#     """    
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.gca()
#     ax.set_aspect('equal', adjustable='box')
#     ax.set_title("Prediction", fontsize=20)
#     p = ax.contourf(X, Y, Z, levels=_CLEVELS)
#     plt.colorbar(p, ax=ax)
    
#     plot_circle(plt, radius, circle_res, color="red")
#     if save:
#         save_fig(fig_dir, name, extension)
#     if log:
#         log_fig(log_dir, name, extension, step)
#     plt.clf()
#     return

# def plot_potential_sides(x, y, z, *,
#                          fig_dir,
#                          name,
#                          extension="png"):
#     pass


# def plot_stress(X, Y, Z, Z_true, *, fig_dir, name,
#                 extension="png",
#                 radius = _DEFAULT_RADIUS,
#                 circle_res = _DEFAULT_CIRCLE_RES,
#                 figsize = (35, 30),
#                 log_dir="", 
#                 save=True, 
#                 log=False,
#                 step=None):
#     """
#     Function for plotting stresses in cartesian coordinates.
#     """
    
#     vmin0 = min(jnp.min(Z_true[0]),jnp.min(Z[0]))
#     vmin1 = min(jnp.min(Z_true[1]),jnp.min(Z[1]))
#     vmin2 = min(jnp.min(Z_true[2]),jnp.min(Z[2]))
#     vmin3 = min(jnp.min(Z_true[3]),jnp.min(Z[3]))
    
#     vmax0 = max(jnp.max(Z_true[0]), jnp.max(Z[0]))
#     vmax1 = max(jnp.max(Z_true[1]), jnp.max(Z[1]))
#     vmax2 = max(jnp.max(Z_true[2]), jnp.max(Z[2]))
#     vmax3 = max(jnp.max(Z_true[3]), jnp.max(Z[3]))
    

#     fig, ax = plt.subplots(3, 3, figsize=figsize)
#     ax[0, 0].set_aspect('equal', adjustable='box')
#     ax[0, 0].set_title("XX stress", fontsize=_FONTSIZE)
#     p1 = ax[0, 0].contourf(X , Y, Z[0], levels=_CLEVELS, vmin=vmin0, vmax=vmax0)
#     plt.colorbar(p1, ax=ax[0, 0])

#     ax[0, 1].set_aspect('equal', adjustable='box')
#     ax[0, 1].set_title("XY stress", fontsize=_FONTSIZE)
#     p2 = ax[0, 1].contourf(X, Y, Z[1], levels=_CLEVELS, vmin=vmin1, vmax=vmax1)
#     plt.colorbar(p2, ax=ax[0, 1])

#     ax[0, 2].set_aspect('equal', adjustable='box')
#     ax[0, 2].set_title("YY stress", fontsize=_FONTSIZE)
#     p4 = ax[0, 2].contourf(X, Y, Z[3], levels=_CLEVELS, vmin=vmin3, vmax=vmax3)
#     plt.colorbar(p4, ax=ax[0, 2])



#     ax[1, 0].set_aspect('equal', adjustable='box')
#     ax[1, 0].set_title("True XX stress", fontsize=_FONTSIZE)
#     p1 = ax[1, 0].contourf(X, Y, Z_true[0], levels=_CLEVELS, vmin=vmin0, vmax=vmax0)
#     plt.colorbar(p1, ax=ax[1, 0])

#     ax[1, 1].set_aspect('equal', adjustable='box')
#     ax[1, 1].set_title("True XY stress", fontsize=_FONTSIZE)
#     p2 = ax[1, 1].contourf(X, Y, Z_true[1], levels=_CLEVELS, vmin=vmin1, vmax=vmax1)
#     plt.colorbar(p2, ax=ax[1, 1])

#     ax[1, 2].set_aspect('equal', adjustable='box')
#     ax[1, 2].set_title("True YY stress", fontsize=_FONTSIZE)
#     p4 = ax[1, 2].contourf(X, Y, Z_true[3], levels=_CLEVELS, vmin=vmin3, vmax=vmax3)
#     plt.colorbar(p4, ax=ax[1, 2])



#     ax[2, 0].set_aspect('equal', adjustable='box')
#     ax[2, 0].set_title("Abs. error of XX stress", fontsize=_FONTSIZE)
#     p1 = ax[2, 0].contourf(X, Y, jnp.abs(Z[0]-Z_true[0]), levels=_CLEVELS)
#     plt.colorbar(p1, ax=ax[2, 0])

#     ax[2, 1].set_aspect('equal', adjustable='box')
#     ax[2, 1].set_title("Abs. error of XY stress", fontsize=_FONTSIZE)
#     p2 = ax[2, 1].contourf(X, Y, jnp.abs(Z[1]-Z_true[1]), levels=_CLEVELS)
#     plt.colorbar(p2, ax=ax[2, 1])

#     ax[2, 2].set_aspect('equal', adjustable='box')
#     ax[2, 2].set_title("Abs. error of YY stress", fontsize=_FONTSIZE)
#     p4 = ax[2, 2].contourf(X, Y, jnp.abs(Z[3]-Z_true[3]), levels=_CLEVELS)
#     plt.colorbar(p4, ax=ax[2, 2])



#     [plot_circle(ax[i, j], radius, circle_res, color="red") for i in range(3) for j in range(3)]
#     if save:
#         save_fig(fig_dir, name, extension)
#     if log:
#         log_fig(log_dir, name, extension, step)
#     plt.clf()
#     return
    

# def plot_polar_stress(X, Y, Z, Z_true, *, fig_dir, name, 
#                       extension="png", 
#                       figsize=(35, 30),
#                       log_dir="", 
#                       save=True, 
#                       log=False,
#                       step=None):
#     """
#     Function for plotting stresses in polar coordinates.
#     """

#     vmin0 = min(jnp.min(Z_true[0]),jnp.min(Z[0]))
#     vmin1 = min(jnp.min(Z_true[1]),jnp.min(Z[1]))
#     vmin2 = min(jnp.min(Z_true[2]),jnp.min(Z[2]))
#     vmin3 = min(jnp.min(Z_true[3]),jnp.min(Z[3]))
    
#     vmax0 = max(jnp.max(Z_true[0]), jnp.max(Z[0]))
#     vmax1 = max(jnp.max(Z_true[1]), jnp.max(Z[1]))
#     vmax2 = max(jnp.max(Z_true[2]), jnp.max(Z[2]))
#     vmax3 = max(jnp.max(Z_true[3]), jnp.max(Z[3]))
    
#     fig, ax = plt.subplots(3, 3, figsize=figsize)
#     ax[0, 0].set_aspect('equal', adjustable='box')
#     ax[0, 0].set_title("RR stress", fontsize=_FONTSIZE)
#     p1 = ax[0, 0].contourf(X , Y, Z[0], levels=_CLEVELS, vmin=vmin0, vmax=vmax0)
#     plt.colorbar(p1, ax=ax[0, 0])

#     ax[0, 1].set_aspect('equal', adjustable='box')
#     ax[0, 1].set_title("RT stress", fontsize=_FONTSIZE)
#     p2 = ax[0, 1].contourf(X, Y, Z[1], levels=_CLEVELS, vmin=vmin1, vmax=vmax1)
#     plt.colorbar(p2, ax=ax[0, 1])

#     ax[0, 2].set_aspect('equal', adjustable='box')
#     ax[0, 2].set_title("TT stress", fontsize=_FONTSIZE)
#     p4 = ax[0, 2].contourf(X, Y, Z[3], levels=_CLEVELS, vmin=vmin3, vmax=vmax3)
#     plt.colorbar(p4, ax=ax[0, 2])



#     ax[1, 0].set_aspect('equal', adjustable='box')
#     ax[1, 0].set_title("True RR stress", fontsize=_FONTSIZE)
#     p1 = ax[1, 0].contourf(X, Y, Z_true[0], levels=_CLEVELS, vmin=vmin0, vmax=vmax0)
#     plt.colorbar(p1, ax=ax[1, 0])

#     ax[1, 1].set_aspect('equal', adjustable='box')
#     ax[1, 1].set_title("True RT stress", fontsize=_FONTSIZE)
#     p2 = ax[1, 1].contourf(X, Y, Z_true[1], levels=_CLEVELS, vmin=vmin1, vmax=vmax1)
#     plt.colorbar(p2, ax=ax[1, 1])

#     ax[1, 2].set_aspect('equal', adjustable='box')
#     ax[1, 2].set_title("True TT stress", fontsize=_FONTSIZE)
#     p4 = ax[1, 2].contourf(X, Y, Z_true[3], levels=_CLEVELS, vmin=vmin3, vmax=vmax3)
#     plt.colorbar(p4, ax=ax[1, 2])



#     ax[2, 0].set_aspect('equal', adjustable='box')
#     ax[2, 0].set_title("Abs. error of RR stress", fontsize=_FONTSIZE)
#     p1 = ax[2, 0].contourf(X, Y, jnp.abs(Z[0]-Z_true[0]), levels=_CLEVELS)
#     plt.colorbar(p1, ax=ax[2, 0])

#     ax[2, 1].set_aspect('equal', adjustable='box')
#     ax[2, 1].set_title("Abs. error of RT stress", fontsize=_FONTSIZE)
#     p2 = ax[2, 1].contourf(X, Y, jnp.abs(Z[1]-Z_true[1]), levels=_CLEVELS)
#     plt.colorbar(p2, ax=ax[2, 1])

#     ax[2, 2].set_aspect('equal', adjustable='box')
#     ax[2, 2].set_title("Abs. error of TT stress", fontsize=_FONTSIZE)
#     p4 = ax[2, 2].contourf(X, Y, jnp.abs(Z[3]-Z_true[3]), levels=_CLEVELS)
#     plt.colorbar(p4, ax=ax[2, 2])

#     if fig:
#         save_fig(fig_dir, name, extension)
#     if log:
#         print("logging step " + str(step))
#         log_fig(log_dir, name, extension, step)
#     plt.clf()
#     return

def log_plot(X, Y, Z, name, log_dir, step=None, vmin=0, vmax=1, log=False):
    fig = plt.figure(figsize=(10,10), dpi=500)
    plt.contourf(X, Y, Z, vmin=vmin, vmax=vmax, levels=_CLEVELS)
    
    fig.canvas.draw()
    
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    data2 = data.reshape((int(h), int(w), -1))
    im = data2.transpose((2, 0, 1))
    
    log_img(log_dir, name, im, step=step)
    plt.close()