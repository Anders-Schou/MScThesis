import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np

from . import analytic
from models.networks import netmap
from utils.plotting import (
    save_fig,
    plot_circle,
    log_plot,
    get_plot_variables
)
from utils.transforms import (
    cart2polar_tensor,
    xy2r,
    rtheta2xy,
    vrtheta2xy,
    vxy2rtheta
)



_DEFAULT_RADIUS = 2
_DEFAULT_CIRCLE_RES = 100
_CLEVELS = 101
_FONTSIZE = 40



def plot_loss(
    loss_arr: jax.Array,
    loss_map: dict,
    *,
    fig_dir,
    name,
    epoch_step = None,
    extension="png",
    figsize = (35, 30)
) -> None:
    """
    Plots losses from array in different subplots according to the specified dict.
    """
    
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
        
    save_fig(fig_dir, name, extension, fig=fig)
    plt.clf()
    return

def get_plot_data(geometry_settings, hessian, params, grid, mesh_data=None, **kwargs):
    if mesh_data is not None:
        radius = geometry_settings["domain"]["circle"]["radius"]
        
        X = mesh_data["X"]
        Y = mesh_data["Y"]
        R = mesh_data["R"]
        THETA = mesh_data["THETA"]
        plotpoints = mesh_data["plotpoints"]
        plotpoints2 = mesh_data["plotpoints2"]
        
        sigma_cart_true_list = mesh_data["sigma_cart_true_list"]
        sigma_polar_true_list = mesh_data["sigma_polar_true_list"]
        
        assert(jnp.allclose(plotpoints, vrtheta2xy(vxy2rtheta(plotpoints)), atol=1e-4))

        # Hessian prediction
        phi_pp = netmap(hessian)(params, plotpoints).reshape(-1, 4)

        # Calculate stress from phi function: phi_xx = sigma_yy, phi_yy = sigma_xx, phi_xy = -sigma_xy
        sigma_cart = phi_pp[:, [3, 1, 2, 0]]
        sigma_cart = sigma_cart.at[:, [1, 2]].set(-phi_pp[:, [1, 2]])

        # List and reshape the four components
        sigma_cart_list = [sigma_cart[:, i].reshape(X.shape)*(xy2r(X, Y) >= radius) for i in range(4)]

        # Repeat for the other set of points (polar coords converted to cartesian coords)
        phi_pp2 = netmap(hessian)(params, plotpoints2).reshape(-1, 4)

        # Calculate stress from phi function
        sigma_cart2 = phi_pp2[:, [3, 1, 2, 0]]
        sigma_cart2 = sigma_cart2.at[:, [1, 2]].set(-phi_pp2[:, [1, 2]])

        # Convert these points to polar coordinates before listing and reshaping
        sigma_polar = jax.vmap(cart2polar_tensor, in_axes=(0, 0))(sigma_cart2.reshape(-1, 2, 2), plotpoints2).reshape(-1, 4)
        sigma_polar_list = [sigma_polar[:, i].reshape(R.shape)*(R >= radius) for i in range(4)]
        
        return X, Y, R, THETA, sigma_cart_list, sigma_cart_true_list, sigma_polar_list, sigma_polar_true_list

    else:
        radius = geometry_settings["domain"]["circle"]["radius"]
        xlim = geometry_settings["domain"]["rectangle"]["xlim"]
        ylim = geometry_settings["domain"]["rectangle"]["ylim"]

        X, Y, plotpoints = get_plot_variables(xlim, ylim, grid=grid)
        R, THETA, plotpoints_polar = get_plot_variables([radius, max(xlim[1], ylim[1])], [0, 4*np.pi], grid=grid)
        plotpoints2 = jax.vmap(rtheta2xy)(plotpoints_polar)

        assert(jnp.allclose(plotpoints, vrtheta2xy(vxy2rtheta(plotpoints)), atol=1e-4))

        # Hessian prediction
        phi_pp = netmap(hessian)(params, plotpoints).reshape(-1, 4)

        # Calculate stress from phi function: phi_xx = sigma_yy, phi_yy = sigma_xx, phi_xy = -sigma_xy
        sigma_cart = phi_pp[:, [3, 1, 2, 0]]
        sigma_cart = sigma_cart.at[:, [1, 2]].set(-phi_pp[:, [1, 2]])

        # List and reshape the four components
        sigma_cart_list = [sigma_cart[:, i].reshape(X.shape)*(xy2r(X, Y) >= radius) for i in range(4)]

        # Repeat for the other set of points (polar coords converted to cartesian coords)
        phi_pp2 = netmap(hessian)(params, plotpoints2).reshape(-1, 4)

        # Calculate stress from phi function
        sigma_cart2 = phi_pp2[:, [3, 1, 2, 0]]
        sigma_cart2 = sigma_cart2.at[:, [1, 2]].set(-phi_pp2[:, [1, 2]])

        # Convert these points to polar coordinates before listing and reshaping
        sigma_polar = jax.vmap(cart2polar_tensor, in_axes=(0, 0))(sigma_cart2.reshape(-1, 2, 2), plotpoints2).reshape(-1, 4)
        sigma_polar_list = [sigma_polar[:, i].reshape(R.shape)*(R >= radius) for i in range(4)]

        # Calculate true stresses (cartesian and polar)
        sigma_cart_true = jax.vmap(analytic.cart_stress_true)(plotpoints, **kwargs)
        sigma_cart_true_list = [sigma_cart_true.reshape(-1, 4)[:, i].reshape(X.shape)*(xy2r(X, Y) >= radius) for i in range(4)]
        sigma_polar_true = jax.vmap(analytic.polar_stress_true)(plotpoints_polar, **kwargs)
        sigma_polar_true_list = [sigma_polar_true.reshape(-1, 4)[:, i].reshape(R.shape)*(R >= radius) for i in range(4)]

        return X, Y, R, THETA, sigma_cart_list, sigma_cart_true_list, sigma_polar_list, sigma_polar_true_list, plotpoints, plotpoints2


def plot_results(geometry_settings, hessian, params, fig_dir, log_dir, save=True, log=False, step=None, grid=201, dpi=50, mesh_data=None, **kwargs):

    X, Y, R, THETA, sigma_cart_list, sigma_cart_true_list, sigma_polar_list, sigma_polar_true_list = get_plot_data(geometry_settings, hessian, params, grid=grid, mesh_data=mesh_data, **kwargs)
    radius = geometry_settings["domain"]["circle"]["radius"]

    if save:
        plot_stress(X, Y, sigma_cart_list, sigma_cart_true_list, fig_dir=fig_dir, name="Cart_stress", radius=radius)
        plot_polar_stress(R, THETA, sigma_polar_list, sigma_polar_true_list, fig_dir=fig_dir, name="Polar_stress")
    if log:        
        log_stress(X, Y, sigma_cart_list, sigma_cart_true_list, log_dir=log_dir, name="Cart_stress", varnames="XY", step=step, dpi=dpi)
        log_stress(R, THETA, sigma_polar_list, sigma_polar_true_list, log_dir=log_dir, name="Polar_stress", varnames="RT", step=step, dpi=dpi)

    return

def plot_potential(X, Y, Z, *, fig_dir, name,
                   extension="png",
                   radius = _DEFAULT_RADIUS,
                   circle_res = _DEFAULT_CIRCLE_RES):
    """
    Function for plotting potential function.
    """    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Prediction", fontsize=20)
    p = ax.contourf(X, Y, Z, levels=_CLEVELS)
    plt.colorbar(p, ax=ax)
    
    plot_circle(plt, radius, circle_res, color="red")
    save_fig(fig_dir, name, extension)
    plt.clf()
    return


def plot_stress(X, Y, Z, Z_true, *, fig_dir, name,
                extension="png",
                radius = _DEFAULT_RADIUS,
                circle_res = _DEFAULT_CIRCLE_RES,
                figsize = (35, 30)):
    """
    Function for plotting stresses in cartesian coordinates.
    """
    
    vmin0 = min(jnp.min(Z_true[0]),jnp.min(Z[0]))
    vmin1 = min(jnp.min(Z_true[1]),jnp.min(Z[1]))
    vmin3 = min(jnp.min(Z_true[3]),jnp.min(Z[3]))
    
    vmax0 = max(jnp.max(Z_true[0]), jnp.max(Z[0]))
    vmax1 = max(jnp.max(Z_true[1]), jnp.max(Z[1]))
    vmax3 = max(jnp.max(Z_true[3]), jnp.max(Z[3]))
    

    fig, ax = plt.subplots(3, 3, figsize=figsize)
    ax[0, 0].set_aspect('equal', adjustable='box')
    ax[0, 0].set_title("XX stress", fontsize=_FONTSIZE)
    p1 = ax[0, 0].contourf(X , Y, Z[0], levels=_CLEVELS, vmin=vmin0, vmax=vmax0)
    plt.colorbar(p1, ax=ax[0, 0])

    ax[0, 1].set_aspect('equal', adjustable='box')
    ax[0, 1].set_title("XY stress", fontsize=_FONTSIZE)
    p2 = ax[0, 1].contourf(X, Y, Z[1], levels=_CLEVELS, vmin=vmin1, vmax=vmax1)
    plt.colorbar(p2, ax=ax[0, 1])

    ax[0, 2].set_aspect('equal', adjustable='box')
    ax[0, 2].set_title("YY stress", fontsize=_FONTSIZE)
    p4 = ax[0, 2].contourf(X, Y, Z[3], levels=_CLEVELS, vmin=vmin3, vmax=vmax3)
    plt.colorbar(p4, ax=ax[0, 2])



    ax[1, 0].set_aspect('equal', adjustable='box')
    ax[1, 0].set_title("True XX stress", fontsize=_FONTSIZE)
    p1 = ax[1, 0].contourf(X, Y, Z_true[0], levels=_CLEVELS, vmin=vmin0, vmax=vmax0)
    plt.colorbar(p1, ax=ax[1, 0])

    ax[1, 1].set_aspect('equal', adjustable='box')
    ax[1, 1].set_title("True XY stress", fontsize=_FONTSIZE)
    p2 = ax[1, 1].contourf(X, Y, Z_true[1], levels=_CLEVELS, vmin=vmin1, vmax=vmax1)
    plt.colorbar(p2, ax=ax[1, 1])

    ax[1, 2].set_aspect('equal', adjustable='box')
    ax[1, 2].set_title("True YY stress", fontsize=_FONTSIZE)
    p4 = ax[1, 2].contourf(X, Y, Z_true[3], levels=_CLEVELS, vmin=vmin3, vmax=vmax3)
    plt.colorbar(p4, ax=ax[1, 2])



    ax[2, 0].set_aspect('equal', adjustable='box')
    ax[2, 0].set_title("Abs. error of XX stress", fontsize=_FONTSIZE)
    p1 = ax[2, 0].contourf(X, Y, jnp.abs(Z[0]-Z_true[0]), levels=_CLEVELS)
    plt.colorbar(p1, ax=ax[2, 0])

    ax[2, 1].set_aspect('equal', adjustable='box')
    ax[2, 1].set_title("Abs. error of XY stress", fontsize=_FONTSIZE)
    p2 = ax[2, 1].contourf(X, Y, jnp.abs(Z[1]-Z_true[1]), levels=_CLEVELS)
    plt.colorbar(p2, ax=ax[2, 1])

    ax[2, 2].set_aspect('equal', adjustable='box')
    ax[2, 2].set_title("Abs. error of YY stress", fontsize=_FONTSIZE)
    p4 = ax[2, 2].contourf(X, Y, jnp.abs(Z[3]-Z_true[3]), levels=_CLEVELS)
    plt.colorbar(p4, ax=ax[2, 2])



    [plot_circle(ax[i, j], radius, circle_res, color="red") for i in range(3) for j in range(3)]
    save_fig(fig_dir, name, extension)
    plt.clf()
    return
    

def plot_polar_stress(X, Y, Z, Z_true, *, fig_dir, name, 
                      extension="png", 
                      figsize=(35, 30)):
    """
    Function for plotting stresses in polar coordinates.
    """

    vmin0 = min(jnp.min(Z_true[0]),jnp.min(Z[0]))
    vmin1 = min(jnp.min(Z_true[1]),jnp.min(Z[1]))
    vmin3 = min(jnp.min(Z_true[3]),jnp.min(Z[3]))
    
    vmax0 = max(jnp.max(Z_true[0]), jnp.max(Z[0]))
    vmax1 = max(jnp.max(Z_true[1]), jnp.max(Z[1]))
    vmax3 = max(jnp.max(Z_true[3]), jnp.max(Z[3]))
    
    fig, ax = plt.subplots(3, 3, figsize=figsize)
    ax[0, 0].set_aspect('equal', adjustable='box')
    ax[0, 0].set_title("RR stress", fontsize=_FONTSIZE)
    p1 = ax[0, 0].contourf(X , Y, Z[0], levels=_CLEVELS, vmin=vmin0, vmax=vmax0)
    plt.colorbar(p1, ax=ax[0, 0])

    ax[0, 1].set_aspect('equal', adjustable='box')
    ax[0, 1].set_title("RT stress", fontsize=_FONTSIZE)
    p2 = ax[0, 1].contourf(X, Y, Z[1], levels=_CLEVELS, vmin=vmin1, vmax=vmax1)
    plt.colorbar(p2, ax=ax[0, 1])

    ax[0, 2].set_aspect('equal', adjustable='box')
    ax[0, 2].set_title("TT stress", fontsize=_FONTSIZE)
    p4 = ax[0, 2].contourf(X, Y, Z[3], levels=_CLEVELS, vmin=vmin3, vmax=vmax3)
    plt.colorbar(p4, ax=ax[0, 2])



    ax[1, 0].set_aspect('equal', adjustable='box')
    ax[1, 0].set_title("True RR stress", fontsize=_FONTSIZE)
    p1 = ax[1, 0].contourf(X, Y, Z_true[0], levels=_CLEVELS, vmin=vmin0, vmax=vmax0)
    plt.colorbar(p1, ax=ax[1, 0])

    ax[1, 1].set_aspect('equal', adjustable='box')
    ax[1, 1].set_title("True RT stress", fontsize=_FONTSIZE)
    p2 = ax[1, 1].contourf(X, Y, Z_true[1], levels=_CLEVELS, vmin=vmin1, vmax=vmax1)
    plt.colorbar(p2, ax=ax[1, 1])

    ax[1, 2].set_aspect('equal', adjustable='box')
    ax[1, 2].set_title("True TT stress", fontsize=_FONTSIZE)
    p4 = ax[1, 2].contourf(X, Y, Z_true[3], levels=_CLEVELS, vmin=vmin3, vmax=vmax3)
    plt.colorbar(p4, ax=ax[1, 2])



    ax[2, 0].set_aspect('equal', adjustable='box')
    ax[2, 0].set_title("Abs. error of RR stress", fontsize=_FONTSIZE)
    p1 = ax[2, 0].contourf(X, Y, jnp.abs(Z[0]-Z_true[0]), levels=_CLEVELS)
    plt.colorbar(p1, ax=ax[2, 0])

    ax[2, 1].set_aspect('equal', adjustable='box')
    ax[2, 1].set_title("Abs. error of RT stress", fontsize=_FONTSIZE)
    p2 = ax[2, 1].contourf(X, Y, jnp.abs(Z[1]-Z_true[1]), levels=_CLEVELS)
    plt.colorbar(p2, ax=ax[2, 1])

    ax[2, 2].set_aspect('equal', adjustable='box')
    ax[2, 2].set_title("Abs. error of TT stress", fontsize=_FONTSIZE)
    p4 = ax[2, 2].contourf(X, Y, jnp.abs(Z[3]-Z_true[3]), levels=_CLEVELS)
    plt.colorbar(p4, ax=ax[2, 2])

    save_fig(fig_dir, name, extension)

    plt.clf()
    return


def log_stress(X, Y, Z, Z_true, *, log_dir, name, step=None, varnames="XY", dpi=50):
        
    # Log plots
    log_plot(X, Y, Z[0], name=name+"/Surrogate/"+varnames[0]+varnames[0], log_dir=log_dir, step=step,
            vmin=min(jnp.min(Z_true[0]),jnp.min(Z[0])), 
            vmax=max(jnp.max(Z_true[0]),jnp.max(Z[0])), dpi=dpi)
    
    log_plot(X, Y, Z[1], name=name+"/Surrogate/"+varnames[0]+varnames[1], log_dir=log_dir, step=step,
            vmin=min(jnp.min(Z_true[1]),jnp.min(Z[1])), 
            vmax=max(jnp.max(Z_true[1]),jnp.max(Z[1])), dpi=dpi)
            
    log_plot(X, Y, Z[3], name=name+"/Surrogate/"+varnames[1]+varnames[1], log_dir=log_dir, step=step,
            vmin=min(jnp.min(Z_true[3]),jnp.min(Z[3])), 
            vmax=max(jnp.max(Z_true[3]),jnp.max(Z[3])), dpi=dpi)
    
    
    log_plot(X, Y, jnp.abs(Z_true[0] - Z[0]), name=name+"/Error/"+varnames[0]+varnames[0], log_dir=log_dir, step=step, logscale=True, dpi=dpi)
    
    log_plot(X, Y, jnp.abs(Z_true[1] - Z[1]), name=name+"/Error/"+varnames[0]+varnames[1], log_dir=log_dir, step=step, logscale=True, dpi=dpi)

    log_plot(X, Y, jnp.abs(Z_true[3] - Z[3]), name=name+"/Error/"+varnames[1]+varnames[1], log_dir=log_dir, step=step, logscale=True, dpi=dpi)
            
    
        # These are redundant after first time being logged
    if step == 0:
        log_plot(X, Y, Z_true[0], name=name+"/True/"+varnames[0]+varnames[0], log_dir=log_dir, step=step,
                vmin=min(jnp.min(Z_true[0]),jnp.min(Z[0])), 
                vmax=max(jnp.max(Z_true[0]),jnp.max(Z[0])), dpi=dpi)
        
        log_plot(X, Y, Z_true[1], name=name+"/True/"+varnames[0]+varnames[1], log_dir=log_dir, step=step,
                vmin=min(jnp.min(Z_true[1]),jnp.min(Z[1])), 
                vmax=max(jnp.max(Z_true[1]),jnp.max(Z[1])), dpi=dpi)
                
        log_plot(X, Y, Z_true[3], name=name+"/True/"+varnames[1]+varnames[1], log_dir=log_dir, step=step,
                vmin=min(jnp.min(Z_true[3]),jnp.min(Z[3])), 
                vmax=max(jnp.max(Z_true[3]),jnp.max(Z[3])), dpi=dpi)
            
            
            
            
def plot_boundaries(geometry_settings, hessian, params, fig_dir, log_dir, save=True, log=False, step=None, grid=201, dpi=50, **kwargs):
        xlim = geometry_settings["domain"]["rectangle"]["xlim"]
        ylim = geometry_settings["domain"]["rectangle"]["ylim"]

        b0x = jnp.linspace(xlim[0], xlim[1], grid).reshape(-1,1)
        b0y = jnp.full_like(b0x, ylim[0])
        b0 = jnp.hstack((b0x, b0y))
        
        b1y = jnp.linspace(ylim[0], ylim[1], grid).reshape(-1,1)
        b1x = jnp.full_like(b1y, xlim[1])
        b1 = jnp.hstack((b1x, b1y))
        
        b2x = jnp.linspace(xlim[1], xlim[0], grid).reshape(-1,1)
        b2y = jnp.full_like(b2x, ylim[1])
        b2 = jnp.hstack((b2x, b2y))

        b3y = jnp.linspace(ylim[1], ylim[0], grid).reshape(-1,1)
        b3x = jnp.full_like(b3y, xlim[0])
        b3 = jnp.hstack((b3x, b3y))

        x = jnp.concatenate((b0x, b1x, b2x, b3x))
        y = jnp.concatenate((b0y, b1y, b2y, b3y))
        xy = jnp.concatenate((x, y),axis=1)
        u = netmap(hessian)(params, jnp.concatenate((b0, b1, b2, b3))).reshape(-1, 4)
        
        u_true = jax.vmap(analytic.cart_stress_true)(xy, **kwargs).reshape(-1, 4)
        
        fig, ax = plt.subplots(2, 3, figsize=(30, 20))
        ax[0, 0].set_title("XX stress", fontsize=_FONTSIZE)
        ax[0, 0].plot(u_true[:, 0], linewidth=2)
        ax[0, 0].plot(u[:, 0], linewidth=2)
        ax[0, 0].vlines([grid, 2*grid, 3*grid], ymin = min(min(u_true[:, 0]), min(u[:, 0])), ymax = max(max(u_true[:, 0]), max(u[:, 0])), colors='black')
        
        ax[0, 1].set_title("XY stress", fontsize=_FONTSIZE)
        ax[0, 1].plot(u_true[:, 1], linewidth=2)
        ax[0, 1].plot(u[:, 1], linewidth=2)
        ax[0, 1].vlines([grid, 2*grid, 3*grid], ymin = min(min(u_true[:, 1]), min(u[:, 1])), ymax = max(max(u_true[:, 1]), max(u[:, 1])), colors='black')
                
        ax[0, 2].set_title("YY stress", fontsize=_FONTSIZE)
        ax[0, 2].plot(u_true[:, 3], linewidth=2)
        ax[0, 2].plot(u[:, 3], linewidth=2)
        ax[0, 2].vlines([grid, 2*grid, 3*grid], ymin = min(min(u_true[:, 3]), min(u[:, 3])), ymax = max(max(u_true[:, 3]), max(u[:, 3])), colors='black')
        
        ax[1, 0].set_title("XX error", fontsize=_FONTSIZE)
        ax[1, 0].semilogy(jnp.abs(u_true[:, 0] - u[:, 0]), linewidth=2)
        ax[1, 0].vlines([grid, 2*grid, 3*grid], ymin = 0, ymax = max(jnp.abs(u_true[:, 0] - u[:, 0])), colors='black')
         
        ax[1, 1].set_title("XY error", fontsize=_FONTSIZE)
        ax[1, 1].semilogy(jnp.abs(u_true[:, 1] - u[:, 1]), linewidth=2)
        ax[1, 1].vlines([grid, 2*grid, 3*grid], ymin = 0, ymax = max(jnp.abs(u_true[:, 1] - u[:, 1])), colors='black')
        
        ax[1, 2].set_title("YY error", fontsize=_FONTSIZE)
        ax[1, 2].semilogy(jnp.abs(u_true[:, 3] - u[:, 3]), linewidth=2)
        ax[1, 2].vlines([grid, 2*grid, 3*grid], ymin = 0, ymax = max(jnp.abs(u_true[:, 3] - u[:, 3])), colors='black')
        
        for i in ax.ravel():
            for item in ([i.xaxis.label, i.yaxis.label] + i.get_xticklabels() + i.get_yticklabels()):
                item.set_fontsize(20)
        
        save_fig(fig_dir, "boundaries", "png")
