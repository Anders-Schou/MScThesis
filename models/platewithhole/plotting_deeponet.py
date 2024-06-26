import jax
import jax.numpy as jnp

from . import analytic
from models.networks import deeponetmap
from utils.plotting import get_plot_variables
from models.platewithhole.plotting import plot_stress, plot_polar_stress, log_stress, plot_vm_stress
from utils.transforms import (
    cart2polar_tensor,
    xy2r,
    rtheta2xy,
    vrtheta2xy,
    vxy2rtheta
)

def get_plot_data(geometry_settings, hessian, params, branch_point, grid, **kwargs):
    # Does not work for DeepONet with (radius, tension) as branch - then it should be radius = branch_point[0]
    radius = geometry_settings["domain"]["circle"]["radius"]
    xlim = geometry_settings["domain"]["rectangle"]["xlim"]
    ylim = geometry_settings["domain"]["rectangle"]["ylim"]
    angle = geometry_settings["domain"]["circle"].get("angle")
    if angle is None:
        angle = [0, 2*jnp.pi]

    X, Y, plotpoints = get_plot_variables(xlim, ylim, grid=grid)
    R, THETA, plotpoints_polar = get_plot_variables([radius, max(xlim[1], ylim[1])], angle, grid=grid)
    plotpoints2 = jax.vmap(rtheta2xy)(plotpoints_polar)

    assert(jnp.allclose(plotpoints, vrtheta2xy(vxy2rtheta(plotpoints)), atol=1e-4))

    # Hessian prediction
    phi_pp = deeponetmap(hessian)(params, branch_point, plotpoints).reshape(-1, 4)

    # Calculate stress from phi function: phi_xx = sigma_yy, phi_yy = sigma_xx, phi_xy = -sigma_xy
    sigma_cart = phi_pp[:, [3, 1, 2, 0]]
    sigma_cart = sigma_cart.at[:, [1, 2]].set(-phi_pp[:, [1, 2]])

    # List and reshape the four components
    sigma_cart_list = [sigma_cart[:, i].reshape(X.shape)*(xy2r(X, Y) >= radius) for i in range(4)]

    # Repeat for the other set of points (polar coords converted to cartesian coords)
    phi_pp2 = deeponetmap(hessian)(params, branch_point, plotpoints2).reshape(-1, 4)

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

    return X, Y, R, THETA, sigma_cart_list, sigma_cart_true_list, sigma_polar_list, sigma_polar_true_list


def plot_results(geometry_settings, hessian, params, branch_point, fig_dir, log_dir, save=True, log=False, step=None, grid=201, dpi=50, **kwargs):

    X, Y, R, THETA, sigma_cart_list, sigma_cart_true_list, sigma_polar_list, sigma_polar_true_list = get_plot_data(geometry_settings, hessian, params, branch_point, grid=grid, **kwargs)
    # Does not work for DeepONet with (radius, tension) as branch - then it should be radius = branch_point[0]
    radius = geometry_settings["domain"]["circle"]["radius"]
    angle = geometry_settings["domain"]["circle"].get("angle")

    if save:
        plot_stress(X, Y, sigma_cart_list, sigma_cart_true_list, fig_dir=fig_dir, name="Cart_stress", radius=radius, angle=angle)
        plot_vm_stress(X, Y, sigma_cart_list, sigma_cart_true_list, fig_dir=fig_dir, name="VM_stress", radius=radius, angle=angle)
        plot_polar_stress(R, THETA, sigma_polar_list, sigma_polar_true_list, fig_dir=fig_dir, name="Polar_stress")
    if log:        
        log_stress(X, Y, sigma_cart_list, sigma_cart_true_list, log_dir=log_dir, name="Cart_stress", varnames="XY", step=step, dpi=dpi)
        log_stress(R, THETA, sigma_polar_list, sigma_polar_true_list, log_dir=log_dir, name="Polar_stress", varnames="RT", step=step, dpi=dpi)

    return