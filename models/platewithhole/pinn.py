from collections.abc import Callable
from typing import override

import jax
import jax.numpy as jnp

from . import analytic
from . import plotting as pwhplot

from datahandlers.generators import (
    generate_rectangle_with_hole,
    generate_collocation_points_with_hole,
    resample
)
from models import PINN
from models.networks import netmap
from utils.plotting import get_plot_variables
from utils.transforms import (
    cart2polar_tensor,
    xy2r,
    rtheta2xy,
    vrtheta2xy,
    vxy2rtheta
)

_TENSION = 10
_OUTER_RADIUS = 10

class PWHPINN(PINN):
    """
    PINN class specifically for solving the plate-with-hole problem.

    The methods in this class are common for all experiments made.
    They include:
    
        self.predict():
            The method use for external calls of the models.
        
        self.sample_points():
            The method used for sampling the points on the
            boundaries and in the domain.
        
        self.resample():
            The method used for resampling.
        
        self.plot_results():
            The method used for plotting the potential and
            the cartesian/polar stresses.
    
    The methods are NOT general methods for any PINN (see the PINN class),
    but relates to this specific problem. The inheritance is as follows:
    ```
                    Model   ________________________
                                                    |
                      |                             |
                      V                             V

                    PINN             << Other models, e.g. DeepONet >>

                      |
                      V

                   PWHPINN

                      |
                      V
                
        << Specific implementations >>
    
    ```
    """

    def __init__(self, settings: dict, *args, **kwargs):
        super().__init__(settings, *args, **kwargs)
        self.forward_input_coords = "cartesian"
        self.forward_output_coords = "cartesian"
        return
    
    def hessian(self):
        raise NotImplementedError("Method 'hessian' is not defined.")

    @override
    def predict(self,
                input: jax.Array,
                in_coords: str = "cartesian",
                out_coords: str = "cartesian"
                ) -> jax.Array:
        """
        Function for predicting without inputting parameters.
        For external use.
        """
        in_forward = self.forward_input_coords.lower()
        out_forward = self.forward_output_coords.lower()
        in_coords = in_coords.lower()
        out_coords = out_coords.lower()

        if in_coords == in_forward:
            output = self.forward(self.params, input)
        elif in_coords == "cartesian" and in_forward == "polar":
            output = self.forward(self.params, vxy2rtheta(input))
        elif in_coords == "polar" and in_forward == "cartesian":
            output = self.forward(self.params, vrtheta2xy(input))
        else:
            raise NotImplementedError("Unknown type of coordinates.")

        if out_coords == out_forward:
            return output
        elif out_coords == "cartesian" and out_forward == "polar":
            return vrtheta2xy(output)
        elif out_coords == "polar" and out_forward == "cartesian":
            return vxy2rtheta(output)
        else:
            raise NotImplementedError("Unknown type of coordinates.")

    def sample_points(self):
        """
        Method used for sampling points on boundaries and in domain.
        """

        self._key, train_key, eval_key = jax.random.split(self._key, 3)
        radius = self.geometry_settings["domain"]["circle"]["radius"]
        xlim = self.geometry_settings["domain"]["rectangle"]["xlim"]
        ylim = self.geometry_settings["domain"]["rectangle"]["ylim"]
        train_sampling = self.train_settings.sampling if self.do_train is not None else None
        eval_sampling = self.eval_settings.sampling if self.do_eval is not None else None

        # Sampling points in domain and on boundaries
        self.train_points = generate_rectangle_with_hole(train_key, radius, xlim, ylim,
                                                         train_sampling["coll"],
                                                         train_sampling["rect"],
                                                         train_sampling["circ"])
        self.eval_points = generate_rectangle_with_hole(eval_key, radius, xlim, ylim,
                                                        eval_sampling["coll"],
                                                        eval_sampling["rect"],
                                                        eval_sampling["circ"])
        
        # Get corresponding function values
        self.train_true_val = analytic.get_true_vals(self.train_points, ylim=ylim)
        self.eval_true_val = analytic.get_true_vals(self.eval_points, ylim=ylim)
        return

    def resample(self, loss_fun: Callable):
        """
        Method for resampling training points

        input:
            loss_fun(params, inputs, true_val):
                A function for calculating the loss of each point.
                This could be the square of some residual, for example.
        """
        radius = self.geometry_settings["domain"]["circle"]["radius"]
        xlim = self.geometry_settings["domain"]["rectangle"]["xlim"]
        ylim = self.geometry_settings["domain"]["rectangle"]["ylim"]
        self._key, resample_key = jax.random.split(self._key)
        
        # Get resample parameters
        loss_emphasis = self.train_settings.resampling["loss_emphasis"]
        resample_num = self.train_settings.resampling["resample_num"]
        if isinstance(resample_num, int):
            resample_num = [resample_num]
        
        coll_num = sum(self.train_settings.sampling["coll"])

        print(f"Resampling {sum(resample_num)} out of {coll_num} points with loss emphasis = {loss_emphasis} ...")

        resample_points = [int(loss_emphasis*r) for r in resample_num]

        # Generate points and choose the ones with highest loss
        new_coll = generate_collocation_points_with_hole(resample_key, radius, xlim, ylim, resample_points)

        # Get true values of new collocation points
        new_true = analytic.get_true_vals(self.train_points, exclude=["rect", "circ", "diri"])["coll"]
        
        # Calculate values
        new_loss = loss_fun(self.params, new_coll, true_val=new_true)
        
        # Choose subset of sampled points to keep
        new_coll = resample(new_coll, new_loss, sum(resample_num))

        # Set new training points
        self.train_points["coll"].at[:sum(resample_num)].set(new_coll)

        # Recalculate true values for collocation points
        self.train_true_val["coll"] = analytic.get_true_vals(self.train_points, exclude=["rect", "circ", "diri"])["coll"]
        return
    
    def plot_results(self):
        
        radius = self.geometry_settings["domain"]["circle"]["radius"]
        xlim = self.geometry_settings["domain"]["rectangle"]["xlim"]
        ylim = self.geometry_settings["domain"]["rectangle"]["ylim"]

        X, Y, plotpoints = get_plot_variables(xlim, ylim, grid=101)
        R, THETA, plotpoints_polar = get_plot_variables([radius, _OUTER_RADIUS], [0, 4*jnp.pi], grid=101)
        plotpoints2 = jax.vmap(rtheta2xy)(plotpoints_polar)
        
        assert(jnp.allclose(plotpoints, vrtheta2xy(vxy2rtheta(plotpoints)), atol=1e-4))

        phi = self.predict(plotpoints).reshape(X.shape)*(xy2r(X, Y) >= radius)

        # Hessian prediction
        phi_pp = netmap(self.hessian)(self.params, plotpoints).reshape(-1, 4)
        
        # Calculate stress from phi function: phi_xx = sigma_yy, phi_yy = sigma_xx, phi_xy = -sigma_xy
        sigma_cart = phi_pp[:, [3, 1, 2, 0]]
        sigma_cart = sigma_cart.at[:, [1, 2]].set(-phi_pp[:, [1, 2]])

        # List and reshape the four components
        sigma_cart_list = [sigma_cart[:, i].reshape(X.shape)*(xy2r(X, Y) >= radius) for i in range(4)]

        # Repeat for the other set of points (polar coords converted to cartesian coords)
        phi_pp2 = netmap(self.hessian)(self.params, plotpoints2).reshape(-1, 4)

        # Calculate stress from phi function
        sigma_cart2 = phi_pp2[:, [3, 1, 2, 0]]
        sigma_cart2 = sigma_cart2.at[:, [1, 2]].set(-phi_pp2[:, [1, 2]])
        
        # Convert these points to polar coordinates before listing and reshaping
        sigma_polar = jax.vmap(cart2polar_tensor, in_axes=(0, 0))(sigma_cart2.reshape(-1, 2, 2), plotpoints2).reshape(-1, 4)
        sigma_polar_list = [sigma_polar[:, i].reshape(R.shape)*(R >= radius) for i in range(4)]

        # Calculate true stresses (cartesian and polar)
        sigma_cart_true = jax.vmap(analytic.cart_stress_true)(plotpoints)
        sigma_cart_true_list = [sigma_cart_true.reshape(-1, 4)[:, i].reshape(X.shape)*(xy2r(X, Y) >= radius) for i in range(4)]
        sigma_polar_true = jax.vmap(analytic.polar_stress_true)(plotpoints_polar)
        sigma_polar_true_list = [sigma_polar_true.reshape(-1, 4)[:, i].reshape(R.shape)*(R >= radius) for i in range(4)]

        pwhplot.plot_potential(X, Y, phi,
                               fig_dir=self.dir.figure_dir, name="potential", radius=radius)
        pwhplot.plot_stress(X, Y, sigma_cart_list, sigma_cart_true_list,
                            fig_dir=self.dir.figure_dir, name="stress", radius=radius)
        pwhplot.plot_polar_stress(R, THETA, sigma_polar_list, sigma_polar_true_list,
                                  fig_dir=self.dir.figure_dir, name="stress_polar")
        return