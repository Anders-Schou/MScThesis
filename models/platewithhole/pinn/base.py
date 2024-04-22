from collections.abc import Callable
from typing import override
from functools import partial

import jax

from datahandlers.generators import (
    generate_rectangle_with_hole,
    generate_collocation_points_with_hole,
    resample,
    resample_idx
)
from models import PINN
from models.derivatives import hessian
from models.networks import netmap
from models.platewithhole import analytic
from models.platewithhole import loss as pwhloss
from models.platewithhole import plotting as pwhplot
from utils.transforms import (
    vrtheta2xy,
    vxy2rtheta
)
from utils.utils import timer

class PlateWithHolePINN(PINN):
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

               PlateWithHolePINN

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
    
    def hessian(self, params, input: jax.Array) -> jax.Array:
        return hessian(self.forward)(params, input)
    
    @partial(jax.jit, static_argnums=(0,))
    def jitted_hessian(self, params, input: jax.Array) -> jax.Array:
        return self.hessian(params, input)

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

    def loss_rect(self, params, input: tuple[jax.Array], true_val: dict[str, jax.Array] | None = None):
        """
        Computes the loss of the BC residuals on the four sides of the rectangle.
        """
        """
        Layout of rectangle sides:

                            ^
                          y |
                            |
                            *------>
                                 x
                                
                                        2
                                _________________
               xx stress  <-   |                 |   -> xx stress
                               |                 |
                            3  |        O        |  1
                               |                 |
               xx stress  <-   |_________________|   -> xx stress
                                
                                        0
        """

        # Compute Hessian values for each of the four sides
        out0 = netmap(self.hessian)(params, input[0]).reshape(-1, 4) # horizontal lower
        out1 = netmap(self.hessian)(params, input[1]).reshape(-1, 4) # vertical right
        out2 = netmap(self.hessian)(params, input[2]).reshape(-1, 4) # horizontal upper
        out3 = netmap(self.hessian)(params, input[3]).reshape(-1, 4) # vertical left

        # Compute losses for all four sides of rectangle
        losses  = pwhloss.loss_rect(out0, out1, out2, out3, true_val=true_val)
        return losses

    def loss_circ(self, params, input: jax.Array, true_val: dict[str, jax.Array] | None = None):
        """
        Computes the loss of the residuals on the circle.
        """

        # Compute cartesian output
        output = netmap(self.hessian)(params, input).reshape(-1, 4)

        # Compute polar stresses and return loss
        losses = pwhloss.loss_circ_rr_rt(input, output, true_val=true_val)
        return losses

    def loss_diri(self, params, input: tuple[jax.Array], true_val: dict[str, jax.Array] | None = None):

        # Compute potential output
        out0 = netmap(self.forward)(params, input[0])
        out1 = netmap(self.forward)(params, input[1])
        out2 = netmap(self.forward)(params, input[2])
        out3 = netmap(self.forward)(params, input[3])

        # Compute losses for all four sides of rectangle
        losses  = pwhloss.loss_dirichlet(out0, out1, out2, out3, true_val=true_val)
        return losses
    
    def loss_data(self, params, input: jax.Array, true_val: dict[str, jax.Array] | None = None):
        
        # Compute model output
        out = netmap(self.hessian)(params, input).reshape(-1, 4)

        # Compute losses
        losses = pwhloss.loss_data(out, true_val=true_val)
        return losses

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
        
        # Generate data points
        self._key, data_train_key, data_eval_key = jax.random.split(self._key, 3)
        self.train_points["data"] = generate_collocation_points_with_hole(data_train_key, radius,
                                                                          xlim, ylim, train_sampling.get("data"))
        self.eval_points["data"] = generate_collocation_points_with_hole(data_eval_key, radius,
                                                                         xlim, ylim, eval_sampling.get("data"))

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
        self._key, resample_key, perm_key = jax.random.split(self._key, 3)
        
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
        new_true = analytic.get_true_vals({"coll": new_coll}, exclude=["rect", "circ", "diri", "data"])["coll"]
        
        # Calculate values
        new_loss = loss_fun(self.params, new_coll, true_val=new_true)
        
        # Choose subset of sampled points to keep
        new_coll = resample(new_coll, new_loss, sum(resample_num))

        # Find indices to swap out new training points with
        old_coll = self.train_points["coll"]
        old_true = self.train_true_val["coll"]
        old_loss = loss_fun(self.params, old_coll, true_val=old_true)
        replace_idx = resample_idx(old_coll, old_loss, sum(resample_num))

        # Set new training points
        self.train_points["coll"] = self.train_points["coll"].at[replace_idx].set(new_coll)
        
        # Recalculate true values for collocation points
        self.train_true_val["coll"] = analytic.get_true_vals(self.train_points, exclude=["rect", "circ", "diri", "data"])["coll"]
        return
    
    @timer
    def plot_results(self, save=True, log=False, step=None):
        if not hasattr(self, 'mesh_data'):
            values = pwhplot.get_plot_data(self.geometry_settings, 
                                           self.hessian, self.params, 
                                           grid=self.plot_settings["grid"])
            keys = ["X", "Y", "R", "THETA", "sigma_cart_list", "sigma_cart_true_list", "sigma_polar_list", "sigma_polar_true_list", "plotpoints", "plotpoints2"]
            self.mesh_data = {key: val for key, val in zip(keys, values)}
            
        pwhplot.plot_results(self.geometry_settings, self.jitted_hessian, self.params, 
                             self.dir.figure_dir, self.dir.log_dir, save=save, log=log, step=step, 
                             grid=self.plot_settings["grid"], dpi=self.plot_settings["dpi"], mesh_data=self.mesh_data)
