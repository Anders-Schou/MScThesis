from collections.abc import Callable
from typing import override
from functools import partial

import jax
import jax.numpy as jnp

from datahandlers.generators import (
    generate_collocation_points,
    generate_rectangle_with_hole,
    generate_collocation_points_with_hole,
    resample,
    resample_idx
)
from models import DeepONet
from models.derivatives import hessian
from models.networks import deeponetmap
from models.platewithhole import analytic
from models.platewithhole import loss as pwhloss
from models.platewithhole import plotting_deeponet as pwhplot
from models.loss import L2rel
from utils.transforms import (
    vrtheta2xy,
    vxy2rtheta
)
from utils.utils import timer

class PlateWithHoleDeepONet(DeepONet):
    """
    DeepONet class specifically for solving the plate-with-hole problem.

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
    
    The methods are NOT general methods for any DeepONet (see the DeepONet class),
    but relates to this specific problem. The inheritance is as follows:
    ```
                    Model   ________________________
                                                    |
                      |                             |
                      V                             V

                    DeepONet

                      |
                      V

               PlateWithHoleDeepONet

                      |
                      V
                
        << Specific implementations >>
    
    ```
    """

    def __init__(self, settings: dict, *args, **kwargs):
        super().__init__(settings, *args, **kwargs)
        self._register_static_loss_arg("update_key")
        self.forward_input_coords = "cartesian"
        self.forward_output_coords = "cartesian"
        return
    
    def hessian(self, params, branch_input: jax.Array, trunk_input: jax.Array) -> jax.Array:
        return hessian(self.forward, argnums=2)(params, branch_input, trunk_input)
    
    @partial(jax.jit, static_argnums=(0,))
    def jitted_hessian(self, params, branch_input: jax.Array, trunk_input: jax.Array) -> jax.Array:
        return self.hessian(params, branch_input, trunk_input)

    @override
    def predict(self,
                branch_input: jax.Array, 
                trunk_input: jax.Array,
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
            output = self.forward(self.params, branch_input, trunk_input)
        elif in_coords == "cartesian" and in_forward == "polar":
            output = self.forward(self.params, branch_input, vxy2rtheta(trunk_input))
        elif in_coords == "polar" and in_forward == "cartesian":
            output = self.forward(self.params, branch_input, vrtheta2xy(trunk_input))
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

    def loss_rect(self, params, branch_input: tuple[jax.Array], trunk_input: tuple[jax.Array], true_val: dict[str, jax.Array] | None = None):
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
        out0 = deeponetmap(self.hessian)(params, branch_input, trunk_input[0]).reshape(-1, 4) # horizontal lower
        out1 = deeponetmap(self.hessian)(params, branch_input, trunk_input[1]).reshape(-1, 4) # vertical right
        out2 = deeponetmap(self.hessian)(params, branch_input, trunk_input[2]).reshape(-1, 4) # horizontal upper
        out3 = deeponetmap(self.hessian)(params, branch_input, trunk_input[3]).reshape(-1, 4) # vertical left

        # Compute losses for all four sides of rectangle
        losses  = pwhloss.loss_rect(out0, out1, out2, out3, true_val=true_val, loss_fn=self.loss_fn)
        return losses

    def loss_circ(self, params, branch_input: jax.Array, trunk_input: jax.Array, true_val: dict[str, jax.Array] | None = None):
        """
        Computes the loss of the residuals on the circle.
        """

        # Compute cartesian output
        output = deeponetmap(self.hessian)(params, branch_input, trunk_input).reshape(-1, 4)

        # Compute polar stresses and return loss
        losses = pwhloss.loss_circ_rr_rt(trunk_input, output, true_val=true_val, loss_fn=self.loss_fn)
        # losses = pwhloss.loss_circ_xx_xy_yy(output, true_val=true_val, loss_fn=self.loss_fn)
        return losses

    def loss_diri(self, params, branch_input: tuple[jax.Array], trunk_input: tuple[jax.Array], true_val: dict[str, jax.Array] | None = None):

        # Compute potential output
        out0 = deeponetmap(self.forward)(params, branch_input, trunk_input[0])
        out1 = deeponetmap(self.forward)(params, branch_input, trunk_input[1])
        out2 = deeponetmap(self.forward)(params, branch_input, trunk_input[2])
        out3 = deeponetmap(self.forward)(params, branch_input, trunk_input[3])

        # Compute losses for all four sides of rectangle
        losses  = pwhloss.loss_dirichlet(out0, out1, out2, out3, true_val=true_val, loss_fn=self.loss_fn)
        return losses
    
    def loss_data(self, params, branch_input: jax.Array, trunk_input: jax.Array, true_val: dict[str, jax.Array] | None = None):
        
        # Compute model output
        out = deeponetmap(self.hessian)(params, branch_input, trunk_input).reshape(-1, 4)

        # Compute losses
        losses = pwhloss.loss_data(out, true_val=true_val, loss_fn=self.loss_fn)
        return losses
    
    def loss_rect_extra(self, params, branch_input: tuple[jax.Array], trunk_input: tuple[jax.Array], true_val: dict[str, jax.Array] | None = None):
        # Compute Hessian values for each of the four sides
        out0 = deeponetmap(self.hessian)(params, branch_input, trunk_input[0]).reshape(-1, 4) # horizontal lower
        out1 = deeponetmap(self.hessian)(params, branch_input, trunk_input[1]).reshape(-1, 4) # vertical right
        out2 = deeponetmap(self.hessian)(params, branch_input, trunk_input[2]).reshape(-1, 4) # horizontal upper
        out3 = deeponetmap(self.hessian)(params, branch_input, trunk_input[3]).reshape(-1, 4) # vertical left

        # Compute losses for all four sides of rectangle
        losses  = pwhloss.loss_rect_extra(out0, out1, out2, out3, true_val=true_val, loss_fn=self.loss_fn)
        return losses
    
    
    def sample_points(self):
        """
        Method used for sampling points on boundaries and in domain.
        """

        self._key, train_branch_key, eval_branch_key, train_trunk_key, eval_trunk_key = jax.random.split(self._key, 5)
        self._key, data_trunk_key = jax.random.split(self._key, 2)
        radius_interval = self.geometry_settings["trunk"]["radius_interval"]
        tension_interval = self.geometry_settings["trunk"]["tension_interval"]
        xlim = self.geometry_settings["domain"]["rectangle"]["xlim"]
        ylim = self.geometry_settings["domain"]["rectangle"]["ylim"]
        train_sampling = self.train_settings.sampling if self.do_train is not None else None
        eval_sampling = self.eval_settings.sampling if self.do_eval is not None else None
        
        self.train_points_branch = generate_collocation_points(train_branch_key, radius_interval, tension_interval, train_sampling["branch"])
        
        # Sampling points in domain and on boundaries
        self.train_points_trunk = []
        self.train_true_val = []
        for i in range(train_sampling["branch"]):
            train_trunk_key, train_trunk_key2 = jax.random.split(train_trunk_key, 2)
            
            trunk_points = generate_rectangle_with_hole(train_trunk_key2, self.train_points_branch[i, 0], xlim, ylim,
                                                         train_sampling["coll"],
                                                         train_sampling["rect"],
                                                         train_sampling["circ"])
            
            # Generate data points
            data_trunk_key, data_trunk_key2 = jax.random.split(data_trunk_key, 2)
            trunk_points["data"] = generate_collocation_points_with_hole(data_trunk_key2, self.train_points_branch[i, 0],
                                                                            xlim, ylim, train_sampling.get("data"))
            
            self.train_points_trunk.append(trunk_points)
            
            # Get corresponding function values
            self.train_true_val.append(analytic.get_true_vals(trunk_points, radius=self.train_points_branch[i, 0], tension=self.train_points_branch[i, 1], exclude=["diri"]))
        
        self.eval_points_branch = jnp.array(self.eval_settings.sampling["branch_point"])
        self.eval_points_trunk = generate_rectangle_with_hole(eval_trunk_key, self.eval_points_branch[0], xlim, ylim,
                                                         eval_sampling["coll"],
                                                         eval_sampling["rect"],
                                                         eval_sampling["circ"])
        return
    
    def eval(self, point_type: str = "coll", metric: str  = "L2-rel", **kwargs):
        """
        Evaluates the Cartesian stresses using the specified metric.
        """
        match metric.lower():
            case "l2-rel":
                metric_fun = jax.jit(L2rel)
            case _:
                print(f"Unknown metric: '{metric}'. Default ('L2-rel') is used for evaluation.")
                metric_fun = jax.jit(L2rel)

        u = jnp.squeeze(deeponetmap(self.jitted_hessian)(self.params, self.eval_points_branch, self.eval_points_trunk[point_type]))
        u_true = jax.vmap(partial(analytic.cart_stress_true, a=self.eval_points_branch[0], S=self.eval_points_branch[1]))(self.eval_points_trunk[point_type])

        err = jnp.array([[metric_fun(u[:, i, j], u_true[:, i, j]) for i in range(2)] for j in range(2)])

        attr_name = "eval_result"

        if hasattr(self, attr_name):
            if isinstance(self.eval_result, dict):
                self.eval_result[metric] = err
            else:
                raise TypeError(f"Attribute '{attr_name}' is not a dictionary. "
                                f"Evaluation error cannot be added.")
        else:
            self.eval_result = {metric: err}
        
        return err
    
    def plot_results(self, save=True, log=False, step=None):
        pwhplot.plot_results(self.geometry_settings, self.jitted_hessian, self.params, self.eval_points_branch,
                             self.dir.figure_dir, self.dir.log_dir, save=save, log=log, step=step, 
                             grid=self.plot_settings["grid"], dpi=self.plot_settings["dpi"])
        
        
    def do_every(self, epoch: int | None = None, loss_terms: jax.Array | None = None):
        
        plot_every = self.result_plots.plot_every
        log_every = self.logging.log_every
        do_log = self.logging.do_logging
        checkpoint_every = self.train_settings.checkpoint_every

        if do_log and epoch % log_every == log_every-1:
            if epoch // log_every == 0:
                self.all_losses = jnp.zeros((0, loss_terms.shape[0]))
            self.all_losses = self.log_scalars(loss_terms, self.loss_names, all_losses=self.all_losses, log=False)

        if plot_every and epoch % plot_every == plot_every-1:
            self.plot_results(save=False, log=True, step=epoch)

        if epoch % checkpoint_every == checkpoint_every-1:
            self.write_model(step=epoch+1)
            if hasattr(self, "all_losses"):
                with open(self.dir.log_dir.joinpath('all_losses.npy'), "wb") as f:
                    jnp.save(f, self.all_losses)
        
        return