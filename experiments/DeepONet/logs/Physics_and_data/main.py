from functools import partial
from typing import override
from time import perf_counter
import shutil
import sys

import numpy as np
import jax
import jax.numpy as jnp

from flax.linen import Sequential

from models.loss import ms, mse, sq, sqe
from models.networks import netmap
from models.platewithhole.deeponet import BiharmonicDeepONet
from models.platewithhole import analytic
from setup.parsers import parse_arguments
from utils.utils import timer, get_gpu_model
from utils.checkpoint import write_model, load_model

from datahandlers.generators import (
    generate_collocation_points,
    generate_rectangle_with_hole,
    generate_collocation_points_with_hole,
    generate_rectangle_points,
    resample,
    resample_idx,
    JaxDataset
)

# jax.config.update("jax_enable_x64", True)

class DeepONet(BiharmonicDeepONet):
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._set_loss(loss_term_fun_name="loss_terms")
        return

    def sample_points(self):
        """
        Method used for sampling points on boundaries and in domain.
        """

        self._key, train_branch_key, eval_branch_key, train_trunk_key, eval_trunk_key, dataset_key, perm_key = jax.random.split(self._key, 7)
        radius = self.geometry_settings["domain"]["circle"]["radius"]
        tension_interval = self.geometry_settings["trunk"]["tension_interval"]
        num_sensors = self.train_settings.sampling["num_sensors"]
        xlim = self.geometry_settings["domain"]["rectangle"]["xlim"]
        ylim = self.geometry_settings["domain"]["rectangle"]["ylim"]
        train_sampling = self.train_settings.sampling if self.do_train is not None else None
        eval_sampling = self.eval_settings.sampling if self.do_eval is not None else None
        
        num_branch = train_sampling["branch"]
        self.train_branch_tensions = jax.random.permutation(perm_key, jnp.linspace(tension_interval[0], tension_interval[1], num_branch))
        self.train_branch_points = generate_rectangle_points(train_branch_key, xlim, ylim, [num_sensors // 4]*4, radius=radius)
        
        # Sampling points in domain and on boundaries
        self.train_trunk_points = generate_rectangle_with_hole(train_trunk_key, radius, xlim, ylim,
                                                        train_sampling["coll"],
                                                        train_sampling["rect"],
                                                        train_sampling["circ"])
        self.eval_trunk_points = generate_rectangle_with_hole(eval_trunk_key, radius, xlim, ylim,
                                                        eval_sampling["coll"],
                                                        eval_sampling["rect"],
                                                        eval_sampling["circ"])
        
        # Keys for data points
        self._key, data_train_key, data_eval_key = jax.random.split(self._key, 3)
        self.train_trunk_points["data"] = generate_collocation_points_with_hole(data_train_key, radius,
                                                                          xlim, ylim, train_sampling.get("data"))
        self.eval_trunk_points["data"] = generate_collocation_points_with_hole(data_eval_key, radius,
                                                                         xlim, ylim, eval_sampling.get("data"))

        self.full_batch_dataset = JaxDataset(key=dataset_key, xy=self.train_trunk_points["coll"], u = None, batch_size=sum(train_sampling["coll"]) // num_branch)
        
        # self.plot_training_points()
        return

    def get_branch_inputs(self, inputs, tension):
        true_rect = [jax.vmap(analytic.cart_stress_true, in_axes=(0, None, None))(inputs[i], tension, self.geometry_settings["domain"]["circle"]["radius"]) for i in range(4)]

        return jnp.array((true_rect[0][:, 1, 1],
                         -true_rect[0][:, 0, 1],
                          true_rect[1][:, 0, 0],
                         -true_rect[1][:, 1, 0],
                          true_rect[2][:, 1, 1],
                         -true_rect[2][:, 0, 1],
                          true_rect[3][:, 0, 0],
                         -true_rect[3][:, 1, 0]))

    def loss_terms(self,
                   params,
                   branch_inputs: dict[str, jax.Array],
                   trunk_inputs: dict[str, jax.Array],
                   true_val: dict[str, jax.Array],
                   update_key: int | None = None,
                   **kwargs
                   ) -> jax.Array:
        """
        Retrieves all loss values and packs them together in a jax.Array.

        This function is ultimately called by an update function that is jitted and
        recompiled when the update_key argument changes. Therefore, one can write
        multiple different loss functions using if statements as below.
        """
        
        if update_key == 1:
            loss_diri = self.loss_diri(params, branch_inputs, trunk_inputs["rect"], true_val=true_val.get("diri"))
            self.loss_names = [f"diri{i}" for i, _ in enumerate(loss_diri)]
            return jnp.array((*loss_diri,))
        
        if update_key == 2:
            loss_data = self.loss_data(params, branch_inputs, trunk_inputs["data"], true_val=true_val.get("data"))
            loss_coll = self.loss_coll(params, branch_inputs, trunk_inputs["coll"], true_val=true_val.get("coll"))
            loss_rect = self.loss_rect(params, branch_inputs, trunk_inputs["rect"], true_val=true_val.get("rect"))
            loss_circ = self.loss_circ(params, branch_inputs, trunk_inputs["circ"], true_val=true_val.get("circ"))
            sum_rect = jnp.sum(jnp.array(loss_rect))
            self.loss_names = ["phi", "rect"] + [f"circ{i}" for i, _ in enumerate(loss_circ)] + [f"data{i}" for i, _ in enumerate(loss_data)]
            return jnp.array((loss_coll, sum_rect, *loss_circ, *loss_data))
        
        if update_key == 3:
            loss_data = self.loss_data(params, branch_inputs, trunk_inputs["data"], true_val=true_val.get("data"))
            self.loss_names = [f"data{i}" for i, _ in enumerate(loss_data)]
            return jnp.array((*loss_data,))
        
        if update_key == 4:
            loss_coll = self.loss_coll(params, branch_inputs, trunk_inputs["coll"], true_val=true_val.get("coll"))
            loss_rect = self.loss_rect(params, branch_inputs, trunk_inputs["rect"], true_val=true_val.get("rect"))
            loss_circ = self.loss_circ(params, branch_inputs, trunk_inputs["circ"], true_val=true_val.get("circ"))
            sum_rect = jnp.sum(jnp.array(loss_rect))
            self.loss_names = ["phi", "rect"] + [f"circ{i}" for i, _ in enumerate(loss_circ)]
            return jnp.array((loss_coll, sum_rect, *loss_circ))
        
        # Default update
        # Computes losses for domain and boundaries
        loss_coll = self.loss_coll(params, branch_inputs, trunk_inputs["coll"], true_val=true_val.get("coll"))
        loss_rect = self.loss_rect(params, branch_inputs, trunk_inputs["rect"], true_val=true_val.get("rect"))
        loss_circ = self.loss_circ(params, branch_inputs, trunk_inputs["circ"], true_val=true_val.get("circ"))

        # loss_rect = tuple([10*i for i in loss_rect])
        # loss_circ = tuple([10*i for i in loss_circ])
        # Return 1D array of all loss values in the following order
        self.loss_names = ["phi"] + [f"rect{i}" for i, _ in enumerate(loss_rect)] + [f"circ{i}" for i, _ in enumerate(loss_circ)]
        return jnp.array((loss_coll, *loss_rect, *loss_circ))

    def train(self, update_key = None, epochs: int | None = None, new_init: bool = False) -> None:
        """
        Method for training the model.

        This method initializes the optimzer and state,
        and then calls the update function for a number
        of times specified in the train_settings.
        """

        if not self.do_train:
            print("Model is not set to train")
            return
        
        with open(self.dir.log_dir.joinpath("GPU_model.txt"), "w") as f:
            f.write(get_gpu_model())
            
        shutil.copy(self.dir.settings_path, self.dir.log_dir)
        shutil.copy(sys.path[0] + '/main.py', self.dir.log_dir)
        
        
        if new_init:
            del self.weights
        
        max_epochs = self.train_settings.iterations if epochs is None else epochs
        log_every = self.logging.log_every
        do_log = self.logging.do_logging
        num_batches = self.train_settings.sampling.get("branch", 1)
        
        jitted_loss = jax.jit(self.loss_terms, static_argnames=self._static_loss_args)
        
        # Start time
        t0 = perf_counter()
        for epoch in range(max_epochs):
            
            self.get_weights(epoch, 
                             jitted_loss, 
                             self.params, 
                             self.train_points_branch[0], 
                             self.train_points_trunk[0], 
                             true_val=self.train_true_val[0], 
                             update_key=update_key)
            
            for batch_num, (xy_batch, u_batch) in enumerate(iter(self.full_batch_dataset)):
                
                tension = self.branch_tensions[batch_num]
                self.train_points_trunk["coll"] = xy_batch
                self.train_true_val_batch = analytic.get_true_vals(self.train_points_trunk, exclude=["diri"], tension=tension)
                branch_inputs = analytic.get_true_vals(self.train_points_branch, tension=tension)
            
                # Update step
                self.params, self.opt_state, total_loss, loss_terms = self.update(opt_state=self.opt_state,
                                                                                                params=self.params,
                                                                                                branch_inputs=branch_inputs,
                                                                                                trunk_inputs=self.train_points_trunk,
                                                                                                weights=self.weights,
                                                                                                true_val=self.train_true_val_batch,
                                                                                                update_key=update_key,
                                                                                                start_time=t0,
                                                                                                epoch=epoch,
                                                                                                batch_num=batch_num,
                                                                                                learning_rate=self.schedule(epoch)
                                                                                                )
            
            self.do_every(epoch=epoch, loss_terms=loss_terms)
                
        if do_log and epoch > log_every:
            self.plot_loss(self.all_losses, {f"{loss_name}": key for key, loss_name in enumerate(self.loss_names)}, fig_dir=self.dir.figure_dir, name="losses.png", epoch_step=log_every)
        return

if __name__ == "__main__":
    t1 = perf_counter()
    
    raw_settings = timer(parse_arguments)()
    deeponet = timer(DeepONet)(raw_settings)
    timer(deeponet.sample_points)()
    timer(deeponet.train)(update_key=2)
    timer(deeponet.write_model)()
    # timer(deeponet.load_model)()
    timer(deeponet.plot_results)()
    
    t2 = perf_counter()
        
    f = open(deeponet.dir.log_dir.joinpath('time_and_eval.dat'), "w")
    f.write(f'Time taken for the whole training process: {t2-t1:.1f} s \t or \t {(t2-t1)/60.0:.1f} min\n')
    
    for metric_fun in ["mse", "maxabse", "L2rel"]:
        timer(deeponet.eval)(metric=metric_fun)
        
        f.write(f'{metric_fun} xx error: {deeponet.eval_result[metric_fun][0, 0]:.4f}\n')
        f.write(f'{metric_fun} xy error: {deeponet.eval_result[metric_fun][0, 1]:.4f}\n')
        f.write(f'{metric_fun} yy error: {deeponet.eval_result[metric_fun][1, 1]:.4f}\n')
    
    for metric_fun in ["mse", "maxabse", "L2rel"]:
        vm_error = deeponet.eval(metric=metric_fun, cartesian=False)
            
        f.write(f'{metric_fun} vm_stress error: {vm_error:.4f}\n')