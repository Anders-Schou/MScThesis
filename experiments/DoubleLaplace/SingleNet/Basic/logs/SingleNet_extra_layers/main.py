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
from models.platewithhole.pinn import DoubleLaplacePINN
from setup.parsers import parse_arguments
from utils.utils import timer, get_gpu_model
from utils.checkpoint import write_model, load_model

# jax.config.update("jax_enable_x64", True)

class PINN01(DoubleLaplacePINN):
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._set_loss(loss_term_fun_name="loss_terms")
        return

    def loss_terms(self,
                   params,
                   inputs: dict[str, jax.Array],
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
            loss_diri = self.loss_diri(params, inputs["rect"], true_val=true_val.get("diri"))
            self.loss_names = [f"diri{i}" for i, _ in enumerate(loss_diri)]
            return jnp.array((*loss_diri,))
        
        if update_key == 2:
            loss_data = self.loss_data(params, inputs["data"], true_val=true_val.get("data"))
            loss_coll = self.loss_coll(params, inputs["coll"], true_val=true_val.get("coll"))
            loss_rect = self.loss_rect(params, inputs["rect"], true_val=true_val.get("rect"))
            loss_circ = self.loss_circ(params, inputs["circ"], true_val=true_val.get("circ"))
            sum_rect = jnp.sum(jnp.array(loss_rect))
            self.loss_names = ["phi", "rect"] + [f"circ{i}" for i, _ in enumerate(loss_circ)] + [f"data{i}" for i, _ in enumerate(loss_data)]
            return jnp.array((loss_coll, sum_rect, *loss_circ, *loss_data))
        
        if update_key == 3:
            loss_data = self.loss_data(params, inputs["data"], true_val=true_val.get("data"))
            self.loss_names = [f"data{i}" for i, _ in enumerate(loss_data)]
            return jnp.array((*loss_data,))
        
        if update_key == 4:
            loss_coll = self.loss_coll(params, inputs["coll"], true_val=true_val.get("coll"))
            loss_rect = self.loss_rect(params, inputs["rect"], true_val=true_val.get("rect"))
            loss_circ = self.loss_circ(params, inputs["circ"], true_val=true_val.get("circ"))
            sum_rect = jnp.sum(jnp.array(loss_rect))
            self.loss_names = ["phi", "rect"] + [f"circ{i}" for i, _ in enumerate(loss_circ)]
            return jnp.array((loss_coll, sum_rect, *loss_circ))
        
        # Default update
        # Computes losses for domain and boundaries
        loss_coll = self.loss_coll(params, inputs["coll"], true_val=true_val.get("coll"))
        loss_rect = self.loss_rect(params, inputs["rect"], true_val=true_val.get("rect"))
        loss_circ = self.loss_circ(params, inputs["circ"], true_val=true_val.get("circ"))

        # Return 1D array of all loss values in the following order
        self.loss_names = ["phi", "psi"] + [f"rect{i}" for i, _ in enumerate(loss_rect)] + [f"circ{i}" for i, _ in enumerate(loss_circ)]
        return jnp.array((*loss_coll, *loss_rect, *loss_circ))

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
        
        jitted_loss = jax.jit(self.loss_terms, static_argnames=self._static_loss_args)
        
        # Start time
        t0 = perf_counter()
        for epoch in range(max_epochs):
            
            self.get_weights(epoch, 
                             jitted_loss, 
                             self.params, 
                             self.train_points, 
                             true_val=self.train_true_val, 
                             update_key=update_key)
            
            for batch_num, (xy_batch, u_batch) in enumerate(iter(self.full_batch_dataset)):
                
                self.train_points_batch["coll"] = xy_batch
                self.train_true_val_batch["coll"] = u_batch
                
                # Update step
                self.params, self.opt_state, total_loss, loss_terms = self.update(opt_state=self.opt_state,
                                                                                                params=self.params,
                                                                                                inputs=self.train_points_batch,
                                                                                                weights=self.weights,
                                                                                                true_val=self.train_true_val_batch,
                                                                                                update_key=update_key,
                                                                                                start_time=t0,
                                                                                                epoch=epoch,
                                                                                                learning_rate=self.schedule(epoch),
                                                                                                batch_num=batch_num
                                                                                                )
            
            
            self.do_every(epoch=epoch, loss_term_fun=jitted_loss, params=self.params, inputs=self.train_points, true_val=self.train_true_val, update_key=update_key)
            
        if do_log and epoch > log_every:
            self.plot_loss(self.all_losses, {f"{loss_name}": key for key, loss_name in enumerate(self.loss_names)}, fig_dir=self.dir.figure_dir, name="losses", epoch_step=log_every)
        return


if __name__ == "__main__":
    t1 = perf_counter()
    
    raw_settings = timer(parse_arguments)()
    pinn = timer(PINN01)(raw_settings)
    timer(pinn.sample_points)()
    timer(pinn.train)()
    timer(pinn.write_model)()
    # timer(pinn.load_model)()
    timer(pinn.plot_results)()
    
    t2 = perf_counter()

    f = open(pinn.dir.log_dir.joinpath('time_and_eval.dat'), "w")
    f.write(f'Time taken for the whole training process: {t2-t1:.1f} s \t or \t {(t2-t1)/60.0:.1f} min\n')
    
    for metric_fun in ["mse", "maxabse", "L2rel"]:
        timer(pinn.eval)(metric=metric_fun)
        
        f.write(f'{metric_fun} xx error: {pinn.eval_result[metric_fun][0, 0]:.4f}\n')
        f.write(f'{metric_fun} xy error: {pinn.eval_result[metric_fun][0, 1]:.4f}\n')
        f.write(f'{metric_fun} yy error: {pinn.eval_result[metric_fun][1, 1]:.4f}\n')