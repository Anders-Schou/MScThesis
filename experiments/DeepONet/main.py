from functools import partial
from typing import override
from time import perf_counter

import numpy as np
import jax
import jax.numpy as jnp

from flax.linen import Sequential

from models.loss import ms, mse, sq, sqe
from models.networks import netmap
from models.platewithhole.deeponet import BiharmonicDeepONet
from setup.parsers import parse_arguments
from utils.utils import timer
from utils.checkpoint import write_model, load_model

# jax.config.update("jax_enable_x64", True)

class DeepONet(BiharmonicDeepONet):
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._set_loss(loss_term_fun_name="loss_terms")
        return

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
        
        if new_init:
            del self.weights
        
        max_epochs = self.train_settings.iterations if epochs is None else epochs
        plot_every = self.result_plots.plot_every
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
            
            for batch_num in range(num_batches):
                # Update step
                self.params, self.opt_state, total_loss, self.prevlosses = self.update(opt_state=self.opt_state,
                                                                                                params=self.params,
                                                                                                branch_inputs=self.train_points_branch[batch_num],
                                                                                                trunk_inputs=self.train_points_trunk[batch_num],
                                                                                                weights=self.weights,
                                                                                                true_val=self.train_true_val[batch_num],
                                                                                                update_key=update_key,
                                                                                                start_time=t0,
                                                                                                epoch=epoch,
                                                                                                batch_num=batch_num,
                                                                                                learning_rate=self.schedule(epoch)
                                                                                                )
            
            if do_log and epoch % log_every == log_every-1:
                if epoch // log_every == 0:
                    self.all_losses = jnp.zeros((0, self.prevlosses.shape[0]))
                self.all_losses = self.log_scalars(self.prevlosses, self.loss_names, all_losses=self.all_losses, log=False)
            if plot_every and epoch % plot_every == plot_every-1:
                self.plot_results(save=False, log=True, step=epoch)
                
        if do_log and epoch > log_every:
            self.plot_loss(self.all_losses, {f"{loss_name}": key for key, loss_name in enumerate(self.loss_names)}, fig_dir=self.dir.figure_dir, name="losses.png", epoch_step=log_every)
        return

if __name__ == "__main__":
    t1 = perf_counter()
    
    raw_settings = timer(parse_arguments)()
    deeponet = timer(DeepONet)(raw_settings)
    timer(deeponet.sample_points)()
    timer(deeponet.train)()
    timer(deeponet.write_model)()
    timer(deeponet.plot_results)()
    timer(deeponet.eval)()
    
    t2 = perf_counter()
        
    f = open(deeponet.dir.log_dir.joinpath('time.dat'), "w+")
    f.write(f'Time taken for the whole training process: {t2-t1:.1f} s \t or \t {(t2-t1)/60.0:.1f} min\n')
    f.write(f'L2-rel xx error: {deeponet.eval_result["L2-rel"][0, 0]:.4f}\n')
    f.write(f'L2-rel xy error: {deeponet.eval_result["L2-rel"][0, 1]:.4f}\n')
    f.write(f'L2-rel yy error: {deeponet.eval_result["L2-rel"][1, 1]:.4f}\n')