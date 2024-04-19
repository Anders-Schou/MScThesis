from functools import partial
from typing import override
from time import perf_counter

import numpy as np
import jax
import jax.numpy as jnp

from flax.linen import Sequential

from models.loss import ms, mse, sq, sqe
from models.networks import netmap
from models.platewithhole.pinn import DoubleLaplacePINN
from setup.parsers import parse_arguments

class PINN01(DoubleLaplacePINN):
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._set_loss(loss_term_fun_name="loss_terms")
        return

    def loss_terms(self,
                   params,
                   inputs: dict[str, jax.Array],
                   true_val: dict[str, jax.Array],
                   update_key: int | None = None
                   ) -> jax.Array:
        """
        Retrieves all loss values and packs them together in a jax.Array.

        This function is ultimately called by an update function that is jitted and
        recompiled when the update_key argument changes. Therefore, one can write
        multiple different loss functions using if statements as below.
        """
        
        if update_key == 1:
            loss_diri = self.loss_diri(params, inputs["rect"], true_val=true_val.get("diri"))
            return jnp.array((*loss_diri,))
        
        if update_key == 2:
            loss_data = self.loss_data(params, inputs["data"], true_val=true_val.get("data"))
            loss_coll = self.loss_coll(params, inputs["coll"], true_val=true_val.get("coll"))
            loss_rect = self.loss_rect(params, inputs["rect"], true_val=true_val.get("rect"))
            loss_circ = self.loss_circ(params, inputs["circ"], true_val=true_val.get("circ"))
            sum_rect = jnp.sum(jnp.array(loss_rect))
            return jnp.array((*loss_coll, sum_rect, *loss_circ, *loss_data))
        
        if update_key == 3:
            loss_data = self.loss_data(params, inputs["data"], true_val=true_val.get("data"))
            return jnp.array((*loss_data,))
        
        # Default update
        # Computes losses for domain and boundaries
        loss_coll = self.loss_coll(params, inputs["coll"], true_val=true_val.get("coll"))
        loss_rect = self.loss_rect(params, inputs["rect"], true_val=true_val.get("rect"))
        loss_circ = self.loss_circ(params, inputs["circ"], true_val=true_val.get("circ"))

        # Return 1D array of all loss values in the following order
        # [phi, psi, xx0, xy0, yy1, xy1, xx2, xy2, yy3, xy3, rr, rt, di0, di1, di2, di3]
        return jnp.array((*loss_coll, *loss_rect, *loss_circ))

    @partial(jax.jit, static_argnums=(0,))
    def eval_loss(self,
                  params,
                  inputs: dict[str, jax.Array],
                  true_val: dict[str, jax.Array]
                  ) -> float:
        return jnp.sum(self.loss_terms(params, inputs, true_val))

    def train(self, update_key = None, epochs: int | None = None) -> None:
        """
        Method for training the model.

        This method initializes the optimzer and state,
        and then calls the update function for a number
        of times specified in the train_settings.
        """

        if not self.do_train:
            print("Model is not set to train")
            return
        
        max_epochs = self.train_settings.iterations if epochs is None else epochs
        plot_every = self.result_plots.plot_every
        sample_every = self.train_settings.resampling["resample_steps"]
        do_resample = self.train_settings.resampling["do_resampling"]

        self._init_prevlosses(self.loss_terms, update_key=update_key)
        
        # Start time
        t0 = perf_counter()
        for epoch in range(max_epochs):
            
            # Update step
            self.params, self.opt_state, total_loss, self.prevlosses, weights  = self.update(self.opt_state,
                                                                                             self.params,
                                                                                             self.train_points,
                                                                                             true_val=self.train_true_val,
                                                                                             update_key=update_key,
                                                                                             prevlosses=self.prevlosses,
                                                                                             start_time=t0,
                                                                                             epoch=epoch,
                                                                                             learning_rate=self.schedule(epoch)
                                                                                             )
            
            if plot_every and epoch % plot_every == 0:
                self.plot_results(save=False, log=True, step=epoch)
            if do_resample:
                if (epoch % sample_every == (sample_every-1)):
                    if epoch < (max_epochs-1):
                        self.resample(self.resample_eval)
            
        return

    def eval(self):
        pass


if __name__ == "__main__":
    raw_settings = parse_arguments()
    pinn = PINN01(raw_settings)
    pinn.sample_points()
    pinn.train(update_key=None)
    # pinn.train(update_key=None, epochs=10000)
    pinn.plot_results()