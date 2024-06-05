from functools import partial
from typing import override
from time import perf_counter

import matplotlib.pyplot as plt

import numpy as np
import jax
import jax.numpy as jnp

import flax
from flax.linen import Sequential

from models.loss import L2rel
from models.platewithhole.pinn import BiharmonicPINN
from setup.parsers import parse_arguments

# jax.config.update("jax_enable_x64", True)


class PINN01(BiharmonicPINN):
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
            # sum_rect = jnp.sum(jnp.array(loss_rect))
            return jnp.array((loss_coll, *loss_rect, *loss_circ, *loss_data))
        
        if update_key == 3:
            loss_data = self.loss_data(params, inputs["data"], true_val=true_val.get("data"))
            return jnp.array((*loss_data,))
        
        if update_key == 4:
            loss_coll = self.loss_coll(params, inputs["coll"], true_val=true_val.get("coll"))
            loss_rect = self.loss_rect(params, inputs["rect"], true_val=true_val.get("rect"))
            loss_circ = self.loss_circ(params, inputs["circ"], true_val=true_val.get("circ"))
            loss_extra = self.loss_rect_extra(params, inputs["rect"], true_val=true_val.get("rect"))
            return jnp.array((loss_coll, *loss_rect, *loss_circ, *loss_extra))
        
        if update_key == 5:
            loss_rect = self.loss_rect(params, inputs["rect"], true_val=true_val.get("rect"))
            loss_circ = self.loss_circ(params, inputs["circ"], true_val=true_val.get("circ"))
            return jnp.array((*loss_rect, *loss_circ))
        
        if update_key == 6:
            loss_rect = self.loss_rect(params, inputs["rect"], true_val=true_val.get("rect"))
            loss_circ = self.loss_circ(params, inputs["circ"], true_val=true_val.get("circ"))
            loos_circ_extra = self.loss_circ_extra(params, inputs["circ"], true_val=true_val.get("circ"))
            return jnp.array((*loss_rect, *loss_circ, loos_circ_extra))

        if update_key == 7:
            loss_coll = self.loss_coll(params, inputs["coll"], true_val=true_val.get("coll"))
            loss_coll_extra = self.loss_coll(params, inputs["coll_extra"], true_val=true_val.get("coll_extra"))
            loss_rect = self.loss_rect(params, inputs["rect"], true_val=true_val.get("rect"))
            loss_circ = self.loss_circ(params, inputs["circ"], true_val=true_val.get("circ"))
            return jnp.array((loss_coll, loss_coll_extra, *loss_rect, *loss_circ))
        
        if update_key == 8:
            loss_rect = self.loss_rect(params, inputs["rect"], true_val=true_val.get("rect"))
            loss_circ = self.loss_circ(params, inputs["circ"], true_val=true_val.get("circ"))
            return jnp.array((*loss_rect, *loss_circ))

        # Default update
        # Computes losses for domain and boundaries
        loss_coll = self.loss_coll(params, inputs["coll"], true_val=true_val.get("coll"))
        loss_rect = self.loss_rect(params, inputs["rect"], true_val=true_val.get("rect"))
        loss_circ = self.loss_circ(params, inputs["circ"], true_val=true_val.get("circ"))

        # Return 1D array of all loss values in the following order
        # [phi, psi, xx0, xy0, yy1, xy1, xx2, xy2, yy3, xy3, rr, rt, di0, di1, di2, di3]
        return jnp.array((loss_coll, *loss_rect, *loss_circ))

    @partial(jax.jit, static_argnums=(0,))
    def eval_loss(self,
                  params,
                  inputs: dict[str, jax.Array],
                  true_val: dict[str, jax.Array]
                  ) -> float:
        return jnp.sum(self.loss_terms(params, inputs, true_val))

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
        sample_every = self.train_settings.resampling["resample_steps"]
        do_resample = self.train_settings.resampling["do_resampling"]

        # self._init_prevlosses(self.loss_terms, update_key=update_key)
        
        fig = plt.figure()
        plt.scatter(*[self.train_points["coll"][:, i] for i in [0, 1]], s=5)
        if self.train_settings.sampling["separate_coll"]:
            plt.scatter(*[self.train_points["coll_extra"][:, i] for i in [0, 1]], s=5)
        [plt.scatter(*[self.train_points["rect"][r][:, i] for i in [0, 1]], c="red", s=5) for r in range(4)]
        plt.scatter(*[self.train_points["circ"][:, i] for i in [0, 1]], c="green", s=5)
        fig.savefig(self.dir.figure_dir / ("sampling.png"))

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
            # Update step
            self.params, self.opt_state, total_loss, loss_terms  = self.update(self.opt_state,
                                                                               self.params,
                                                                               self.train_points,
                                                                               true_val=self.train_true_val,
                                                                               update_key=update_key,
                                                                               start_time=t0,
                                                                               epoch=epoch,
                                                                               weights=self.weights,
                                                                               learning_rate=self.schedule(epoch)
                                                                               )
            
            if plot_every and epoch % plot_every == 0:
                self.plot_results(save=False, log=True, step=epoch)
            if do_resample:
                if (epoch % sample_every == (sample_every-1)):
                    if epoch < (max_epochs-1):
                        self.resample(self.resample_eval)
                        fig = plt.figure()
                        plt.scatter(self.train_points["coll"][:, 0], self.train_points["coll"][:, 1], s=5)
                        fig.savefig(self.dir.figure_dir / ("sobol" + str(epoch+1) + ".png"))
            
        return

    def eval(self):
        pass


if __name__ == "__main__":
    raw_settings = parse_arguments()
    pinn = PINN01(raw_settings)
    pinn.sample_points()
    pinn.train(update_key=7)
    # for u, e in zip([7], [10000]):
    #     pinn.train(update_key=u, epochs=e, new_init=True)
    pinn.plot_results()
    with open(pinn.dir.log_dir / "rel_loss.txt", "a+") as file:
        file.writelines([*[str(L2rel(pinn.mesh_data["sigma_cart_list"][i].ravel(), pinn.mesh_data["sigma_cart_true_list"][i].ravel())) for i in [0, 1, 3]], "\n\n"])