from time import perf_counter
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from datahandlers.generators import generate_interval_points
from models.derivatives import gradient, hessian
from models.loss import mse, maxabse
from models.networks import netmap
from models.pinn import PINN
from setup.parsers import parse_arguments
from utils.plotting import save_fig


class Sin1DPINN(PINN):
    def __init__(self, settings: dict, *args, **kwargs):
        super().__init__(settings, *args, **kwargs)
        self.init_model(settings["model"]["pinn"]["network"])
        self._set_loss(loss_term_fun_name="loss_terms")
        self._set_update(loss_fun_name="_total_loss", optimizer_name="optimizer")
        self.network = self.net[0]
        self.opt_state = self.optimizer.init(self.params)
        return

    def forward(self, params, input: jax.Array) -> jax.Array:
        return self.network.apply(params["net0"], input)
    
    def grad1(self, params, input: jax.Array) -> jax.Array:
        return gradient(self.forward)(params, input)

    def grad2(self, params, input: jax.Array) -> jax.Array:
        return hessian(self.forward)(params, input)
    
    def grad3(self, params, input: jax.Array) -> jax.Array:
        return gradient(hessian(self.forward))(params, input)
    
    def grad4(self, params, input: jax.Array) -> jax.Array:
        return hessian(hessian(self.forward))(params, input)

    def loss_terms(self,
                   params,
                   inputs: dict[str, jax.Array],
                   true_val: dict[str, jax.Array],
                   update_key: int | None = None,
                   loss_fn: Callable | None = None,
                   **kwargs
                   ) -> jax.Array:
        
        if update_key == 0:
            loss = self.loss0(params, inputs["coll"], true_val=true_val["0"], loss_fn=loss_fn)
            return jnp.array(loss)

        if update_key == 1:
            loss1 = self.loss1(params, inputs["coll"], true_val=true_val["1"], loss_fn=loss_fn)
            return jnp.array(loss1)
        
        if update_key == 2:
            loss2 = self.loss2(params, inputs["coll"], true_val=true_val["2"], loss_fn=loss_fn)
            return jnp.array(loss2)
        
        if update_key == 3:
            loss3 = self.loss3(params, inputs["coll"], true_val=true_val["3"], loss_fn=loss_fn)
            return jnp.array(loss3)
        
        if update_key == 4:
            loss4 = self.loss4(params, inputs["coll"], true_val=true_val["4"], loss_fn=loss_fn)
            return jnp.array(loss4)
        
        if update_key == 5:
            loss4 = self.loss4(params, inputs["coll"], true_val=true_val["4"], loss_fn=loss_fn)
            bcloss2 = self.loss2(params, inputs["bc"], true_val=true_val["bc2"], loss_fn=loss_fn)
            return jnp.array((loss4, bcloss2))
        
        if update_key == 6:
            loss4 = self.loss4(params, inputs["coll"], true_val=true_val["4"], loss_fn=loss_fn)
            bcloss2 = self.loss2(params, inputs["bc"], true_val=true_val["bc2"], loss_fn=loss_fn)
            bcloss0 = self.loss0(params, inputs["bc"], true_val=true_val["bc0"], loss_fn=loss_fn)
            return jnp.array((loss4, bcloss2, bcloss0))

        loss0 = self.loss0(params, inputs["coll"], true_val=true_val["0"], loss_fn=loss_fn)
        loss1 = self.loss1(params, inputs["coll"], true_val=true_val["1"], loss_fn=loss_fn)
        loss2 = self.loss2(params, inputs["coll"], true_val=true_val["2"], loss_fn=loss_fn)
        loss3 = self.loss3(params, inputs["coll"], true_val=true_val["3"], loss_fn=loss_fn)
        loss4 = self.loss4(params, inputs["coll"], true_val=true_val["4"], loss_fn=loss_fn)
        return jnp.array((loss0, loss1, loss2, loss3, loss4))
    
    def loss0(self,
              params,
              input: dict[str, jax.Array],
              true_val: dict[str, jax.Array],
              loss_fn: Callable | None = None
              ) -> jax.Array:

        out0 = netmap(self.forward)(params, input)
        if loss_fn is not None:
            return loss_fn(out0, true_val)
        
        return mse(out0, true_val)
    
    def loss1(self,
              params,
              input: dict[str, jax.Array],
              true_val: dict[str, jax.Array],
              loss_fn: Callable | None = None
              ) -> jax.Array:

        out1 = netmap(self.grad1)(params, input)
        if loss_fn is not None:
            return loss_fn(out1, true_val)
        
        return mse(out1, true_val)
    
    def loss2(self,
              params,
              input: dict[str, jax.Array],
              true_val: dict[str, jax.Array],
              loss_fn: Callable | None = None
              ) -> jax.Array:

        out2 = netmap(self.grad2)(params, input)
        if loss_fn is not None:
            return loss_fn(out2, true_val)
        
        return mse(out2, u_true=true_val)
    
    def loss3(self,
              params,
              input: dict[str, jax.Array],
              true_val: dict[str, jax.Array],
              loss_fn: Callable | None = None
              ) -> jax.Array:

        out3 = netmap(self.grad3)(params, input)
        if loss_fn is not None:
            return loss_fn(out3, true_val)
        
        return mse(out3, true_val)
    
    def loss4(self,
              params,
              input: dict[str, jax.Array],
              true_val: dict[str, jax.Array],
              loss_fn: Callable | None = None
              ) -> jax.Array:

        out4 = netmap(self.grad4)(params, input)
        
        if loss_fn is not None:
            return loss_fn(out4, true_val)
        
        return mse(out4, true_val)
    
    def sample_points(self):
        """
        Method used for sampling points on boundaries and in domain.
        """

        self._key, train_key, eval_key = jax.random.split(self._key, 3)
        xlim = self.geometry_settings["domain"]["interval"]["xlim"]
        train_sampling = self.train_settings.sampling if self.do_train is not None else None
        eval_sampling = self.eval_settings.sampling if self.do_eval is not None else None
        
        # Sampling points in domain and on boundaries
        self.train_points = {}
        self.eval_points = {}

        self.train_points["coll"] = generate_interval_points(train_key, xlim, train_sampling["coll"], sobol=True)
        self.eval_points["coll"] = generate_interval_points(eval_key, xlim, eval_sampling["coll"], sobol=True)
        self.train_points["bc"] = jnp.array(xlim).reshape(-1, 1)
        self.eval_points["bc"] = jnp.array(xlim).reshape(-1, 1)

        # Get corresponding function values
        self.train_true_val = {}
        self.eval_true_val = {}

        self.train_true_val["0"] =  jnp.sin(self.train_points["coll"])
        self.eval_true_val["0"]  =  jnp.sin(self.eval_points["coll"])
        self.train_true_val["1"] =  jnp.cos(self.train_points["coll"])
        self.eval_true_val["1"]  =  jnp.cos(self.eval_points["coll"])
        self.train_true_val["2"] = -jnp.sin(self.train_points["coll"])
        self.eval_true_val["2"]  = -jnp.sin(self.eval_points["coll"])
        self.train_true_val["3"] = -jnp.cos(self.train_points["coll"])
        self.eval_true_val["3"]  = -jnp.cos(self.eval_points["coll"])
        self.train_true_val["4"] =  jnp.sin(self.train_points["coll"])
        self.eval_true_val["4"]  =  jnp.sin(self.eval_points["coll"])

        
        self.train_true_val["bc0"] =  jnp.sin(self.train_points["bc"])
        self.eval_true_val["bc0"]  =  jnp.sin(self.eval_points["bc"])
        self.train_true_val["bc1"] =  jnp.cos(self.train_points["bc"])
        self.eval_true_val["bc1"]  =  jnp.cos(self.eval_points["bc"])
        self.train_true_val["bc2"] = -jnp.sin(self.train_points["bc"])
        self.eval_true_val["bc2"]  = -jnp.sin(self.eval_points["bc"])
        self.train_true_val["bc3"] = -jnp.cos(self.train_points["bc"])
        self.eval_true_val["bc3"]  = -jnp.cos(self.eval_points["bc"])
        self.train_true_val["bc4"] =  jnp.sin(self.train_points["bc"])
        self.eval_true_val["bc4"]  =  jnp.sin(self.eval_points["bc"])

        return
    
    def get_true_vals(self, x):
        true_vals = {}
        
        true_vals["0"] =  jnp.sin(x)
        true_vals["1"] =  jnp.cos(x)
        true_vals["2"] = -jnp.sin(x)
        true_vals["3"] = -jnp.cos(x)
        true_vals["4"] =  jnp.sin(x)
        
        return true_vals
        
        
    
    def train(self, update_key: int | None = None):
        if not self.do_train:
            print("Model is not set to train")
            return
        
        max_epochs = self.train_settings.iterations
        plot_every = self.result_plots.plot_every
        sample_every = self.train_settings.resampling["resample_steps"]
        do_resample = self.train_settings.resampling["do_resampling"]

        log_every = self.logging.log_every
        
        # Create arrays for losses for function values and 1st-4th order gradients
        self.loss_log_train = np.zeros((max_epochs // log_every + 1, 5))
        self.loss_log_eval = np.zeros((max_epochs // log_every + 1, 5))
        self.loss_log_epochs = np.arange(0, max_epochs+log_every, log_every)
        # Loss counter
        l = 0
        
        jitted_loss = jax.jit(self.loss_terms, static_argnames=("update_key", "loss_fn"))
        
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
            self.params, self.opt_state, total_loss, loss_terms = self.update(opt_state=self.opt_state,
                                                                                             params=self.params,
                                                                                             inputs=self.train_points,
                                                                                             weights=self.weights,
                                                                                             true_val=self.train_true_val,
                                                                                             update_key=update_key,
                                                                                             start_time=t0,
                                                                                             epoch=epoch,
                                                                                             learning_rate=self.schedule(epoch)
                                                                                             )
            
            
            if (epoch % log_every == 0):
                self.loss_log_train[l] = jitted_loss(self.params, self.train_points, true_val=self.train_true_val, update_key=None)
                self.loss_log_eval[l] = jitted_loss(self.params, self.eval_points, true_val=self.eval_true_val, update_key=None, loss_fn=maxabse)
                l += 1
        
        # Log latest model loss
        self.loss_log_train[-1] = jitted_loss(self.params, self.train_points, true_val=self.train_true_val, update_key=None)
        self.loss_log_eval[-1] = jitted_loss(self.params, self.eval_points, true_val=self.eval_true_val, update_key=None, loss_fn=maxabse)
            
        return

    def plot_losses(self):

        fig = plt.figure()
        plt.semilogy(self.loss_log_epochs, self.loss_log_train)
        plt.legend(["Diff" + str(i) for i in range(5)])
        save_fig(self.dir.figure_dir, "train.pdf", format="pdf", fig=fig)
        
        fig = plt.figure()
        plt.semilogy(self.loss_log_epochs, self.loss_log_eval)
        plt.legend(["Diff" + str(i) for i in range(5)])
        save_fig(self.dir.figure_dir, "eval.pdf", format="pdf", fig=fig)

        return
    
    def plot_derivatives2(self):
        xlim = self.geometry_settings["domain"]["interval"]["xlim"]
        xx = jnp.linspace(xlim[0], xlim[1], 501)

        fig = plt.figure()
        plt.plot(xx, netmap(self.forward)(self.params, xx.reshape(-1, 1)).ravel())
        plt.plot(xx, netmap(self.grad1  )(self.params, xx.reshape(-1, 1)).ravel())
        plt.plot(xx, netmap(self.grad2  )(self.params, xx.reshape(-1, 1)).ravel())
        plt.plot(xx, netmap(self.grad3  )(self.params, xx.reshape(-1, 1)).ravel())
        plt.plot(xx, netmap(self.grad4  )(self.params, xx.reshape(-1, 1)).ravel())
        plt.legend(["Diff" + str(i) for i in range(5)])
        save_fig(self.dir.figure_dir, "diff.pdf", format="pdf", fig=fig)
        

        fig = plt.figure()
        true_vals = self.get_true_vals(xx)
        
        plt.semilogy(xx, jnp.abs(netmap(self.forward)(self.params, xx.reshape(-1, 1)).ravel() - true_vals["0"].ravel()))
        plt.semilogy(xx, jnp.abs(netmap(self.grad1  )(self.params, xx.reshape(-1, 1)).ravel() - true_vals["1"].ravel()))
        plt.semilogy(xx, jnp.abs(netmap(self.grad2  )(self.params, xx.reshape(-1, 1)).ravel() - true_vals["2"].ravel()))
        plt.semilogy(xx, jnp.abs(netmap(self.grad3  )(self.params, xx.reshape(-1, 1)).ravel() - true_vals["3"].ravel()))
        plt.semilogy(xx, jnp.abs(netmap(self.grad4  )(self.params, xx.reshape(-1, 1)).ravel() - true_vals["4"].ravel()))
        plt.legend(["Diff_error" + str(i) for i in range(5)])
        save_fig(self.dir.figure_dir, "diff_error.pdf", format="pdf", fig=fig)
        
    
    def plot_derivatives(self):
        xlim = self.geometry_settings["domain"]["interval"]["xlim"]
        xx = jnp.linspace(xlim[0], xlim[1], 501)

        fig = plt.figure()
        plt.plot(xx, netmap(self.forward)(self.params, xx.reshape(-1, 1)).ravel())
        plt.plot(xx, netmap(self.grad1  )(self.params, xx.reshape(-1, 1)).ravel())
        plt.plot(xx, netmap(self.grad2  )(self.params, xx.reshape(-1, 1)).ravel())
        plt.plot(xx, netmap(self.grad3  )(self.params, xx.reshape(-1, 1)).ravel())
        plt.plot(xx, netmap(self.grad4  )(self.params, xx.reshape(-1, 1)).ravel(), linestyle="--")
        plt.legend(["Diff" + str(i) for i in range(5)])
        save_fig(self.dir.figure_dir, "diff.pdf", format="pdf", fig=fig)
        
        fig = plt.figure()
        plt.plot(xx,  jnp.sin(xx))
        plt.plot(xx,  jnp.cos(xx))
        plt.plot(xx, -jnp.sin(xx))
        plt.plot(xx, -jnp.cos(xx))
        plt.plot(xx,  jnp.sin(xx), linestyle="--")
        plt.legend(["Diff" + str(i) for i in range(5)])
        save_fig(self.dir.figure_dir, "diff_true.pdf", format="pdf", fig=fig)

        fig = plt.figure()
        true_vals = self.get_true_vals(xx)
        
        plt.semilogy(xx, jnp.abs(netmap(self.forward)(self.params, xx.reshape(-1, 1)).ravel() - true_vals["0"].ravel()))
        plt.semilogy(xx, jnp.abs(netmap(self.grad1  )(self.params, xx.reshape(-1, 1)).ravel() - true_vals["1"].ravel()))
        plt.semilogy(xx, jnp.abs(netmap(self.grad2  )(self.params, xx.reshape(-1, 1)).ravel() - true_vals["2"].ravel()))
        plt.semilogy(xx, jnp.abs(netmap(self.grad3  )(self.params, xx.reshape(-1, 1)).ravel() - true_vals["3"].ravel()))
        plt.semilogy(xx, jnp.abs(netmap(self.grad4  )(self.params, xx.reshape(-1, 1)).ravel() - true_vals["4"].ravel()))
        plt.legend(["Diff" + str(i) for i in range(5)])
        save_fig(self.dir.figure_dir, "diff_error.pdf", format="pdf", fig=fig)


    def eval(self):
        pass
