from time import perf_counter
from collections.abc import Callable
from functools import partial
from typing import override

import numpy as np
from scipy.stats.qmc import Sobol
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from datahandlers.generators import generate_collocation_points, generate_rectangle_points
from models.derivatives import gradient, hessian, laplacian
from models.loss import mse, ms
from models.networks import netmap
from models.pinn import PINN
import models.platewithhole.loss as pwhloss
from models.platewithhole.pinn import DoubleLaplacePINN, BiharmonicPINN
from models.platewithhole.plotting import plot_stress, plot_polar_stress
from setup.parsers import parse_arguments
from utils.plotting import save_fig, get_plot_variables

_FIG_NAME = "Cart_sol_Biharmonic"
_NUM_TERMS = 13

class Square2DPINN(BiharmonicPINN):
    def __init__(self, settings: dict, *args, **kwargs):
        super().__init__(settings, *args, **kwargs)
        self._set_loss(loss_term_fun_name="loss_terms")
        return

    def loss_terms(self,
                   params,
                   inputs: dict[str, jax.Array],
                   true_val: dict[str, jax.Array],
                   update_key: int | None = None
                   ) -> jax.Array:
        loss_coll = self.loss_coll(params, inputs["coll"], true_val=true_val.get("coll"))
        loss_rect = self.loss_rect(params, inputs["rect"], true_val=true_val.get("rect"))
        loss_extra = self.loss_rect_extra(params, inputs["rect"], true_val=true_val.get("rect"))
        # loss_diri = self.loss_diri(params, inputs["rect"], true_val=true_val.get("rect"))
        return jnp.array((loss_coll, *loss_rect, *loss_extra))

    @override
    def sample_points(self):
        """
        Method used for sampling points on boundaries and in domain.
        """

        def analytic(xy: jax.Array) -> jax.Array:
            # return jnp.multiply(jnp.sin(xy[0]), jnp.sin(xy[1]))
            omega = 0.05*jnp.pi
            return jnp.multiply(jnp.sin(omega*xy[0]), jnp.sin(omega*xy[1])) / omega**2

        true_func = jax.vmap(analytic)
        true_hess = lambda xy: jax.vmap(jax.hessian(analytic))(xy).reshape(-1, 2, 2)

        self.true_hess_func = true_hess


        self._key, train_key, eval_key, bc_train_key, bc_eval_key = jax.random.split(self._key, 5)
        xlim = self.geometry_settings["domain"]["rectangle"]["xlim"]
        ylim = self.geometry_settings["domain"]["rectangle"]["ylim"]
        train_sampling = self.train_settings.sampling if self.do_train is not None else None
        eval_sampling = self.eval_settings.sampling if self.do_eval is not None else None
        
        # Sampling points in domain and on boundaries
        self.train_points = {}
        self.eval_points = {}

        # self.train_points["coll"] = generate_collocation_points(train_key, xlim, ylim, train_sampling["coll"])
        # self.eval_points["coll"] = generate_collocation_points(eval_key, xlim, ylim, eval_sampling["coll"])
        self.train_points["coll"] = jnp.array(Sobol(2, seed=self._seed).random_base2(10))
        self.eval_points["coll"] = jnp.array(Sobol(2, seed=self._seed).random_base2(10))
        self.train_points["rect"] = generate_rectangle_points(bc_train_key, xlim, ylim, train_sampling["rect"])
        self.eval_points["rect"] = generate_rectangle_points(bc_eval_key, xlim, ylim, eval_sampling["rect"])

        # Get corresponding function values
        self.train_true_val = {}
        self.eval_true_val = {}

        self.train_true_val["coll"] = true_func(self.train_points["coll"])
        self.eval_true_val["coll"] = true_func(self.eval_points["coll"])
        
        true_rect = [true_hess(self.train_points["rect"][i]) for i in range(4)]
        rect = {
            "xx0":  true_rect[0][:, 0, 0],
            "xy0":  true_rect[0][:, 0, 1],
            "yy0":  true_rect[0][:, 1, 1], # extra

            "xx1":  true_rect[1][:, 0, 0], # extra
            "xy1":  true_rect[1][:, 1, 0],
            "yy1":  true_rect[1][:, 1, 1],

            "xx2":  true_rect[2][:, 0, 0],
            "xy2":  true_rect[2][:, 0, 1],
            "yy2":  true_rect[2][:, 1, 1], # extra

            "xx3":  true_rect[3][:, 0, 0], # extra
            "xy3":  true_rect[3][:, 1, 0],
            "yy3":  true_rect[3][:, 1, 1],


            "di0":  true_func(self.train_points["rect"][0]),
            "di1":  true_func(self.train_points["rect"][1]),
            "di2":  true_func(self.train_points["rect"][2]),
            "di3":  true_func(self.train_points["rect"][3])
        }
        self.train_true_val["rect"] = rect

        true_rect = [true_hess(self.eval_points["rect"][i]) for i in range(4)]
        rect = {
            "xx0":  true_rect[0][:, 0, 0],
            "xy0":  true_rect[0][:, 0, 1],
            "yy0":  true_rect[0][:, 1, 1], # extra

            "xx1":  true_rect[1][:, 0, 0], # extra
            "xy1":  true_rect[1][:, 1, 0],
            "yy1":  true_rect[1][:, 1, 1],

            "xx2":  true_rect[2][:, 0, 0],
            "xy2":  true_rect[2][:, 0, 1],
            "yy2":  true_rect[2][:, 1, 1], # extra

            "xx3":  true_rect[3][:, 0, 0], # extra
            "xy3":  true_rect[3][:, 1, 0],
            "yy3":  true_rect[3][:, 1, 1],

            "di0":  true_func(self.eval_points["rect"][0]),
            "di1":  true_func(self.eval_points["rect"][1]),
            "di2":  true_func(self.eval_points["rect"][2]),
            "di3":  true_func(self.eval_points["rect"][3])
        }
        self.eval_true_val["rect"] = rect

        return
    
    def param_hessian(self):
        return jax.hessian(self.loss_terms)(self.params, self.train_points, true_val=self.train_true_val, update_key=None)
        # jax.hessian(self._total_loss, has_aux=True)(self.params, self.train_points, true_val=self.train_true_val, update_key=None, prevlosses=None)
        # jax.jacrev(self.loss_terms, has_aux=False)(self.params, self.train_points, true_val=self.train_true_val)["net0"]["params"]

    def train(self, update_key: int | None = None):
        if not self.do_train:
            print("Model is not set to train")
            return
        
        jitted_eval = jax.jit(self.loss_terms, static_argnames=("update_key"))
        self.num_loss_terms = _NUM_TERMS
        max_epochs = self.train_settings.iterations
        plot_every = self.result_plots.plot_every
        sample_every = self.train_settings.resampling["resample_steps"]
        do_resample = self.train_settings.resampling["do_resampling"]

        self._init_prevlosses(self.loss_terms, update_key=update_key)
        
        log_every = self.logging.log_every
        
        # Create arrays for losses for function values and 1st-4th order gradients
        self.loss_log_train = np.zeros((max_epochs // log_every + 1, self.num_loss_terms))
        self.loss_log_eval = np.zeros((max_epochs // log_every + 1, self.num_loss_terms))
        self.loss_log_epochs = np.arange(0, max_epochs+log_every, log_every)
        # Loss counter
        l = 0

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
            
            
            if (epoch % log_every == 0):
                self.loss_log_train[l] = jitted_eval(self.params, self.train_points, true_val=self.train_true_val, update_key=None)
                self.loss_log_eval[l] = jitted_eval(self.params, self.eval_points, true_val=self.eval_true_val, update_key=None)
                l += 1
        
        # Log latest model loss
        self.loss_log_train[-1] = jitted_eval(self.params, self.train_points, true_val=self.train_true_val, update_key=None)
        self.loss_log_eval[-1] = jitted_eval(self.params, self.eval_points, true_val=self.eval_true_val, update_key=None)

            # if do_resample:
            #     if (epoch % sample_every == (sample_every-1)):
            #         if epoch < (max_epochs-1):
            #             self.resample(self.resample_eval)
            
        return

    def plot_losses(self):

        fig = plt.figure()
        plt.semilogy(self.loss_log_epochs, self.loss_log_train)
        plt.legend([str(i) for i in range(self.num_loss_terms)], ncol=2)
        save_fig(self.dir.figure_dir, "train.pdf", format="pdf", fig=fig)
        
        fig = plt.figure()
        plt.semilogy(self.loss_log_epochs, self.loss_log_eval)
        plt.legend([str(i) for i in range(self.num_loss_terms)], ncol=2)
        save_fig(self.dir.figure_dir, "eval.pdf", format="pdf", fig=fig)

        return
    
    def plot_stresses(self, *,
                      extension="png",
                      figsize = (35, 30)):
        """
        Function for plotting stresses in cartesian coordinates.
        """

        fig_dir = self.dir.figure_dir
        name = _FIG_NAME
        
        xlim = self.geometry_settings["domain"]["rectangle"]["xlim"]
        ylim = self.geometry_settings["domain"]["rectangle"]["ylim"]

        X, Y, plotpoints = get_plot_variables(xlim, ylim, grid=101)

        Z_true = [self.true_hess_func(plotpoints).reshape(-1, 4)[:, i].reshape(X.shape) for i in range(4)]

        Z = [netmap(self.hessian)(self.params, plotpoints).reshape(-1, 4)[:, i].reshape(X.shape) for i in range(4)]

        _CLEVELS = 101
        _FONTSIZE = 40

        vmin0 = min(jnp.min(Z_true[0]),jnp.min(Z[0]))
        vmin1 = min(jnp.min(Z_true[1]),jnp.min(Z[1]))
        vmin3 = min(jnp.min(Z_true[3]),jnp.min(Z[3]))
        
        vmax0 = max(jnp.max(Z_true[0]), jnp.max(Z[0]))
        vmax1 = max(jnp.max(Z_true[1]), jnp.max(Z[1]))
        vmax3 = max(jnp.max(Z_true[3]), jnp.max(Z[3]))
        

        fig, ax = plt.subplots(3, 3, figsize=figsize)
        ax[0, 0].set_aspect('equal', adjustable='box')
        ax[0, 0].set_title("XX stress", fontsize=_FONTSIZE)
        p1 = ax[0, 0].contourf(X , Y, Z[0], levels=_CLEVELS, vmin=vmin0, vmax=vmax0)
        plt.colorbar(p1, ax=ax[0, 0])

        ax[0, 1].set_aspect('equal', adjustable='box')
        ax[0, 1].set_title("XY stress", fontsize=_FONTSIZE)
        p2 = ax[0, 1].contourf(X, Y, Z[1], levels=_CLEVELS, vmin=vmin1, vmax=vmax1)
        plt.colorbar(p2, ax=ax[0, 1])

        ax[0, 2].set_aspect('equal', adjustable='box')
        ax[0, 2].set_title("YY stress", fontsize=_FONTSIZE)
        p4 = ax[0, 2].contourf(X, Y, Z[3], levels=_CLEVELS, vmin=vmin3, vmax=vmax3)
        plt.colorbar(p4, ax=ax[0, 2])



        ax[1, 0].set_aspect('equal', adjustable='box')
        ax[1, 0].set_title("True XX stress", fontsize=_FONTSIZE)
        p1 = ax[1, 0].contourf(X, Y, Z_true[0], levels=_CLEVELS, vmin=vmin0, vmax=vmax0)
        plt.colorbar(p1, ax=ax[1, 0])

        ax[1, 1].set_aspect('equal', adjustable='box')
        ax[1, 1].set_title("True XY stress", fontsize=_FONTSIZE)
        p2 = ax[1, 1].contourf(X, Y, Z_true[1], levels=_CLEVELS, vmin=vmin1, vmax=vmax1)
        plt.colorbar(p2, ax=ax[1, 1])

        ax[1, 2].set_aspect('equal', adjustable='box')
        ax[1, 2].set_title("True YY stress", fontsize=_FONTSIZE)
        p4 = ax[1, 2].contourf(X, Y, Z_true[3], levels=_CLEVELS, vmin=vmin3, vmax=vmax3)
        plt.colorbar(p4, ax=ax[1, 2])



        ax[2, 0].set_aspect('equal', adjustable='box')
        ax[2, 0].set_title("Abs. error of XX stress", fontsize=_FONTSIZE)
        p1 = ax[2, 0].contourf(X, Y, jnp.abs(Z[0]-Z_true[0]), levels=_CLEVELS)
        plt.colorbar(p1, ax=ax[2, 0])

        ax[2, 1].set_aspect('equal', adjustable='box')
        ax[2, 1].set_title("Abs. error of XY stress", fontsize=_FONTSIZE)
        p2 = ax[2, 1].contourf(X, Y, jnp.abs(Z[1]-Z_true[1]), levels=_CLEVELS)
        plt.colorbar(p2, ax=ax[2, 1])

        ax[2, 2].set_aspect('equal', adjustable='box')
        ax[2, 2].set_title("Abs. error of YY stress", fontsize=_FONTSIZE)
        p4 = ax[2, 2].contourf(X, Y, jnp.abs(Z[3]-Z_true[3]), levels=_CLEVELS)
        plt.colorbar(p4, ax=ax[2, 2])



        save_fig(fig_dir, name, extension)
        plt.clf()
            



    def eval(self):
        pass
