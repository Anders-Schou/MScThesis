from functools import partial
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from models.derivatives import biharmonic, hessian
from models.loss import ms, mse, sq, sqe
from models.networks import deeponetmap
import models.platewithhole.loss as pwhloss
from . import PlateWithHoleDeepONet

class BiharmonicDeepONet(PlateWithHoleDeepONet):
    net_bi: nn.Module
    optimizer: optax.GradientTransformation

    def __init__(self, settings: dict):
        super().__init__(settings)
        # Call init_model() which must be implemented and set the params
        self.init_model(settings["model"]["deeponet"]["networks"])
        self._set_update(loss_fun_name="_total_loss", optimizer_name="optimizer")

        # Only use one branch network
        self.branch = self.branch_nets[0]
        self.trunk = self.trunk_net
        self.opt_state = self.optimizer.init(self.params)

        return
    
    def forward(self, params, branch_input: jax.Array, trunk_input: jax.Array):
        """
        Function defining the forward pass of the model. The same as phi in this case.
        """
        return jnp.dot(self.branch_forward(params, branch_input), self.trunk_forward(params, trunk_input))
    
    def branch_forward(self, params, input: jax.Array) -> jax.Array:
        return self.branch.apply(params["branch0"], input)

    def trunk_forward(self, params, input: jax.Array) -> jax.Array:
        return self.trunk.apply(params["trunk"], input)

    def biharmonic(self, params, branch_input: jax.Array, trunk_input: jax.Array) -> jax.Array:
        return biharmonic(self.forward, argnums=2)(params, branch_input, trunk_input)
    
    def loss_coll(self, params, branch_input: jax.Array, trunk_input: jax.Array, true_val: jax.Array | None = None):
        """
        Computes the loss of the PDE residual on the domain.
        """
        
        # Compute biharmonic values
        bi_out = deeponetmap(self.biharmonic)(params, branch_input, trunk_input)

        # Return loss
        if true_val is None:
            return self.loss_fn(bi_out)
        return self.loss_fn(bi_out, true_val)

    def resample_eval(self,
                      params,
                      branch_input: jax.Array, 
                      trunk_input: jax.Array,
                      true_val: jax.Array | None = None
                      ) -> jax.Array:
        """
        Function for evaluating loss in resampler.
        Should return the loss for each input point, i.e.
        not aggregating them with sum, mean, etc.
        """

        # Compute biharmonic values
        bi_out = deeponetmap(self.biharmonic)(params, branch_input, trunk_input)

        # Return loss
        if true_val is None:
            return sq(bi_out)
        return sqe(bi_out, true_val)