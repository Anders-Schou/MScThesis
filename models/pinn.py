from functools import partial
from collections.abc import Callable

import jax
import optax
import jax.numpy as jnp

from models.networks import MLP


class PINN:
    net: MLP
    loss_fn: Callable
    optimizer: Callable
    
    def __init__(self, net: MLP = None, loss_fn = None, lr = 1e-4):
        self.net = net
        self.params = self.net.init(jax.random.key(0), jnp.ones((1, net.input_dim)))
        self.optimizer = optax.adam(learning_rate=lr)
        self.opt_state = self.optimizer.init(self.params)
        # loss_fn = loss_fn
        return None

    def forward(self, params, x: jnp.ndarray):
        return self.net.apply(params, x)

    def predict(self, x: jnp.ndarray):
        return self.net.apply(self.params, x)
    
    def BC(self, params, x: jnp.ndarray, y: jnp.ndarray):
        y_model = self.forward(params, x)
        y = y.reshape(y_model.shape)
        return ((y_model - y)**2).mean()
    
    def PDE(self, params, x: jnp.ndarray):
        y_model = self.forward
        d1 = jax.vmap(jax.jacrev(y_model, 1), in_axes=(None,0))
        return ((jnp.sum(d1(params, x),axis=2))**2).mean()
    
    def loss(self, params, x_PDE: jnp.ndarray, x_BC: jnp.ndarray, y_BC: jnp.ndarray):
        return self.BC(params, x_BC, y_BC) + self.PDE(params, x_PDE)
    
    @partial(jax.jit, static_argnums=(0))
    def update(self, params, opt_state, x_PDE: jnp.ndarray, x_BC: jnp.ndarray, y_BC: jnp.ndarray):
        loss, grads = jax.value_and_grad(self.loss)(params, x_PDE, x_BC, y_BC)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    
    def train(self, max_iter: int, print_every: int, x_PDE: jnp.ndarray, x_BC: jnp.ndarray, y_BC: jnp.ndarray):
        for i in range(max_iter):
            self.params, self.opt_state = self.update(self.params, self.opt_state, x_PDE, x_BC, y_BC)
            if (i % print_every == 0):
                print(i)
        return self.predict(x_BC)
    
    pass
