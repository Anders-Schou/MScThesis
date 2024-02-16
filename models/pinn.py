from dataclasses import dataclass
from models.networks import MLP
import jax, jax.numpy as jnp
import flax.linen as nn
from jax import grad
import optax
from functools import partial

class Pinn:
    net: MLP
    loss_fn: callable
    optimizer: callable
    
    def __init__(self, num_neurons_per_layer=[1], activation=jnp.tanh, loss_fn=None, lr=1e-4):
        self.net = MLP(num_neurons_per_layer=num_neurons_per_layer, activation=activation)
        self.params = self.net.init(jax.random.key(0), jnp.ones([1, num_neurons_per_layer[0]]))
        self.optimizer = optax.adam(learning_rate=lr)
        self.opt_state = self.optimizer.init(self.params)
        # loss_fn = loss_fn
        return None

    def forward(self, params, x):
        return self.net.apply(params, x)

    def predict(self, x):
        return self.net.apply(self.params, x)
    
    def BC(self, params, x, y):
        y_model = self.forward(params, x)
        y = y.reshape(y_model.shape)
        return ((y_model - y)**2).mean()
    
    def PDE(self, params, x):
        y_model = self.forward
        d1 = jax.vmap(jax.jacrev(y_model, 1), in_axes=(None,0))
        return ((jnp.sum(d1(params, x),axis=2))**2).mean()
    
    def loss(self, params, x_PDE, x_BC, y_BC):
        return self.BC(params, x_BC, y_BC) + self.PDE(params, x_PDE)
    
    @partial(jax.jit, static_argnums=(0))
    def update(self, params, opt_state, x_PDE, x_BC, y_BC):
        loss, grads = jax.value_and_grad(self.loss)(params, x_PDE, x_BC, y_BC)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    
    def train(self, max_iter, print_every, x_PDE, x_BC, y_BC):
        for i in range(max_iter):
            self.params, self.opt_state = self.update(self.params, self.opt_state, x_PDE, x_BC, y_BC)
            if (i % print_every == 0):
                print(i)
        return self.predict(x_BC)
    
    pass

class MPinn:
    pass

@dataclass
class XPinn:
    net: MLP