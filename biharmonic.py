import os
from functools import partial
from collections.abc import Callable
from time import perf_counter

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

from models.networks import setup_network
from setup.parsers import parse_arguments
from utils.plotting import save_fig
from models.networks import MLP
from utils.utils import out_shape


def test_fun(a, b):
    return a + b


class PPINN:
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
    
    def laplacian(self, model):
        hess = jax.hessian(model, argnums=1)
        tr = lambda p, xx: jnp.trace(hess(p, xx), axis1=1, axis2=2)
        return tr

    def BC(self, params, x: jnp.ndarray, y: jnp.ndarray):
        y_model = self.forward(params, x)
        y = y.reshape(y_model.shape)
        return ((y_model - y)**2).mean()

    def BC2(self, params, x: jnp.ndarray, y: jnp.ndarray):
        y_model = self.forward
        lap = self.laplacian(y_model)
        vBC = jax.vmap(lap, in_axes=(None, 0))
        out = vBC(params, x)
        y = y.reshape(out.shape)
        return ((out - y)**2).mean()
    
    def RHS(self, x):
        return 4 * jnp.multiply(jnp.sin(x[:, 0]), jnp.sin(x[:, 1]))
    
    # def PDE(self, params, x: jnp.ndarray):
    #     y_model = self.forward
    #     hess = jax.hessian(y_model, argnums=1)
    #     tr = lambda p, xx: jnp.trace(hess(p, xx), axis1=1, axis2=2)
    #     hess2 = jax.hessian(tr, argnums=1)
    #     tr2 = lambda p, xx: jnp.trace(hess2(p, xx), axis1=1, axis2=2)
    #     vPDE = jax.vmap(tr2, in_axes=(None, 0))
    #     return (jnp.square(vPDE(params, x).ravel() - self.RHS(x).ravel())).mean()
    def PDE(self, params, x: jnp.ndarray):
        y_model = self.forward
        lap = self.laplacian(y_model)
        lap2 = self.laplacian(lap)
        vPDE = jax.vmap(lap2, in_axes=(None, 0))
        return (jnp.square(vPDE(params, x).ravel() - self.RHS(x).ravel())).mean()
    
    def loss(self, params, x_PDE: jnp.ndarray, x_BC: jnp.ndarray, true_BC: jnp.ndarray):
        u_BC = true_BC[0]
        upp_BC = true_BC[1]
        return self.BC(params, x_BC, u_BC) + self.BC2(params, x_BC, upp_BC) + self.PDE(params, x_PDE)
    
    @partial(jax.jit, static_argnums=(0))
    def update(self, params, opt_state, x_PDE: jnp.ndarray, x_BC: jnp.ndarray, u_BC: jnp.ndarray):
        loss, grads = jax.value_and_grad(self.loss)(params, x_PDE, x_BC, u_BC)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    
    def train(self, max_iter: int, print_every: int, x_PDE: jnp.ndarray, x_BC: jnp.ndarray, u_BC: jnp.ndarray):
        for i in range(max_iter):
            self.params, self.opt_state = self.update(self.params, self.opt_state, x_PDE, x_BC, u_BC)
            if (i % print_every == 0):
                print(i)
        return self.predict(x_BC)
    
    pass

if __name__ == "__main__":
    raw_settings = parse_arguments()
    network_settings = raw_settings["model"]["pinn"]["network"]
    MLP_instance = setup_network(network_settings)
    fig_dir = raw_settings["IO"]["figure_dir"]
    print(network_settings)

    CLEVELS = 100
    CTICKS = [t for t in jnp.linspace(0, 1, 11)]
    seed = 1234
    key = jax.random.PRNGKey(seed)

    xlim = [0, jnp.pi]
    ylim = [0, jnp.pi]

    true_sol = lambda x, y: jnp.multiply(jnp.sin(x), jnp.sin(y)).reshape(x.shape)
    true_lap = lambda x, y: -2*jnp.multiply(jnp.sin(x), jnp.sin(y)).reshape(x.shape)

    num_bc = 100
    num_c = 10000
    num_test = 100

    # Per dimension
    shape_bc = (num_bc, 1)
    shape_pde = (num_c, 1)
    shape_test = (num_test, 1)

    x_BC1 = jax.random.uniform(jax.random.PRNGKey(0), shape_bc, minval=xlim[0], maxval=xlim[1])
    y_BC1 = jnp.full(shape_bc, ylim[0])

    x_BC2 = jnp.full(shape_bc, xlim[1])
    y_BC2 = jax.random.uniform(jax.random.PRNGKey(0), shape_bc, minval=ylim[0], maxval=ylim[1])

    x_BC3 = jax.random.uniform(jax.random.PRNGKey(0), shape_bc, minval=xlim[0], maxval=xlim[1])
    y_BC3 = jnp.full(shape_bc, ylim[1])

    x_BC4 = jnp.full(shape_bc, xlim[0])
    y_BC4 = jax.random.uniform(jax.random.PRNGKey(0), shape_bc, minval=ylim[0], maxval=ylim[1])

    x_BC = jnp.concatenate([x_BC1, x_BC2, x_BC3, x_BC4])
    y_BC = jnp.concatenate([y_BC1, y_BC2, y_BC3, y_BC4])

    BC = jnp.stack([x_BC, y_BC], axis=1).reshape((-1,2))

    BC_true = (true_sol(BC[:,0], BC[:,1]), true_lap(BC[:,0], BC[:,1]))

    key, x_key, y_key, x_key_test, y_key_test = jax.random.split(key, 5)

    x_PDE = jax.random.uniform(x_key, shape_pde, minval=xlim[0], maxval=xlim[1])
    y_PDE = jax.random.uniform(y_key, shape_pde, minval=ylim[0], maxval=ylim[1])
    PDE = jnp.stack([x_PDE, y_PDE], axis=1).reshape((-1,2))

    x_test = jax.random.uniform(x_key_test, shape_test, minval=xlim[0], maxval=xlim[1])
    y_test = jax.random.uniform(y_key_test, shape_test, minval=ylim[0], maxval=ylim[1])
    test_points = jnp.stack([x_test, y_test], axis=1).reshape((-1,2))

    # Plot training points
    plt.scatter(PDE[:,0], PDE[:,1])
    plt.scatter(BC[:,0], BC[:,1], c = 'green')
    save_fig(fig_dir, "training_points", format="png")
    plt.clf()

    # Plot test points
    plt.scatter(test_points[:,0], test_points[:,1], c = 'green')
    save_fig(fig_dir, "test_points", format="png")
    plt.clf()

    # Create and train PINN
    pinn = PPINN(net=MLP_instance)
    
    print(pinn.PDE(pinn.params, PDE))
    print(pinn.RHS(PDE).shape)

    # a = jnp.ones((100, 1, 1))
    # # b = jnp.linspace(1, 100, 100).reshape((100, 1))
    # b = jnp.ones((1, 100))
    # print(a.shape)
    # print(b.shape)
    # print(jax.eval_shape(jax.vmap(test_fun, in_axes=(0, 1), out_axes=2), a, b))
    # print(test_fun(a, b)[:5, :5])
    # exit()
    t1 = perf_counter()
    pinn.train(10000, 1000, PDE, BC, BC_true)
    t2 = perf_counter()
    print(f"Time: {t2-t1:.2f}")
    
    # Print MSE based on test points
    print(f"MSE: {((pinn.predict(test_points).ravel() - true_sol(test_points[:,0], test_points[:,1]).flatten())**2).mean():2.2e}")

    x = jnp.arange(xlim[0], xlim[1], 0.01)
    y = jnp.arange(ylim[0], ylim[1], 0.01)
    # x = jnp.linspace(-0.5*jnp.pi, 1.5*jnp.pi, 501)
    # y = jnp.linspace(-0.5*jnp.pi, 1.5*jnp.pi, 501)
    X, Y = jnp.meshgrid(x, y)
    plotpoints = np.concatenate((X.reshape((-1, 1)), Y.reshape((-1, 1))), axis=1)
    z = pinn.predict(plotpoints)
    Z = z.reshape((x.size, y.size))

    fig, ax = plt.subplots(1, 3, figsize=(22, 5))
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_title("True solution")
    p1 = ax[0].contourf(X, Y, true_sol(X, Y), levels=CLEVELS)
    plt.colorbar(p1, ax=ax[0], ticks=CTICKS)

    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title("Prediction")
    p2 = ax[1].contourf(X, Y, Z, levels=CLEVELS)
    plt.colorbar(p2, ax=ax[1], ticks=CTICKS)
    
    ax[2].set_aspect('equal', adjustable='box')
    ax[2].set_title("Abs. error")
    p3 = ax[2].contourf(X, Y, jnp.abs(Z - true_sol(X, Y)), levels=CLEVELS)
    plt.colorbar(p3, ax=ax[2])
    
    save_fig(fig_dir, "subplot", "png")
