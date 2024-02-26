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
from utils.utils import out_shape, generate_circle_points, generate_rectangle_points


def rm_points(arr: jnp.ndarray, fun: Callable) -> jnp.ndarray:
    print(jnp.invert(fun(arr)).shape)
    return arr[jnp.invert(fun(arr))].copy()

def plot_circle(ax, radius: float, resolution: int) -> None:
    theta = jnp.linspace(0, 2*jnp.pi, resolution+1)
    x = radius * jnp.cos(theta)
    y = radius * jnp.sin(theta)
    ax.plot(x, y, color="red")
    return

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

    def forward(self, params, x: jnp.ndarray) -> jnp.ndarray:
        return self.net.apply(params, x)

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.net.apply(self.params, x)
    
    def hessian(self, x):
        u = self.forward
        hess = jax.hessian(u, argnums=1)
        v_bc = jax.vmap(hess, in_axes=(None, 0))
        return v_bc(self.params, x).reshape((-1, 4))
    
    def laplacian(self, model) -> Callable:
        hess = jax.hessian(model, argnums=1)
        tr = lambda p, xx: jnp.trace(hess(p, xx), axis1=1, axis2=2)
        return tr
    
    def diagonal(self, model) -> Callable:
        hess = jax.hessian(model, argnums=1)
        diag = lambda p, xx: jnp.diagonal(hess(p, xx), axis1=1, axis2=2)
        return diag

    def rect_bc0(self, params, x: jnp.ndarray, u_bc: jnp.ndarray) -> float:
        u = self.forward(params, x)
        u_bc = u_bc.reshape(u.shape)
        return ((u - u_bc)**2).mean()

    def rect_bc2(self, params, x: jnp.ndarray, u_bc: jnp.ndarray) -> float:
        # u = self.forward
        # lap = self.laplacian(u)
        # vBC = jax.vmap(lap, in_axes=(None, 0))
        u = self.forward
        hess = jax.hessian(u, argnums=1)
        v_bc = jax.vmap(hess, in_axes=(None, 0))
        out = v_bc(params, x).reshape((-1, 4))
        # print("OUT SHAPE:", out.shape)
        # print("UBC SHAPE:", u_bc.shape)
        u_bc = u_bc.reshape(out.shape)
        return ((out - u_bc[:, [3, 1, 2, 0]])**2).mean(axis=(0, 1)) # switch xx and yy stresses according to eq. 7.5 (a-b)
    
    def circle_bc2(self, params, x: jnp.ndarray, u_bc: jnp.ndarray) -> float:
        u = self.forward
        hess = jax.hessian(u, argnums=1)
        v_bc = jax.vmap(hess, in_axes=(None, 0))
        out = v_bc(params, x).reshape((-1, 4))
        # print("OUT SHAPE:", out.shape)
        # print("UBC SHAPE:", u_bc.shape)
        u_bc = u_bc.reshape(out.shape)
        return ((out - u_bc[:, [3, 1, 2, 0]])**2).mean(axis=(0, 1))

    def rhs(self, x):
        return jnp.zeros((x.shape[0], 1))#4 * jnp.multiply(jnp.sin(x[:, 0]), jnp.sin(x[:, 1]))
    
    # def PDE(self, params, x: jnp.ndarray):
    #     y_model = self.forward
    #     hess = jax.hessian(y_model, argnums=1)
    #     tr = lambda p, xx: jnp.trace(hess(p, xx), axis1=1, axis2=2)
    #     hess2 = jax.hessian(tr, argnums=1)
    #     tr2 = lambda p, xx: jnp.trace(hess2(p, xx), axis1=1, axis2=2)
    #     vPDE = jax.vmap(tr2, in_axes=(None, 0))
    #     return (jnp.square(vPDE(params, x).ravel() - self.RHS(x).ravel())).mean()
    def pde(self, params, x: jnp.ndarray):
        y_model = self.forward
        lap = self.laplacian(y_model)
        lap2 = self.laplacian(lap)
        vPDE = jax.vmap(lap2, in_axes=(None, 0))
        # print(vPDE(params, x).shape)
        return (jnp.square(vPDE(params, x).ravel() - self.rhs(x).ravel())).mean()
    
    def loss(self,
             params,
             x: tuple[jnp.ndarray],
             u_bc: tuple[jnp.ndarray],
             upp_bc: tuple[jnp.ndarray]
             ) -> float:
        xp, xr, xc = x
        ur, uc = u_bc
        urpp, ucpp = upp_bc
        return self.rect_bc0(params, xr, ur) + \
               self.rect_bc2(params, xr, urpp) + \
               self.circle_bc2(params, xc, ucpp) + \
               self.pde(params, xp)
    
    @partial(jax.jit, static_argnums=(0,))
    def update(self,
               params,
               opt_state,
               x: tuple[jnp.ndarray],
               u_bc: tuple[jnp.ndarray],
               upp_bc: tuple[jnp.ndarray]
               ):
        loss, grads = jax.value_and_grad(self.loss)(params, x, u_bc, upp_bc)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    
    def train(self,
              max_iter: int,
              print_every: int,
              x: tuple[jnp.ndarray],
              u_bc: tuple[jnp.ndarray],
              upp_bc: tuple[jnp.ndarray]
              ) -> None:
        for i in range(max_iter):
            self.params, self.opt_state = self.update(self.params, self.opt_state, x, u_bc, upp_bc)
            if (i % print_every == 0):
                print(i)
        return
    
if __name__ == "__main__":
    raw_settings = parse_arguments()
    network_settings = raw_settings["model"]["pinn"]["network"]
    MLP_instance = setup_network(network_settings)
    fig_dir = raw_settings["IO"]["figure_dir"]
    print(network_settings)

    CLEVELS = 100
    CTICKS = [t for t in jnp.linspace(-2, 2, 41)]
    SEED = 1234
    key = jax.random.PRNGKey(SEED)

    # [xx, xy, yx, yy]
    OUTER_STRESS = ([0, 0], [0, 1])
    INNER_STRESS = ([0, 0], [0, 0])

    XLIM = [-1, 1]
    YLIM = [-1, 1]
    RADIUS = 0.25

    true_sol = lambda x, y: jnp.zeros_like(x) # jnp.multiply(jnp.sin(x), jnp.sin(y)).reshape(x.shape)
    true_lap = lambda x, y: jnp.zeros_like(x) # -2*jnp.multiply(jnp.sin(x), jnp.sin(y)).reshape(x.shape)

    num_bc = 100
    num_c = 10000
    num_test = 100

    # Per dimension
    shape_bc = (num_bc, 1)
    shape_stress = (num_bc, 4)
    shape_pde = (num_c, 1)
    shape_test = (num_test, 1)

    # x_BC = jnp.concatenate([x_BC1, x_BC2, x_BC3, x_BC4])
    # y_BC = jnp.concatenate([y_BC1, y_BC2, y_BC3, y_BC4])
    # BC = jnp.stack([x_BC, y_BC], axis=1).reshape((-1,2))


    # POINTS
    rkey, ckey, key = jax.random.split(key, 3)
    rect_points = generate_rectangle_points(rkey, XLIM, YLIM, num_bc) 
    circ_points = generate_circle_points(ckey, RADIUS, num_bc)
    
    xr = jnp.concatenate(rect_points)
    xc = circ_points

    key, x_key, y_key, x_key_test, y_key_test = jax.random.split(key, 5)

    x_train = jax.random.uniform(x_key, shape_pde, minval=XLIM[0], maxval=XLIM[1])
    y_train = jax.random.uniform(y_key, shape_pde, minval=YLIM[0], maxval=YLIM[1])
    xp = jnp.stack([x_train, y_train], axis=1).reshape((-1,2))

    x_test = jax.random.uniform(x_key_test, shape_test, minval=XLIM[0], maxval=XLIM[1])
    y_test = jax.random.uniform(y_key_test, shape_test, minval=YLIM[0], maxval=YLIM[1])
    xt = jnp.stack([x_test, y_test], axis=1).reshape((-1,2))

    # Remove sample points inside circle
    xp = rm_points(xp, lambda p: jnp.linalg.norm(p, axis=-1) <= RADIUS)
    xt = rm_points(xt, lambda p: jnp.linalg.norm(p, axis=-1) <= RADIUS)

    x = (xt, xr, xc)

    # FUNCTIONS VALUES
    rect_stress = (jnp.concatenate([jnp.full((num_bc, 1), OUTER_STRESS[i][j]) for i in range(2) for j in range(2)], axis=1),
                   jnp.concatenate([jnp.full((num_bc, 1), 0) for i in range(2) for j in range(2)], axis=1),
                   jnp.concatenate([jnp.full((num_bc, 1), OUTER_STRESS[i][j]) for i in range(2) for j in range(2)], axis=1),
                   jnp.concatenate([jnp.full((num_bc, 1), 0) for i in range(2) for j in range(2)], axis=1))

    circ_stress = jnp.concatenate([jnp.full((num_bc, 1), 0) for i in range(2) for j in range(2)], axis=1)

    urpp = jnp.concatenate(rect_stress)
    ucpp = circ_stress
    upp = (urpp, ucpp)
    
    ur = jnp.zeros((num_bc*4, 1))
    uc = jnp.zeros((num_bc, 1))
    u = (ur, uc)
    
    # Plot training points
    plt.scatter(xp[:, 0], xp[:, 1])
    [plt.scatter(bc[:, 0], bc[:, 1], c='green') for bc in rect_points]
    plt.scatter(circ_points[:, 0], circ_points[:, 1], c='red')
    save_fig(fig_dir, "training_points", format="png")
    plt.clf()

    # Plot test points
    plt.scatter(xt[:,0], xt[:,1], c = 'green')
    save_fig(fig_dir, "test_points", format="png")
    plt.clf()

    # Create and train PINN
    pinn = PPINN(net=MLP_instance)
    

    t1 = perf_counter()
    pinn.train(20000, 1000, x, u, upp)
    t2 = perf_counter()
    print(f"Time: {t2-t1:.2f}")
    
    # Print MSE based on test points
    print(f"MSE: {((pinn.predict(xt).ravel() - true_sol(xt[:,0], xt[:,1]).flatten())**2).mean():2.2e}")

    x = jnp.arange(XLIM[0], XLIM[1], 0.01)
    y = jnp.arange(YLIM[0], YLIM[1], 0.01)
    # x = jnp.linspace(-0.5*jnp.pi, 1.5*jnp.pi, 501)
    # y = jnp.linspace(-0.5*jnp.pi, 1.5*jnp.pi, 501)
    X, Y = jnp.meshgrid(x, y)
    plotpoints = np.concatenate((X.reshape((-1, 1)), Y.reshape((-1, 1))), axis=1)
    z = pinn.predict(plotpoints)
    Z = z.reshape((x.size, y.size))
    zpp = pinn.hessian(plotpoints)
    Zpp = [zpp[:, i].reshape((x.size, y.size)) for i in range(4)]

    fig, ax = plt.subplots(1, 3, figsize=(22, 5))
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_title("True solution")
    p1 = ax[0].contourf(X, Y, true_sol(X, Y), levels=CLEVELS)
    plt.colorbar(p1, ax=ax[0], ticks=CTICKS)

    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title("Prediction")
    p2 = ax[1].contourf(X, Y, Z, levels=CLEVELS)
    plt.colorbar(p2, ax=ax[1]) #, ticks=CTICKS)
    plot_circle(ax[1], RADIUS, 100)
    
    ax[2].set_aspect('equal', adjustable='box')
    ax[2].set_title("Abs. error")
    p3 = ax[2].contourf(X, Y, jnp.abs(Z - true_sol(X, Y)), levels=CLEVELS)
    plt.colorbar(p3, ax=ax[2])
    
    save_fig(fig_dir, "subplot", "png")
    plt.clf()

    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    ax[0, 0].set_aspect('equal', adjustable='box')
    ax[0, 0].set_title("XX")
    p1 = ax[0, 0].contourf(X, Y, Zpp[0], levels=CLEVELS)
    plt.colorbar(p1, ax=ax[0, 0])

    ax[0, 1].set_aspect('equal', adjustable='box')
    ax[0, 1].set_title("XY")
    p2 = ax[0, 1].contourf(X, Y, Zpp[1], levels=CLEVELS)
    plt.colorbar(p2, ax=ax[0, 1])
    
    ax[1, 0].set_aspect('equal', adjustable='box')
    ax[1, 0].set_title("YX")
    p3 = ax[1, 0].contourf(X, Y, Zpp[2], levels=CLEVELS)
    plt.colorbar(p3, ax=ax[1, 0])

    ax[1, 1].set_aspect('equal', adjustable='box')
    ax[1, 1].set_title("YY")
    p4 = ax[1, 1].contourf(X, Y, Zpp[3], levels=CLEVELS)
    plt.colorbar(p4, ax=ax[1, 1])

    [plot_circle(ax[i, j], RADIUS, 100) for i in range(2) for j in range(2)]

    save_fig(fig_dir, "stresses", "png")
