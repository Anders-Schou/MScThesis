import os
from functools import partial
from collections.abc import Callable
from time import perf_counter

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter

from models.networks import MLP, setup_network, setup_run
from models.derivatives import diff_xx, diff_xy, diff_yy, biharmonic
from datahandlers.generators import generate_circle_points, generate_rectangle_points
from setup.parsers import parse_arguments
from utils.plotting import save_fig, plot_potential, plot_stress, plot_circle, get_plot_variables
from utils.transforms import cart2polar_tensor, xy2r
from utils.platewithhole import cart_sol_true, cart_stress_true
from utils.utils import out_shape, remove_points

class PPINN:
    net: MLP
    loss_fn: Callable
    optimizer: Callable
    
    def __init__(self, net: MLP, run_settings: dict, loss_fn = None):
        self.net = net
        self.params = self.net.init(jax.random.key(0), jnp.ones((1, net.input_dim)))
        self.schedule = optax.exponential_decay(run_settings["learning_rate"],
                                                run_settings["decay_steps"],
                                                run_settings["decay_rate"])
        self.optimizer = run_settings["optimizer"](learning_rate=self.schedule)
        self.opt_state = self.optimizer.init(self.params)
        self.loss_fn = loss_fn
        # self.writer = SummaryWriter(log_dir="/zhome/e8/9/147091/MSc/results/logs")
        return None

    def forward(self, params, x: jnp.ndarray) -> jnp.ndarray:
        return self.net.apply(params, x)

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.net.apply(self.params, x)
    
    def hessian(self, params, x):
        u = self.forward
        hess = jax.hessian(u, argnums=1)
        v_bc = jax.vmap(hess, in_axes=(None, 0))
        return v_bc(params, x)
    
    def hessian_flatten(self, params, x):
        return self.hessian(params, x).reshape((-1, 4))
    
    def laplacian(self, model) -> Callable:
        hess = jax.hessian(model, argnums=1)
        tr = lambda p, xx: jnp.trace(hess(p, xx), axis1=1, axis2=2)
        return tr
    
    def biharmonic(self, model) -> Callable:
        lap = self.laplacian(model)
        lap2 = self.laplacian(lap)
        vPDE = jax.vmap(lap2, in_axes=(None, 0))
        return vPDE
    
    def diagonal(self, model) -> Callable:
        hess = jax.hessian(model, argnums=1)
        diag = lambda p, xx: jnp.diagonal(hess(p, xx), axis1=1, axis2=2)
        return diag

    def rect_bc0(self, params, x: jnp.ndarray, u_bc: jnp.ndarray) -> float:
        u = self.forward(params, x)
        u_bc = u_bc.reshape(u.shape)
        return ((u - u_bc).ravel()**2).mean()

    def rect_bc2h(self, params, x):
        out = self.hessian(params, x)
        return ((out[:, 0, 0] - 10)**2).mean() + ((out[:, 0, 1])**2).mean()

    def rect_bc2v(self, params, x):
        out = self.hessian(params, x)
        return ((out[:, 1, 0])**2).mean() + ((out[:, 1, 1])**2).mean()
    
    def rect_bc2(self, params, x):
        out0 = self.hessian(params, x[0]) # horizontal lower
        out1 = self.hessian(params, x[1]) # vertical right
        out2 = self.hessian(params, x[2]) # horizontal upper
        out3 = self.hessian(params, x[3]) # vertical left
        hl = ((out0[:, 0, 0, 0]     )**2).mean() + ((out0[:, 0, 0, 1])**2).mean() # horizontal lower
        vr = ((out1[:, 0, 1, 1] - 10)**2).mean() + ((out1[:, 0, 1, 0])**2).mean() # vertical right
        hu = ((out2[:, 0, 0, 0]     )**2).mean() + ((out2[:, 0, 0, 1])**2).mean() # horizontal upper
        vl = ((out3[:, 0, 1, 1] - 10)**2).mean() + ((out3[:, 0, 1, 0])**2).mean() # vertical left
        return hl + vr + hu + vl

    def circle_bc2(self, params, x: jnp.ndarray) -> float:
        out = self.hessian(params, x)
        rtheta_stress = jax.vmap(cart2polar_tensor, in_axes=(0, 0))(out.reshape(-1, 2, 2), x)
        return ((rtheta_stress[:, 0, 0].ravel())**2).mean() + ((rtheta_stress[:, 0, 1].ravel())**2).mean()

    def rhs(self, x):
        return jnp.zeros((x.shape[0], 1))#4 * jnp.multiply(jnp.sin(x[:, 0]), jnp.sin(x[:, 1]))
    
    def pde(self, params, x: jnp.ndarray):
        phi = self.forward
        vPDE = self.biharmonic(phi)
        return (jnp.square(vPDE(params, x).ravel())).mean() # Homogeneous RHS
    
    def loss(self,
              params,
              x: tuple[jnp.ndarray]
              ) -> float:
        xp, xr, xc = x
        # ur, uc, up = u_bc
        # urpp, ucpp = upp_bc
        
        # return (jnp.square(jnp.subtract(self.forward(params, xp), up))).mean()
        return self.circle_bc2(params, xc) + self.pde(params, xp) + self.rect_bc2(params, xr)
        # return self.rect_bc2h(params, xr[0]) + \
        #        self.rect_bc2v(params, xr[1]) + \
        #        self.rect_bc2h(params, xr[2]) + \
        #        self.rect_bc2v(params, xr[3]) + \
        #        self.circle_bc2(params, xc, ucpp) + \
        #        self.pde(params, xp)
    
    @partial(jax.jit, static_argnums=(0,))
    def update(self,
               params,
               opt_state,
               x: tuple[jnp.ndarray],
               u_bc: tuple[jnp.ndarray],
               upp_bc: tuple[jnp.ndarray],
               ):
        loss, grads = jax.value_and_grad(self.loss)(params, x)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    def train(self,
              max_iter: int,
              print_every: int,
              x: tuple[jnp.ndarray],
              u_bc: tuple[jnp.ndarray],
              upp_bc: tuple[jnp.ndarray]
              ) -> None:
        
        losses = np.zeros(int(max_iter/print_every))
        l = 0
        for i in range(max_iter):
            self.params, self.opt_state, loss = self.update(self.params, self.opt_state, x, u_bc, upp_bc)
            if (i % print_every == 0):
                print(f"Epoch {i:>6}    MSE: {loss:2.2e}")
                # losses[l] = loss
                # self.writer.add_scalar("custom_loss", np.array(loss), i)
                l += 1
        
        self.losses = losses
        return
    

def print_info(network_settings, run_settings):
    print("")
    [print(f"{key+":":<30}", value) for key, value in network_settings.items()]
    print("")
    [print(f"{key+":":<30}", value) for key, value in run_settings.items()]
    print("")

if __name__ == "__main__":
    raw_settings = parse_arguments()
    network_settings = raw_settings["model"]["pinn"]["network"]
    MLP_instance = setup_network(network_settings)
    run_settings = setup_run(raw_settings["run"])
    fig_dir = raw_settings["IO"]["figure_dir"]
    do_sample_data_plots = raw_settings["plotting"]["do_sample_data_plots"]
    do_result_plots = raw_settings["plotting"]["do_result_plots"]
    print_info(network_settings, run_settings)
    

    CLEVELS = 101
    CTICKS = [t for t in jnp.linspace(-2, 2, 41)]
    SEED = 1234
    key = jax.random.PRNGKey(SEED)

    TENSION = 10
    R = 10
    XLIM = [-R, R]
    YLIM = [-R, R]
    RADIUS = 2.0



    
    A = -TENSION*RADIUS**2/2
    B = 0
    C = TENSION*RADIUS**2/2
    D = -TENSION*RADIUS**4/4
    true_sol_polar = lambda r, theta: ((TENSION*r**2/4 - TENSION*r**2*jnp.cos(2*theta)/4) + A*jnp.log(r) + B*theta + C*jnp.cos(2*theta) + D*jnp.pow(r,-2)*jnp.cos(2*theta))
    true_sol_cart = lambda x, y: true_sol_polar(jnp.sqrt(x**2 + y**2), jnp.arctan2(y, x))
    



    num_r = 1000
    num_c = 2000
    num_pde = 20000
    num_test = 100

    # Per dimension
    shape_bc = (num_r, 1)
    shape_stress = (num_r, 4)
    shape_pde = (num_pde, 1)
    shape_test = (num_test, 1)

    # x_BC = jnp.concatenate([x_BC1, x_BC2, x_BC3, x_BC4])
    # y_BC = jnp.concatenate([y_BC1, y_BC2, y_BC3, y_BC4])
    # BC = jnp.stack([x_BC, y_BC], axis=1).reshape((-1,2))


    # POINTS
    rkey, ckey, key = jax.random.split(key, 3)
    rect_points = generate_rectangle_points(rkey, XLIM, YLIM, num_r) 
    circ_points = generate_circle_points(ckey, RADIUS, num_c)

    xr = jnp.concatenate(rect_points)
    
    # [print(jax.vmap(partial(cart_stress_true, a=RADIUS))(r[:5])) for r in rect_points]
    # exit()
    xc = circ_points

    key, x_key, y_key, x_key_test, y_key_test = jax.random.split(key, 5)

    x_train = jax.random.uniform(x_key, shape_pde, minval=XLIM[0], maxval=XLIM[1])
    y_train = jax.random.uniform(y_key, shape_pde, minval=YLIM[0], maxval=YLIM[1])
    xp = jnp.stack([x_train, y_train], axis=1).reshape((-1,2))

    x_test = jax.random.uniform(x_key_test, shape_test, minval=XLIM[0], maxval=XLIM[1])
    y_test = jax.random.uniform(y_key_test, shape_test, minval=YLIM[0], maxval=YLIM[1])
    xt = jnp.stack([x_test, y_test], axis=1).reshape((-1,2))

    # Remove sample points inside circle
    xp = remove_points(xp, lambda p: jnp.linalg.norm(p, axis=-1) <= RADIUS)
    xt = remove_points(xt, lambda p: jnp.linalg.norm(p, axis=-1) <= RADIUS)

    x = (xp, rect_points, xc)
   
    u_stress = jnp.full((num_r, 1), TENSION)
    u_zero = jnp.zeros((num_c, 1))
    rect_stress = (u_stress, u_zero, u_stress, u_zero)

    circ_stress = jnp.concatenate([jnp.full((num_c, 1), 0) for i in range(2) for j in range(2)], axis=1)

    urpp = jnp.concatenate(rect_stress)
    ucpp = circ_stress
    upp = (urpp, ucpp)
    
    # ur = jnp.zeros((num_bc*4, 1))
    ur = cart_sol_true(xr[:, 0], xr[:, 1], S=TENSION, a=RADIUS).reshape(-1, 1)
    uc = jnp.zeros((num_c, 1))
    up = cart_sol_true(xp[:, 0], xp[:, 1], S=TENSION, a=RADIUS).reshape(-1, 1)
    u = (ur, uc, up)
    
    # exit()
    if do_sample_data_plots:
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
    print("Creating PINN:")
    pinn = PPINN(MLP_instance, run_settings["specifications"])
    print("PINN created!")

    # Collect all data in list of tuples (1 tuple for each loss term)
    # losses = [jax.vmap(biharmonic, in_axes=(None, 0)), ]
    # batch = [(xp, up), (xc, uc), *[(x, u) for x, u in zip(rect_points, rect_stress)]]

    # for b in batch:
    #     print("TYPE:      ", type(b))
    #     print("LENGTH:    ", len(b))
    #     print("SHAPE:     ", b[0].shape, b[1].shape)

    print("Entering training phase:")
    t1 = perf_counter()
    pinn.train(run_settings["specifications"]["iterations"], 200, x, u, upp)
    t2 = perf_counter()
    print("Training done!")
    print(f"Time: {t2-t1:.2f}")
    

    if do_result_plots:
        X, Y, plotpoints = get_plot_variables(XLIM, YLIM, grid=201)
        
        z = pinn.predict(plotpoints)
        Z = z.reshape(X.shape)
        #Ztrue = cart_sol_true(X, Y, S=TENSION, a=RADIUS)*(xy2r(X, Y) >= RADIUS)
        Ztrue = true_sol_cart(X, Y)*(xy2r(X, Y) >= RADIUS)

        zpp = pinn.hessian_flatten(pinn.params, plotpoints)
        Zpp = [zpp[:, i].reshape(X.shape)*(xy2r(X, Y) >= RADIUS) for i in range(4)]
        Zhesstrue = jax.vmap(cart_stress_true)(plotpoints)
        Zpptrue = [Zhesstrue.reshape(-1, 4)[:, i].reshape(X.shape)*(xy2r(X, Y) >= RADIUS) for i in [3, 1, 2, 0]]
        
        plot_potential(X, Y, Z, Ztrue, fig_dir, "potential", radius=RADIUS)
        plot_stress(X, Y, Zpp, Zpptrue, fig_dir, "stress", radius=RADIUS)


