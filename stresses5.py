import os
from functools import partial
from collections.abc import Callable
from time import perf_counter

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from models.networks import MLP, setup_network, setup_run
# from models.derivatives import diff_xx, diff_xy, diff_yy, biharmonic
from datahandlers.generators import generate_rectangle_with_hole
from setup.parsers import parse_arguments
from utils.plotting import save_fig, plot_potential, plot_stress, plot_polar_stress, plot_circle, get_plot_variables
from utils.transforms import cart2polar_tensor, xy2r
from utils.transforms import *
from utils.platewithhole import cart_sol_true, cart_stress_true, polar_stress_true
from utils.utils import out_shape, remove_points

class PPINN:
    net: MLP
    loss_fn: Callable
    optimizer: Callable
    
    def __init__(self, net: MLP, run_settings: dict, loss_fn = None, logging = False):
        self.net = net
        self.params = self.net.init(jax.random.key(0), jnp.ones((1, net.input_dim)))
        self.schedule = optax.exponential_decay(run_settings["learning_rate"],
                                                run_settings["decay_steps"],
                                                run_settings["decay_rate"])
        self.optimizer = run_settings["optimizer"](learning_rate=self.schedule)
        self.opt_state = self.optimizer.init(self.params)
        self.loss_fn = loss_fn
        self.logging = logging
        self.writer = SummaryWriter(log_dir="/zhome/e8/9/147091/MSc/results/logs")
        print("Logging:", self.logging)
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

    def rect_bc0(self, params, x: jnp.ndarray, u: jnp.ndarray) -> float:
        out0 = self.forward(params, x[0]) # horizontal lower
        out1 = self.forward(params, x[1]) # vertical right
        out2 = self.forward(params, x[2]) # horizontal upper
        out3 = self.forward(params, x[3]) # vertical left
        hl = ((out0.ravel() - u[0].ravel())**2).mean() # horizontal lower
        vr = ((out1.ravel() - u[1].ravel())**2).mean() # vertical right
        hu = ((out2.ravel() - u[2].ravel())**2).mean() # horizontal upper
        vl = ((out3.ravel() - u[3].ravel())**2).mean() # vertical left
        return hl + vr + hu + vl
    
    def rect_bc2(self, params, x):
        out0 = self.hessian(params, x[0]) # horizontal lower
        out1 = self.hessian(params, x[1]) # vertical right
        out2 = self.hessian(params, x[2]) # horizontal upper
        out3 = self.hessian(params, x[3]) # vertical left
        hl = ((out0[:, 0, 0, 0].ravel()     )**2).mean() + ((out0[:, 0, 0, 1].ravel())**2).mean() # horizontal lower
        vr = ((out1[:, 0, 1, 1].ravel() - 10)**2).mean() + ((out1[:, 0, 1, 0].ravel())**2).mean() # vertical right
        hu = ((out2[:, 0, 0, 0].ravel()     )**2).mean() + ((out2[:, 0, 0, 1].ravel())**2).mean() # horizontal upper
        vl = ((out3[:, 0, 1, 1].ravel() - 10)**2).mean() + ((out3[:, 0, 1, 0].ravel())**2).mean() # vertical left
        return hl + vr + hu + vl

    def circle_bc0(self, params, x, u):
        out = self.forward(params, x)
        return ((out.ravel()-u.ravel())**2).mean()

    def circle_bc2(self, params, x: jnp.ndarray) -> float:
        out = self.hessian_flatten(params, x)
        sigmas = out[:, [3, 1, 2, 0]]
        sigmas = sigmas.at[:, [1, 2]].set(jnp.negative(out[:, [1, 2]]))
        rtheta_stress = jax.vmap(cart2polar_tensor, in_axes=(0, 0))(sigmas.reshape(-1, 2, 2), x)
        return ((rtheta_stress[:, 0, 0].ravel())**2).mean() + ((rtheta_stress[:, 0, 1].ravel())**2).mean()

    def coll(self, params, x, u):
        out = self.forward(params, x)
        return ((out.ravel() - u.ravel())**2).mean()

    def pde(self, params, x: jnp.ndarray):
        phi = self.forward
        vPDE = self.biharmonic(phi)
        return (jnp.square(vPDE(params, x).ravel())).mean() # Homogeneous RHS
    
    def loss(self,
              params,
              x: tuple[jnp.ndarray],
              u: tuple[jnp.ndarray]
              ) -> float:
        xy_coll, xy_rect, xy_circ = x
        u_coll, u_rect, u_circ = u
        # sigma_rect, sigma_circ = sigma_bc
        
        total_loss = self.circle_bc2(params, xy_circ) + \
            2*self.rect_bc2(params, xy_rect) + \
            5*self.pde(params, xy_coll)# + \
            # self.rect_bc0(params, xy_rect, u_rect) + \
            # self.circle_bc0(params, xy_circ, u_circ)
        return total_loss, (self.circle_bc0(params, xy_circ, u_circ),
                            self.rect_bc0(params, xy_rect, u_rect),
                            self.circle_bc2(params, xy_circ),
                            self.rect_bc2(params, xy_rect),
                            self.pde(params, xy_coll))

        # out = self.forward(params, xy_coll)
        # this_loss = ((out.ravel() - u_coll.ravel())**2).mean()
        
        # out = self.forward(params, xy_coll).reshape(u_coll.shape)
        # this_loss = ((out - u_coll)**2).mean()
        


        # total_loss = 5 * self.coll(params, xy_coll, u_coll) + \
        #     self.circle_bc0(params, xy_circ, u_circ) + \
        #     2 * self.rect_bc0(params, xy_rect, u_rect)
        # # total_loss = self.rect_bc0(params, xy_rect, u_rect)
        # # total_loss = self.circle_bc0(params, xy_circ, u_circ)

        # return total_loss, (self.coll(params, xy_coll, u_coll),
        #                     self.rect_bc0(params, xy_rect, u_rect),
        #                     self.circle_bc0(params, xy_circ, u_circ))
    
        # return total_loss, (self.rect_bc0(params, xy_rect, u_rect), 0., 0.)
        # return total_loss, (self.circle_bc0(params, xy_circ, u_circ), 0., 0.)

    @partial(jax.jit, static_argnums=(0,))
    def update(self,
               params,
               opt_state,
               x: tuple[jnp.ndarray],
               u_bc: tuple[jnp.ndarray],
               upp_bc: tuple[jnp.ndarray]):
        (loss, otherloss), grads = jax.value_and_grad(self.loss, argnums=0, has_aux=True)(params, x, u_bc)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, (loss, otherloss)
    
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
                print(f"Epoch {i:>6}    MSE: {loss[0]:2.2e}    (C0 = {loss[1][0]:2.2e}, R0 = {loss[1][1]:2.2e}, C2 = {loss[1][2]:2.2e}, R2 = {loss[1][3]:2.2e}, PDE = {loss[1][4]:2.2e})")
                # print(f"Epoch {i:>6}    MSE: {loss[0]:2.2e}    (Coll = {loss[1][0]:2.2e},    R = {loss[1][1]:2.2e},    C = {loss[1][2]:2.2e})")
                # losses[l] = loss
                
                if self.logging:
                    self.writer.add_scalar("loss/total", np.array(loss[0]   ), i)
                    self.writer.add_scalar("loss/circ0", np.array(loss[1][0]), i)
                    self.writer.add_scalar("loss/rect0", np.array(loss[1][1]), i)
                    self.writer.add_scalar("loss/circ2", np.array(loss[1][2]), i)
                    self.writer.add_scalar("loss/rect2", np.array(loss[1][3]), i)
                    self.writer.add_scalar("loss/pde",   np.array(loss[1][4]), i)
                    self.writer.add_scalar("learning_rate", np.array(self.schedule(i)), i)
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
    do_logging = raw_settings["logging"]
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
    
    # [xx, xy, yx, yy]
    OUTER_STRESS = ([0, 0], [0, TENSION])
    # [rr, rt, tr, tt]
    INNER_STRESS = ([0, 0], [0, 0])



    
    A = -TENSION*RADIUS**2/2
    B = 0
    C = TENSION*RADIUS**2/2
    D = -TENSION*RADIUS**4/4
    true_sol_polar = lambda r, theta: ((TENSION*r**2/4 - TENSION*r**2*jnp.cos(2*theta)/4) + A*jnp.log(r) + B*theta + C*jnp.cos(2*theta) + D*jnp.pow(r,-2)*jnp.cos(2*theta))
    true_sol_cart = lambda x, y: true_sol_polar(jnp.sqrt(x**2 + y**2), jnp.arctan2(y, x))

    
    num_coll = 2500
    num_rBC = 500
    num_cBC = 1000
    num_test = 100
    
    key, subkey = jax.random.split(key, 2)
    
    xy_coll, xy_rect_tuple, xy_circ, xy_test = generate_rectangle_with_hole(subkey, RADIUS, XLIM, YLIM, num_coll, num_rBC, num_cBC, num_test)
    xy_rect = jnp.concatenate(xy_rect_tuple)
    
    x = (xy_coll, xy_rect_tuple, xy_circ)
    
    sigma = (OUTER_STRESS, INNER_STRESS)
    
    u_rect_tuple = tuple([cart_sol_true(xy_rect_tuple[i][:, 0], xy_rect_tuple[i][:, 1], S=TENSION, a=RADIUS).reshape(-1, 1) for i in range(4)])
    u_circ = cart_sol_true(xy_circ[:, 0], xy_circ[:, 1], S=TENSION, a=RADIUS).reshape(-1, 1)
    u_coll = cart_sol_true(xy_coll[:, 0], xy_coll[:, 1], S=TENSION, a=RADIUS).reshape(-1, 1)
    u_rect = jnp.concatenate(u_rect_tuple)
    u = (u_coll, u_rect_tuple, u_circ)
    

    if do_sample_data_plots:
        # Plot training points
        plt.scatter(xy_coll[:, 0], xy_coll[:, 1])
        plt.scatter(xy_rect[:, 0], xy_rect[:, 1], c='green')
        plt.scatter(xy_circ[:, 0], xy_circ[:, 1], c='red')
        save_fig(fig_dir, "training_points", format="png")
        plt.clf()

        # Plot test points
        plt.scatter(xy_test[:,0], xy_test[:,1], c = 'green')
        save_fig(fig_dir, "test_points", format="png")
        plt.clf()
    
    # Create and train PINN
    print("Creating PINN:")
    pinn = PPINN(MLP_instance, run_settings["specifications"], logging=do_logging)
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
    pinn.train(run_settings["specifications"]["iterations"], 200, x, u, sigma)
    t2 = perf_counter()
    print("Training done!")
    print(f"Time: {t2-t1:.2f}")



    if do_result_plots:
        X, Y, plotpoints = get_plot_variables(XLIM, YLIM, grid=201)
        R, THETA, plotpoints_polar = get_plot_variables([RADIUS, R], [0, 4*jnp.pi], grid=201)
        plotpoints2 = jax.vmap(rtheta2xy)(plotpoints_polar)
        
        phi = pinn.predict(plotpoints).reshape(X.shape)*(xy2r(X, Y) >= RADIUS)
        phi_polar = pinn.predict(plotpoints2).reshape(R.shape)
        phi_true = cart_sol_true(X, Y, S=TENSION, a=RADIUS)*(xy2r(X, Y) >= RADIUS)

        # Hessian prediction
        phi_pp = pinn.hessian_flatten(pinn.params, plotpoints)
        
        # Calculate stress from phi function: phi_xx = sigma_yy, phi_yy = sigma_xx, phi_xy = -sigma_xy
        sigma_cart = phi_pp[:, [3, 1, 2, 0]]
        sigma_cart = sigma_cart.at[:, [1, 2]].set(-phi_pp[:, [1, 2]])

        # List and reshape the four components
        sigma_cart_list = [sigma_cart[:, i].reshape(X.shape)*(xy2r(X, Y) >= RADIUS) for i in range(4)]

        # Repeat for the other set of points (polar coords converted to cartesian coords)
        phi_pp2 = pinn.hessian_flatten(pinn.params, plotpoints2)

        # Calculate stress from phi function
        sigma_cart2 = phi_pp2[:, [3, 1, 2, 0]]
        sigma_cart2 = sigma_cart2.at[:, [1, 2]].set(-phi_pp2[:, [1, 2]])
        
        # Convert these points to polar coordinates before listing and reshaping
        sigma_polar = jax.vmap(cart2polar_tensor, in_axes=(0, 0))(sigma_cart2.reshape(-1, 2, 2), plotpoints2).reshape(-1, 4)
        sigma_polar_list = [sigma_polar[:, i].reshape(R.shape)*(R >= RADIUS) for i in range(4)]

        # Calculate true stresses (cartesian and polar)
        sigma_cart_true = jax.vmap(cart_stress_true)(plotpoints)
        sigma_cart_true_list = [sigma_cart_true.reshape(-1, 4)[:, i].reshape(X.shape)*(xy2r(X, Y) >= RADIUS) for i in range(4)]
        sigma_polar_true = jax.vmap(polar_stress_true)(plotpoints_polar)
        sigma_polar_true_list = [sigma_polar_true.reshape(-1, 4)[:, i].reshape(R.shape)*(R >= RADIUS) for i in range(4)]
        
        plot_potential(X, Y, phi, phi_true, fig_dir, "potential", radius=RADIUS)
        plot_stress(X, Y, sigma_cart_list, sigma_cart_true_list, fig_dir, "stress", radius=RADIUS)
        plot_polar_stress(R, THETA, sigma_polar_list, sigma_polar_true_list, fig_dir, "stress_polar", radius=RADIUS)


