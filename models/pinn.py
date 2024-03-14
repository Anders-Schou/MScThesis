import os
from functools import partial
from collections.abc import Callable

import numpy as np
import jax
import jax.numpy as jnp
import optax
# from torch.utils.tensorboard import SummaryWriter

from models.networks import MLP
from utils.transforms import *

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
        # self.writer = SummaryWriter(log_dir="/zhome/e8/9/147091/MSc/results/logs")
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
    
    def rect_bc2(self, params, x, sigma_bc):
        out0 = self.hessian(params, x[0]) # horizontal lower
        out1 = self.hessian(params, x[1]) # vertical right
        out2 = self.hessian(params, x[2]) # horizontal upper
        out3 = self.hessian(params, x[3]) # vertical left
        hl = ((out0[:, 0, 0, 0].ravel() - sigma_bc[1][1])**2).mean() + ((out0[:, 0, 0, 1].ravel() - sigma_bc[1][0])**2).mean() # horizontal lower
        vr = ((out1[:, 0, 1, 1].ravel() - sigma_bc[0][0])**2).mean() + ((out1[:, 0, 1, 0].ravel() - sigma_bc[0][1])**2).mean() # vertical right
        hu = ((out2[:, 0, 0, 0].ravel() - sigma_bc[1][1])**2).mean() + ((out2[:, 0, 0, 1].ravel() - sigma_bc[1][0])**2).mean() # horizontal upper
        vl = ((out3[:, 0, 1, 1].ravel() - sigma_bc[0][0])**2).mean() + ((out3[:, 0, 1, 0].ravel() - sigma_bc[0][1])**2).mean() # vertical left
        return hl + vr + hu + vl

    def circle_bc0(self, params, x, u):
        out = self.forward(params, x)
        return ((out.ravel()-u.ravel())**2).mean()

    def circle_bc2(self, params, x: jnp.ndarray, sigma_bc) -> float:
        out = self.hessian_flatten(params, x)
        sigmas = out[:, [3, 1, 2, 0]]
        sigmas = sigmas.at[:, [1, 2]].set(jnp.negative(out[:, [1, 2]]))
        rtheta_stress = jax.vmap(cart2polar_tensor, in_axes=(0, 0))(sigmas.reshape(-1, 2, 2), x)
        return ((rtheta_stress[:, 0, 0].ravel() - sigma_bc[0][0])**2).mean() + ((rtheta_stress[:, 0, 1].ravel() - sigma_bc[0][1])**2).mean()

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
              u: tuple[jnp.ndarray],
              sigma_bc: tuple[jnp.ndarray]):
        xy_coll, xy_rect, xy_circ = x
        u_coll, u_rect, u_circ = u
        sigma_rect, sigma_circ = sigma_bc
        
        circ = self.circle_bc2(params, xy_circ, sigma_circ)
        rect = self.rect_bc2(params, xy_rect, sigma_rect)
        pde = self.pde(params, xy_coll)
        
        total_loss = circ + rect + pde
            # self.rect_bc0(params, xy_rect, u_rect) + \
            # self.circle_bc0(params, xy_circ, u_circ)
        
        return total_loss, (circ, rect, pde)

    def weighted_loss(self, params, x: tuple[jnp.ndarray], u: tuple[jnp.ndarray], sigma_bc: tuple[jnp.ndarray], prevlosses = (1., 1., 1.), it = 0):
        
        total, circ, rect, pde = self.loss(params, x, u, sigma_bc)
        
        
        

    @partial(jax.jit, static_argnums=(0,))
    def update(self,
               params,
               opt_state,
               x: tuple[jnp.ndarray],
               u_bc: tuple[jnp.ndarray],
               sigma_bc: tuple[jnp.ndarray]):
        (loss, otherloss), grads = jax.value_and_grad(self.loss, argnums=0, has_aux=True)(params, x, u_bc, sigma_bc)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, (loss, otherloss)
    
    def train(self,
              max_iter: int,
              print_every: int,
              x: tuple[jnp.ndarray],
              u_bc: tuple[jnp.ndarray],
              sigma_bc: tuple[jnp.ndarray]
              ) -> None:
        
        losses = np.zeros(int(max_iter/print_every))
        l = 0
        for i in range(max_iter):
            self.params, self.opt_state, loss = self.update(self.params, self.opt_state, x, u_bc, sigma_bc)
            if (i % print_every == 0):
                print(f"Epoch {i:>6}    MSE: {loss[0]:2.2e}    lr:  {self.schedule(i):2.2e}    (C2 = {loss[1][0]:2.2e}, R2 = {loss[1][1]:2.2e}, PDE = {loss[1][2]:2.2e})")
                # print(f"Epoch {i:>6}    MSE: {loss[0]:2.2e}    (Coll = {loss[1][0]:2.2e},    R = {loss[1][1]:2.2e},    C = {loss[1][2]:2.2e})")
                # losses[l] = loss
                
                # if self.logging:
                #     self.writer.add_scalar("loss/total", np.array(loss[0]   ), i)
                #     self.writer.add_scalar("loss/circ0", np.array(loss[1][0]), i)
                #     self.writer.add_scalar("loss/rect0", np.array(loss[1][1]), i)
                #     self.writer.add_scalar("loss/circ2", np.array(loss[1][2]), i)
                #     self.writer.add_scalar("loss/rect2", np.array(loss[1][3]), i)
                #     self.writer.add_scalar("loss/pde",   np.array(loss[1][4]), i)
                #     self.writer.add_scalar("learning_rate", np.array(self.schedule(i)), i)
                #     l += 1
        
        self.losses = losses
        return