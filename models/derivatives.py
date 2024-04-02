from collections.abc import Callable

import numpy as np
import jax
import jax.numpy as jnp


def hessian(model: Callable, argnums=1) -> Callable:
    return jax.hessian(model, argnums=argnums)


def laplacian(model: Callable, axis1=-2, axis2=-1, argnums=1) -> Callable:
    hess = jax.hessian(model, argnums=argnums)
    tr = lambda p, xx: jnp.trace(hess(p, xx), axis1=axis1, axis2=axis2)
    return tr


def biharmonic(model: Callable, axis1=-2, axis2=-1) -> Callable:
    lap = laplacian(model, axis1=axis1, axis2=axis2)
    lap2 = laplacian(lap, axis1=axis1, axis2=axis2)
    return lap2


def diff_xx(model) -> Callable:
    def xx(params, input):
        return jax.vmap(jax.hessian(model, argnums=1), in_axes=(None, 0))(params, input)
    return xx


def diff_yy(model) -> Callable:
    def yy(params, input):
        return jax.vmap(hessian(model), in_axes=(None, 0))(params, input)[:, 1, 1]
    return yy


def diff_xy(model) -> Callable:
    def xy(params, input):
        return jax.vmap(hessian(model), in_axes=(None, 0))(params, input)[:, 0, 1]
    return xy


def finite_diff_xx(model, h=None) -> Callable:
    if h is None:
        h = np.cbrt(np.finfo(np.float32).eps)
    
    h2 = h**2
    def fdmx2d(params, input):
        dx = jnp.array([[h, 0]])
        xx = (model(params, jnp.add(input, dx)) + model(params, jnp.subtract(input, dx)) - 2*model(params, input)) / h2
        return xx
    return fdmx2d


def finite_diff_yy(model, h=None) -> Callable:
    if h is None:
        h = np.cbrt(np.finfo(np.float32).eps)
        
    h2 = h**2
    def fdmy2d(params, input):
        dy = jnp.array([[0, h]])
        xx = (model(params, jnp.add(input, dy)) + model(params, jnp.subtract(input, dy)) - 2*model(params, input)) / h2
        return xx
    return fdmy2d