from collections.abc import Callable

import jax
import jax.numpy as jnp

def hessian(model) -> Callable:
    return jax.hessian(model, argnums=1)


def laplacian(model, axis1=1, axis2=2) -> Callable:
    hess = jax.hessian(model, argnums=1)
    tr = lambda p, xx: jnp.trace(hess(p, xx), axis1=axis1, axis2=axis2)
    return tr


def biharmonic(model, axis1=1, axis2=2) -> Callable:
    lap = laplacian(model, axis1=axis1, axis2=axis2)
    lap2 = laplacian(lap, axis1=axis1, axis2=axis2)
    return lap2