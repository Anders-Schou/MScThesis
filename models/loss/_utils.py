import jax
import jax.numpy as jnp


def sq(r: jax.Array) -> jax.Array:
    return jnp.square(r)


def sqe(u: jax.Array, u_true: jax.Array | None = None):
    if u_true is None:
        return sq(u)
    return sq(jnp.subtract(u.ravel(), u_true.ravel()))


def ms(r: jax.Array) -> float:
    return jnp.mean(jnp.square(r))


def mse(u: jax.Array, u_true: jax.Array | None = None):
    if u_true is None:
        return ms(u)
    return ms(jnp.subtract(u.ravel(), u_true.ravel()))


def ma(r: jax.Array) -> float:
    return jnp.mean(jnp.abs(r))


def mae(u: jax.Array, u_true: jax.Array | None = None):
    if u_true is None:
        return ma(u)
    return ma(jnp.subtract(u.ravel(), u_true.ravel()))

def maxabse(u: jax.Array, u_true: jax.Array | None = None):
    if u_true is None:
        return jnp.max(jnp.abs(u))
    return jnp.max(jnp.abs(jnp.subtract(u.ravel(), u_true.ravel())))


# def get_loss(loss_terms: list[Callable], loss_type: str = "mse"):
#     """
#     INPUTS:
#         loss_terms: List of the loss_terms (callables)

#         loss_type: Type of outer loss (default: MSE), such that:

#             loss_fn(params, batch) = outer_loss(fun1(params, input1), output1) + outer_loss(fun2(params, input2), output2)) + ...
    
#     OUTPUTS:
#         loss_fn: A function returning the total loss, when given params and batch
#     """

#     # Get outer loss type (default: MSE)
#     outer_loss = parse_loss_type(loss_type)

#     # Define specific loss function to return
#     def loss_fn(params, batch: list[tuple[jnp.ndarray, jnp.ndarray]]):

#         total_loss = 0

#         # Run through loss terms
#         for i, loss in enumerate(loss_terms):
#             inputs, outputs = batch[i]
#             total_loss += outer_loss(loss(params, inputs), outputs)
        
#         return total_loss

#     return loss_fn
