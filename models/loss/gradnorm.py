from functools import wraps
from collections.abc import Callable

from scipy.special import gamma
import jax
import jax.numpy as jnp

from setup.settings import DefaultSettings, GradNormSettings



def running_average_old(sett: GradNormSettings) -> Callable[..., Callable]:
    """
    Function implementing the Running_average algorithm.
    RETURNS a DECORATOR for the function calculating
    the loss terms.

    Usage:
        new_loss_terms = running_average(order=...)(loss_terms)
    or  @running_average(order=...)
        def loss_terms(...):
            ...

    """
    
    alpha = sett.alpha
    normalized = sett.normalized

    def running_average_decorator(loss_terms: Callable[..., jax.Array]):
        """
        This is the decorator.
        """
        
        @wraps(loss_terms)
        def wrapper(*args, prevlosses=None, weights=None, **kwargs):
            """
            This is the wrapper.
            """
            
            # Calculate losses in original loss_terms function (returns array of loss values)
            detached_params = jax.lax.stop_gradient(args[0])
            
            grads = jax.jacrev(loss_terms)(detached_params, *args[1:], **kwargs)
            losses = loss_terms(*args, **kwargs)
            
            grad_per_loss = jnp.concatenate([i.reshape(len(losses), -1) for i in jax.tree.leaves(grads)], axis=1)
            
            grad_norms = jnp.linalg.norm(grad_per_loss, axis=1)
            
            grad_sum = jnp.sum(grad_norms)
            
            weights_new = jax.tree_util.tree_map(lambda x: (grad_sum / x), grad_norms)

            weights = alpha*weights + (1-alpha)*weights_new
            
            if normalized:
                weights = jnp.divide(weights, jnp.sum(weights))
            
            # Weight by loss values
            # if loss_weighted:
            avg_weights = jnp.multiply(jnp.mean(prevlosses, axis=0), weights)
            weights = jnp.divide(avg_weights, jnp.sum(avg_weights))
            
            weighted_losses = jnp.dot(weights, losses)

            return weighted_losses, prevlosses, weights
        return wrapper
    return running_average_decorator


def gradnorm(sett: GradNormSettings, grads, num_losses: int, loss_weights: jax.Array | None = None) -> jax.Array:
    grad_per_loss = jnp.concatenate([i.reshape(num_losses, -1) for i in jax.tree.leaves(grads)], axis=1)
    grad_norms = jnp.linalg.norm(grad_per_loss, axis=1)
    grad_sum = jnp.mean(grad_norms)
    
    weights = jax.tree_util.tree_map(lambda x: (grad_sum / x), grad_norms)
    
    if sett.loss_weighted:
        if loss_weights is not None:
            weights = jnp.divide(w:=jnp.multiply(loss_weights, weights), jnp.sum(w))

    if sett.normalized:
        weights = jnp.divide(weights, jnp.sum(weights))
    
    return jax.lax.stop_gradient(weights)


# def gradnorm()
#     grads = jacrev(self.losses)(params, batch, *args)

#     # Compute the grad norm of each loss
#     grad_norm_dict = {}
#     for key, value in grads.items():
#         flattened_grad = flatten_pytree(value)
#         grad_norm_dict[key] = jnp.linalg.norm(flattened_grad)

#     # Compute the mean of grad norms over all losses
#     mean_grad_norm = jnp.mean(jnp.stack(tree_leaves(grad_norm_dict)))
#     # Grad Norm Weighting
#     w = tree_map(lambda x: (mean_grad_norm / x), grad_norm_dict)
