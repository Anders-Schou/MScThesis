from functools import wraps
from collections.abc import Callable

import jax
import jax.numpy as jnp

from setup.settings import UnweightedSettings


def unweighted(sett: UnweightedSettings) -> Callable[..., Callable]:
    """
    Decorator factory for unweighted loss.
    If normalize is true, the loss terms are
    divided by the number of loss terms.
    """
    
    normalized = sett.normalized
    save_last = sett.save_last

    if save_last < 1:
        raise ValueError(f"Arguments 'save_last' must be at least 1, but was {save_last}.")


    def unweighted_decorator(loss_terms: Callable[..., jax.Array]):
        """
        This is the decorator.
        """

        @wraps(loss_terms)
        def wrapper(*args, prevlosses=None, **kwargs):
            """
            This is the wrapper.
            """
            
            # Calculate losses in original loss_terms function (returns array of loss values)
            losses = loss_terms(*args, **kwargs)

            if prevlosses is None:
                prevlosses = jnp.tile(losses, save_last).reshape((save_last, losses.shape[0]))

            # Insert newest loss values and push out oldest loss values
            prevlosses = jnp.roll(prevlosses, -1, axis=0).at[-1, :].set(losses)
            
            weights = jnp.ones_like(losses)

            if normalized:
                weights = jnp.divide(weights, jnp.sum(weights))
            
            weighted_losses = jnp.dot(weights, losses)

            return weighted_losses, prevlosses, weights
        return wrapper
    return unweighted_decorator