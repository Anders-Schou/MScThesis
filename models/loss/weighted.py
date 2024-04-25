from functools import wraps
from collections.abc import Callable
import warnings

import jax
import jax.numpy as jnp

from .unweighted import unweighted
from setup.settings import WeightedSettings


def weighted(sett: WeightedSettings) -> Callable[..., Callable]:
    """
    Decorator factory for (fixed) weighted loss.
    If normalize is true, the loss terms are
    divided by the number of loss terms.
    """
    _weights = sett.weights
    normalized = sett.normalized
    save_last = sett.save_last

    if save_last < 1:
        raise ValueError(f"Arguments 'save_last' must be at least 1, but was {save_last}.")

    if _weights is None:
        # No weights provided: Use unweighted loss
        warnings.warn("No weights found. Using unweighted loss instead of weighted loss.")
        return unweighted(normalized=normalized, save_last=save_last)
    
    _weights = jnp.array(_weights)
    if jnp.any(_weights < 0.):
        raise ValueError("Negative weights are not allowed.")

    w = _weights.shape[0]

    def weighted_decorator(loss_terms: Callable[..., jax.Array]):
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
            l = losses.shape[0]

            if prevlosses is None:
                prevlosses = jnp.tile(losses, save_last).reshape((save_last, losses.shape[0]))

            # Insert newest loss values and push out oldest loss values
            prevlosses = jnp.roll(prevlosses, -1, axis=0).at[-1, :].set(losses)
            
            if w < l:
                weights = jnp.concatenate((_weights, jnp.ones((l-w,))))
            else:
                weights = _weights[:l]

            if normalized:
                weights = jnp.divide(weights, jnp.sum(weights))

            weighted_losses = jnp.dot(weights, losses)

            return weighted_losses, prevlosses, weights
        return wrapper
    return weighted_decorator