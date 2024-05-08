from functools import wraps
from collections.abc import Callable

from scipy.special import gamma
import jax
import jax.numpy as jnp

from setup.settings import SoftAdaptSettings

def compute_losses(sett: SoftAdaptSettings | None = None) -> Callable[..., Callable]:
    """
    RETURNS a DECORATOR for the function calculating
    the loss terms.
    """
    
    if sett is not None:
        order = sett.order
    else:
        order = 2
    
    def loss_decorator(loss_terms: Callable[..., jax.Array]):
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
                prevlosses = jnp.tile(losses, order+1).reshape((order+1, losses.shape[0]))

            # # Insert newest loss values and push out oldest loss values
            prevlosses = jnp.roll(prevlosses, -1, axis=0).at[-1, :].set(losses)
            
            # Calculate weighted loss
            weighted_losses = jnp.multiply(kwargs["weights"], losses)

            return weighted_losses, prevlosses
        return wrapper
    return loss_decorator