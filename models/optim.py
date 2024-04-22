from collections.abc import Callable
from functools import wraps
from time import perf_counter

import jax
import jax.numpy as jnp
import optax

from setup.settings import LoggingSettings


def get_update(
    loss_fun: Callable,
    optimizer: optax.GradientTransformation,
    jitted: bool,
    verbose: bool = False,
    verbose_kwargs: dict | None = None
    ) -> Callable:
    """
    Function that returns an update function
    for training a model.

    'update_scheme' can be one of the following:
        'unweighted'
        'softadapt'
    """

    update_fun = _get_update(loss_fun=loss_fun, optimizer=optimizer)
    if jitted:
        update_fun = jax.jit(update_fun, static_argnames=("update_key",))
    if verbose:
        if verbose_kwargs is None:
            verbose_kwargs = {}
        update_fun = _verbose_update(**verbose_kwargs)(update_fun)
    return update_fun


def _get_update(loss_fun: Callable,
                optimizer: optax.GradientTransformation,
                ) -> Callable:
    """
    Method for updating the model by applying the chosen optimizer.

    This method computes the gradients and apply them using the
    specified optimizer.

    Input:
        params:         The parameters.
        opt_state:      The model state.
        inputs:         The input points.
        true_val:       The true function values (if not zero).
        update_key:     A key that can be passed through to the
                        loss_term function in order to evaluate
                        loss differently. Treated as a static
                        argument, so recompiling occurs when
                        it is changed.

    Output:
        params:         The updated parameters.
        opt_state:      The updated model state.
        total_loss:     The total loss of the iteration.
        *aux:           Other output such as each loss term
                        or other info.
        
    """
    
    def update(
        opt_state: optax.OptState,
        params: optax.Params,
        inputs: dict[str],
        true_val: dict[str] | None = None,
        update_key: int | None = None,
        prevlosses: jax.Array | None = None
    ) -> tuple[optax.Params, optax.OptState, float, jax.Array, jax.Array]:
        """
        Update function for loss.
        """

        # Compute loss and gradients
        (total_loss, aux), grads = jax.value_and_grad(loss_fun, has_aux=True)(
            params, inputs, true_val=true_val, update_key=update_key, prevlosses=prevlosses)

        # Apply updates
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Return updated params and state as well as the losses
        return params, opt_state, total_loss, *aux
    return update


def _verbose_update(print_every = LoggingSettings.print_every):
    def decorator(update_func: Callable):
        if print_every is None or print_every < 1:
            return update_func
        
        @wraps(update_func)
        def wrapper(*args, epoch, learning_rate, start_time, **kwargs):
            
            # Call update function
            params, opt_state, total_loss, prevlosses, weights = update_func(*args, **kwargs)

            if epoch % print_every == 0:
                tcurr = perf_counter()
                if len(weights.shape) == 0:
                    ww = [weights]
                else:
                    ww = weights
                print(f"Training time: {tcurr-start_time:>7.2f} s    "
                      f"Epoch: {epoch:>6}    "
                      f"Learning rate: {learning_rate:2.2e}    "
                      f"Weighted loss: {total_loss:2.2e}    "
                      f"Unweighted loss: {jnp.sum(prevlosses[-1]):2.2e}")
                print("Weights:     ", end="")
                [print(f"{w:2.2e}", end="  ") for w in ww]
                print("")
                print("Loss terms:  ", end="")
                [print(f"{l:2.2e}", end="  ") for l in prevlosses[-1]]
                print("\n\n")

            return params, opt_state, total_loss, prevlosses, weights
        return wrapper
    return decorator