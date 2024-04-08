from functools import partial
from collections.abc import Callable

import jax
import optax

from .softadapt import softadapt


def get_update(loss_func: Callable,
               optimizer: optax.GradientTransformation,
               *args,
               update_scheme: str = "unweighted",
               jit_compile: bool = True,
               **kwargs
               ) -> Callable:
    """
    Function that returns an update function
    for training a model.

    'update_scheme' can be one of the following:
        'unweighted'
        'softadapt'
    """

    update_scheme = update_scheme.lower()

    match update_scheme:
        case "unweighted":
            update_func = partial(_unweighted_update,
                                  loss_func=loss_func,
                                  optimizer=optimizer)
            if jit_compile:
                return jax.jit(update_func,
                               *args,
                               static_argnames=("update_key",),
                               **kwargs)
            return update_func
        
        case "softadapt":
            update_func = partial(_softadapt_update,
                                  loss_func=loss_func,
                                  optimizer=optimizer)
            if jit_compile:
                return jax.jit(update_func,
                               *args,
                               static_argnames=("prevlosses", "update_key")
                               **kwargs)
            return update_func

        case _:
            raise ValueError(f"Unknown update scheme: {update_scheme}.")


def _unweighted_update(loss_func: Callable,
                       optimizer: optax.GradientTransformation,
                       opt_state: optax.OptState,
                       params: optax.Params,
                       inputs: dict[str],
                       true_val: dict[str] | None = None,
                       update_key: int | None = None):
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

        # Compute graients
        (total_loss, aux), grads = jax.value_and_grad(
            loss_func, has_aux=True)(
                params, inputs, true_val=true_val, update_key=update_key)

        # Apply updates
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Return updated params and state as well as the losses
        return params, opt_state, total_loss, *aux


def _softadapt_update(loss_func: Callable,
                      optimizer: optax.GradientTransformation,
                      opt_state: optax.OptState,
                      params: optax.Params,
                      inputs: dict[str],
                      true_val: dict[str] | None = None,
                      update_key: int | None = None,
                      prevlosses: jax.Array | None = None):
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
        prevlosses:     Previous losses used for SoftAdapt algorithm.

    Output:
        params:         The updated parameters.
        opt_state:      The updated model state.
        total_loss:     The total loss of the iteration.
        *aux:           Other output such as that from SoftAdapt.
        
    """
    (total_loss, aux), grads = jax.value_and_grad(
        loss_func, has_aux=True)(
            params, inputs, true_val=true_val, update_key=update_key, prevlosses=prevlosses)

    # Apply updates
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    # Return updated params and state as well as the losses
    return params, opt_state, total_loss, *aux