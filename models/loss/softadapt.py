from functools import wraps
from collections.abc import Callable

from scipy.special import gamma
import jax
import jax.numpy as jnp

from setup.settings import DefaultSettings, SoftAdaptSettings



def get_fdm_coeff(order: int,
                  backward: bool = True
                  ) -> jax.Array:
    """
    Function for calculating finite difference coefficients.
    """
    if backward:
        lins = jnp.linspace(-order, 0, order+1)
    else:
        lins = jnp.linspace(0, order, order+1)
    exponents = jnp.linspace(0, order, order+1)
    powers = jnp.power(lins.reshape(-1, 1), exponents)
    factorials = jnp.array([gamma(i+1) for i in range(order+1)])
    
    # Set up linear system
    RHS = jnp.zeros((order+1,)).at[1].set(1.) # Set position of 1st derivative to 1.
    M = jnp.divide(powers, factorials) # Taylor expansion terms
    return jnp.linalg.solve(M.T, RHS)


def softmax(input: jax.Array,
            beta: float = 0.1,
            numerator_weights: jax.Array | None = None,
            shift_by_max_value: bool = True
            ) -> jax.Array:
    if shift_by_max_value:
        exp_of_input = jnp.exp(beta * (input - jnp.max(input)))
    else:
        exp_of_input = jnp.exp(beta * input)
        
    if numerator_weights is not None:
        exp_of_input = jnp.multiply(numerator_weights, exp_of_input)
        
    return exp_of_input / jnp.sum(exp_of_input + DefaultSettings.SOFTMAX_TOLERANCE)


def softadapt_old(sett: SoftAdaptSettings) -> Callable[..., Callable]:
    """
    Function implementing the SoftAdapt algorithm.
    RETURNS a DECORATOR for the function calculating
    the loss terms.

    Usage:
        new_loss_terms = softadapt(order=...)(loss_terms)
    or  @softadapt(order=...)
        def loss_terms(...):
            ...

    """
    
    order = sett.order
    beta = sett.beta
    normalized = sett.normalized
    loss_weighted = sett.loss_weighted
    delta_time = sett.delta_time
    shift_by_max_val = sett.shift_by_max_val

    # Get finite difference coefficients for calculating rates of change for loss terms
    fdm_coeff = get_fdm_coeff(order, backward=True)

    def softadapt_decorator(loss_terms: Callable[..., jax.Array]):
        """
        This is the decorator.
        """
        
        @wraps(loss_terms)
        def wrapper(*args, prevlosses=None, **kwargs):
            """
            This is the wrapper.
            """
            
            # Calculate losses in original loss_terms function (returns array of loss values)
            detached_params = jax.lax.stop_gradient(args[0])
            
            losses = loss_terms(detached_params, *args[1:], **kwargs)
            
            if prevlosses is None:
                prevlosses = jnp.tile(losses, order+1).reshape((order+1, losses.shape[0]))

            # Insert newest loss values and push out oldest loss values
            prevlosses = jnp.roll(prevlosses, -1, axis=0).at[-1, :].set(losses)
            
            # Calculate loss slopes using finite difference
            rates_of_change = jnp.matmul(fdm_coeff, prevlosses)
            
            # Normalize slopes
            if normalized:
                rates_of_change = jnp.divide(rates_of_change, jnp.sum(jnp.abs(rates_of_change)))
            elif delta_time is not None:
                rates_of_change = jnp.divide(rates_of_change, delta_time)

            # Call custom SoftMax function
            weights = softmax(rates_of_change, beta=beta, shift_by_max_value=shift_by_max_val)

            # Weight by loss values
            if loss_weighted:
                avg_weights = jnp.multiply(jnp.mean(prevlosses, axis=0), weights)
                weights = jnp.divide(avg_weights, jnp.sum(avg_weights))
            
            # Calculate weighted loss
            weighted_losses = jnp.multiply(weights, losses)

            return weighted_losses, prevlosses, weights
        return wrapper
    return softadapt_decorator

def softadapt(sett: SoftAdaptSettings,
              prevlosses = None,
              fdm_coeff = None
              ) -> jax.Array:

    if prevlosses is None:
        return None
    
    if fdm_coeff is None:
        fdm_coeff = get_fdm_coeff(sett.order, backward=True)
    
    # Calculate loss slopes using finite difference
    rates_of_change = jnp.matmul(fdm_coeff, prevlosses)
    
    # Normalize slopes
    if sett.normalized_rates:
        rates_of_change = jnp.divide(rates_of_change, jnp.sum(jnp.abs(rates_of_change)))
    elif sett.delta_time is not None:
        rates_of_change = jnp.divide(rates_of_change, sett.delta_time)

    # Call custom SoftMax function
    weights = jnp.clip(softmax(rates_of_change, beta=sett.beta, shift_by_max_value=sett.shift_by_max_val), 0.01, 100)

    # Weight by loss values
    if sett.loss_weighted:
        avg_weights = jnp.multiply(jnp.mean(prevlosses, axis=0), weights)
        weights = jnp.divide(avg_weights, jnp.sum(avg_weights))
    
    if sett.normalized:
        weights = jnp.divide(weights, jnp.sum(weights))

    return jax.lax.stop_gradient(weights)


# def compute_rates_of_change(input: jnp.ndarray, order: int = 6):
#     """
#     Implements a finite difference stencil.
#     """

#     # order = len(input) - 1
#     if order + 1 > len(input):
#         raise ValueError("Order must be no more than len(input) - 1.")
    
#     if   order == 1:
#         constants = _FIRST_ORDER_COEFFICIENTS
#     elif order == 2:
#         constants = _SECOND_ORDER_COEFFICIENTS
#     elif order == 3:
#         constants = _THIRD_ORDER_COEFFICIENTS
#     elif order == 4:
#         constants = _FOURTH_ORDER_COEFFICIENTS
#     elif order == 5: 
#         constants = _FIFTH_ORDER_COEFFICIENTS
#     elif order == 6: 
#         constants = _SIXTH_ORDER_COEFFICIENTS
#     else:
#         raise ValueError("Invalid order of finite difference for softmax. "
#                          "Supports only order 6 or lower.")
#     # print("INPUT:", input[-(order+1):])
#     # print("CONST:", constants)
#     return -jnp.dot(input[-(order+1):], constants)
#     # return sum([inp * constants[-(i+1)] for i, inp in enumerate(input)])

# def softadapt_weights(prevlosses: tuple,
#                       beta: float = 0.1,
#                       order: int = 6,
#                       normalized_slopes: bool = False,
#                       loss_weighted: bool = False,
#                       delta_time = None):
#     if len(prevlosses) == 1:
#         warnings.warn("You have only passed on the values of one loss"
#                       " component, which will result in trivial weighting.")
    
#     # rates_of_change = jnp.array([compute_rates_of_change(loss_points, order=order) for loss_points in loss_component_values])
#     # print("ORDER:", order)
#     # print("RATES:", rates_of_change)

#     if delta_time is not None:
#         rates_of_change = jnp.divide(rates_of_change, delta_time)

#     if loss_weighted:
#         average_loss_values = jnp.array([jnp.mean(loss_points) for loss_points in loss_component_values])
    
#     if normalized_slopes:
#         rates_of_change = rates_of_change/jnp.sum(rates_of_change)
    
#     return softmax(rates_of_change, beta = beta)


# _FIRST_ORDER_COEFFICIENTS  = jnp.flip(jnp.array((-1,      1)))
# _SECOND_ORDER_COEFFICIENTS = jnp.flip(jnp.array((-3/2,    2, -1/2)))
# _THIRD_ORDER_COEFFICIENTS  = jnp.flip(jnp.array((-11/6,   3, -3/2, 1/3)))
# _FOURTH_ORDER_COEFFICIENTS = jnp.flip(jnp.array((-25/12,  4, -3, 4/3, -1/4)))
# _FIFTH_ORDER_COEFFICIENTS  = jnp.flip(jnp.array((-137/60, 5, -5, 10/3, -5/4, 1/5)))
# _SIXTH_ORDER_COEFFICIENTS  = jnp.flip(jnp.array((-49/20,  6, -15/2, 20/3, -15/4, 6/5, -1/6)))
 