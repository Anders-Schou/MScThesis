import jax.numpy as jnp

_FIRST_ORDER_COEFFICIENTS  = jnp.array((-1, 1))
_SECOND_ORDER_COEFFICIENTS = jnp.array((-3/2, 2, -1/2))
_THIRD_ORDER_COEFFICIENTS  = jnp.array((-11/6, 3, -3/2, 1/3))
_FOURTH_ORDER_COEFFICIENTS = jnp.array((-25/12, 4, -3, 4/3, -1/4))
_FIFTH_ORDER_COEFFICIENTS  = jnp.array((-137/60, 5, -5, 10/3, -5/4, 1/5))
_SIXTH_ORDER_COEFFICIENTS  = jnp.array((-49/20, 6, -15/2, 20/3, -15/4, 6/5, -1/6))

def softmax(input, beta = 1, numerator_weights = None, shift_by_max_value = True):
    if shift_by_max_value:
        exp_of_input = jnp.exp(beta * (input - input.max()))
    else:
        exp_of_input = jnp.exp(beta * input)
        
    if numerator_weights is not None:
        exp_of_input = jnp.multiply(numerator_weights, exp_of_input)
        
    return exp_of_input / jnp.sum(exp_of_input + 1e-8)


def compute_rates_of_change(input):
    order = len(input) - 1
    
    if order == 1:
        constants = _FIRST_ORDER_COEFFICIENTS
    elif order == 2:
        constants = _SECOND_ORDER_COEFFICIENTS
    elif order == 3:
        constants = _THIRD_ORDER_COEFFICIENTS
    elif order == 4:
        constants = _FOURTH_ORDER_COEFFICIENTS
    elif order == 5: 
        constants = _FIFTH_ORDER_COEFFICIENTS
    elif order == 6: 
        constants = _SIXTH_ORDER_COEFFICIENTS
    else:
        print("Invalid order of finite difference for softmax. Currently only supports order up to 6")
    
    pointwise_multiplication = [input[i] * constants[i] for i in range(len(constants))]
    
    return jnp.sum(pointwise_multiplication)

def softadapt(*loss_component_values, beta: int = 0.1, normalized_slopes = False, loss_weighted = False):
    if len(loss_component_values) == 1:
        print("==> Warning: You have only passed on the values of one loss"
                  " component, which will result in trivial weighting.")
        
    rates_of_change     = []
    average_loss_values = []
    
    for loss_points in loss_component_values:
        rates_of_change.append(compute_rates_of_change(loss_points))
        if loss_weighted:
            average_loss_values.append(jnp.mean(loss_points))
        
    rates_of_change = jnp.array(rates_of_change)
    
    if normalized_slopes:
        rates_of_change = rates_of_change/jnp.sum(rates_of_change)
    
    return softmax(rates_of_change, beta = beta)