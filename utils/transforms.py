import jax.numpy as jnp


def cart2polar(sigma_xx: jnp.ndarray,
               sigma_xy: jnp.ndarray,
               sigma_yy: jnp.ndarray,
               theta: jnp.ndarray
               ):
    
    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)
    costheta2 = jnp.square(costheta)
    sintheta2 = jnp.square(sintheta)
    sincos = jnp.multiply(costheta, sintheta)

    sigma_rr = jnp.multiply(sigma_xx, costheta2) + \
               jnp.multiply(sigma_yy, sintheta2) + \
               jnp.multiply(sigma_xy, sincos) * 2
    
    sigma_rt = jnp.multiply(sincos, jnp.subtract(sigma_yy, sigma_xx)) + \
               jnp.multiply(sigma_xy, jnp.subtract(costheta2, sintheta2))
    
    sigma_tt = jnp.multiply(sigma_xx, sintheta2) + \
               jnp.multiply(sigma_yy, costheta2) - \
               jnp.multiply(sigma_xy, sincos) * 2
    
    return sigma_rr, sigma_rt, sigma_tt