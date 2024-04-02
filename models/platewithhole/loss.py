import jax
import jax.numpy as jnp

from models.loss import mse
from utils.transforms import xy2theta


def loss_rect(*output: jax.Array, true_val: dict[str, jax.Array] | None = None):
    """
    Computes loss of the BC residuals on the four sides of the rectangle.
    
    Layout of rectangle sides:
    ```
                        ^
                      y |
                        |
                        *------>
                                x
                            
                                     2
                             _________________
            xx stress  <-   |                 |   -> xx stress
                            |                 |
                         3  |        O        |  1
                            |                 |
            xx stress  <-   |_________________|   -> xx stress
                            
                                     0
    ```
    """
    # Unpack outputs
    out0, out1, out2, out3 = output

    # Return all losses
    if true_val is None:
        return mse(out0[:, 0]), mse(out0[:, 1]), \
               mse(out1[:, 3]), mse(out1[:, 2]), \
               mse(out2[:, 0]), mse(out2[:, 1]), \
               mse(out3[:, 3]), mse(out3[:, 2])
    return mse(out0[:, 0], true_val.get("xx0")), mse(out0[:, 1], true_val.get("xy0")), \
           mse(out1[:, 3], true_val.get("yy1")), mse(out1[:, 2], true_val.get("xy1")), \
           mse(out2[:, 0], true_val.get("xx2")), mse(out2[:, 1], true_val.get("xy2")), \
           mse(out3[:, 3], true_val.get("yy3")), mse(out3[:, 2], true_val.get("xy3"))


def loss_circ_rr_rt(input: jax.Array, output: jax.Array, true_val: dict[str, jax.Array] | None = None):

    # Get polar angle
    theta = xy2theta(input[:, 1], input[:, 0])
    
    # Compute polar stresses (sigma_rr, sigma_rt)
    srr = jnp.multiply(output[:, 3], jnp.square(jnp.cos(theta))) + \
          jnp.multiply(output[:, 0], jnp.square(jnp.sin(theta))) - \
          2*jnp.multiply(output[:, 1], jnp.sin(theta)*jnp.cos(theta))
    srt = jnp.multiply(jnp.subtract(output[:, 0], output[:, 3]), jnp.sin(theta)*jnp.cos(theta)) - \
          jnp.multiply(output[:, 1], jnp.square(jnp.cos(theta))-jnp.square(jnp.sin(theta)))
    
    # Return both losses
    if true_val is None:
        return mse(srr), mse(srt)
    return mse(srr, true_val.get("srr")), mse(srt, true_val.get("srt"))


def loss_dirichlet(*output: jax.Array, true_val: dict[str, jax.Array] | None = None):

    # Unpack outputs
    out0, out1, out2, out3 = output
    
    # Return all losses
    if true_val is None:
        return mse(out0), mse(out1), mse(out2), mse(out3)
    return mse(out0, true_val.get("di0")), \
           mse(out1, true_val.get("di1")), \
           mse(out2, true_val.get("di2")), \
           mse(out3, true_val.get("di3"))
