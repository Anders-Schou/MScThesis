from collections.abc import Sequence

import jax
import jax.numpy as jnp

def cyclic_diff(x: jnp.ndarray):
    return jnp.subtract(x, jnp.roll(x, -1))


def out_shape(fun, *args):
    print(f"{fun.__name__:<15}    {jax.eval_shape(fun, *args).shape}")


def generate_rectangle_points(key: jax.random.PRNGKey,
                              xlim: Sequence[float],
                              ylim: Sequence[float],
                              num_points: int | tuple
                              ) -> tuple:
    """
    Order of rectangle sides: Lower horizontal, left vertical, upper horizontal, right vertical.
    """
    if isinstance(num_points, int):
        N1 = num_points
        N2 = num_points
        N3 = num_points
        N4 = num_points
        NC = num_points
    elif isinstance(num_points, tuple):
        if len(num_points == 1):
            N1 = num_points[0]
            N2 = num_points[0]
            N3 = num_points[0]
            N4 = num_points[0]
        elif len(num_points == 4):
            N1 = num_points[0]
            N2 = num_points[1]
            N3 = num_points[2]
            N4 = num_points[3]
        else:
            raise ValueError(f"Wrong tuple length: f{len(tuple)}. Tuple length must be either 1 or 4.")
    else:
        raise ValueError("Argument 'num_points' must be int or tuple.")
    
    key1, key2, key3, key4 = jax.random.split(key, 4)

    x_BC1 = jax.random.uniform(key1, (N1, 1), minval=xlim[0], maxval=xlim[1])
    y_BC1 = jnp.full((N1, 1), ylim[0])
    xy1 = jnp.stack([x_BC1, y_BC1], axis=1).reshape((-1,2))

    x_BC2 = jnp.full((N2, 1), xlim[0])
    y_BC2 = jax.random.uniform(key2, (N2, 1), minval=ylim[0], maxval=ylim[1])
    xy2 = jnp.stack([x_BC2, y_BC2], axis=1).reshape((-1,2))

    x_BC3 = jax.random.uniform(key3, (N3, 1), minval=xlim[0], maxval=xlim[1])
    y_BC3 = jnp.full((N3, 1), ylim[1])
    xy3 = jnp.stack([x_BC3, y_BC3], axis=1).reshape((-1,2))

    x_BC4 = jnp.full((N4, 1), xlim[1])
    y_BC4 = jax.random.uniform(key4, (N4, 1), minval=ylim[0], maxval=ylim[1])
    xy4 = jnp.stack([x_BC4, y_BC4], axis=1).reshape((-1,2))

    return (xy1, xy2, xy3, xy4)


def generate_circle_points(key: jax.random.PRNGKey,
                           radius: float,
                           num_points: int
                           ) -> int:
    
    theta = jax.random.uniform(key, (num_points, 1), minval=0, maxval=2*jnp.pi)
    xc = radius*jnp.cos(theta)
    yc = radius*jnp.sin(theta)
    xyc = jnp.stack([xc, yc], axis=1).reshape((-1,2))

    return xyc