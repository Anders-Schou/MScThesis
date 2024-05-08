from collections.abc import Sequence

import jax
import jax.numpy as jnp

from utils.transforms import xy2r, xy2theta, xy2rtheta, polar2cart_tensor, cart2polar_tensor

_TENSION = 10
_RADIUS = 2

def cart_stress_true(xy, **kwargs):

    x = xy[0]
    y = xy[1]
    diff_xx = -jnp.sin(x)*jnp.sin(y)
    diff_xy = jnp.cos(x)*jnp.cos(y)
    diff_yy = -jnp.sin(x)*jnp.sin(y)
    
    return jnp.array([diff_xx, diff_xy, diff_xy, diff_yy]).reshape(2, 2)


def get_true_vals(points: dict[str, jax.Array | tuple[dict[str, jax.Array]] | None],
                  *,
                  exclude: Sequence[str] | None = None,
                  ylim = None,
                  noise: float | None = None,
                  key = None
                  ) -> dict[str, jax.Array | dict[str, jax.Array] | None]:
    vals = {}
    if exclude is None:
        exclude = []
    
    # Homogeneous PDE ==> RHS is zero
    if "coll" not in exclude:
        coll = None
        vals["coll"] = coll

    # True stresses in domain
    if "data" not in exclude:
        true_data = jax.vmap(cart_stress_true)(points["data"])
        vals["data"] = {}
        if noise is None:
            # Exact data
            vals["data"]["true_xx"] = true_data[:, 0, 0]
            vals["data"]["true_xy"] = true_data[:, 0, 1]
            vals["data"]["true_yy"] = true_data[:, 1, 1]
        else:
            # Noisy data
            if key is None:
                raise ValueError(f"PRNGKey must be specified.")
            keys = jax.random.split(key, 3)

            xx_noise = jax.random.normal(keys[0], true_data[:, 0, 0].shape)
            vals["data"]["true_xx"] = true_data[:, 0, 0] + \
                noise * (jnp.linalg.norm(true_data[:, 0, 0]) / jnp.linalg.norm(xx_noise)) * xx_noise
            xy_noise = jax.random.normal(keys[1], true_data[:, 0, 1].shape)
            vals["data"]["true_xy"] = true_data[:, 0, 1] + \
                noise * (jnp.linalg.norm(true_data[:, 0, 1]) / jnp.linalg.norm(xy_noise)) * xy_noise
            yy_noise = jax.random.normal(keys[2], true_data[:, 1, 1].shape)
            vals["data"]["true_yy"] = true_data[:, 1, 1] + \
                noise * (jnp.linalg.norm(true_data[:, 1, 1]) / jnp.linalg.norm(yy_noise)) * yy_noise
    
    # Only inhomogeneous BCs at two sides of rectangle
    if "rect" not in exclude:
        # rect_points = [p.shape[0] for p in points["rect"]]
        # rect = {"yy1": jnp.full((rect_points[1],), _TENSION),
        #         "yy3": jnp.full((rect_points[3],), _TENSION)}
        
        true_rect = [jax.vmap(cart_stress_true)(points["rect"][i]) for i in range(4)]

        rect = {
                "xx0":  true_rect[0][:, 1, 1],
                "xy0": -true_rect[0][:, 0, 1],
                "yy1":  true_rect[1][:, 0, 0],
                "xy1": -true_rect[1][:, 1, 0],
                "xx2":  true_rect[2][:, 1, 1],
                "xy2": -true_rect[2][:, 0, 1],
                "yy3":  true_rect[3][:, 0, 0],
                "xy3": -true_rect[3][:, 1, 0],
                "yy0":  true_rect[0][:, 0, 0], # extra
                "xx1":  true_rect[1][:, 1, 1], # extra
                "yy2":  true_rect[2][:, 0, 0], # extra
                "xx3":  true_rect[3][:, 1, 1]  # extra
                }


        vals["rect"] = rect

    return vals