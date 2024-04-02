from collections.abc import Sequence

import jax
import jax.numpy as jnp

from utils.transforms import xy2r, xy2theta, xy2rtheta, polar2cart_tensor

_TENSION = 10


def polar_sol_true(r, theta, S = 10, a = 2):
    a2 = a*a
    r2 = jnp.square(r)
    A = -S * 0.25
    B = 0
    C = -a2**2 * S * 0.25
    D = a2 * S * 0.5
    return (A*r2 + B*r2**2 + C / r2 + D) * jnp.cos(2*theta)


def cart_sol_true(x, y, **kwargs):
    return polar_sol_true(xy2r(x, y), xy2theta(x, y), **kwargs)


def sigma_rr_true(r, theta, S = 10, a = 2):
    a2 = jnp.array(a*a)
    r2 = jnp.square(r)
    a2divr2 = jnp.divide(a2, r2)
    return 0.5*S * ((1 + 3 * jnp.square(a2divr2) - 4 * a2divr2) * jnp.cos(2*theta) + (1 - a2divr2))


def sigma_tt_true(r, theta, S = 10, a = 2):
    a2 = jnp.array(a*a)
    r2 = jnp.square(r)
    a2divr2 = jnp.divide(a2, r2)
    return 0.5*S * ((1 + a2divr2) - (1 + 3 * jnp.square(a2divr2)) * jnp.cos(2*theta))


def sigma_rt_true(r, theta, S = 10, a = 2):
    a2 = jnp.array(a*a)
    r2 = jnp.square(r)
    a2divr2 = jnp.divide(a2, r2)
    return -0.5*S * (1 - 3 * jnp.square(a2divr2) + 2 * a2divr2) * jnp.sin(2*theta)


def polar_stress_true(rtheta, **kwargs):

    # Compute polar stresses from analytical solutions
    rr_stress = sigma_rr_true(rtheta[0], rtheta[1], **kwargs)
    rt_stress = sigma_rt_true(rtheta[0], rtheta[1], **kwargs)
    tt_stress = sigma_tt_true(rtheta[0], rtheta[1], **kwargs)

    # Format as stress tensor
    return jnp.array([[rr_stress, rt_stress], [rt_stress, tt_stress]])


def cart_stress_true(xy, **kwargs):

    # Map cartesian coordinates to polar coordinates
    rtheta = xy2rtheta(xy)
    
    # Compute true stress tensor in polar coordinates
    polar_stress_hessian = polar_stress_true(rtheta, **kwargs)

    # Convert polar stress tensor to cartesian stress tensor
    return polar2cart_tensor(polar_stress_hessian, rtheta)


def get_true_vals(points: dict[str, jax.Array | tuple[dict[str, jax.Array]] | None],
                  exclude: Sequence[str] | None = None,
                  ylim = None) -> dict[str, jax.Array | dict[str, jax.Array] | None]:
    vals = {}
    if exclude is None:
        exclude = []
    
    # Homogeneous PDE ==> RHS is zero
    if "coll" not in exclude:
        coll = None
        vals["coll"] = coll
    
    # Only inhomogeneous BCs at two sides of rectangle
    if "rect" not in exclude:
        rect_points = [p.shape[0] for p in points["rect"]]
        rect = {"yy1": jnp.full((rect_points[1],), _TENSION),
                "yy3": jnp.full((rect_points[3],), _TENSION)}
        vals["rect"] = rect
    
    # Homogeneous BC at inner circle
    if "circ" not in exclude:
        circ = None
        vals["circ"] = circ
    
    # If used, constrains the solution to a specific one, namely
    # the one where the terms of order 0 and 1 vanish.
    """
                    _..._
                ..**     **..
              .*             *.
            .*                 *.
           /                   /
          /    _..._          /
         / ..**     **..     /
        /.*             *.  /
       /*                 */
    
    """
    if "diri" not in exclude:
        if ylim is None:
            print("Y-limit defaults to [-10.0, 10.0]")
            ylim = [-10.0, 10.0]
        diri = {"di1": 0.5*_TENSION*(points["rect"][1][:, 1]-ylim[0])*(points["rect"][1][:, 1]-ylim[1]),
                "di3": 0.5*_TENSION*(points["rect"][3][:, 1]-ylim[0])*(points["rect"][3][:, 1]-ylim[1])}
        vals["diri"] = diri

    return vals