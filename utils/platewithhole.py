from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from models.loss import ms, mse
from utils.transforms import xy2r, xy2theta, xy2rtheta, polar2cart_tensor, cart2polar_tensor
from utils.plotting import plot_circle


"""
DEPRECATED: TO BE REMOVED IN NEXT COMMIT
"""



# _TENSION = 10

# def polar_sol_true(r, theta, S = 10, a = 2):
#     a2 = a*a
#     r2 = jnp.square(r)
#     A = -S * 0.25
#     B = 0
#     C = -a2**2 * S * 0.25
#     D = a2 * S * 0.5
#     return (A*r2 + B*r2**2 + C / r2 + D) * jnp.cos(2*theta)


# def cart_sol_true(x, y, **kwargs):
#     return polar_sol_true(xy2r(x, y), xy2theta(x, y), **kwargs)


# def sigma_rr_true(r, theta, S = 10, a = 2):
#     a2 = jnp.array(a*a)
#     r2 = jnp.square(r)
#     a2divr2 = jnp.divide(a2, r2)
#     return 0.5*S * ((1 + 3 * jnp.square(a2divr2) - 4 * a2divr2) * jnp.cos(2*theta) + (1 - a2divr2))


# def sigma_tt_true(r, theta, S = 10, a = 2):
#     a2 = jnp.array(a*a)
#     r2 = jnp.square(r)
#     a2divr2 = jnp.divide(a2, r2)
#     return 0.5*S * ((1 + a2divr2) - (1 + 3 * jnp.square(a2divr2)) * jnp.cos(2*theta))


# def sigma_rt_true(r, theta, S = 10, a = 2):
#     a2 = jnp.array(a*a)
#     r2 = jnp.square(r)
#     a2divr2 = jnp.divide(a2, r2)
#     return -0.5*S * (1 - 3 * jnp.square(a2divr2) + 2 * a2divr2) * jnp.sin(2*theta)


# def polar_stress_true(rtheta, **kwargs):

#     # Compute polar stresses from analytical solutions
#     rr_stress = sigma_rr_true(rtheta[0], rtheta[1], **kwargs)
#     rt_stress = sigma_rt_true(rtheta[0], rtheta[1], **kwargs)
#     tt_stress = sigma_tt_true(rtheta[0], rtheta[1], **kwargs)

#     # Format as stress tensor
#     return jnp.array([[rr_stress, rt_stress], [rt_stress, tt_stress]])


# def cart_stress_true(xy, **kwargs):

#     # Map cartesian coordinates to polar coordinates
#     rtheta = xy2rtheta(xy)
    
#     # Compute true stress tensor in polar coordinates
#     polar_stress_hessian = polar_stress_true(rtheta, **kwargs)

#     # Convert polar stress tensor to cartesian stress tensor
#     return polar2cart_tensor(polar_stress_hessian, rtheta)


# def loss_rect(*output: jax.Array, true_val: dict[str, jax.Array] | None = None):
    
#     # Unpack outputs
#     out0, out1, out2, out3 = output

#     # Return all losses
#     if true_val is None:
#         return mse(out0[:, 0]), mse(out0[:, 1]), \
#                mse(out1[:, 3]), mse(out1[:, 2]), \
#                mse(out2[:, 0]), mse(out2[:, 1]), \
#                mse(out3[:, 3]), mse(out3[:, 2])
#     return mse(out0[:, 0], true_val.get("xx0")), mse(out0[:, 1], true_val.get("xy0")), \
#            mse(out1[:, 3], true_val.get("yy1")), mse(out1[:, 2], true_val.get("xy1")), \
#            mse(out2[:, 0], true_val.get("xx2")), mse(out2[:, 1], true_val.get("xy2")), \
#            mse(out3[:, 3], true_val.get("yy3")), mse(out3[:, 2], true_val.get("xy3"))


# def loss_circ_rr_rt(input: jax.Array, output: jax.Array, true_val: dict[str, jax.Array] | None = None):

#     # Get polar angle
#     theta = xy2theta(input[:, 1], input[:, 0])
    
#     # Compute polar stresses (sigma_rr, sigma_rt)
#     srr = jnp.multiply(output[:, 3], jnp.square(jnp.cos(theta))) + \
#           jnp.multiply(output[:, 0], jnp.square(jnp.sin(theta))) - \
#           2*jnp.multiply(output[:, 1], jnp.sin(theta)*jnp.cos(theta))
#     srt = jnp.multiply(jnp.subtract(output[:, 0], output[:, 3]), jnp.sin(theta)*jnp.cos(theta)) - \
#           jnp.multiply(output[:, 1], jnp.square(jnp.cos(theta))-jnp.square(jnp.sin(theta)))
    
#     # Return both losses
#     if true_val is None:
#         return mse(srr), mse(srt)
#     return mse(srr, true_val.get("srr")), mse(srt, true_val.get("srt"))


# def loss_dirichlet(*output: jax.Array, true_val: dict[str, jax.Array] | None = None):

#     # Unpack outputs
#     out0, out1, out2, out3 = output
    
#     # Return all losses
#     if true_val is None:
#         return mse(out0), mse(out1), mse(out2), mse(out3)
#     return mse(out0, true_val.get("di0")), \
#            mse(out1, true_val.get("di1")), \
#            mse(out2, true_val.get("di2")), \
#            mse(out3, true_val.get("di3"))


# def get_true_vals(points: dict[str, jax.Array | tuple[dict[str, jax.Array]] | None],
#                   exclude: Sequence[str] | None = None,
#                   ylim = None) -> dict[str, jax.Array | dict[str, jax.Array] | None]:
#     vals = {}
#     if exclude is None:
#         exclude = []
    
#     # Homogeneous PDE ==> RHS is zero
#     if "coll" not in exclude:
#         coll = None
#         vals["coll"] = coll
    
#     # Only inhomogeneous BCs at two sides of rectangle
#     if "rect" not in exclude:
#         rect_points = [p.shape[0] for p in points["rect"]]
#         rect = {"yy1": jnp.full((rect_points[1],), _TENSION),
#                 "yy3": jnp.full((rect_points[3],), _TENSION)}
#         vals["rect"] = rect
    
#     # Homogeneous BC at inner circle
#     if "circ" not in exclude:
#         circ = None
#         vals["circ"] = circ
    
#     if "diri" not in exclude:
#         if ylim is None:
#             print("Y-limit defaults to [-10.0, 10.0]")
#             ylim = [-10.0, 10.0]
#         diri = {"di1": 0.5*_TENSION*(points["rect"][1][:, 1]-ylim[0])*(points["rect"][1][:, 1]-ylim[1]),
#                 "di3": 0.5*_TENSION*(points["rect"][3][:, 1]-ylim[0])*(points["rect"][3][:, 1]-ylim[1])}
#         vals["diri"] = diri

#     return vals