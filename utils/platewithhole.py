import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils.transforms import xy2r, xy2theta, xy2rtheta, polar2cart_tensor, cart2polar_tensor
from utils.plotting import plot_circle


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


def polar_stress_true(rt):

    # Compute polar stresses from analytical solutions
    rr_stress = sigma_rr_true(rt[0], rt[1])
    rt_stress = sigma_rt_true(rt[0], rt[1])
    tt_stress = sigma_tt_true(rt[0], rt[1])

    # Format as stress tensor
    return jnp.array([[rr_stress, rt_stress], [rt_stress, tt_stress]])


def cart_stress_true(xy):

    # Map cartesian coordinates to polar coordinates
    rtheta = xy2rtheta(xy)
    
    # Compute true stress tensor in polar coordinates
    polar_stress_hessian = polar_stress_true(rtheta)

    # Convert polar stress tensor to cartesian stress tensor
    return polar2cart_tensor(polar_stress_hessian, rtheta)


# def true_sol_cart(x, y, constants):
#     return true_sol_polar(xy2r(x, y), xy2theta(x, y), constants)

# def true_sol_polar(r, theta, constants):
#     TENSION, A, B, C, D = constants
#     r2 = jnp.square(2)
#     return (TENSION*r2/4 - TENSION*r2*jnp.cos(2*theta)/4) + A*jnp.log(r) + B*theta + C*jnp.cos(2*theta) + D*jnp.divide(jnp.cos(2*theta), r2)


# def sigma_xx_true(x, y, tension, radius, mask=True):
#     sigma_xx = (tension - tension*(radius/xy2r(x, y))**2*(1.5*jnp.cos(2*xy2theta(x, y))+jnp.cos(4*xy2theta(x, y)))+tension*1.5*(radius/xy2r(x, y))**4*jnp.cos(4*xy2theta(x,y))) * (xy2r(x, y) >= radius)


# sigma_yy_true = lambda x, y: (-TENSION*(RADIUS/r_func(x, y))**2*(0.5*jnp.cos(2*t_func(x, y))-jnp.cos(4*t_func(x, y)))-TENSION*1.5*(RADIUS/r_func(x, y))**4*jnp.cos(4*t_func(x,y))) * (r_func(x, y) >= RADIUS)
# sigma_xy_true = lambda x, y: (-TENSION*(RADIUS/r_func(x, y))**2*(0.5*jnp.sin(2*t_func(x, y))+jnp.sin(4*t_func(x, y)))+TENSION*1.5*(RADIUS/r_func(x, y))**4*jnp.sin(4*t_func(x,y))) * (r_func(x, y) >= RADIUS)

if __name__ == "__main__":
    x = jnp.linspace(-10, 10, 201)
    y = x.copy()
    RADIUS = 2

    X, Y = jnp.meshgrid(x, y)

    P = jnp.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
    
    true_sol = lambda p: cart_sol_true(p[0], p[1])

    fun = jax.vmap(jax.hessian(true_sol, argnums=0))


    ptest = jnp.array([[ 1,  0],
                       [ 1,  1],
                       [ 0,  1],
                       [-1,  1],
                       [-1,  0],
                       [-1, -1],
                       [ 0, -1],
                       [ 1, -1]])

    rtest = jnp.array([[ 1,  0],
                       [ 1,  0.5*jnp.pi],
                       [ 1, -0.5*jnp.pi]])

    # ftest = jax.vmap(cart2polar_tensor, in_axes=(None, 0))

    Sxy = jnp.array([[1, 0], [0, 0]])
    Srt = jnp.array([[1, 0], [0, 0]])

    print(jnp.round(cart2polar_tensor(Sxy, ptest), decimals=2))
    print("")
    print(jnp.round(polar2cart_tensor(Srt, rtest), decimals=2))
    # Z = fun(P)

    