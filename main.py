from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter

from models.networks import setup_network, setup_run
from models.pinn import PINN
from datahandlers.generators import generate_rectangle_with_hole
from setup.parsers import parse_arguments
from utils.plotting import save_fig, plot_potential, plot_stress, plot_polar_stress, get_plot_variables
from utils.transforms import *
from utils.platewithhole import cart_sol_true, cart_stress_true, polar_stress_true



def print_info(network_settings, run_settings):
    print("")
    [print(f"{key+":":<30}", value) for key, value in network_settings.items()]
    print("")
    [print(f"{key+":":<30}", value) for key, value in run_settings.items()]
    print("")


raw_settings = parse_arguments()
network_settings = raw_settings["model"]["pinn"]["network"]
MLP_instance = setup_network(network_settings)
run_settings = setup_run(raw_settings["run"])
fig_dir = raw_settings["IO"]["figure_dir"]
do_sample_data_plots = raw_settings["plotting"]["do_sample_data_plots"]
do_result_plots = raw_settings["plotting"]["do_result_plots"]
do_logging = raw_settings["logging"]
print_info(network_settings, run_settings)

CLEVELS = 101
SEED = 1234
key = jax.random.PRNGKey(SEED)



TENSION = 10
R = 10
XLIM = [-R, R]
YLIM = [-R, R]
RADIUS = 2.0

# [xx, xy, yx, yy]
OUTER_STRESS = ([TENSION, 0], [0, 0])
# [rr, rt, tr, tt]
INNER_STRESS = ([0, 0], [0, 0])




A = -TENSION*RADIUS**2/2
B = 0
C = TENSION*RADIUS**2/2
D = -TENSION*RADIUS**4/4
true_sol_polar = lambda r, theta: ((TENSION*r**2/4 - TENSION*r**2*jnp.cos(2*theta)/4) + A*jnp.log(r) + B*theta + C*jnp.cos(2*theta) + D*jnp.pow(r,-2)*jnp.cos(2*theta))
true_sol_cart = lambda x, y: true_sol_polar(jnp.sqrt(x**2 + y**2), jnp.arctan2(y, x))


num_coll = 10000
num_rBC = 500
num_cBC = 1000
num_test = 100

key, subkey = jax.random.split(key, 2)

xy_coll, xy_rect_tuple, xy_circ, xy_test = generate_rectangle_with_hole(subkey, RADIUS, XLIM, YLIM, num_coll, num_rBC, num_cBC, num_test)
xy_rect = jnp.concatenate(xy_rect_tuple)

x = (xy_coll, xy_rect_tuple, xy_circ)

sigma = (OUTER_STRESS, INNER_STRESS)

u_rect_tuple = tuple([cart_sol_true(xy_rect_tuple[i][:, 0], xy_rect_tuple[i][:, 1], S=TENSION, a=RADIUS).reshape(-1, 1) for i in range(4)])
u_circ = cart_sol_true(xy_circ[:, 0], xy_circ[:, 1], S=TENSION, a=RADIUS).reshape(-1, 1)
u_coll = cart_sol_true(xy_coll[:, 0], xy_coll[:, 1], S=TENSION, a=RADIUS).reshape(-1, 1)
u_rect = jnp.concatenate(u_rect_tuple)
u = (u_coll, u_rect_tuple, u_circ)


if do_sample_data_plots:
    # Plot training points
    plt.scatter(xy_coll[:, 0], xy_coll[:, 1])
    plt.scatter(xy_rect[:, 0], xy_rect[:, 1], c='green')
    plt.scatter(xy_circ[:, 0], xy_circ[:, 1], c='red')
    save_fig(fig_dir, "training_points", format="png")
    plt.clf()

    # Plot test points
    plt.scatter(xy_test[:,0], xy_test[:,1], c = 'green')
    save_fig(fig_dir, "test_points", format="png")
    plt.clf()

# Create and train PINN
print("Creating PINN:")
pinn = PINN(MLP_instance, run_settings["specifications"], logging=do_logging)
print("PINN created!")

# Collect all data in list of tuples (1 tuple for each loss term)
# losses = [jax.vmap(biharmonic, in_axes=(None, 0)), ]
# batch = [(xp, up), (xc, uc), *[(x, u) for x, u in zip(rect_points, rect_stress)]]

# for b in batch:
#     print("TYPE:      ", type(b))
#     print("LENGTH:    ", len(b))
#     print("SHAPE:     ", b[0].shape, b[1].shape)

print("Entering training phase:")
t1 = perf_counter()
pinn.train(run_settings["specifications"]["iterations"], 200, x, u, sigma)
t2 = perf_counter()
print("Training done!")
print(f"Time: {t2-t1:.2f}")



if do_result_plots:
    X, Y, plotpoints = get_plot_variables(XLIM, YLIM, grid=201)
    R, THETA, plotpoints_polar = get_plot_variables([RADIUS, R], [0, 4*jnp.pi], grid=201)
    plotpoints2 = jax.vmap(rtheta2xy)(plotpoints_polar)
    
    phi = pinn.predict(plotpoints).reshape(X.shape)*(xy2r(X, Y) >= RADIUS)
    phi_polar = pinn.predict(plotpoints2).reshape(R.shape)
    phi_true = cart_sol_true(X, Y, S=TENSION, a=RADIUS)*(xy2r(X, Y) >= RADIUS)

    # Hessian prediction
    phi_pp = pinn.hessian_flatten(pinn.params, plotpoints)
    
    # Calculate stress from phi function: phi_xx = sigma_yy, phi_yy = sigma_xx, phi_xy = -sigma_xy
    sigma_cart = phi_pp[:, [3, 1, 2, 0]]
    sigma_cart = sigma_cart.at[:, [1, 2]].set(-phi_pp[:, [1, 2]])

    # List and reshape the four components
    sigma_cart_list = [sigma_cart[:, i].reshape(X.shape)*(xy2r(X, Y) >= RADIUS) for i in range(4)]

    # Repeat for the other set of points (polar coords converted to cartesian coords)
    phi_pp2 = pinn.hessian_flatten(pinn.params, plotpoints2)

    # Calculate stress from phi function
    sigma_cart2 = phi_pp2[:, [3, 1, 2, 0]]
    sigma_cart2 = sigma_cart2.at[:, [1, 2]].set(-phi_pp2[:, [1, 2]])
    
    # Convert these points to polar coordinates before listing and reshaping
    sigma_polar = jax.vmap(cart2polar_tensor, in_axes=(0, 0))(sigma_cart2.reshape(-1, 2, 2), plotpoints2).reshape(-1, 4)
    sigma_polar_list = [sigma_polar[:, i].reshape(R.shape)*(R >= RADIUS) for i in range(4)]

    # Calculate true stresses (cartesian and polar)
    sigma_cart_true = jax.vmap(cart_stress_true)(plotpoints)
    sigma_cart_true_list = [sigma_cart_true.reshape(-1, 4)[:, i].reshape(X.shape)*(xy2r(X, Y) >= RADIUS) for i in range(4)]
    sigma_polar_true = jax.vmap(polar_stress_true)(plotpoints_polar)
    sigma_polar_true_list = [sigma_polar_true.reshape(-1, 4)[:, i].reshape(R.shape)*(R >= RADIUS) for i in range(4)]
    
    plot_potential(X, Y, phi, phi_true, fig_dir, "potential", radius=RADIUS)
    plot_stress(X, Y, sigma_cart_list, sigma_cart_true_list, fig_dir, "stress", radius=RADIUS)
    plot_polar_stress(R, THETA, sigma_polar_list, sigma_polar_true_list, fig_dir, "stress_polar", radius=RADIUS)


