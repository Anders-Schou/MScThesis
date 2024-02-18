import os

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from models.pinn import PINN
from models.networks import setup_network
from setup.parsers import parse_arguments
from utils.plotting import save_fig

# Run function: python main.py --settings="settings.json"
# or:           . run.sh
raw_settings = parse_arguments()
network_settings = raw_settings["model"]["pinn"]["network"]
MLP_instance = setup_network(network_settings)
fig_dir = os.path.join(raw_settings["output_dir"], raw_settings["figure_sub_dir"])

# Alternative (while developing): Comment out above 4 lines and use below instead
#
# network_settings = {
#     "input_dim": 2,
#     "output_dim": 1,
#     "hidden_dims": [16, 16],
#     "activation": "tanh",
#     "initialization": "glorot_normal"
# }
# MLP_instance = setup_network(settings_dict)
# fig_dir = "path/to/figures"

print(network_settings)

seed = 1234
key = jax.random.PRNGKey(seed)

xlim = [0, 1]
ylim = [0, 1]

true_sol = lambda x, y: jnp.sin(x - y).reshape(x.shape)

num_bc = 100
num_c = 10000
num_test = 100

# Per dimension
shape_bc = (num_bc, 1)
shape_pde = (num_c, 1)
shape_test = (num_test, 1)

x_BC1 = jax.random.uniform(jax.random.PRNGKey(0), shape_bc, minval=xlim[0], maxval=xlim[1])
y_BC1 = jnp.full(shape_bc, ylim[0])

x_BC2 = jnp.full(shape_bc, xlim[1])
y_BC2 = jax.random.uniform(jax.random.PRNGKey(0), shape_bc, minval=ylim[0], maxval=ylim[1])

x_BC3 = jax.random.uniform(jax.random.PRNGKey(0), shape_bc, minval=xlim[0], maxval=xlim[1])
y_BC3 = jnp.full(shape_bc, ylim[1])

x_BC4 = jnp.full(shape_bc, xlim[0])
y_BC4 = jax.random.uniform(jax.random.PRNGKey(0), shape_bc, minval=ylim[0], maxval=ylim[1])

x_BC = jnp.concatenate([x_BC1, x_BC2, x_BC3, x_BC4])
y_BC = jnp.concatenate([y_BC1, y_BC2, y_BC3, y_BC4])

BC = jnp.stack([x_BC, y_BC], axis=1).reshape((-1,2))
BC_true = true_sol(BC[:,0], BC[:,1])

key, x_key, y_key, x_key_test, y_key_test = jax.random.split(key, 5)

x_PDE = jax.random.uniform(x_key, shape_pde, minval=xlim[0], maxval=xlim[1])
y_PDE = jax.random.uniform(y_key, shape_pde, minval=ylim[0], maxval=ylim[1])
PDE = jnp.stack([x_PDE, y_PDE], axis=1).reshape((-1,2))

x_test = jax.random.uniform(x_key_test, shape_test, minval=xlim[0], maxval=xlim[1])
y_test = jax.random.uniform(y_key_test, shape_test, minval=ylim[0], maxval=ylim[1])
test_points = jnp.stack([x_test, y_test], axis=1).reshape((-1,2))

# Plot training points
plt.scatter(PDE[:,0], PDE[:,1])
plt.scatter(BC[:,0], BC[:,1], c = 'green')
save_fig(fig_dir, "training_points", format="png")
plt.clf()

# Plot test points
plt.scatter(test_points[:,0], test_points[:,1], c = 'green')
save_fig(fig_dir, "test_points", format="png")
plt.clf()

# Create and train PINN
pinn = PINN(net=MLP_instance)
pinn.train(10000, 1000, PDE, BC, BC_true)

# Print MSE based on test points
print(f"MSE: {((pinn.predict(test_points).flatten() - true_sol(test_points[:,0], test_points[:,1]).flatten())**2).mean():2.2e}")

x = jnp.arange(xlim[0], xlim[1], 0.01)
y = jnp.arange(ylim[0], ylim[1], 0.01)
X, Y = jnp.meshgrid(x, y)
plotpoints = np.concatenate((X.reshape((-1, 1)), Y.reshape((-1, 1))), axis=1)
z = pinn.predict(plotpoints)
Z = z.reshape((x.size, y.size))

# Plot true values
plt.contourf(X, Y, true_sol(X, Y))
plt.colorbar()
save_fig(fig_dir, "true", format="png")
plt.clf()

# Plot prediction
plt.contourf(X, Y, Z)
plt.colorbar()
save_fig(fig_dir, "prediction", format="png")
plt.clf()

# Plot absolute error
plt.contourf(X, Y, jnp.abs(Z - true_sol(X, Y)))
plt.colorbar()
save_fig(fig_dir, "error_abs", format="png")
plt.clf()

# plt.savefig(os.path.join(dir, file_name)
