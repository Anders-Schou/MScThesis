import os

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import matplotlib.pyplot as plt

from setup.parsers import parse_arguments
from models.pinn import Pinn
from models.networks import setup_network

# Run function: python main.py --settings="settings.json"
# or:           . run.sh
raw_settings = parse_arguments()
MLP_instance = setup_network(raw_settings["model"]["pinn"]["network"])
fig_dir = os.path.join(raw_settings["output_dir"], raw_settings["figure_sub_dir"])

# Alternative (while developing): Comment out above 3 lines and use below instead
#
# settings_dict = {
#     "input_dim": 2,
#     "output_dim": 1,
#     "hidden_dims": [16, 16],
#     "activation": "tanh",
#     "initialization": "glorot_normal"
# }
# MLP_instance = setup_network(settings_dict)
# fig_dir = "path/to/figures"

xlim = [0, 1]
ylim = [0, 1]

true_sol = lambda x, y: jnp.sin(x - y).reshape(x.shape)

num_bc = 100
num_c = 10000

x_BC1 = jax.random.uniform(jax.random.PRNGKey(0), (num_bc,1), minval=0, maxval=1)
y_BC1 = jax.random.uniform(jax.random.PRNGKey(0), (num_bc,1), minval=0, maxval=0)

x_BC2 = jax.random.uniform(jax.random.PRNGKey(0), (num_bc,1), minval=1, maxval=1)
y_BC2 = jax.random.uniform(jax.random.PRNGKey(0), (num_bc,1), minval=0, maxval=1)

x_BC3 = jax.random.uniform(jax.random.PRNGKey(0), (num_bc,1), minval=0, maxval=1)
y_BC3 = jax.random.uniform(jax.random.PRNGKey(0), (num_bc,1), minval=1, maxval=1)

x_BC4 = jax.random.uniform(jax.random.PRNGKey(0), (num_bc,1), minval=0, maxval=0)
y_BC4 = jax.random.uniform(jax.random.PRNGKey(0), (num_bc,1), minval=0, maxval=1)

x_BC = jnp.concatenate([x_BC1, x_BC2, x_BC3, x_BC4])
y_BC = jnp.concatenate([y_BC1, y_BC2, y_BC3, y_BC4])


BC = jnp.stack([x_BC, y_BC], axis=1).reshape((-1,2))
BC_true = true_sol(BC[:,0], BC[:,1])

PDE = jax.random.uniform(jax.random.PRNGKey(12), (num_c, 2), minval=0, maxval=1)

plt.scatter(PDE[:,0], PDE[:,1])
plt.scatter(BC[:,0], BC[:,1], c = 'green')
plt.savefig(fig_dir)
plt.clf()

pinn = Pinn(net=MLP_instance)

pinn.train(10000, 1000, PDE, BC, BC_true)


test = jax.random.uniform(jax.random.PRNGKey(123), (100, 2), minval=0, maxval=1)

plt.scatter(test[:,0], test[:,1], c = 'green')
plt.savefig(fig_dir)
plt.clf()

print(((pinn.predict(test) - true_sol(test[:,0], test[:,1]))**2).mean())

x = jnp.arange(0, 1, 0.01)
y = jnp.arange(0, 1, 0.01)
X, Y = jnp.meshgrid(x, y)
plotpoints = np.concatenate((X.reshape((-1, 1)), Y.reshape((-1, 1))), axis=1)
z = pinn.predict(plotpoints)
Z = z.reshape((x.size, y.size))


plt.contourf(X, Y, jnp.abs(Z - true_sol(X, Y)))
plt.colorbar()
plt.savefig(fig_dir)
plt.clf()

plt.contourf(X, Y, true_sol(X, Y))
plt.colorbar()
plt.savefig(fig_dir)
plt.clf()

plt.contourf(X, Y, Z)
plt.colorbar()
plt.savefig(fig_dir)
plt.clf()