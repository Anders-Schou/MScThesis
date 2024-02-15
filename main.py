from models.pinn import Pinn
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

xlim = [0, 1]
ylim = [0, 1]



pinn = Pinn([2, 10, 10, 1], jnp.tanh)








x_PDE = jax.random.normal(jax.random.key(0),(1000,2))
x_BC = jax.random.normal(jax.random.key(0),(1000,2))
y_BC = jnp.ones((1000,1))
print(pinn.predict(x_BC))

print(pinn.train(10000, 10, x_PDE, x_BC, y_BC))
jax.tree_map(print, pinn.params)