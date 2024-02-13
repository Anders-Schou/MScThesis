import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    u = x[0]**2 + x[1]**3
    return u

v = jnp.array([1.0, 2.0])
v2 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
a, b = jax.jvp(f, (v,), (v,))

g = jax.grad(f)
g2 = jax.jacrev(g)
g3 = jax.jacrev(g2)
g4 = jax.jacrev(g3)

print(b)
print(g(v))
print(g2(v))
print(g3(v))
print(g4(v))

print(jax.devices())

plt.scatter(np.linspace(0, 5, 51), np.random.rand(51))
plt.savefig("../figures/testfig3.pdf", format="pdf")
