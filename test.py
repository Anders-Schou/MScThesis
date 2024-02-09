import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    u = x[0]**2 + x[1]**3
    return u

v = jnp.array([1.0, 2.0])
a, b = jax.jvp(f, (v,), (v,))

g = jax.grad(f)

print(b)
print(g(v))
print(jax.devices())

plt.scatter(np.linspace(0, 5, 51), np.random.rand(51))
plt.savefig("testfig2.pdf", format="pdf")
