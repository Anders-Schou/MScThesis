import jax
import jax.numpy as jnp
from functools import partial


class Counter:
    """A simple counter."""

    def __init__(self):
        self.n = 0

    @partial(jax.jit, static_argnums=(0,))
    def count(self) -> int:
        """Increments the counter and returns the new value."""
        self.n += 1
        return self.n

    def reset(self):
        """Resets the counter to zero."""
        self.n = 0


counter = Counter()

for _ in range(3):
    print(counter.count())


counter.reset()
fast_count = counter.count

for _ in range(3):
    print(fast_count())