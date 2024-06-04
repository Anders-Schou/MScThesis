from collections.abc import Sequence

import jax
import jax.numpy as jnp

# from setup.parsers import convert_sampling_distribution


def sample_line(key,
                end_points: Sequence[Sequence], # Outer sequence length should be 2 (there are 2 end points)
                *args,
                ref_scale: float = 1.0,
                ref_offset: float = 0.0,
                distribution: str = "uniform",
                **kwargs) -> jnp.ndarray:
    if len(end_points) != 2:
        raise ValueError(f"Length of argument 'end_points' should be 2 but was {len(end_points)}.")
    if distribution == "uniform":
        sample_fun = jax.random.uniform
    else:
        raise ValueError("Unknown sampling distribution.")
    sample_points = sample_fun(key, *args, **kwargs)
    ref_points = (sample_points - ref_offset) / ref_scale
    p1 = jnp.array(end_points[0], dtype=jnp.float32)
    p2 = jnp.array(end_points[1], dtype=jnp.float32)
    return p1*ref_points + p2*(1-ref_points)


