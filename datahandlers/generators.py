from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import torch.utils.data

from datahandlers.samplers import sample_line
from utils.utils import limits2vertices



class Dataset(torch.utils.data.Dataset):

    def __init__(self, data: dict) -> None:
        # Set data and "labels"
        # Input structure: One pair for each condition
        # data  =  {
        #           "pde": (jnp.ndarray <- collocation points, jnp.ndarray <- RHS if any),
        #           "xy": (jnp.ndarray <- [x, y, z, ...], jnp.ndarray <- [u_xy(x, y, z, ...), v_xy(x, ...), ...]),
        #           "xx": (jnp.ndarray <-   ---||---    , jnp.ndarray <-   ---||---                       ),
        #           .
        #           .
        #           .
        #           "val", (jnp.ndarray <- x_val, jnp.ndarray <- u_val)
        #           }
        #
        self.data = data
        pass
    
    def __len__(self):
        # Define "length" of dataset
        return 150

    def __getitem__(self, index):
        # Define which data to fetch based on input index
        return index, index
    
    def __sample_data(self, key):
        pass

    def __repr__(self):
        num_points = 5
        r = 4
        repr_str = "\n" + "--------" * r + "  DATASET  " + "--------" * r + "\n\n"
        repr_str += f"Printing first {num_points} data points for each entry:\n\n"
        for key, (x, u) in self.data.items():
            n = min(len(x), num_points)
            repr_str += (16*r+11) * "-"
            repr_str += f"\nType: '{key}'"
            repr_str += "\n\nData points:\n"
            repr_str += str(x[:n])
            repr_str += "\n\nFunction values\n"
            repr_str += str(u[:n])
            repr_str += "\n\n\n"
        repr_str += "\n"
        repr_str += (16*r+11) * "-"
        repr_str += "\n"
        return repr_str


def numpy_collate(batch):
    return tree_map(jnp.asarray, torch.utils.data.default_collate(batch))


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
              batch_size=batch_size,
              shuffle=shuffle,
              sampler=sampler,
              batch_sampler=batch_sampler,
              num_workers=num_workers,
              collate_fn=numpy_collate,
              pin_memory=pin_memory,
              drop_last=drop_last,
              timeout=timeout,
              worker_init_fn=worker_init_fn)


def generate_rectangle_points(key: jax.random.PRNGKey,
                              xlim: Sequence[float],
                              ylim: Sequence[float],
                              num_points: int | Sequence
                              ) -> tuple:
    """
    Order of rectangle sides: Lower horizontal, right vertical, upper horizontal, left vertical.
    """
    if isinstance(num_points, int):
        N = [num_points] * 4
    elif isinstance(num_points, Sequence):
        if len(num_points) == 1:
            N = num_points * 4
        elif len(num_points) == 4:
            N = num_points
        else:
            raise ValueError(f"Wrong length of 'num_points': f{len(num_points)}. Sequence length must be either 1 or 4.")
    else:
        raise ValueError("Argument 'num_points' must be int or tuple.")
    
    key, *keys = jax.random.split(key, 5)
    end_points = limits2vertices(xlim, ylim)
    points = [sample_line(keys[i], end_points[i], shape=(N[i], 1)) for i in range(4)]

    return tuple(points)


def generate_circle_points(key: jax.random.PRNGKey,
                           radius: float,
                           num_points: int
                           ) -> jnp.ndarray:
    theta = jax.random.uniform(key, (num_points, 1), minval=0, maxval=2*jnp.pi)
    xc = radius*jnp.cos(theta)
    yc = radius*jnp.sin(theta)
    xyc = jnp.stack([xc, yc], axis=1).reshape((-1,2))
    return xyc


if __name__ == "__main__":
    from utils.utils import out_shape

    key = jax.random.PRNGKey(123)
    points = generate_rectangle_points(key, [0, 1], [0, 1], 150)
    test_fun = lambda p: (p[:, 0]**2 + p[:, 1]**2).reshape(-1, 1)
    data_dict = {"yy": (points[0], test_fun(points[0])),
                 "xy": (points[1], test_fun(points[1])),
                 "xx": (points[0], test_fun(points[0])),}
    dataset = Dataset(data_dict)
    batch_size = 16
    print(dataset)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for i, batch in enumerate(dataloader):
        print("Batch", i, "with indices", batch[0], "\n\n")