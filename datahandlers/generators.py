from functools import partial
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import jax
import jax.numpy as jnp
from jax import tree_util
import torch.utils.data
import flax

from datahandlers.samplers import sample_line
from utils.utils import limits2vertices, remove_points, keep_points


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

def generate_collocation_points(key: jax.random.PRNGKey,
                                xlim: Sequence[float],
                                ylim: Sequence[float],
                                num_coll: int) -> jnp.ndarray:
    shape_pde = (num_coll, 1)
    
    x_key, y_key = jax.random.split(key, 2)

    x_train = jax.random.uniform(x_key, shape_pde, minval=xlim[0], maxval=xlim[1])
    y_train = jax.random.uniform(y_key, shape_pde, minval=ylim[0], maxval=ylim[1])
    xp = jnp.stack([x_train, y_train], axis=1).reshape((-1,2))
    
    return xp

def generate_extra_points(keyr, keytheta, radius, num_extra):
    theta_rand = jax.random.uniform(keytheta, (num_extra, 1), minval=0, maxval=2*jnp.pi)
    r_rand = jax.random.chisquare(keyr, df=2, shape=(num_extra, 1)) / 10 + radius
    xy_extra = jnp.stack([r_rand*jnp.cos(theta_rand), r_rand*jnp.sin(theta_rand)], axis=1).reshape((-1,2))
    return xy_extra


def generate_collocation_points_with_hole(key: jax.random.PRNGKey,
                                          radius: float, 
                                          xlim: Sequence[float],
                                          ylim: Sequence[float],
                                          num_coll: int
                                          ):
    num_extra = num_coll

    key, key_coll = jax.random.split(key)
    

    xy_coll = generate_collocation_points(key_coll, xlim, ylim, num_coll)
    xy_coll = remove_points(xy_coll, lambda p: jnp.linalg.norm(p, axis=-1) <= radius)
    
    key, keytheta, keyr = jax.random.split(key, 3)
    xy_extra = generate_extra_points(keyr, keytheta, radius, num_extra)
    xy_extra = keep_points(xy_extra, lambda p: jnp.logical_and(jnp.logical_and(p[:, 0] >= xlim[0], p[:, 0] <= xlim[1]),
                                                               jnp.logical_and(p[:, 1] >= ylim[0], p[:, 1] <= ylim[1])))

    xy_coll = jnp.concatenate((xy_coll, xy_extra))
    return xy_coll

def generate_rectangle_with_hole(key: jax.random.PRNGKey,
                                radius: float, 
                                xlim: Sequence[float],
                                ylim: Sequence[float],
                                num_coll: int,
                                num_rBC: int,
                                num_cBC: int,
                                num_test: int = 0):

    num_extra = num_coll

    key, rkey, ckey, collkey, testkey = jax.random.split(key, 5)
    

    xy_coll = generate_collocation_points(collkey, xlim, ylim, num_coll)
    xy_coll = remove_points(xy_coll, lambda p: jnp.linalg.norm(p, axis=-1) <= radius)
    xy_rect = generate_rectangle_points(rkey, xlim, ylim, num_rBC)
    xy_circ = generate_circle_points(ckey, radius, num_cBC)
    xy_test = generate_collocation_points(testkey, xlim, ylim, num_test)
    xy_test = remove_points(xy_test, lambda p: jnp.linalg.norm(p, axis=-1) <= radius)
    
    key, keytheta, keyr = jax.random.split(key, 3)
    theta_rand = jax.random.uniform(keytheta, (num_extra, 1), minval=0, maxval=2*jnp.pi)
    r_rand = jax.random.chisquare(keyr, df=2, shape=(num_extra, 1)) / 10 + radius
    xy_extra = jnp.stack([r_rand*jnp.cos(theta_rand), r_rand*jnp.sin(theta_rand)], axis=1).reshape((-1,2))

    xy_extra = keep_points(xy_extra, lambda p: jnp.logical_and(jnp.logical_and(p[:, 0] >= xlim[0], p[:, 0] <= xlim[1]),
                                                               jnp.logical_and(p[:, 1] >= ylim[0], p[:, 1] <= ylim[1])))

    xy_coll = jnp.concatenate((xy_coll, xy_extra))

    return tuple([xy_coll, xy_rect, xy_circ, xy_test]) #Skal key returnes her og i gen_collocation?
    

def numpy_collate(batch):
    return tree_util.tree_map(np.asarray, torch.utils.data.default_collate(batch))


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dict: dict, seed: int = 0) -> None:
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

        self.key = jax.random.PRNGKey(seed)

        # Data types 
        self.data_keys = [key for key in data_dict.keys()]
        
        # Extract data lengths and calculate 'global' indices of data
        length = [value[0].shape[0] for value in data_dict.values()]
        indx = [jnp.arange(sum(length[:i]), sum(length[:i+1])) for i, _ in enumerate(length)]
        self.data_length_accum = [sum(length[:i+1]) for i, _ in enumerate(length)]

        # Create dictionaries with data lengths and indices
        self.data_length = flax.core.FrozenDict([(data_key, data_length) for data_key, data_length in zip(self.data_keys, length)])
        self.data_indx = flax.core.FrozenDict([(data_key, data_indx) for data_key, data_indx in zip(self.data_keys, indx)])
        self.indx_dict = flax.core.FrozenDict([(key, jnp.arange(value[0].shape[0])) for i, (key, value) in enumerate(data_dict.items())])

        # Flattened/concatenated dataset
        self.x = np.vstack([data[0] for data in data_dict.values()])
        self.u = np.vstack([data[1] for data in data_dict.values()])
        
        return

    
    def __len__(self):
        # Define "length" of dataset
        return self.x.shape[0]

    def __getitem__(self, index):
        # Define which data to fetch based on input index
        type_idx = np.searchsorted(self.data_length_accum, index)
        # batch = (self.x[index], self.u[index], type_idx)
        return index, self.x[index], self.u[index], type_idx
        # return index, index

    def __repr__(self):
        num_points = 5
        r = 4
        repr_str = "\n" + "--------" * r + "  DATASET  " + "--------" * r + "\n\n"
        repr_str += f"Printing first {num_points} data points for each entry:\n\n"
        for key, idx in self.data_indx.items():
            n = min(len(idx), num_points)
            repr_str += (16*r+11) * "-"
            repr_str += f"\nType: '{key}'"
            repr_str += "\n\nData points:\n"
            repr_str += str(self.x[idx[:n]])
            repr_str += "\n\nFunction values\n"
            repr_str += str(self.u[idx[:n]])
            repr_str += "\n\n\n"
        repr_str += "\n"
        repr_str += (16*r+11) * "-"
        repr_str += "\n"
        return repr_str


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

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    for i, (global_indices, x, u, tidx) in enumerate(dataloader):
        print("Batch", i, "with indices", global_indices, "\n", x, "\n", u, "\n", tidx, "\n\n")
    