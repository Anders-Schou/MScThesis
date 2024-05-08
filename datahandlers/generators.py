from collections.abc import Sequence
import math

import numpy as np
from scipy.stats.qmc import Sobol
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import flax
import torch.utils.data

from datahandlers.samplers import sample_line
from utils.utils import limits2vertices, remove_points, keep_points


def generate_interval_points(key: jax.random.PRNGKey,
                             xlim: Sequence[float],
                             num_points: int,
                             sobol: bool = True
                             ):
    
    s_key, key = jax.random.split(key, 2)
    if sobol:
        # Use Sobol QMC sampling (convert key to seed, to make sampling "deterministic")
        xp = Sobol(1, seed=int(jax.random.randint(s_key, (), 0,
                                                  jnp.iinfo(jnp.int32).max))
            ).random_base2(math.ceil(jnp.log2(num_points)))
        
        return jnp.array(xp*(xlim[1]-xlim[0]) + xlim[0])
    
    # Use uniform sampling
    return jax.random.uniform(s_key, (num_points, 1), minval=xlim[0], maxval=xlim[1])
    



def generate_rectangle_points(key: jax.random.PRNGKey,
                              xlim: Sequence[float],
                              ylim: Sequence[float],
                              num_points: int | Sequence[int]
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
                           ) -> jax.Array:
    theta = jax.random.uniform(key, (num_points, 1), minval=0, maxval=2*jnp.pi)
    xc = radius*jnp.cos(theta)
    yc = radius*jnp.sin(theta)
    xyc = jnp.stack([xc, yc], axis=1).reshape((-1,2))
    return xyc


def generate_collocation_points(key: jax.random.PRNGKey,
                                xlim: Sequence[float],
                                ylim: Sequence[float],
                                num_coll: int,
                                sobol: bool = True) -> jax.Array:
    
    if sobol:
        # Use Sobol QMC sampling (convert key to seed, to make sampling "deterministic")
        s_key, key = jax.random.split(key, 2)
        xp = Sobol(2, seed=int(jax.random.randint(s_key, (), 0,
                                                  jnp.iinfo(jnp.int32).max))
            ).random_base2(math.ceil(jnp.log2(num_coll)))
        
        xp[:, 0] = xp[:, 0]*(xlim[1]-xlim[0]) + xlim[0]
        xp[:, 1] = xp[:, 1]*(ylim[1]-ylim[0]) + ylim[0]
        return jnp.array(xp)
    
    # Uniform sampling
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
                                          points: int | Sequence[int] | None,
                                          sobol: bool = True
                                          ) -> jax.Array:
    """
    This function samples points in the inner of the domain.
    """
    if points is None:
        return jnp.empty((0,))

    if not isinstance(points, Sequence):
        points = [points]
    
    num_coll = points[0]

    # Initial coll point gen
    key, key_coll = jax.random.split(key)
    xy_coll = generate_collocation_points(key_coll, xlim, ylim, num_coll, sobol=sobol)
    xy_coll = remove_points(xy_coll, lambda p: jnp.linalg.norm(p, axis=-1) <= radius)
    
    # Filler coll point gen
    pnum = xy_coll.shape[0]
    while pnum < num_coll:
        key, key_coll = jax.random.split(key_coll)
        tmp = generate_collocation_points(key_coll, xlim, ylim, num_coll, sobol=sobol)
        tmp = remove_points(tmp, lambda p: jnp.linalg.norm(p, axis=-1) <= radius)
        xy_coll = jnp.concatenate((xy_coll, tmp))
        pnum = xy_coll.shape[0]
    xy_coll = xy_coll[:num_coll]

    # Return if no extra points should be generated
    if len(points) == 1:
        return xy_coll
    
    num_extra = points[1]

    # Initial extra point gen
    key, keytheta, keyr = jax.random.split(key, 3)
    xy_extra = generate_extra_points(keyr, keytheta, radius, num_extra)
    xy_extra = keep_points(xy_extra, lambda p: jnp.logical_and(jnp.logical_and(p[:, 0] >= xlim[0], p[:, 0] <= xlim[1]),
                                                               jnp.logical_and(p[:, 1] >= ylim[0], p[:, 1] <= ylim[1])))
    
    # Filler extra point gen
    pnum = xy_extra.shape[0]
    while pnum < num_extra:
        key, keytheta, keyr = jax.random.split(key, 3)
        tmp = generate_extra_points(keyr, keytheta, radius, num_extra)
        tmp = keep_points(xy_extra, lambda p: jnp.logical_and(jnp.logical_and(p[:, 0] >= xlim[0], p[:, 0] <= xlim[1]),
                                                               jnp.logical_and(p[:, 1] >= ylim[0], p[:, 1] <= ylim[1])))
        xy_extra = jnp.concatenate((xy_extra, tmp))
        pnum = xy_extra.shape[0]
    xy_extra = xy_extra[:num_extra]
    
    # Collect all points
    return jnp.concatenate((xy_coll, xy_extra))
    

def generate_rectangle_with_hole(key: jax.random.PRNGKey,
                                 radius: float, 
                                 xlim: Sequence[float],
                                 ylim: Sequence[float],
                                 num_coll: int | Sequence[int],
                                 num_rect: int | Sequence[int],
                                 num_circ: int,
                                 sobol: bool = True
                                 ) -> dict[str, jax.Array | tuple[jax.Array]]:
    """
    Main function for generating necessary sample points for the plate-with-hole problem.

    The function generates 
    """


    key, rectkey, circkey, collkey, permkey = jax.random.split(key, 5)

    xy_coll = generate_collocation_points_with_hole(collkey, radius, xlim, ylim, num_coll, sobol=sobol)
    xy_coll = jax.random.permutation(permkey, xy_coll)
    xy_rect = generate_rectangle_points(rectkey, xlim, ylim, num_rect)
    xy_circ = generate_circle_points(circkey, radius, num_circ)
    # xy_test = generate_collocation_points_with_hole(testkey, radius, xlim, ylim, num_test)
    return {"coll": xy_coll, "rect": xy_rect, "circ": xy_circ}
    

def generate_rectangle(key: jax.random.PRNGKey,
                                 xlim: Sequence[float],
                                 ylim: Sequence[float],
                                 num_coll: int | Sequence[int],
                                 num_rect: int | Sequence[int]) -> dict[str, jax.Array | tuple[jax.Array]]:
    """
    Main function for generating necessary sample points for the square problem.

    The function generates 
    """


    key, rectkey, collkey, permkey = jax.random.split(key, 4)

    xy_coll = generate_collocation_points(collkey, xlim, ylim, num_coll)
    xy_coll = jax.random.permutation(permkey, xy_coll)
    xy_rect = generate_rectangle_points(rectkey, xlim, ylim, num_rect)
    return {"coll": xy_coll, "rect": xy_rect}


def resample(new_arr: jax.Array, new_loss: jax.Array, num_keep: int):
    """
    Utility function for choosing the points with
    highest loss.

    input:
        new_arr:
            The array to choose points from.
        
        new_loss:
            The losses to base the choice on.
        
        num_keep:
            The number of sampled points to keep.
        
    """

    num_keep = min(num_keep, new_loss.ravel().shape[0])
    idx = jnp.argpartition(new_loss.ravel(), kth=-num_keep)
    return new_arr[idx[-num_keep:]]


def resample_idx(new_arr: jax.Array, new_loss: jax.Array, num_throwaway: int):
    """
    Utility function for finding the indices of the points with the lowest loss

    input:
        new_arr:
            The array to choose points from.
        
        new_loss:
            The losses to base the choice on.
        
        num_throwaway:
            The number of sampled points to not keep.
        
    """

    num_throwaway = min(num_throwaway, new_loss.ravel().shape[0])
    idx = jnp.argpartition(new_loss.ravel(), kth=-num_throwaway)
    return idx[:num_throwaway]
















def numpy_collate(batch):
    return jtu.tree_map(np.asarray, torch.utils.data.default_collate(batch))


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
    