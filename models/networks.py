from dataclasses import dataclass
from collections.abc import Callable

import jax
import flax.linen as nn

from setup.parsers import parse_MLP_settings, parse_training_settings


def netmap(model: Callable, **kwargs) -> Callable:
    """
    Applies the jax.vmap function with in_axes=(None, 0).
    """
    return jax.vmap(model, in_axes=(None, 0), **kwargs)


def setup_run_settings(run_settings: dict, run_type: str):
    run_settings = run_settings.copy()
    if run_type == "train":
        train_settings = parse_training_settings(run_settings["train"])
        return train_settings
    if run_type == "eval":
        raise NotImplementedError("Evaluation run is not implemented yet.")
    raise ValueError(f"Invalid run type: '{run_type}'.")


def setup_network(network_settings: dict):
    arch = network_settings["architecture"].lower()
    if  arch == "mlp":
        parsed_settings = parse_MLP_settings(network_settings["specifications"])
        return MLP(**parsed_settings)
    raise ValueError(f"Invalid network architecture: '{arch}'.")


@dataclass
class MLP(nn.Module):
    """
    A classic multilayer perceptron, also known as
    a feed-forward neural network (FNN).
    
    Args:
        input_dim: ............ The input dimension of the network.
        output_dim: ........... The output dimension of the network.
        hidden_dims: .......... A list containing the number of
                                hidden neurons in each layer.
        activation: ........... A list containing the activation
                                functions for each layer. Must be
                                the same length as hidden_dims.
        initialization: ....... A list of the initializations for
                                the weight matrices. Length must
                                be len(hidden_dims) + 1.
    """
     
    input_dim: int
    output_dim: int
    hidden_dims: list[int]
    activation: list[Callable]
    initialization: list[Callable]
    
    @nn.compact
    def __call__(self, input, transform = None):
        if transform is not None:
            x = transform(input)
        else:
            x = input
        for i, feats in enumerate(self.hidden_dims):
            x = nn.Dense(features=feats, kernel_init=self.initialization[i](), name=f"MLP_linear{i}")(x)
            x = self.activation[i](x)
        x = nn.Dense(features=self.output_dim, kernel_init=self.initialization[-1](), name=f"MLP_linear_output")(x)
        return x

    def __str__(self):
        s = f"\n"
        s += f"input_dim:        {self.input_dim}\n"
        s += f"output_dim:       {self.output_dim}\n"
        s += f"hidden_dims:      {self.hidden_dims}\n"
        s += f"activation:       {[f.__name__ for f in self.activation]}\n"
        s += f"initialization:   {[f.__name__ for f in self.activation]}\n"
        s += f"\n"
        return s