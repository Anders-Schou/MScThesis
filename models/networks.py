from dataclasses import dataclass
from collections.abc import Callable

import jax.numpy as jnp
import flax.linen as nn

from utils.datastructures import MLPArch
from setup.parsers import parse_MLP_settings

def setup_network(network_settings: dict):
    arch = network_settings["architecture"].lower()
    if  arch == "mlp":
        parsed_settings = parse_MLP_settings(network_settings["specifications"])
        return MLP(**parsed_settings)
    raise ValueError(f"Invalid network architecture: '{arch}'")

@dataclass
class MLP(nn.Module):
    input_dim: int
    output_dim: int
    hidden_dims: list[int]
    activation: list[Callable]
    initialization: list[Callable]
    
    @nn.compact
    def __call__(self, input, transform=None):
        
        if transform is not None:
            x = transform(input)
        else:
            x = input

        for i, feats in enumerate(self.hidden_dims):
            x = nn.Dense(features=feats, kernel_init=self.initialization[i](), name=f"MLP_linear{i}")(x)
            x = self.activation[i](x)

        x = nn.Dense(features=self.output_dim, kernel_init=self.initialization[-1](), name=f"MLP_linear_output")(x)

        return x
    
