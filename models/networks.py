from dataclasses import dataclass
from collections.abc import Callable

import flax.linen as nn

from setup.parsers import parse_MLP_settings, parse_training_settings


def setup_run(run_settings: dict):
    run_type = run_settings["type"].lower()
    if run_type == "train":
        run_settings["specifications"] = parse_training_settings(run_settings["specifications"])
        return run_settings
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
