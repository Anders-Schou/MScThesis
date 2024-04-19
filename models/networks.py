from dataclasses import dataclass
from collections.abc import Callable, Sequence

import jax
import flax.linen as nn

from setup.parsers import (
    parse_MLP_settings,
    parse_ResNetBlock_settings
)


class MLP(nn.Module):
    """
    A classic multilayer perceptron, also known as
    a feed-forward neural network (FNN).
    
    Args:
        name: ................. The name of the network.
        input_dim: ............ The input dimension of the network.
        output_dim: ........... The output dimension of the network.
        hidden_dims: .......... A sequence containing the number of
                                hidden neurons in each layer.
        activation: ........... A sequence containing the activation
                                functions for each layer. Must be
                                the same length as hidden_dims.
        initialization: ....... A sequence of the initializations for
                                the weight matrices. Length must
                                be len(hidden_dims) + 1.
    """
    
    name: str
    input_dim: int
    output_dim: int
    hidden_dims: Sequence[int]
    activation: Sequence[Callable]
    initialization: Sequence[Callable]

    @nn.compact
    def __call__(self, input, transform = None):
        
        if transform is not None:
            x = transform(input)
        else:
            x = input
        
        for i, feats in enumerate(self.hidden_dims):
            x = nn.Dense(features=feats,
                         kernel_init=self.initialization[i](),
                         name=f"{self.name}_linear{i}")(x)
            x = self.activation[i](x)
        
        x = nn.Dense(features=self.output_dim,
                     kernel_init=self.initialization[-1](),
                     name=f"{self.name}_linear_output")(x)
        
        return x

    def __str__(self):
        s = f"\n"
        s += f"name:             {self.name}\n"
        s += f"input_dim:        {self.input_dim}\n"
        s += f"output_dim:       {self.output_dim}\n"
        s += f"hidden_dims:      {self.hidden_dims}\n"
        s += f"activation:       {[f.__name__ for f in self.activation]}\n"
        s += f"initialization:   {[f.__name__ for f in self.initialization]}\n"
        s += f"\n"
        return s


class ResNetBlock(nn.Module):
    """
    A ResNet block. This block can be combined with other models.

    This module consists of two paths: A linear one that is just
    an MLP, and a shortcut that adds the input to the MLP output.
    ```
            o---- shortcut ----o
            |                  |
        x --|                  + ---> y
            |                  |
            o------ MLP  ------o
    ```
    The shortcut is an identity mapping if `dim(x) == dim(y)`.
    Else, it is a linear mapping.

    Args:
        name: ................. The name of the network.
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
        pre_act: .............. Activation function to be applied
                                to the MLP input before passing it
                                through the MLP.
        post_act: ............. Activation function to be applied
                                to the output, i.e. the sum of the
                                MLP output and the shortcut output.
        shortcut_init: ........ Initialization for the shortcut if
                                dim(x) != dim(y). If not specified,
                                the first init function in the MLP
                                will be used.
    """
    
    name: str
    input_dim: int
    output_dim: int
    hidden_dims: Sequence[int]
    activation: Sequence[Callable]
    initialization: Sequence[Callable]
    pre_act: Callable | None = None
    post_act: Callable | None = None
    shortcut_init: Callable | None = None

    def __post_init__(self) -> None:
        if self.input_dim != self.output_dim:
            if self.shortcut_init is None:
                self.shortcut_init = self.initialization[0]
        else:
            self.shortcut_init = None
        
        super().__post_init__()
        return

    @nn.compact
    def __call__(self, input):
        
        # MLP input
        x = input
    
        # Shortcut input
        y = input
        # If dimensions match, use identity mapping:
        # Else use linear map between input and output
        if self.input_dim != self.output_dim:
            y = nn.Dense(features=self.output_dim, kernel_init=self.shortcut_init(), name=f"{self.name}_linear_shortcut")(y)

        if self.pre_act is not None:
            x = self.pre_act(x)

        for i, feats in enumerate(self.hidden_dims):
            x = nn.Dense(features=feats, kernel_init=self.initialization[i](), name=f"{self.name}_linear{i}")(x)
            x = self.activation[i](x)
        
        x = nn.Dense(features=self.output_dim, kernel_init=self.initialization[-1](), name=f"{self.name}_linear_output")(x)
        
        # Add the two flows together
        out = x+y
        if self.post_act is not None:
            out = self.post_act(out)
        return out

    def __str__(self):
        s = f"\n"
        s += f"name:             {self.name}\n"
        s += f"input_dim:        {self.input_dim}\n"
        s += f"output_dim:       {self.output_dim}\n"
        s += f"hidden_dims:      {self.hidden_dims}\n"
        s += f"activation:       {[f.__name__ for f in self.activation]}\n"
        s += f"initialization:   {[f.__name__ for f in self.initialization]}\n"
        s += f"pre_activation:   {self.pre_act.__name__ if self.pre_act is not None else str(None)}\n"
        s += f"post_activation:  {self.post_act.__name__ if self.post_act is not None else str(None)}\n"
        if self.shortcut_init is not None:
            print("IF", self.shortcut_init)
        
        s_init = self.shortcut_init.__name__ if self.shortcut_init is not None else self.shortcut_init
        s += f"shortcut_init:    {s_init}\n"
        s += f"\n"
        return s


def netmap(model: Callable[..., jax.Array], **kwargs) -> Callable[..., jax.Array]:
    """
    Applies the jax.vmap function with in_axes=(None, 0).
    """
    return jax.vmap(model, in_axes=(None, 0), **kwargs)


def setup_network(network_settings: dict[str, str | dict]) -> MLP | ResNetBlock:
    """
    Given a dict of network settings, returns a network instance.
    """
    
    arch = network_settings["architecture"].lower()
    
    match arch:
        case "mlp":
            parsed_settings = parse_MLP_settings(network_settings["specifications"])
            return MLP(**parsed_settings)

        case "resnet":
            parsed_settings = parse_ResNetBlock_settings(network_settings["specifications"])
            return ResNetBlock(**parsed_settings)

        case _:
            raise ValueError(f"Invalid network architecture: '{arch}'.")