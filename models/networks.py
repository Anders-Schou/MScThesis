from dataclasses import dataclass
import jax
import jax.numpy as jnp
import flax.linen as nn

@dataclass
class MLP(nn.Module):
    num_neurons_per_layer: list[int]
    activation: callable
    weight_init: callable = nn.initializers.glorot_uniform()
    
    @nn.compact
    def __call__(self, input, transform=None):
        
        if transform is not None:
            x = transform(input)
        else:
            x = input

        for i, feats in enumerate(self.num_neurons_per_layer[1:-1]):
            x = nn.Dense(features=feats, kernel_init=self.weight_init, name=f"MLP_linear{i}")(x)
            x = self.activation(x)

        x = nn.Dense(features=self.num_neurons_per_layer[-1], kernel_init=self.weight_init, name=f"MLP_linear_output")(x)

        return x