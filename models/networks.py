from dataclasses import dataclass

import flax.linen as nn

@dataclass
class MLP(nn.Module):
    layers: list[int]
    activations: list[str]
    weight_init: callable

    @nn.compact
    def __call__(self, input, transform=None):
        
        if transform is not None:
            x = transform(input)
        else:
            x = input

        for i, feats in enumerate(self.layers[:-1]):
            x = nn.Dense(features=feats, kernel_init=self.weight_init, name=f"MLP_linear{i}")(x)
            x = self.activations[i](x)

        x = nn.Dense(features=feats, kernel_init=self.weight_init, name=f"MLP_linear_output")(x)

        return x