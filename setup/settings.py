from dataclasses import dataclass
from collections.abc import Callable

import jax.numpy as jnp
import flax.linen as nn

class Settings:
    pass


def settings2dict(settings: Settings) -> dict:
    return settings.__dict__


class SettingsInterpretationException(Exception):
    pass


class SettingsNotSupported(Exception):
    pass


@dataclass
class DirectorySettings:
    figure_dir: str
    model_dir: str


@dataclass
class PINNSettings:
    pass


@dataclass
class MLPSettings(Settings):
    input_dim: int = 1
    output_dim: int = 1
    hidden_dims: int | list[int] = 32
    activation: str | list[str] = "tanh"
    initialization: str | list[str] = "glorot_normal"


@dataclass
class SupportedActivations:
    tanh: Callable = nn.tanh
    sigmoid: Callable = nn.sigmoid
    silu: Callable = nn.silu
    swish: Callable = nn.silu
