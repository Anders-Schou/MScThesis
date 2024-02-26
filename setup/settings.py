from dataclasses import dataclass
from collections.abc import Callable

import jax.numpy as jnp
import flax.linen as nn

class Settings:
    pass


def settings2dict(settings: Settings) -> dict:
    return settings.__dict__


class SettingsInterpretationError(Exception):
    pass


class SettingsNotSupportedError(Exception):
    pass


@dataclass
class DirectorySettings:
    figure_dir: str
    model_dir: str
    image_dir: str
    log_dir: str


@dataclass
class PINNSettings:
    network: object
    loss: Callable
    equation: str


@dataclass
class MLPSettings(Settings):
    input_dim: int = 1
    output_dim: int = 1
    hidden_dims: int | list[int] = 32
    activation: str | list[str] = "tanh"
    initialization: str | list[str] = "glorot_normal"


@dataclass
class TrainingSettings(Settings):
    iterations: int
    optimizer: str
    learning_rate: float
    batch_size: int


@dataclass
class SupportedActivations:
    tanh: Callable = nn.tanh
    sigmoid: Callable = nn.sigmoid
    silu: Callable = nn.silu
    swish: Callable = nn.silu


@dataclass
class SupportedEquations:
    # laplace: Callable = equations.laplace
    # poisson: Callable = equations.poisson
    # biharmonic: Callable = equations.biharmonic
    #
    # (file 'equations' does not exists at the moment)
    pass


class Model:
    def __init__(self, settings: dict):
        self.parse_settings(settings)

    def parse_settings(self, settings: dict):
        self.dir = DirectorySettings(**settings["IO"])
        self.train = settings["run"]["specification"]

