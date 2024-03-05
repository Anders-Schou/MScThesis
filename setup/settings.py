from dataclasses import dataclass
from collections.abc import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

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
    activation: str | list[str] | Callable | list[Callable] = "tanh"
    initialization: str | list[str] | Callable | list[Callable] = "glorot_normal"


@dataclass
class TrainingSettings(Settings):
    iterations: int = 1000
    optimizer: str | Callable = "adam"
    learning_rate: float = 1e-3
    batch_size: int | None = None
    decay_rate: float | None = None
    decay_steps: int | None = None
    transfer_learning: bool = False


@dataclass
class SupportedActivations:
    tanh: Callable = nn.tanh
    sigmoid: Callable = nn.sigmoid
    silu: Callable = nn.silu
    swish: Callable = nn.silu


@dataclass
class SupportedOptimizers:
    adam: Callable = optax.adam


@dataclass
class SupportedEquations:
    # laplace: Callable = equations.laplace
    # poisson: Callable = equations.poisson
    # biharmonic: Callable = equations.biharmonic
    #
    # (file 'equations' does not exists at the moment)
    pass


@dataclass
class SupportedSamplingDistributions:
    uniform: Callable = jax.random.uniform


class Model:
    def __init__(self, settings: dict):
        self.parse_settings(settings)

    def parse_settings(self, settings: dict):
        self.dir = DirectorySettings(**settings["IO"])
        self.train = settings["run"]["specification"]

