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
class SupportedActivations:
    tanh: Callable = nn.tanh
    sigmoid: Callable = nn.sigmoid
    silu: Callable = nn.silu
    swish: Callable = nn.silu


@dataclass
class SupportedOptimizers:
    adam: Callable = optax.adam
    adamw: Callable = optax.adamw
    set_to_zero: Callable = optax.set_to_zero


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
class TrainingSettings:
    iterations: int = 1000
    optimizer: Callable = SupportedOptimizers.adam
    learning_rate: float = 1e-3
    batch_size: int | None = None
    decay_rate: float | None = None
    decay_steps: int | None = None
    transfer_learning: bool = False


@dataclass
class EvaluationSettings:
    error_metric: str = "L2-rel"
    pass


@dataclass
class PlottingSettings:
    do_plots: bool = True
    overwrite: bool = False
    image_file_type: str = "pdf"
    pass


@dataclass
class MLPSettings:
    input_dim: int = 1
    output_dim: int = 1
    hidden_dims: int | list[int] = 32
    activation: Callable | list[Callable] = SupportedActivations.tanh
    initialization: Callable | list[Callable] = nn.initializers.glorot_normal