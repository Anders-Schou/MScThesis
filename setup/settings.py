from dataclasses import dataclass
from collections.abc import Callable
import pathlib

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
    base_dir: pathlib.Path
    figure_dir: pathlib.Path | None = None
    model_dir: pathlib.Path | None = None
    image_dir: pathlib.Path | None = None
    log_dir: pathlib.Path | None = None


@dataclass
class TrainingSettings(Settings):
    sampling: dict
    iterations: int = 1000
    optimizer: Callable = SupportedOptimizers.adam
    learning_rate: float = 1e-3
    batch_size: int | None = None
    decay_rate: float | None = None
    decay_steps: int | None = None
    transfer_learning: bool = False
    resampling: dict | None = None


@dataclass
class EvaluationSettings(Settings):
    sampling: dict
    error_metric: str = "L2-rel"
    transfer_learning: bool = False
    pass


@dataclass
class PlottingSettings(Settings):
    do_plots: bool = True
    overwrite: bool = False
    image_file_type: str = "pdf"
    pass


@dataclass
class MLPSettings(Settings):
    input_dim: int = 1
    output_dim: int = 1
    hidden_dims: int | list[int] = 32
    activation: Callable | list[Callable] = SupportedActivations.tanh
    initialization: Callable | list[Callable] = nn.initializers.glorot_normal